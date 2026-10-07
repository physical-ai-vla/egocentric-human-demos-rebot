"""RGB-D episode reader for the depth-based HandUMI rigid-body tracker.

Read-only over raw. Three layouts are understood, all yielding the same `RgbdFrame`:

    recorder   <episode>/<stream>.mp4 + <stream>_depth/%06d.png + <stream>_depth/intrinsics.json + sensors.mcap
               (written by collector/recorder.py for a camera with role `aux_depth`)
    flat       <dir>/color/*.{jpg,png} + <dir>/depth/*.png + intrinsics.{yaml,json} [+ timestamps.csv]
               (datasets/HumanRGBD_v1 and the portable packet written by `--export-packet`)

Depth is uint16 in the sensor's own unit; `CameraIntrinsics.depth_unit_m` converts to metres and is ALWAYS recorded
next to any derived output — nothing downstream is allowed to assume millimetres."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import yaml

DEPTH_EXT = ".png"
COLOR_EXTS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics of the stream the depth map is expressed in (Orbbec aligns depth to colour -> colour intrinsics)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_unit_m: float = 1e-3          # metres per raw depth unit (Orbbec depth_scale mm/unit / 1000)
    aligned_to_rgb: bool = True
    source: str = ""

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], np.float64)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        return cls(**{k: d[k] for k in ("fx", "fy", "cx", "cy", "width", "height") if k in d},
                   **{k: d[k] for k in ("depth_unit_m", "aligned_to_rgb", "source") if k in d})

    def scaled(self, factor: float) -> "CameraIntrinsics":
        """Intrinsics after resizing the image by `factor` (0.5 = half size)."""
        return CameraIntrinsics(self.fx * factor, self.fy * factor, (self.cx + 0.5) * factor - 0.5, (self.cy + 0.5) * factor - 0.5,
                                int(round(self.width * factor)), int(round(self.height * factor)),
                                self.depth_unit_m, self.aligned_to_rgb, self.source)


@dataclass
class RgbdFrame:
    index: int                 # index within the episode (0-based, dense)
    t_ns: int                  # capture timestamp (host monotonic clock; synthesised from fps for layouts without one)
    rgb: np.ndarray            # HxWx3 BGR uint8
    depth_raw: np.ndarray      # HxW uint16, sensor units
    depth_m: np.ndarray        # HxW float32 metres, 0 where invalid


def backproject(depth_m: np.ndarray, K: np.ndarray, *, mask: np.ndarray | None = None,
                z_min: float = 0.05, z_max: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Depth map -> (N,3) camera-frame points and their (N,2) int pixel coordinates (v,u). Invalid depth is dropped."""
    h, w = depth_m.shape
    m = (depth_m > z_min) & (depth_m < z_max)
    if mask is not None:
        m &= mask.astype(bool)
    v, u = np.nonzero(m)
    if len(v) == 0:
        return np.zeros((0, 3), np.float64), np.zeros((0, 2), np.int64)
    z = depth_m[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1), np.stack([v, u], axis=1)


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N,3) camera points -> (N,2) pixel uv. Points behind the camera come back as NaN."""
    P = np.asarray(points_cam, np.float64).reshape(-1, 3)
    z = P[:, 2]
    uv = np.full((len(P), 2), np.nan)
    ok = z > 1e-6
    uv[ok, 0] = K[0, 0] * P[ok, 0] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * P[ok, 1] / z[ok] + K[1, 2]
    return uv


class RgbdEpisode:
    """Uniform reader over the raw layouts above. Never writes into the episode directory."""

    def __init__(self, path: Path, layout: str, intrinsics: CameraIntrinsics, t_ns: np.ndarray, *, stream: str | None = None,
                 color_files: list[Path] | None = None, depth_files: list[Path] | None = None, meta: dict | None = None) -> None:
        self.path = Path(path)
        self.layout = layout
        self.intrinsics = intrinsics
        self.t_ns = np.asarray(t_ns, np.int64)
        self.stream = stream
        self._color_files = color_files
        self._depth_files = depth_files
        self._video_frames: list[int] = []       # recorder layout only: the .mp4 frame each depth PNG belongs to
        self.meta = meta or {}

    def __len__(self) -> int:
        return int(len(self.t_ns))

    @property
    def n_frames(self) -> int:
        return len(self)

    @property
    def fps(self) -> float:
        if len(self.t_ns) < 2:
            return 0.0
        dt = np.diff(self.t_ns.astype(np.float64)) / 1e9
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else 0.0

    def describe(self) -> dict:
        return dict(path=str(self.path), layout=self.layout, stream=self.stream, n_frames=self.n_frames,
                    fps_measured=round(self.fps, 3), intrinsics=self.intrinsics.to_dict())

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path, *, stream: str | None = None, fps_fallback: float = 30.0) -> "RgbdEpisode":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if (path / "color").is_dir() and (path / "depth").is_dir():
            return cls._load_flat(path, fps_fallback)
        if (path / "episode_meta.json").exists() or (path / "sensors.mcap").exists():
            return cls._load_recorder(path, stream)
        raise ValueError(f"{path}: not an RGB-D episode (expected color/+depth/ or a recorder episode with *_depth/)")

    @classmethod
    def _load_flat(cls, path: Path, fps_fallback: float) -> "RgbdEpisode":
        color = sorted(p for p in (path / "color").iterdir() if p.suffix.lower() in COLOR_EXTS)
        depth = sorted(p for p in (path / "depth").iterdir() if p.suffix.lower() == DEPTH_EXT)
        if not color or not depth:
            raise ValueError(f"{path}: color/ or depth/ is empty")
        n = min(len(color), len(depth))
        intr = _load_intrinsics_file(path)
        t_ns = _load_timestamps(path, n, fps_fallback)
        meta = {}
        mp = path / "packet_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        return cls(path, "flat", intr, t_ns, color_files=color[:n], depth_files=depth[:n], meta=meta)

    @classmethod
    def _load_recorder(cls, path: Path, stream: str | None) -> "RgbdEpisode":
        depth_dirs = sorted(p for p in path.iterdir() if p.is_dir() and p.name.endswith("_depth"))
        if not depth_dirs:
            raise ValueError(f"{path}: recorder episode without a *_depth/ directory — was the RGB-D camera connected?")
        if stream is None:
            if len(depth_dirs) > 1:
                raise ValueError(f"{path}: several depth streams {[d.name for d in depth_dirs]} — pass --stream")
            stream = depth_dirs[0].name[: -len("_depth")]
        ddir = path / f"{stream}_depth"
        if not ddir.is_dir():
            raise ValueError(f"{path}: no {ddir.name}/")
        intr = _load_intrinsics_file(ddir)
        meta_p = path / "episode_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        from .episode_io import RawEpisode
        raw = RawEpisode.load(path)
        if stream not in raw.frames:
            raise ValueError(f"{path}: stream {stream!r} has no frame_meta in sensors.mcap")
        fm = raw.frames[stream]
        # depth PNGs are named by the VIDEO frame index the recorder wrote (see recorder._pump)
        keep = [(int(vf), int(t)) for vf, t in zip(fm.video_frame, fm.capture_ns) if (ddir / f"{int(vf):06d}.png").exists()]
        if not keep:
            raise ValueError(f"{ddir}: no depth PNG matches any frame_meta.video_frame")
        if intr.width == 0 or intr.height == 0:
            import cv2
            d0 = cv2.imread(str(ddir / f"{keep[0][0]:06d}.png"), cv2.IMREAD_UNCHANGED)
            intr = CameraIntrinsics(intr.fx, intr.fy, intr.cx, intr.cy, int(d0.shape[1]), int(d0.shape[0]),
                                    intr.depth_unit_m, intr.aligned_to_rgb, intr.source)
        ep = cls(path, "recorder", intr, np.array([t for _vf, t in keep], np.int64), stream=stream,
                 depth_files=[ddir / f"{vf:06d}.png" for vf, _t in keep], meta=meta)
        ep._video_frames = [vf for vf, _t in keep]
        return ep

    # ------------------------------------------------------------------ frames
    def iter_frames(self, start: int = 0, stop: int | None = None, step: int = 1):
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        if self.layout == "recorder":
            yield from self._iter_recorder(start, stop, step)
        else:
            yield from self._iter_flat(start, stop, step)

    def _read_depth(self, p: Path) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise RuntimeError(f"cannot read depth {p}")
        if d.ndim != 2:
            raise RuntimeError(f"{p}: depth must be single-channel uint16, got shape {d.shape}")
        return d.astype(np.uint16), (d.astype(np.float32) * np.float32(self.intrinsics.depth_unit_m))

    def _iter_flat(self, start: int, stop: int, step: int):
        import cv2
        for i in range(start, stop, step):
            rgb = cv2.imread(str(self._color_files[i]), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"cannot read colour {self._color_files[i]}")
            raw, m = self._read_depth(self._depth_files[i])
            yield RgbdFrame(i, int(self.t_ns[i]), rgb, raw, m)

    def _iter_recorder(self, start: int, stop: int, step: int):
        import av
        want = {int(self._video_frames[i]): i for i in range(start, stop, step)}
        if not want:
            return
        with av.open(str(self.path / f"{self.stream}.mp4")) as c:
            for vf, frame in enumerate(c.decode(video=0)):
                i = want.get(vf)
                if i is None:
                    continue
                rgb = frame.to_ndarray(format="bgr24")
                raw, m = self._read_depth(self._depth_files[i])
                yield RgbdFrame(i, int(self.t_ns[i]), rgb, raw, m)


def _load_intrinsics_file(d: Path) -> CameraIntrinsics:
    """Accepts the recorder's intrinsics.json ({intrinsics:{fx..}, depth_scale}) and the flat intrinsics.yaml
    ({color:{fx..}, resolution:[w,h], depth_scale_mm}). Missing file is a hard error — a guessed focal length is a silent
    metric-scale error in every pose downstream."""
    for name in ("intrinsics.json", "intrinsics.yaml", "intrinsics.yml"):
        p = d / name
        if not p.exists():
            continue
        raw = json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())
        blk = raw.get("intrinsics") or raw.get("color") or raw
        w, h = 0, 0
        if "resolution" in raw:
            w, h = int(raw["resolution"][0]), int(raw["resolution"][1])
        w = int(blk.get("width", raw.get("width", w)))
        h = int(blk.get("height", raw.get("height", h)))
        if "depth_unit_m" in raw:
            unit = float(raw["depth_unit_m"])
        elif "depth_scale_mm" in raw:
            unit = float(raw["depth_scale_mm"]) * 1e-3
        elif raw.get("depth_scale") is not None:
            unit = float(raw["depth_scale"]) * 1e-3      # Orbbec depth_scale is mm per unit
        else:
            unit = 1e-3
        return CameraIntrinsics(float(blk["fx"]), float(blk["fy"]), float(blk["cx"]), float(blk["cy"]), w, h,
                                unit, bool(raw.get("aligned_to_rgb", True)), source=str(p))
    raise FileNotFoundError(f"{d}: no intrinsics.json / intrinsics.yaml — refusing to guess the camera model")


def _load_timestamps(path: Path, n: int, fps_fallback: float) -> np.ndarray:
    p = path / "timestamps.csv"
    if p.exists() and p.stat().st_size > 0:
        import csv
        rows = list(csv.DictReader(p.read_text().splitlines()))
        for key in ("t_ns", "capture_ns", "timestamp_ns"):
            if rows and key in rows[0]:
                t = np.array([int(float(r[key])) for r in rows[:n]], np.int64)
                if len(t) == n:
                    return t
    return (np.arange(n, dtype=np.int64) * int(1e9 / max(fps_fallback, 1e-6))).astype(np.int64)
