"""Head RGB-D camera: versioned calibration, live Orbbec source, recorded-episode source (spec sections 7, 20, 22).

The head camera is an **observation + human-hand-reconstruction** device. It has no IMU, no 6-DoF tracking role and it
never commands the arm (invariant, spec sections 1/32) — nothing in this module produces a pose for anything but the
operator's own hand.

Depth must correspond to colour before a landmark's depth may be read. The Orbbec SDK does that in hardware/firmware
(`AlignFilter(align_to_stream=COLOR_STREAM)`), which is why `aligned: true` is a recorded property of the calibration
and not an assumption: `HeadRgbdCalibration.require_aligned()` is called by the provider before any depth lookup.

Calibration lives in `configs/calibration/head_rgbd_vNNN.yaml` (same scheme as every other calibration here: latest
wins, never overwritten, the episode records which version it ran on)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml
from handumi_collector.pose.calibration import load_versioned, save_versioned
from .depth import CameraIntrinsics, depth_image_to_m

PREFIX = "head_rgbd"


@dataclass
class HeadRgbdCalibration:
    color: CameraIntrinsics
    depth: CameraIntrinsics
    depth_scale_m: float                       # raw depth unit -> metres (Orbbec Gemini 336: 1 mm -> 0.001)
    aligned: bool                              # depth frame is registered to the colour frame
    fps: float = 30.0
    T_color_depth: np.ndarray = field(default_factory=lambda: np.eye(4))   # colour <- depth, identity when aligned
    model: str = ""
    serial: str = ""
    version: str | None = None
    notes: str = ""

    def require_aligned(self) -> None:
        if not self.aligned:
            raise RuntimeError("head RGB-D calibration says depth is NOT aligned to colour: reading depth[u,v] at a "
                               "colour landmark would be wrong. Enable the SDK align filter or supply T_color_depth "
                               "and reproject explicitly.")

    def depth_to_m(self, raw: np.ndarray) -> np.ndarray:
        return depth_image_to_m(raw, self.depth_scale_m)

    def to_dict(self) -> dict:
        return dict(color=self.color.to_dict(), depth=self.depth.to_dict(), depth_scale_m=float(self.depth_scale_m),
                    aligned=bool(self.aligned), fps=float(self.fps), T_color_depth=np.asarray(self.T_color_depth).tolist(),
                    model=self.model, serial=self.serial, notes=self.notes)

    @classmethod
    def from_dict(cls, d: dict, *, version: str | None = None) -> "HeadRgbdCalibration":
        return cls(CameraIntrinsics.from_dict(d["color"]), CameraIntrinsics.from_dict(d["depth"]),
                   float(d["depth_scale_m"]), bool(d["aligned"]), float(d.get("fps", 30.0)),
                   np.asarray(d.get("T_color_depth", np.eye(4)), np.float64).reshape(4, 4),
                   d.get("model", ""), d.get("serial", ""), version, d.get("notes", ""))

    @classmethod
    def load(cls, which: str = "latest", cal_dir: Path | None = None) -> "HeadRgbdCalibration | None":
        name, d = load_versioned(PREFIX, which, cal_dir)
        return None if not d else cls.from_dict(d, version=name)

    def save(self, cal_dir: Path | None = None) -> Path:
        return save_versioned(PREFIX, self.to_dict(), cal_dir=cal_dir)

    @classmethod
    def from_orbbec_recorder_yaml(cls, path: str | Path, *, model: str = "Orbbec Gemini 336", fps: float = 30.0) -> "HeadRgbdCalibration":
        """Import the `intrinsics.yaml` written by ~/orbbec_recorder.py (colour intrinsics, depth aligned to colour)."""
        d = yaml.safe_load(Path(path).read_text())
        w, h = d["resolution"]
        K = CameraIntrinsics.from_dict(d["color"], size=(w, h))
        return cls(K, K, float(d.get("depth_scale_mm", 1.0)) / 1000.0, True, fps, np.eye(4), model,
                   notes=str(d.get("note", "")) + f" (imported from {Path(path)})")


@dataclass
class ColorExposure:
    """Fixed colour exposure for a LIVE capture — an experimental control, not a tracking feature.

    Auto-exposure changes the shutter between a stationary and a fast-motion segment, which makes "did tracking
    degrade because the hand moved, or because the frame blurred?" unanswerable. Measured on a Gemini 336
    (2026-09-11): **reopening the pipeline turns AE back on**, even though the value it re-converges to is the same
    in unchanged lighting. So a pre-flight instrument that wants the same camera mode as the take has to set it
    itself; it is not inherited from whatever ran before.

    All three fields default to None = leave the device exactly as it is, so an existing capture path is unchanged
    unless it opts in. `~/orbbec_recorder.py` carries the second copy of these property writes because it runs in
    its own venv and cannot import this package; it additionally supports freezing at the auto-settled value, which
    is how the number below gets chosen in the first place."""
    auto: bool | None = None
    exposure: int | None = None           # device units, whatever the device reports (never assume a time unit)
    gain: int | None = None               # settable only once AE is off; there is no separate auto-gain property

    @property
    def requested(self) -> bool:
        return not (self.auto is None and self.exposure is None and self.gain is None)


def apply_color_exposure(device, cfg: "ColorExposure | None") -> dict:
    """Apply `cfg` and return the ACTUAL values read back from the device (never the requested ones)."""
    from pyorbbecsdk import OBPermissionType, OBPropertyID
    AE, EXP, GAIN = (OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
                     OBPropertyID.OB_PROP_COLOR_GAIN_INT)

    def rw(prop):
        try: return bool(device.is_property_supported(prop, OBPermissionType.PERMISSION_READ_WRITE))
        except Exception: return False

    def read():
        out = {}
        try: out["color_auto_exposure"] = bool(device.get_bool_property(AE))
        except Exception: out["color_auto_exposure"] = None
        for k, prop in (("color_exposure", EXP), ("color_gain", GAIN)):
            try: out[k] = int(device.get_int_property(prop))
            except Exception: out[k] = None
        return out

    if cfg is None or not cfg.requested:
        return dict(color_control="untouched", **read())
    if cfg.auto is None:
        raise ValueError("set ColorExposure.auto explicitly: with AE on the device ignores an exposure value")
    if not rw(AE):
        raise RuntimeError("device does not expose COLOR_AUTO_EXPOSURE read/write; cannot fix the exposure")
    if cfg.auto:
        if cfg.exposure is not None or cfg.gain is not None:
            raise ValueError("auto=True cannot be combined with an explicit exposure/gain")
        device.set_bool_property(AE, True)
        control = "auto"
    else:
        device.set_bool_property(AE, False)
        if cfg.exposure is not None: device.set_int_property(EXP, int(cfg.exposure))
        if cfg.gain is not None and rw(GAIN): device.set_int_property(GAIN, int(cfg.gain))
        control = "manual"
    time.sleep(0.4)
    return dict(color_control=control, **read())


@dataclass
class RgbdFrame:
    timestamp_ns: int                    # host monotonic at capture — the teleop clock
    color_bgr: np.ndarray                # (H,W,3) uint8, UNMIRRORED sensor orientation
    depth_m: np.ndarray                  # (H,W) float metres, NaN where the sensor returned nothing
    calib: HeadRgbdCalibration
    source_timestamp_ns: int | None = None   # device clock
    frame_index: int = -1


class RecordedRgbdSource:
    """Replay a `datasets/HumanRGBD_v1/episode_NNNNNN` directory (colour jpg + aligned 16-bit depth png).

    This is what makes A1/A2 runnable and repeatable without the camera plugged in — and what lets the B0/B1
    comparison of spec section 30 run on identical frames instead of two separate live takes."""

    def __init__(self, episode_dir: str | Path, calib: HeadRgbdCalibration | None = None) -> None:
        import cv2
        self._cv2 = cv2
        self.dir = Path(episode_dir)
        self.color = sorted((self.dir / "color").glob("*.jpg"))
        self.depth = sorted((self.dir / "depth").glob("*.png"))
        if not self.color: raise FileNotFoundError(f"no colour frames in {self.dir}/color")
        if len(self.color) != len(self.depth): raise ValueError(f"{len(self.color)} colour vs {len(self.depth)} depth frames in {self.dir}")
        if calib is None:
            p = self.dir / "intrinsics.yaml"
            if not p.exists(): p = self.dir.parent / "_calibration" / "intrinsics.yaml"
            if not p.exists(): raise FileNotFoundError(f"no intrinsics.yaml for {self.dir} (nor in _calibration/)")
            calib = HeadRgbdCalibration.from_orbbec_recorder_yaml(p)
        self.calib = calib
        self.timestamps_ms = self._load_timestamps()

    def _load_timestamps(self) -> list[float] | None:
        p = self.dir / "timestamps.csv"
        if not p.exists(): return None
        rows = [l.split(",") for l in p.read_text().strip().splitlines()[1:]]
        return [float(r[1]) for r in rows] if rows else None

    def __len__(self) -> int: return len(self.color)

    def frame(self, i: int, *, t_ns: int | None = None) -> RgbdFrame:
        img = self._cv2.imread(str(self.color[i]))
        raw = self._cv2.imread(str(self.depth[i]), self._cv2.IMREAD_UNCHANGED)
        if img is None or raw is None: raise IOError(f"unreadable frame {i} in {self.dir}")
        src = None if self.timestamps_ms is None else int(self.timestamps_ms[i] * 1e6)
        # replayed episodes get a synthetic 1/fps host clock so latency/rate maths is exercised the same way
        host = int(i * 1e9 / self.calib.fps) if t_ns is None else int(t_ns)
        return RgbdFrame(host, img, self.calib.depth_to_m(raw), self.calib, src, i)

    def __iter__(self):
        for i in range(len(self)): yield self.frame(i)


class OrbbecHeadCamera:
    """Live Orbbec Gemini 336 colour + depth-aligned-to-colour (pyorbbecsdk, already in this venv).

    macOS note: the SDK needs the UVC device un-claimed; `~/orbbec_recorder.py` runs under sudo for that reason and
    the same applies here. The camera is opened lazily so importing this module never touches hardware."""

    def __init__(self, *, width: int = 848, height: int = 480, fps: int = 30, calib: HeadRgbdCalibration | None = None,
                 exposure: ColorExposure | None = None) -> None:
        self.width, self.height, self.fps = width, height, fps
        self._pipe = self._align = self._cp = None
        self.calib = calib
        self.exposure = exposure                  # None = device untouched (the pre-2026-09-11 behaviour)
        self.applied_exposure: dict = {}

    def open(self) -> "OrbbecHeadCamera":
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, AlignFilter, OBStreamType
        import cv2
        self._cv2 = cv2
        pipe, cfg = Pipeline(), Config()
        cfg.enable_stream(pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(self.width, self.height, OBFormat.UNKNOWN_FORMAT, self.fps))
        cp = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(self.width, self.height, OBFormat.MJPG, self.fps)
        cfg.enable_stream(cp)
        pipe.start(cfg)
        self._pipe, self._align, self._cp = pipe, AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM), cp
        # Warm the stream before touching it. Two reasons, both measured on a Gemini 336 (2026-09-11): the align
        # filter drops framesets until depth and colour are both running, and auto-exposure needs time to settle
        # before "freeze at the value it settled on" means anything. Skipping this raced `_calib_from_device()` into
        # an AttributeError on a cold open.
        settle = 1.5 if (self.exposure is not None and self.exposure.requested) else 0.5
        for _ in range(int(settle * self.fps)): pipe.wait_for_frames(200)
        self.applied_exposure = apply_color_exposure(pipe.get_device(), self.exposure)
        if self.calib is None: self.calib = self._calib_from_device()
        return self

    def _calib_from_device(self) -> HeadRgbdCalibration:
        intr = self._cp.get_intrinsic()
        K = CameraIntrinsics(intr.fx, intr.fy, intr.cx, intr.cy, self._cp.get_width(), self._cp.get_height())
        scale_mm = 1.0
        for _ in range(10):                    # the first aligned frameset after start is often dropped, not late
            fs = self._pipe.wait_for_frames(2000)
            if fs is None: continue
            fs = self._align.process(fs)
            if fs is None: continue            # align returns None until depth and colour are both flowing
            fs = fs.as_frame_set() if hasattr(fs, "as_frame_set") else fs
            df = None if fs is None else fs.get_depth_frame()
            if df is not None:
                scale_mm = float(df.get_depth_scale()); break
        return HeadRgbdCalibration(K, K, scale_mm / 1000.0, True, float(self.fps), np.eye(4), "Orbbec Gemini 336",
                                   notes="intrinsics read from the device; depth aligned to colour by AlignFilter")

    def read(self, timeout_ms: int = 500) -> RgbdFrame | None:
        fs = self._pipe.wait_for_frames(timeout_ms)
        if fs is None: return None
        fs = self._align.process(fs)
        if fs is None: return None          # the align filter drops framesets too; ~/orbbec_recorder.py guards this
        fs = fs.as_frame_set() if hasattr(fs, "as_frame_set") else fs
        if fs is None: return None
        cf, df = fs.get_color_frame(), fs.get_depth_frame()
        if not (cf and df): return None
        img = self._cv2.imdecode(np.frombuffer(cf.get_data(), dtype=np.uint8), self._cv2.IMREAD_COLOR)
        if img is None: return None
        raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
        return RgbdFrame(time.monotonic_ns(), img, self.calib.depth_to_m(raw), self.calib, int(cf.get_timestamp_us() * 1000))

    def close(self) -> None:
        if self._pipe is not None: self._pipe.stop(); self._pipe = None
