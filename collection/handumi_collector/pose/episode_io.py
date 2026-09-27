"""Raw episode reader + derived-output layout. Raw is read-only here; everything the pose pipeline produces goes to
<episode>/derived/pose_<backend>/ so any backend can be re-run without touching raw."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from ..collector.mcap_writer import read_messages


@dataclass
class ImuArrays:
    side: str
    seq: np.ndarray
    device_us: np.ndarray        # Teensy micros (device clock)
    host_ns: np.ndarray          # host receive (jittery, but on the camera clock)
    gyro: np.ndarray             # (N,3) rad/s raw sensor axes (never transformed at rest)
    accel: np.ndarray            # (N,3) m/s^2
    temperature_c: np.ndarray

    def __len__(self) -> int: return len(self.seq)


@dataclass
class GripArrays:
    side: str
    t_ns: np.ndarray
    normalized: np.ndarray       # 0 closed .. 1 open (human raw convention), NaN when uncalibrated
    raw_position: np.ndarray

    def __len__(self) -> int: return len(self.t_ns)


@dataclass
class FrameMeta:
    stream: str
    frame_index: np.ndarray
    video_frame: np.ndarray
    capture_ns: np.ndarray

    def __len__(self) -> int: return len(self.video_frame)


@dataclass
class RawEpisode:
    path: Path
    meta: dict
    events: list[dict]
    frames: dict[str, FrameMeta] = field(default_factory=dict)
    imu: dict[str, ImuArrays] = field(default_factory=dict)
    grip: dict[str, GripArrays] = field(default_factory=dict)
    sync: list[dict] = field(default_factory=list)

    @property
    def t_start_ns(self) -> int: return int(self.meta["t_start_monotonic_ns"])

    @property
    def t_stop_ns(self) -> int: return int(self.meta["t_stop_monotonic_ns"])

    def video_path(self, stream: str) -> Path: return self.path / f"{stream}.mp4"

    def protocol_events(self, kind: str) -> list[int]:
        """Timestamps (ns) of operator protocol marks: home_leave / home_return (optional; QA auto-detects when absent)."""
        return [int(e["t_ns"]) for e in self.events if e.get("kind") == kind]

    @classmethod
    def load(cls, path: str | Path) -> "RawEpisode":
        path = Path(path)
        if (path / ".incomplete").exists(): raise RuntimeError(f"{path.name}: incomplete raw episode (collector crashed) — not processable")
        meta = json.loads((path / "episode_meta.json").read_text())
        ev_p = path / "events.json"
        events = json.loads(ev_p.read_text()) if ev_p.exists() else []
        cols: dict[str, list[dict]] = {}
        for topic, _lt, m in read_messages(path / "sensors.mcap"):
            cols.setdefault(topic, []).append(m)
        ep = cls(path, meta, events)
        for topic, ms in cols.items():
            parts = topic.strip("/").split("/")
            if parts[-1] == "frame_meta":
                ep.frames[parts[0]] = FrameMeta(parts[0], np.array([m["frame_index"] for m in ms], np.int64),
                                                np.array([m["video_frame"] for m in ms], np.int64), np.array([m["capture_ns"] for m in ms], np.int64))
            elif parts[-1] == "imu":
                ep.imu[parts[0]] = ImuArrays(parts[0], np.array([m["seq"] for m in ms], np.int64), np.array([m["device_timestamp_us"] for m in ms], np.int64),
                                             np.array([m["host_receive_ns"] for m in ms], np.int64),
                                             np.array([[m["gx"], m["gy"], m["gz"]] for m in ms], np.float64),
                                             np.array([[m["ax"], m["ay"], m["az"]] for m in ms], np.float64),
                                             np.array([m.get("temperature_c", np.nan) for m in ms], np.float64))
            elif parts[-1] == "gripper":
                ep.grip[parts[0]] = GripArrays(parts[0], np.array([m["sample_ns"] for m in ms], np.int64),
                                               np.array([np.nan if m.get("normalized") is None else m["normalized"] for m in ms], np.float64),
                                               np.array([m["raw_position"] for m in ms], np.int64))
            elif topic == "/system/sync":
                ep.sync = ms
        return ep

    def iter_frames(self, stream: str, *, downscale: int = 1, gray: bool = False):
        """Yield (video_frame, frame_index, capture_ns, image) by decoding <stream>.mp4 in order and joining on video_frame."""
        import av, cv2
        fm = self.frames[stream]
        by_vf = {int(v): i for i, v in enumerate(fm.video_frame)}
        with av.open(str(self.video_path(stream))) as c:
            for vf, frame in enumerate(c.decode(video=0)):
                i = by_vf.get(vf)
                if i is None: continue       # encoder wrote a frame without meta (should not happen; skip, never guess a time)
                img = frame.to_ndarray(format="gray" if gray else "bgr24")
                if downscale > 1: img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale), interpolation=cv2.INTER_AREA)
                yield vf, int(fm.frame_index[i]), int(fm.capture_ns[i]), img


# --------------------------------------------------------------------------------------------------- derived layout
def derived_dir(ep_path: Path, backend: str) -> Path:
    return Path(ep_path) / "derived" / f"pose_{backend}"


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False); return path.with_suffix(".parquet")
    except Exception:                                   # no pyarrow -> csv (still self-describing)
        df.to_csv(path.with_suffix(".csv"), index=False); return path.with_suffix(".csv")


def read_table(path_no_suffix: Path) -> pd.DataFrame:
    for suf in (".parquet", ".csv"):
        p = path_no_suffix.with_suffix(suf)
        if p.exists(): return pd.read_parquet(p) if suf == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(path_no_suffix)


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _default(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, Path): return str(o)
        raise TypeError(type(o))
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=_default)); tmp.replace(path); return path
