"""Visual-teleop episode recorder (2026-10-14, spec 31, 33). Records at the robot command rate with MONOTONIC timestamps on every
stream; nothing is resampled at collection time.

Episode dir:
    ticks.npz        per tick: t_ns, mode, sent/accepted, human_T (L/R), target_T, cmd_T, exec_T (FK of accepted q),
                     obs_T (FK of OBSERVED q = robot-executed TCP, the label source), q_exec, obs_q, g_target/g_exec/obs_g,
                     hand status, pinch, latencies
    events.jsonl     every HOLD / reject / clamp reason with its tick time (safety events)
    frames/<stream>.mp4 + frames/<stream>_t_ns.npy   camera observations (rgbd color, depth png dir, wrists)
    imu_<side>.npz   raw IMU samples (device_us, host_ns, acc, gyro)
    metadata.json    data_source, skill_id, config snapshot, time-sync estimates, init alignment, calibration provenance"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

SIDES = ("left", "right")


def _T(x) -> np.ndarray: return np.full((4, 4), np.nan) if x is None else np.asarray(x, np.float64)


class VideoStream:
    def __init__(self, path: Path, fps: float) -> None:
        self.path, self.fps, self.w, self.t = path, fps, None, []

    def add(self, t_ns: int, img) -> None:
        import cv2
        if self.w is None:
            h, w = img.shape[:2]; self.w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.w.write(img); self.t.append(int(t_ns))

    def close(self) -> None:
        if self.w is not None: self.w.release()
        if self.t: np.save(str(self.path).replace(".mp4", "_t_ns.npy"), np.asarray(self.t, np.int64))   # host monotonic per frame
        np.save(self.path.with_name(self.path.stem + "_t_ns.npy"), np.asarray(self.t, np.int64))


class EpisodeRecorder:
    def __init__(self, root: str | Path, *, skill_id: str, data_source: str = "VISUAL_TELEOP", config: dict | None = None,
                 name: str | None = None) -> None:
        self.dir = Path(root).expanduser() / (name or time.strftime("episode_%Y%m%d_%H%M%S"))
        (self.dir / "frames").mkdir(parents=True, exist_ok=False)
        self.meta = dict(data_source=data_source, skill_id=skill_id, created_wall=time.time(), t0_monotonic_ns=time.monotonic_ns(),
                         clock="time.monotonic_ns (host)", config=config or {}, time_sync={}, init={}, notes=[])
        self.ticks = []; self.videos: dict[str, VideoStream] = {}; self._events = open(self.dir / "events.jsonl", "w"); self.closed = False

    def add_tick(self, rec) -> None:
        self.ticks.append(rec)
        evs = {k: v for k, v in rec.reasons.items() if v}
        if evs or not rec.accepted and rec.sent: self._events.write(json.dumps(dict(t_ns=rec.t_ns, mode=rec.mode.value, reasons=evs, sent=rec.sent, accepted=rec.accepted)) + "\n")

    def add_frame(self, stream: str, t_ns: int, img, fps: float = 30.0) -> None:
        if stream not in self.videos: self.videos[stream] = VideoStream(self.dir / "frames" / f"{stream}.mp4", fps)
        self.videos[stream].add(t_ns, img)

    def add_imu(self, side: str, samples) -> None:
        if not samples: return
        f = self.dir / f"imu_{side}.npz"
        arr = dict(device_us=np.array([s.device_timestamp_us for s in samples], np.int64), host_ns=np.array([s.host_receive_ns for s in samples], np.int64),
                   acc=np.array([[s.ax, s.ay, s.az] for s in samples]), gyro=np.array([[s.gx, s.gy, s.gz] for s in samples]))
        if f.exists():
            old = dict(np.load(f)); arr = {k: np.concatenate([old[k], arr[k]]) for k in arr}
        np.savez(f, **arr)

    def tick_arrays(self) -> dict:
        R = self.ticks; n = len(R); out = dict(t_ns=np.array([r.t_ns for r in R], np.int64), mode=np.array([r.mode.value for r in R]),
                                              sent=np.array([r.sent for r in R]), accepted=np.array([r.accepted for r in R]))
        for s in SIDES:
            for k in ("human_T", "target_T", "cmd_T", "exec_T", "obs_T"): out[f"{k}_{s}"] = np.array([_T(getattr(r, k).get(s)) for r in R]).reshape(n, 4, 4)
            for k in ("q_exec", "obs_q"): out[f"{k}_{s}"] = np.array([np.full(6, np.nan) if getattr(r, k).get(s) is None else getattr(r, k)[s] for r in R]).reshape(n, 6)
            for k in ("g_target", "g_exec", "obs_g"): out[f"{k}_{s}"] = np.array([np.nan if getattr(r, k).get(s) is None else float(getattr(r, k)[s]) for r in R])
            out[f"hand_status_{s}"] = np.array([r.hand_status.get(s, "NONE") for r in R])
            out[f"pinch_{s}"] = np.array([-1 if r.pinch.get(s) is None else int(r.pinch[s]) for r in R], np.int8)
            out[f"held_{s}"] = np.array([r.held(s) for r in R])
        for k in sorted({k for r in R for k in r.latency_ms}): out[f"lat_{k}"] = np.array([r.latency_ms.get(k, np.nan) for r in R])
        return out

    def close(self) -> Path:
        if self.closed: return self.dir
        for v in self.videos.values(): v.close()
        self._events.close(); np.savez_compressed(self.dir / "ticks.npz", **self.tick_arrays())
        self.meta["n_ticks"] = len(self.ticks); (self.dir / "metadata.json").write_text(json.dumps(self.meta, indent=1, default=str))
        self.closed = True; return self.dir
