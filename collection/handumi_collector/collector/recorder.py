"""EpisodeRecorder: drains every device buffer on one thread and writes the raw episode
(<stream>.mp4 + sensors.mcap + events.json). Raw is never transformed here. Crash-safe: the episode directory
carries `.incomplete` until finalize() has written episode_meta.json last."""
from __future__ import annotations
import json
from collections import deque
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from .ownership import give_back
from .mcap_writer import SensorMcapWriter
from .timeline import ClockAnchor, GapDetector, index_gaps
from .video_writer import VideoStreamWriter
from ..config import AppConfig
from ..devices.base import now_ns
from ..devices.manager import DeviceManager

log = logging.getLogger("handumi.recorder")
PROTOCOL_MARKS = ("home_leave", "home_return")     # operator marks: events, but not hardware events


@dataclass
class StreamStats:
    frames: int = 0
    skipped_capture_indices: int = 0     # camera/driver-side drops (gaps in frame_index)
    first_ns: int | None = None
    last_ns: int | None = None
    encoder_backlog_max: int = 0

    @property
    def duration_s(self) -> float:
        return 0.0 if self.first_ns is None or self.last_ns is None else (self.last_ns - self.first_ns) / 1e9


@dataclass
class EpisodeRecorder:
    cfg: AppConfig
    devices: DeviceManager
    episode_dir: Path
    order: str
    instruction: str
    session_anchor: ClockAnchor
    extra_meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        (self.episode_dir / ".incomplete").touch()
        self.events: list[dict] = []
        self.stats: dict[str, StreamStats] = {}
        self.sensor_counts: dict[str, int] = {}
        self._videos: dict[str, VideoStreamWriter] = {}
        self._prev_index: dict[str, int | None] = {}
        self._gaps: dict[str, GapDetector] = {}
        self._depth_dir: Path | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t_start_ns: int | None = None
        self.t_stop_ns: int | None = None
        self.finalized = False
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        cams = [c for c in self.devices.cameras.values() if c.status().connected and c.cfg.record]
        self.streams = [c.name for c in cams]
        self.mcap = SensorMcapWriter(self.episode_dir / "sensors.mcap", streams=self.streams)
        for c in cams:
            a = c.status().detail.get("actual") or {}
            w, h = int(a.get("width", c.cfg.width)), int(a.get("height", c.cfg.height))
            self._videos[c.name] = VideoStreamWriter(self.episode_dir / f"{c.name}.mp4", width=w, height=h, fps=c.cfg.fps,
                                                     codec=c.cfg.codec, bitrate_kbps=c.cfg.bitrate_kbps)
            self.stats[c.name] = StreamStats(); self._prev_index[c.name] = None
            if c.cfg.role == "aux_depth":
                self._depth_dir = self.episode_dir / f"{c.name}_depth"; self._depth_dir.mkdir(exist_ok=True)
                intr = c.status().detail.get("intrinsics"); ds = c.status().detail.get("depth_scale")
                if intr: (self._depth_dir / "intrinsics.json").write_text(json.dumps(dict(intrinsics=intr, depth_scale=ds), indent=1))
        self._c922 = None
        if getattr(self.cfg.collector, "record_c922_view", False) and "right_wrist" in self._videos:
            # [2026-10-06 user "두 버전 다 동시 녹화"] right_wrist_c922.mp4: every right_wrist frame remapped to the measured robot
            # C922 (640x480). Frame i of right_wrist_c922.mp4 == frame i of right_wrist.mp4 (same timestamps in sensors.mcap).
            from ..robotlike.c922_view import C922View
            rcfg = next(c for c in self.devices.cameras.values() if c.name == "right_wrist").cfg
            self._c922 = C922View()
            self._videos["right_wrist_c922"] = VideoStreamWriter(self.episode_dir / "right_wrist_c922.mp4", width=640, height=480, fps=rcfg.fps,
                                                                 codec=rcfg.codec, bitrate_kbps=rcfg.bitrate_kbps)
            self.extra_meta = dict(self.extra_meta, derived_streams=dict(right_wrist_c922=dict(source="right_wrist", frames="1:1 with right_wrist.mp4",
                                   model="robotlike/c922_view.C922View (configs/calibration/robot_right_wrist_c922_v001.json)", size=[640, 480])))
        cc = self.cfg.collector
        # the newest ~1 s of IMU samples per side, kept AFTER the buffer is drained into the mcap, so the session's post-REC
        # hold-still check can read what is being recorded without stealing from (or racing) this loop
        self.recent_imu = {s: deque(maxlen=int(max(getattr(d.cfg, "rate_hz", 400), 50) * 1.0)) for s, d in self.devices.imus.items()}
        self.grip_stats: dict = {}
        for s in self.devices.imus: self._gaps[f"imu_{s}"] = GapDetector(cc.warn_imu_gap_ms)
        for s in self.devices.grippers: self._gaps[f"gripper_{s}"] = GapDetector(cc.warn_gripper_gap_ms)
        for n in self.streams: self._gaps[n] = GapDetector(cc.warn_timestamp_jump_ms)
        # drivers push their own integrity events (seq gap, ts wrap, reconnect, error) while recording
        for d in [*self.devices.imus.values(), *self.devices.grippers.values()]:
            if hasattr(d, "on_event"):
                d.on_event = (lambda kind, detail, t_ns, _n=d.name: self.event(kind, _n, detail, t_ns))
        # discard everything buffered before START so the episode begins at t_start
        for d in self.devices.all_devices(): d.buffer.drain()
        self.device_status_at_start = {n: dict(connected=st.connected, error=st.error, rate_hz=round(st.rate_hz, 1)) for n, st in self.devices.statuses().items()}
        self.t_start_ns = now_ns()
        self.mcap.sync(self.t_start_ns, dict(kind="episode_start", session_anchor=self.session_anchor.to_dict(),
                                             t_start_monotonic_ns=self.t_start_ns, t_start_wall_ns=self.session_anchor.to_wall_ns(self.t_start_ns),
                                             devices={n: st.detail for n, st in self.devices.statuses().items()}))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="episode-recorder", daemon=True); self._thread.start()

    def stop(self) -> None:
        if self._thread is None: return
        self.t_stop_ns = now_ns()
        self._stop.set(); self._thread.join(10.0); self._thread = None
        for d in [*self.devices.imus.values(), *self.devices.grippers.values()]:
            if hasattr(d, "on_event"): d.on_event = None
        self._pump()                                   # final drain
        self.mcap.sync(self.t_stop_ns, dict(kind="episode_stop", t_stop_monotonic_ns=self.t_stop_ns))
        for v in self._videos.values(): v.close()
        self.mcap.close()

    def event(self, kind: str, device: str, detail: dict | None = None, t_ns: int | None = None) -> None:
        t_ns = t_ns or now_ns()
        e = dict(t_ns=t_ns, t_rel_s=(t_ns - (self.t_start_ns or t_ns)) / 1e9, kind=kind, device=device, detail=detail or {})
        with self._lock: self.events.append(e)
        try: self.mcap.event(t_ns, kind, device, detail)
        except Exception: pass
        log.warning("[event] %s %s %s", kind, device, detail or "")

    # ---------------------------------------------------------------- pump
    def _pump(self) -> None:
        for name, cam in self.devices.cameras.items():
            if name not in self._videos: continue
            for f in cam.buffer.drain():
                st = self.stats[name]
                skipped = index_gaps(self._prev_index[name], f.frame_index); self._prev_index[name] = f.frame_index
                if skipped:
                    st.skipped_capture_indices += skipped
                    if skipped >= self.cfg.collector.warn_camera_drop_frames:
                        self.event("camera_drop", name, dict(skipped=skipped, frame_index=f.frame_index), f.capture_ns)
                g = self._gaps[name].update(f.capture_ns)
                if g is not None: self.event("timestamp_jump", name, dict(gap_ms=round(g, 1)), f.capture_ns)
                vf = self._videos[name].put(f.image)
                if name == "right_wrist" and self._c922 is not None: self._videos["right_wrist_c922"].put(self._c922.render(f.image, overlay=False))
                if f.depth is not None and self._depth_dir is not None:
                    cv2.imwrite(str(self._depth_dir / f"{vf:06d}.png"), np.asarray(f.depth, np.uint16))
                self.mcap.frame_meta(name, f.frame_index, vf, f.capture_ns, skipped)
                st.frames += 1; st.first_ns = st.first_ns or f.capture_ns; st.last_ns = f.capture_ns
                st.encoder_backlog_max = max(st.encoder_backlog_max, self._videos[name].backlog())
        for side, imu in self.devices.imus.items():
            for s in imu.buffer.drain():
                # timeouts are judged on the DEVICE clock (host receive time jitters with USB/GIL scheduling but loses nothing);
                # sequence gaps / timestamp wraps are reported by the driver itself (on_event)
                g = self._gaps[f"imu_{side}"].update(s.device_timestamp_us * 1000)
                prev = self._prev_index.get(f"imu_{side}")
                if g is not None and g > 0 and (prev is None or s.seq == (prev + 1) & 0xFFFFFFFF):
                    self.event("imu_timeout", f"imu_{side}", dict(gap_ms=round(g, 1)), s.host_receive_ns)
                self._prev_index[f"imu_{side}"] = s.seq
                self.mcap.imu(s); self.recent_imu[side].append(s)
        for side, gr in self.devices.grippers.items():
            for s in gr.buffer.drain():
                g = self._gaps[f"gripper_{side}"].update(s.sample_ns)
                if g is not None: self.event("gripper_gap", f"gripper_{side}", dict(gap_ms=round(g, 1)), s.sample_ns)
                self.mcap.gripper(s)
                # running raw/normalized range per jaw: the review gate reads it (a frozen reading, or a whole episode outside
                # the calibrated tick range after a servo power cycle, must not pass as a healthy gripper)
                st = self.grip_stats.setdefault(side, dict(n=0, raw_min=s.raw_position, raw_max=s.raw_position, raw_first=s.raw_position, raw_changes=0,
                                                           norm_min=float("inf"), norm_max=float("-inf")))
                if st["n"] and s.raw_position != st["raw_last"]: st["raw_changes"] += 1
                st["n"] += 1; st["raw_last"] = s.raw_position; st["raw_min"] = min(st["raw_min"], s.raw_position); st["raw_max"] = max(st["raw_max"], s.raw_position)
                if s.normalized == s.normalized: st["norm_min"] = min(st["norm_min"], float(s.normalized)); st["norm_max"] = max(st["norm_max"], float(s.normalized))
        # device health -> events (once per state change)
        for n, st in self.devices.statuses().items():
            was = self.sensor_counts.get(f"_err_{n}")
            if st.error and not was:
                self.event("device_error", n, dict(error=st.error)); self.sensor_counts[f"_err_{n}"] = 1
            elif not st.error and was:
                self.sensor_counts[f"_err_{n}"] = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try: self._pump()
            except Exception as exc:
                log.exception("recorder pump failed"); self.event("recorder_error", "recorder", dict(error=str(exc)))
            self._stop.wait(0.02)

    # ---------------------------------------------------------------- finalize
    def summary(self) -> dict:
        streams = {n: dict(frames=s.frames, skipped_capture_indices=s.skipped_capture_indices, duration_s=round(s.duration_s, 3),
                           fps_measured=round((s.frames - 1) / s.duration_s, 2) if s.duration_s > 0 and s.frames > 1 else 0.0,
                           encoder=self._videos[n].codec_used, encoder_error=self._videos[n].error,
                           encoder_backlog_max=s.encoder_backlog_max) for n, s in self.stats.items()}
        return dict(streams=streams, sensor_messages={k: v for k, v in self.mcap.counts.items()},
                    n_events=len(self.events), event_kinds=sorted({e["kind"] for e in self.events}))

    def finalize(self, *, status: str, quality: str | None = None, notes: str = "") -> dict:
        """Write events.json then episode_meta.json (last) and clear the .incomplete marker."""
        assert self.t_stop_ns is not None, "stop() before finalize()"
        (self.episode_dir / "events.json").write_text(json.dumps(self.events, indent=1))
        dur = (self.t_stop_ns - self.t_start_ns) / 1e9
        meta = dict(
            schema="handumi_episode_meta/v1", episode_dir=self.episode_dir.name, order=self.order, instruction=self.instruction,
            status=status, quality=quality, notes=notes,
            t_start_monotonic_ns=self.t_start_ns, t_stop_monotonic_ns=self.t_stop_ns,
            t_start_wall_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.session_anchor.to_wall_ns(self.t_start_ns) / 1e9)),
            duration_s=round(dur, 3), hw_event_in_episode=any(e["kind"] not in PROTOCOL_MARKS for e in self.events),
            device_status_at_start=getattr(self, "device_status_at_start", {}),
            aux_available={n: (n in self.streams) for n in self.devices.cameras if getattr(self.devices.cameras[n].cfg, "role", "") == "aux_depth"},
            imu_present=bool(self.devices.imus), imu_sides=sorted(self.devices.imus),
            # Also in the mcap's episode_start, but a serial nobody can read without opening the mcap is a serial nobody
            # checks. Which board was on which wrist is the one thing that cannot be recovered after the fact.
            imu_serials={side: getattr(d, "serial_number", None) for side, d in self.devices.imus.items()},
            # Decided here, from the devices actually recorded, so every caller gets it -- hardware_check builds its own
            # extra_meta and would otherwise produce an episode nothing could tell apart from a real one.
            synthetic=any(getattr(d, "cfg", None) is not None and getattr(d.cfg, "backend", "") == "mock"
                          for d in self.devices.all_devices()),
            **self.summary(), **self.extra_meta)
        tmp = self.episode_dir / "episode_meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1)); tmp.replace(self.episode_dir / "episode_meta.json")
        (self.episode_dir / ".incomplete").unlink(missing_ok=True)
        if status == "KEEP": (self.episode_dir / ".complete").touch()
        give_back(self.episode_dir)
        self.finalized = True
        return meta
