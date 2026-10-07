"""Segmented relative VIO (2026-09-16): one episode is NOT one continuous VIO trajectory.

Why: in bimanual stacking one hand waits while the other works. A stationary camera has no parallax, so the filter gets
no visual updates, drifts on the IMU, and by the time the hand moves again its velocity state is garbage (4-7 s waits ->
30-230 cm/s "speeds", coverage 0.27-0.72 on the AUTO dry run). ZUPT tuning did not help and is off the table. What we
actually output is short-window relative motion, and for a hand the IMU says is not moving that motion is zero by
definition. So:

    STATIC window  (IMU: low gyro AND low per-axis accelerometer variance AND no sample gaps AND >= min duration, plus an image-motion
                    guard -- a slow pure translation has no gyro and no acceleration, only the picture moves)
        delta p = 0,  delta R = bias-corrected gyro integration (in the camera frame)
    MOVING segment  a FRESH inner backend (OpenVINS) that was already fed the preceding static prefix, so it initializes
                    on the still->motion transition exactly as the episode start does
        delta p = VIO, delta R = VIO (IMU-checked downstream as before)
    stitching       each segment's first valid pose is anchored to the previous output pose composed with the gyro rotation
                    over the gap -- absolute values mean nothing, consecutive relative delta T is what is preserved

This is a PoseEstimator wrapper: run_side feeds it IMU and frames exactly as it feeds any backend, and it routes them to
per-segment inner backends. Frames inside a moving segment before its backend becomes valid (the init lag) are reported
INITIALIZING -- honestly missing, not held. Everything tunable lives in pose.yaml `segmented`."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
from ..estimator import BackendInfo, PoseEstimate, PoseEstimator, TrackingState
from ..imu import integrate_gyro
from ..se3 import inv_T

log = logging.getLogger("handumi.pose.segmented")
DEFAULTS = dict(enabled=False, min_static_s=0.5, gyro_max_deg_s=4.0, accel_std_max=0.25, max_gap_ms=25.0, window_s=0.2,
                prefix_s=2.0, tail_s=0.5, flow_width_px=120, flow_max_dn=6.0, flow_frames=2, starve_frames=0, starve_min_features=5)
# starve_frames: a moving segment whose inner filter has had < starve_min_features visual features for this many consecutive
#   frames (10 = 0.33 s at 30 fps) is DEAD -- it is running on the IMU alone and its velocity state is no longer trusted (left
#   wrist over the white plate, 2026-09-16: 5-37 m/s "speeds", hundreds of jumps). From then on the segment reports LOST until
#   the next static window re-initialises a fresh backend. An honest gap instead of invented motion. 0 disables.
#   DISABLED by default: OpenVINS' n_msckf+n_slam per update is 1-4 during perfectly good tracking, so this signal cannot separate
#   starvation from health (min 5 destroyed both wrists' coverage; min 1 left the jumps). Needs a better starvation signal first.


def static_windows(imu_t_ns, gyro, accel, cfg: dict) -> list[dict]:
    """IMU-still intervals with the per-window gyro bias. Same discipline as the collector gate: gyro AND accel AND gaps AND
    duration -- never gyro alone."""
    t = np.asarray(imu_t_ns, np.int64); g = np.asarray(gyro, float); a = np.asarray(accel, float)
    if len(t) < 10: return []
    gn = np.degrees(np.linalg.norm(g, axis=1))
    hz = len(t) / max((t[-1] - t[0]) / 1e9, 1e-9); w = max(int(hz * float(cfg["window_s"])), 3); k = np.ones(w) / w
    gm = np.convolve(gn, k, "same")
    # accelerometer variance per AXIS, summed (= trace of the windowed covariance: rotation-invariant). The std of |a| is
    # blind to acceleration perpendicular to gravity -- a hand shaking sideways at 0.6 m/s^2 changes |a| by 0.02.
    am = np.sqrt(sum(np.maximum(np.convolve(a[:, i] * a[:, i], k, "same") - np.convolve(a[:, i], k, "same") ** 2, 0.0) for i in range(3)))
    gap = np.concatenate([[0.0], np.diff(t) / 1e6]) > float(cfg["max_gap_ms"])
    still = (gm < float(cfg["gyro_max_deg_s"])) & (am < float(cfg["accel_std_max"])) & ~gap
    out = []; i = 0; n = len(still)
    while i < n:
        if not still[i]: i += 1; continue
        j = i
        while j < n and still[j]: j += 1
        if (t[j - 1] - t[i]) / 1e9 >= float(cfg["min_static_s"]):
            out.append(dict(t0_ns=int(t[i]), t1_ns=int(t[j - 1]), gyro_bias=g[i:j].mean(axis=0)))
        i = j
    # the moving-average window cannot judge the first/last window_s of the stream; a still window that starts (ends) that
    # close to the stream edge is taken to reach the edge, so an episode that begins still does not open with a moving sliver
    edge = int(float(cfg["window_s"]) * 1e9)
    if out and out[0]["t0_ns"] - t[0] <= edge: out[0]["t0_ns"] = int(t[0])
    if out and t[-1] - out[-1]["t1_ns"] <= edge: out[-1]["t1_ns"] = int(t[-1])
    return out


def image_motion(prev_small: np.ndarray | None, img, width_px: int) -> tuple[float | None, np.ndarray]:
    """Mean absolute difference between consecutive grey frames resized to ~width_px wide (DN). Cheap, and enough to tell
    'the picture is moving' from 'the picture is not' -- the guard against calling a slow pure translation static."""
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = max(1, g.shape[1] // int(width_px))
    small = cv2.resize(g, (max(g.shape[1] // f, 8), max(g.shape[0] // f, 8)), interpolation=cv2.INTER_AREA).astype(np.float32)
    if prev_small is None or prev_small.shape != small.shape: return None, small
    return float(np.abs(small - prev_small).mean()), small


@dataclass
class _Segment:
    backend: PoseEstimator
    feed_from_ns: int
    feed_until_ns: int           # +inf until the following static window is known to have started
    motion_start_ns: int
    anchor: np.ndarray | None = None
    n_valid: int = 0
    starve_run: int = 0
    dead: bool = False


class SegmentedEstimator(PoseEstimator):
    """See module docstring. `inner_factory()` returns a fresh, uninitialised inner backend."""

    def __init__(self, inner_factory, *, imu_t_ns, gyro, accel, T_camera_imu, cfg: dict | None = None, name: str = "segmented") -> None:
        self.cfg = dict(DEFAULTS); self.cfg.update(cfg or {})
        self.inner_factory = inner_factory
        self.imu_t = np.asarray(imu_t_ns, np.int64); self.gyro = np.asarray(gyro, float); self.accel = np.asarray(accel, float)
        self.Rci = np.eye(3) if T_camera_imu is None else np.asarray(T_camera_imu, float)[:3, :3]
        self.windows = static_windows(self.imu_t, self.gyro, self.accel, self.cfg)
        self._init_kw: dict = {}
        probe = inner_factory(); self.info = BackendInfo(f"{name}:{probe.info.name}", uses_imu=True, metric_scale=probe.info.metric_scale, online_capable=False,
                                                        provides=probe.info.provides, notes="segmented relative VIO: static = IMU hold, moving = per-segment re-init")
        self.segments: list[_Segment] = []; self.done_segments = 0
        self.T_out: np.ndarray | None = None; self.t_out_ns: int | None = None      # last OUTPUT pose (stitched frame) and its time
        self._prev_small = None; self._flow_hits = 0; self._forced_moving_until_ns: int | None = None
        self.trace: list[dict] = []                                                   # per-frame: mode, segment id, flow
        self._imu_seen = 0; self._last_scheduled = -1; self.starved_segments = 0

    # ------------------------------------------------------------------ PoseEstimator interface
    def initialize(self, *, intrinsics=None, T_camera_imu=None, imu_noise=None, image_size=None) -> None:
        self._init_kw = dict(intrinsics=intrinsics, T_camera_imu=T_camera_imu, imu_noise=imu_noise, image_size=image_size)
        self._ensure_segment_for(int(self.imu_t[0]) if len(self.imu_t) else 0)

    def push_imu(self, t_ns: int, gyro_rad_s, accel_m_s2) -> None:
        self._imu_seen += 1; self._ensure_segment_for(int(t_ns))
        for s in self.segments:
            if s.feed_from_ns <= t_ns <= s.feed_until_ns: s.backend.push_imu(t_ns, gyro_rad_s, accel_m_s2)

    def push_image(self, t_ns: int, frame_index: int, image_bgr) -> PoseEstimate:
        t_ns = int(t_ns); self._ensure_segment_for(t_ns)
        flow, self._prev_small = image_motion(self._prev_small, image_bgr, int(self.cfg["flow_width_px"]))
        win = self._window_at(t_ns)
        # image-motion guard: the IMU says still but the picture moves -> treat the rest of this window as moving
        if win is not None and flow is not None and flow > float(self.cfg["flow_max_dn"]):
            self._flow_hits += 1
            if self._flow_hits >= int(self.cfg["flow_frames"]): self._forced_moving_until_ns = win["t1_ns"]
        elif win is None: self._flow_hits = 0
        forced = self._forced_moving_until_ns is not None and t_ns <= self._forced_moving_until_ns
        static = win is not None and not forced
        if forced:                                                          # keep the owning segment alive through the forced stretch
            own = self._owning_segment(t_ns)
            if own is not None: own.feed_until_ns = max(own.feed_until_ns, int(self._forced_moving_until_ns + float(self.cfg["tail_s"]) * 1e9))
        # every live segment sees every frame (prefix / tail included); only the owning one produces the output
        ests = {id(s): s.backend.push_image(t_ns, frame_index, image_bgr) for s in self.segments if s.feed_from_ns <= t_ns <= s.feed_until_ns}
        if static:
            est = self._static_estimate(t_ns, frame_index, win)
        else:
            seg = self._owning_segment(t_ns)
            inner = ests.get(id(seg)) if seg is not None else None
            est = self._moving_estimate(t_ns, frame_index, seg, inner)
        self.trace.append(dict(t_ns=t_ns, frame_index=int(frame_index), mode="static" if static else "moving", forced_moving=bool(forced), flow_dn=flow,
                               segment=self._segment_index(t_ns), state=est.tracking_state.value))
        self._retire(t_ns)
        return est

    def finish(self):
        for s in self.segments:
            try: s.backend.finish()
            except Exception: pass
        return None

    def get_quality(self) -> dict:
        modes = [r["mode"] for r in self.trace]
        return dict(static_windows=len(self.windows), segments_started=self.done_segments + len(self.segments), frames_static=modes.count("static"),
                    frames_moving=modes.count("moving"), frames_forced_moving=sum(r["forced_moving"] for r in self.trace), imu_samples_seen=self._imu_seen,
                    starved_segments=self.starved_segments)

    # ------------------------------------------------------------------ static
    def _window_at(self, t_ns: int):
        for w in self.windows:
            if w["t0_ns"] <= t_ns <= w["t1_ns"]: return w
        return None

    def _gyro_dT(self, t0_ns: int, t1_ns: int, bias) -> np.ndarray:
        """Camera-frame rotation over [t0, t1] from the gyro, zero translation."""
        dR_imu = integrate_gyro(self.imu_t, self.gyro, int(t0_ns), int(t1_ns), bias)
        T = np.eye(4); T[:3, :3] = self.Rci @ dR_imu @ self.Rci.T; return T

    def _static_estimate(self, t_ns: int, frame_index: int, win) -> PoseEstimate:
        if self.T_out is None: self.T_out = np.eye(4)                              # the episode starts still: origin here
        elif self.t_out_ns is not None: self.T_out = self.T_out @ self._gyro_dT(self.t_out_ns, t_ns, win["gyro_bias"])
        self.t_out_ns = t_ns
        return PoseEstimate(t_ns, int(frame_index), self.T_out.copy(), TrackingState.STATIC, confidence=1.0, num_features=None,
                            extra=dict(mode="static", window_t0_ns=win["t0_ns"]))

    # ------------------------------------------------------------------ moving
    def _ensure_segment_for(self, t_ns: int) -> None:
        """A segment's backend must exist from (motion start - prefix_s), which is inside the preceding static window -- or
        from the episode start. Create it as soon as the clock reaches that point."""
        if not self._init_kw: return
        if not self.segments and self.done_segments == 0:
            self._new_segment(feed_from_ns=t_ns, motion_start_ns=self._first_motion_after(t_ns)); self._last_scheduled = self.segments[-1].motion_start_ns; return
        # the next motion stretch begins where the first static window AFTER the last scheduled motion ends
        nxt = self._next_motion_start_after(self._last_scheduled)
        if nxt is None: return
        feed_from = max(nxt - int(float(self.cfg["prefix_s"]) * 1e9), self._window_start_before(nxt))
        if t_ns >= feed_from:
            self._new_segment(feed_from_ns=feed_from, motion_start_ns=nxt); self._last_scheduled = nxt

    def _new_segment(self, *, feed_from_ns: int, motion_start_ns: int | None) -> None:
        if motion_start_ns is None: motion_start_ns = feed_from_ns
        be = self.inner_factory(); be.initialize(**self._init_kw)
        until = self._window_start_after(motion_start_ns)
        s = _Segment(be, int(feed_from_ns), int(until + float(self.cfg["tail_s"]) * 1e9) if until is not None else np.iinfo(np.int64).max, int(motion_start_ns))
        self.segments.append(s)

    def _next_motion_start_after(self, t_ns: int):
        end = int(self.imu_t[-1]) if len(self.imu_t) else 0
        for w in self.windows:
            if w["t0_ns"] > t_ns: return None if w["t1_ns"] >= end - int(0.1e9) else w["t1_ns"] + 1   # a still window that runs to the end: no motion follows
        return None

    def _first_motion_after(self, t_ns: int):
        """Start of the first moving stretch at or after t_ns: t_ns itself if not inside a static window, else that window's end."""
        w = self._window_at(t_ns)
        if w is None: return t_ns
        return w["t1_ns"] + 1

    def _window_start_before(self, t_ns: int) -> int:
        w = self._window_at(t_ns - 1)
        return w["t0_ns"] if w is not None else 0

    def _window_start_after(self, t_ns: int):
        for w in self.windows:
            if w["t0_ns"] > t_ns: return w["t0_ns"]
        return None

    def _owning_segment(self, t_ns: int):
        own = [s for s in self.segments if s.motion_start_ns <= t_ns <= s.feed_until_ns]
        return max(own, key=lambda s: s.motion_start_ns) if own else None

    def _segment_index(self, t_ns: int) -> int | None:
        s = self._owning_segment(t_ns)
        return None if s is None else self.done_segments + self.segments.index(s)

    def _moving_estimate(self, t_ns: int, frame_index: int, seg, inner: PoseEstimate | None) -> PoseEstimate:
        k = int(self.cfg.get("starve_frames", 0) or 0)
        if seg is not None and inner is not None and k > 0 and not seg.dead:
            nf = inner.num_features if inner.num_features is not None else 999
            starving = inner.valid and nf < int(self.cfg.get("starve_min_features", 1))
            seg.starve_run = seg.starve_run + 1 if starving else 0
            if seg.starve_run >= k:
                seg.dead = True; self.starved_segments += 1
                log.info("segment starting %.2fs: < %d visual features for %d frames -> dead until the next static window", seg.motion_start_ns / 1e9, int(self.cfg.get("starve_min_features", 1)), k)
            if starving:                       # an IMU-only pose is not output, not even once: it must never move T_out
                return PoseEstimate(t_ns, int(frame_index), None, TrackingState.LOST, confidence=0.0, num_features=inner.num_features,
                                    extra=dict(mode="moving", starving=True, starve_run=seg.starve_run))
        if seg is not None and seg.dead:
            return PoseEstimate(t_ns, int(frame_index), None, TrackingState.LOST, confidence=0.0, num_features=inner.num_features if inner else None,
                                extra=dict(mode="moving", starved=True))
        if seg is None or inner is None or not inner.valid:
            st = inner.tracking_state if inner is not None else TrackingState.INITIALIZING
            if st in (TrackingState.TRACKING, TrackingState.DEGRADED, TrackingState.STATIC): st = TrackingState.INITIALIZING
            return PoseEstimate(t_ns, int(frame_index), None, st, confidence=0.0, num_features=inner.num_features if inner else None,
                                extra=dict(mode="moving", segment_init_lag=True))
        if seg.anchor is None:
            # stitch: the new segment's first pose lands where the output was, rotated by the gyro over the gap (zero translation)
            bias = self._bias_before(t_ns)
            prev = np.eye(4) if self.T_out is None else (self.T_out @ self._gyro_dT(self.t_out_ns, t_ns, bias) if self.t_out_ns is not None else self.T_out)
            seg.anchor = prev @ inv_T(inner.T_world_camera)
        T = seg.anchor @ inner.T_world_camera
        self.T_out = T; self.t_out_ns = t_ns; seg.n_valid += 1
        return PoseEstimate(t_ns, int(frame_index), T, inner.tracking_state, confidence=inner.confidence, num_features=inner.num_features,
                            reprojection_error=inner.reprojection_error, extra=dict(inner.extra, mode="moving"))

    def _bias_before(self, t_ns: int):
        prior = [w for w in self.windows if w["t1_ns"] <= t_ns]
        return prior[-1]["gyro_bias"] if prior else None

    def _retire(self, t_ns: int) -> None:
        keep = []
        for s in self.segments:
            if t_ns > s.feed_until_ns:
                try: s.backend.finish()
                except Exception: pass
                self.done_segments += 1
            else: keep.append(s)
        self.segments = keep
