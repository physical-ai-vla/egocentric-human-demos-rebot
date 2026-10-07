"""Live HUD for the RGB-D arm POC — what the pose source is producing, right now, as numbers.

    .venv/bin/python -m ego_teleop.tools.rgbd_arm_hud --episode datasets/HumanRGBD_v1/episode_000001   # no camera
    sudo .venv/bin/python -m ego_teleop.tools.rgbd_arm_hud --live --serve 8712                         # pre-flight

This is a **pre-flight instrument, not part of a take**. It runs the same pose source the analysis runs, so what you
read here is what V0 will compute — but it also runs MediaPipe on every frame, so it must not share a process with a
60 s protocol recording (the take must not drop frames). Check the hand here, close it, then record.

Output is a single composited JPEG (`/tmp/rgbd_arm_hud.jpg`) rather than a cv2 window, because the Orbbec needs sudo
on macOS and a root process generally cannot talk to WindowServer — which is why the existing recorder writes JPEGs
too. `--serve PORT` additionally streams it as MJPEG plus a `/stats.json`, so the numbers can be watched on a second
screen or a phone while both hands are busy in front of the camera.

The four numbers on the panel that decide whether V2 is allowed to happen are marked GATE: stationary jitter,
catastrophic jumps, lost-run length, and arm-pose duty. They are computed live exactly as the report computes them,
so a take that looks bad here will look bad in the report."""
from __future__ import annotations
import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..config import RgbdArmPocCfg, load_teleop_cfg
from ..hand3d import aero_mocap as M
from ..hand3d.head_camera import ColorExposure, OrbbecHeadCamera, RecordedRgbdSource
from ..hand3d.palm_pose import ArmPoseHealth
from ..tracking.interfaces import TrackingHealth

HEALTH_COLOR = {"ARM_POSE_OK": (90, 220, 90), "ARM_POSE_DEGRADED": (60, 200, 250), "ARM_POSE_LOST": (70, 70, 245),
                "TRACKING_OK": (90, 220, 90), "TRACKING_DEGRADED": (60, 200, 250), "TRACKING_LOST": (70, 70, 245)}
FG, DIM, BG, PANEL = (235, 235, 235), (140, 140, 140), (22, 22, 24), (32, 32, 36)
JUMP_MM = 30.0          # the user's gate: a single >30 mm/frame jump is what endangers a real arm, not mean jitter
# "stationary" is judged on NET DRIFT across the window, not on frame-to-frame speed. A hand held still with 2 mm of
# depth noise moves ~2.8 mm between frames = ~85 mm/s of instantaneous speed, so a speed threshold loose enough to
# call that still would also accept real motion, and a tight one never reports the jitter number at all — which is
# the one number this HUD exists for. Noise averages out of a window mean; motion does not.
STILL_DRIFT_MM = 10.0   # |mean(last third) - mean(first third)| below this = stationary


class LiveStats:
    """Rolling + cumulative statistics over the live pose stream, defined the same way the V0 report defines them."""

    def __init__(self, window_s: float = 3.0, rate_hz: float = 30.0) -> None:
        self.n = max(int(window_s * rate_hz), 10)
        self.t: deque[float] = deque(maxlen=self.n)          # seconds
        self.p: deque[np.ndarray] = deque(maxlen=self.n)     # palm xyz, only tracked frames
        self.q: deque[np.ndarray] = deque(maxlen=self.n)     # raw palm quaternion, when available
        self.frames = 0
        self.arm_counts = {h.value: 0 for h in ArmPoseHealth}
        self.jumps = 0                                        # cumulative |dp| > JUMP_MM between consecutive frames
        self.jump_max_mm = 0.0
        self.lost_runs_ms: list[float] = []
        self._lost_start: float | None = None
        self._prev: tuple[float, np.ndarray] | None = None
        self._t_now = 0.0            # latest frame time, tracked or not — `self.t` only holds TRACKED frames, so
                                     # measuring the ongoing outage against it would read negative
        self.proc_ms: deque[float] = deque(maxlen=self.n)
        self.speed_mm_s: deque[float] = deque(maxlen=self.n)

    def push(self, t_s: float, wp, palm, proc_ms: float) -> None:
        self.frames += 1
        self._t_now = t_s
        self.arm_counts[palm.health.value] += 1
        self.proc_ms.append(proc_ms)
        lost = wp.health is TrackingHealth.LOST
        if lost and self._lost_start is None: self._lost_start = t_s
        if not lost and self._lost_start is not None:
            self.lost_runs_ms.append((t_s - self._lost_start) * 1000.0); self._lost_start = None
        if lost or palm.position_m is None:
            self._prev = None                                 # never measure a jump across a gap
            return
        p = np.asarray(palm.position_m, np.float64)
        if self._prev is not None:
            d_mm = float(np.linalg.norm(p - self._prev[1]) * 1000.0)
            self.jump_max_mm = max(self.jump_max_mm, d_mm)
            if d_mm > JUMP_MM: self.jumps += 1
            dt = t_s - self._prev[0]
            self.speed_mm_s.append(d_mm / dt if dt > 1e-6 else float("nan"))
        self._prev = (t_s, p)
        self.t.append(t_s); self.p.append(p)
        if palm.R_palm is not None: self.q.append(Rotation.from_matrix(palm.R_palm).as_quat())

    # ---- derived ------------------------------------------------------------------------------------------
    @property
    def duty(self) -> float:
        ok = self.arm_counts[ArmPoseHealth.OK.value]
        return ok / self.frames if self.frames else float("nan")

    @property
    def peak_speed_mm_s(self) -> float:
        if len(self.p) < 3: return float("nan")
        P, T = np.array(self.p), np.array(self.t)
        dt = np.diff(T)
        ok = dt > 1e-6
        return float(np.max(np.linalg.norm(np.diff(P, axis=0)[ok], axis=1) / dt[ok]) * 1000.0) if ok.any() else float("nan")

    @property
    def drift_mm(self) -> float:
        """Net movement across the window: |mean(last third) - mean(first third)|. Immune to per-frame noise."""
        if len(self.p) < 9: return float("nan")
        P = np.array(self.p); k = max(len(P) // 3, 3)
        return float(np.linalg.norm(P[-k:].mean(0) - P[:k].mean(0)) * 1000.0)

    @property
    def still(self) -> bool:
        d = self.drift_mm
        return bool(np.isfinite(d) and d <= STILL_DRIFT_MM and len(self.p) >= self.n // 2)

    def jitter_mm(self) -> tuple[float, float]:
        """(rms, p95) deviation from the window mean — the GATE number. Only meaningful while `still`."""
        if len(self.p) < 5: return float("nan"), float("nan")
        P = np.array(self.p)
        d = np.linalg.norm(P - P.mean(0), axis=1) * 1000.0
        return float(np.sqrt(np.mean(d ** 2))), float(np.percentile(d, 95))

    def orientation_trace_deg(self) -> np.ndarray:
        """Per-sample angle from the window's mean rotation — the raw orientation noise, measured in every run even
        though V0/V1 command no rotation. Whether 6-DoF is worth building is a question about this series."""
        if len(self.q) < 5: return np.zeros(0)
        R = Rotation.from_quat(np.array(self.q))
        return np.degrees((R.mean().inv() * R).magnitude())

    def orientation_jitter_deg(self) -> float:
        d = self.orientation_trace_deg()
        return float(np.percentile(d, 95)) if d.size else float("nan")

    def lost_ms(self) -> tuple[float, float]:
        cur = 0.0 if self._lost_start is None else max((self._t_now - self._lost_start) * 1000.0, 0.0)
        # the max must include an outage still in progress, or a run that never ends reads as 0
        return cur, max([cur] + self.lost_runs_ms) if (self.lost_runs_ms or cur) else 0.0

    def traces_mm(self) -> dict:
        """Per-axis deviation from the window mean, for the sparklines."""
        if len(self.p) < 3: return {}
        P = np.array(self.p); C = (P - P.mean(0)) * 1000.0
        return {a: C[:, i] for i, a in enumerate("xyz")}

    def to_dict(self) -> dict:
        rms, p95 = self.jitter_mm(); cur, mx = self.lost_ms()
        # `window_*` on purpose, not `stationary_*`: these are the current window's numbers whatever the hand was
        # doing. They are a jitter measurement only while `still` is True, and the gate rows below read them through
        # that flag. Naming them "stationary" in the telemetry invites reading a motion number as sensor noise.
        return dict(frames=self.frames, arm_pose=dict(self.arm_counts), arm_pose_ok_duty=self.duty,
                    still=self.still, window_filled=len(self.p) >= self.n // 2,
                    drift_mm=self.drift_mm, peak_speed_mm_s=self.peak_speed_mm_s,
                    window_jitter_rms_mm=rms, window_jitter_p95_mm=p95,
                    window_orientation_p95_deg=self.orientation_jitter_deg(),
                    catastrophic_jumps=self.jumps, max_frame_step_mm=self.jump_max_mm,
                    lost_run_current_ms=cur, lost_run_max_ms=mx, lost_runs=len(self.lost_runs_ms),
                    proc_ms_p50=float(np.median(self.proc_ms)) if self.proc_ms else float("nan"))


# ---- drawing -----------------------------------------------------------------------------------------------------
def _text(img, s, xy, *, color=FG, scale=0.44, thick=1):
    import cv2
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _sparkline(panel, values, box, *, label, unit, color=(120, 200, 255), symmetric=True, ref=None):
    """A small rolling plot. `ref` draws a threshold line in the same units (e.g. the jump limit)."""
    import cv2
    x0, y0, w, h = box
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (52, 52, 58), -1)
    v = np.asarray([x for x in values if np.isfinite(x)], np.float64)
    if v.size >= 2:
        lim = max(float(np.abs(v).max()), 1e-6)
        if ref is not None: lim = max(lim, abs(ref))
        lo, hi = (-lim, lim) if symmetric else (0.0, lim)
        to_y = lambda a: int(y0 + h - (a - lo) / max(hi - lo, 1e-9) * h)
        if symmetric: cv2.line(panel, (x0, to_y(0.0)), (x0 + w, to_y(0.0)), (80, 80, 88), 1)
        if ref is not None:
            cv2.line(panel, (x0, to_y(ref)), (x0 + w, to_y(ref)), (70, 70, 245), 1)
        xs = np.linspace(x0, x0 + w, v.size).astype(int)
        cv2.polylines(panel, [np.stack([xs, np.array([to_y(a) for a in v])], 1)], False, color, 1, cv2.LINE_AA)
        _text(panel, f"{label} {v[-1]:+.1f}{unit}", (x0 + 4, y0 + 12), color=FG, scale=0.38)
    else:
        _text(panel, f"{label} --", (x0 + 4, y0 + 12), color=DIM, scale=0.38)


def _gate_row(panel, y, name, value, ok, *, fmt="{:.1f}", suffix=""):
    """One GATE line: green when it would pass the provisional V2 gate, red when it would not."""
    txt = "--" if value is None or (isinstance(value, float) and not np.isfinite(value)) else fmt.format(value) + suffix
    _text(panel, name, (10, y), color=DIM, scale=0.40)
    _text(panel, txt, (196, y), color=(90, 220, 90) if ok else (70, 70, 245), scale=0.46, thick=1)


def poc_rate(cfg: RgbdArmPocCfg) -> float:
    """The speed at which one frame-to-frame step equals the catastrophic-jump limit — the red line on the speed plot."""
    return float(cfg.rate_hz)


def render(frame, est_hand, palm, wp, stats: LiveStats, cfg: RgbdArmPocCfg, *, banner: str = "", pw: int = 340,
           window_s: float = 3.0, recycles: int = 0):
    """One composited HUD image: colour + skeleton + palm marker, depth, and the live value panel."""
    import cv2
    from .a1_hand3d import draw_overlay
    img = frame.color_bgr.copy()
    if est_hand is not None and est_hand.n_valid:
        for a, b in M.HAND_EDGES:
            pa, pb = est_hand.landmarks_2d[a], est_hand.landmarks_2d[b]
            if np.isfinite([pa, pb]).all():
                cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), (170, 170, 170), 1, cv2.LINE_AA)
        for j in range(M.N_LANDMARKS):
            p2 = est_hand.landmarks_2d[j]
            if not np.isfinite(p2).all(): continue
            filled = est_hand.filled is not None and est_hand.filled[j]
            col = (90, 220, 90) if est_hand.valid[j] else ((60, 200, 250) if filled else (70, 70, 245))
            cv2.circle(img, tuple(p2.astype(int)), 4 if j in (M.WRIST, *[M.INDEX_MCP, M.MIDDLE_MCP, M.RING_MCP, M.PINKY_MCP]) else 3, col, -1, cv2.LINE_AA)
    hc = HEALTH_COLOR[palm.health.value]
    if palm.position_m is not None:
        uv = frame.calib.color.project(np.asarray(palm.position_m)[None, :])[0]
        if np.isfinite(uv).all():
            c = tuple(uv.astype(int))
            # dark outline first: the health colour is green when OK, which is also the landmark colour — without a
            # contrasting ring the arm-root marker disappears into the skeleton, and it is the thing being judged
            cv2.circle(img, c, 14, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.circle(img, c, 14, hc, 2, cv2.LINE_AA)
            cv2.drawMarker(img, c, (20, 20, 20), cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
            cv2.drawMarker(img, c, hc, cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
    if banner:
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        _text(img, banner, (8, 21), color=(120, 255, 180), scale=0.62, thick=2)

    dep = np.clip(np.nan_to_num(frame.depth_m * 1000.0), 200, 1200)
    dcol = cv2.applyColorMap(((dep - 200) / 1000 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    dcol[np.nan_to_num(frame.depth_m) <= 0] = (25, 25, 25)          # no depth is black, not "far"
    left = np.vstack([img, cv2.resize(dcol, (img.shape[1], img.shape[0] // 2))])

    panel = np.full((left.shape[0], pw, 3), PANEL, np.uint8)
    s = stats.to_dict()
    _text(panel, palm.health.value, (10, 26), color=hc, scale=0.66, thick=2)
    _text(panel, (palm.reason or "-")[:40], (10, 44), color=DIM, scale=0.38)
    _text(panel, wp.health.value, (10, 66), color=HEALTH_COLOR[wp.health.value], scale=0.46)

    y = 92
    z = float(palm.position_m[2]) if palm.position_m is not None else float("nan")
    in_band = cfg.palm.min_origin_depth_m <= z <= cfg.palm.max_origin_depth_m if np.isfinite(z) else False
    for lbl, val, col in (
        ("palm x", f"{palm.position_m[0]:+.3f} m" if palm.position_m is not None else "--", FG),
        ("palm y", f"{palm.position_m[1]:+.3f} m" if palm.position_m is not None else "--", FG),
        ("palm z", f"{z:.3f} m" if np.isfinite(z) else "--", (90, 220, 90) if in_band else (60, 200, 250)),
        ("depth band", f"{cfg.palm.min_origin_depth_m:.2f}-{cfg.palm.max_origin_depth_m:.2f} m", DIM),
        ("palm landmarks", f"{palm.n_palm_landmarks}/5", (90, 220, 90) if palm.n_palm_landmarks >= cfg.palm.min_palm_landmarks_ok else (60, 200, 250)),
        ("palm scale", f"{palm.palm_scale_m*1000:.0f} mm" if np.isfinite(palm.palm_scale_m) else "--", FG),
        ("scale ratio", f"{palm.palm_scale_ratio:.2f}" if np.isfinite(palm.palm_scale_ratio) else "-- (learning)", FG),
        ("depth spread", f"{palm.palm_depth_spread_m*1000:.0f} mm" if np.isfinite(palm.palm_depth_spread_m) else "--", FG),
        ("hand branch", (est_hand.health.value if est_hand else "-").replace("HAND_", ""), DIM),
    ):
        _text(panel, lbl, (10, y), color=DIM, scale=0.40); _text(panel, val, (150, y), color=col, scale=0.42)
        y += 17

    y += 8
    _text(panel, "GATE (provisional, V2 permission)", (10, y), color=(200, 200, 120), scale=0.40); y += 18
    d = s["drift_mm"]
    stat = (f"still (drift {d:.0f} mm)" if s["still"] else f"moving (drift {d:.0f} mm)") if np.isfinite(d) else "filling window"
    _text(panel, f"{window_s:.0f}s window: {stat}", (10, y), color=DIM if s["still"] else (60, 200, 250), scale=0.38); y += 18
    live = s["still"]
    _gate_row(panel, y, "arm_pose duty >=95%", s["arm_pose_ok_duty"] * 100 if np.isfinite(s["arm_pose_ok_duty"]) else None,
              s["arm_pose_ok_duty"] >= 0.95, suffix="%"); y += 19
    _gate_row(panel, y, "jitter RMS <=5mm", s["window_jitter_rms_mm"] if live else None,
              live and s["window_jitter_rms_mm"] <= 5.0, suffix=" mm"); y += 19
    _gate_row(panel, y, "jitter p95 <=10mm", s["window_jitter_p95_mm"] if live else None,
              live and s["window_jitter_p95_mm"] <= 10.0, suffix=" mm"); y += 19
    _gate_row(panel, y, f"jumps >{JUMP_MM:.0f}mm == 0", float(s["catastrophic_jumps"]), s["catastrophic_jumps"] == 0, fmt="{:.0f}"); y += 19
    _gate_row(panel, y, "lost run <=300ms", s["lost_run_max_ms"], s["lost_run_max_ms"] <= 300.0, suffix=" ms"); y += 19
    _gate_row(panel, y, "ori jitter p95 (info)", s["window_orientation_p95_deg"] if live else None,
              live and s["window_orientation_p95_deg"] <= 5.0, suffix=" deg"); y += 24

    tr = stats.traces_mm()
    bw, bh = pw - 20, 32
    for i, a in enumerate("xyz"):
        _sparkline(panel, tr.get(a, []), (10, y + i * (bh + 4), bw, bh), label=f"d{a}", unit=" mm")
    y += 3 * (bh + 4)
    _sparkline(panel, stats.orientation_trace_deg(), (10, y, bw, bh), label="d ori", unit=" deg",
               symmetric=False, color=(200, 160, 255))
    y += bh + 4
    _sparkline(panel, list(stats.speed_mm_s), (10, y, bw, bh), label="speed", unit=" mm/s",
               symmetric=False, color=(140, 230, 200), ref=JUMP_MM * poc_rate(cfg))
    y += bh + 4
    _sparkline(panel, list(stats.proc_ms), (10, y, bw, bh), label="proc", unit=" ms", symmetric=False, color=(180, 180, 120))
    y += bh + 8
    _text(panel, f"{s['frames']} frames   max step {s['max_frame_step_mm']:.0f} mm   "
                 f"lost x{s['lost_runs']}   recycle x{recycles}", (10, y), color=DIM, scale=0.36)
    y += 14
    _text(panel, f"LOST now {s['lost_run_current_ms']:.0f} ms" if s["lost_run_current_ms"] else "", (10, y),
          color=(70, 70, 245), scale=0.38)
    return np.hstack([left, panel])


# ---- viewer page (no server, no threads, no port) ----------------------------------------------------------------
VIEWER = """<title>RGB-D arm HUD</title>
<style>
 html,body{margin:0;height:100%;background:#0d1215;color:#8ba1a9;
   font:12px ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;flex-direction:column}
 #img{flex:1;min-height:0;object-fit:contain;background:#0d1215}
 #bar{padding:6px 10px;display:flex;gap:16px;align-items:center;border-top:1px solid #233238}
 #dot{width:8px;height:8px;border-radius:50%%;background:#4cc38a}
 .stale #dot{background:#ff6f61}
 .stale #state{color:#ff6f61}
</style>
<img id="img" alt="RGB-D arm HUD">
<div id="bar"><span id="dot"></span><span id="state">live</span><span id="fps"></span><span>__PATH__</span></div>
<script>
const SRC = "__NAME__", img = document.getElementById("img"), bar = document.getElementById("bar");
const state = document.getElementById("state"), fpsEl = document.getElementById("fps");
let last = 0, shown = 0, times = [];
function tick() {
  const n = new Image();
  n.onload = () => {
    // a redrawn HUD changes size-on-disk almost every frame; naturalWidth stays constant, so freshness is
    // judged by whether the load actually completed since the previous one
    img.src = n.src; shown++;
    const now = performance.now(); times.push(now); times = times.filter(t => now - t < 2000);
    fpsEl.textContent = (times.length / 2).toFixed(1) + " fps";
    last = now; bar.classList.remove("stale"); state.textContent = "live";
    setTimeout(tick, 120);
  };
  n.onerror = () => { bar.classList.add("stale"); state.textContent = "no image yet"; setTimeout(tick, 600); };
  n.src = SRC + "?t=" + Date.now();
}
setInterval(() => {
  if (last && performance.now() - last > 2000) { bar.classList.add("stale"); state.textContent = "STALE - HUD stopped?"; }
}, 500);
tick();
</script>"""


def write_viewer(jpg: Path) -> Path:
    """A local page that re-loads the HUD image. Deliberately not a server: the MJPEG server did not survive a live
    run, and anything with a socket and a thread is one more thing to debug while the camera is running."""
    html = jpg.with_suffix(".html")
    html.write_text(VIEWER.replace("__NAME__", jpg.name).replace("__PATH__", str(jpg)))
    return html


# ---- optional MJPEG server ----------------------------------------------------------------------------------------
class _Shared:
    def __init__(self): self.jpg: bytes | None = None; self.stats: dict = {}; self.lock = threading.Lock()


def serve(shared: _Shared, port: int) -> threading.Thread:
    """MJPEG + /stats.json on localhost, so the numbers are readable while both hands are in front of the camera."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    PAGE = (b"<!doctype html><title>RGB-D arm HUD</title>"
            b"<style>body{margin:0;background:#16161a;display:grid;place-items:center;min-height:100vh}"
            b"img{max-width:100vw;max-height:100vh;image-rendering:pixelated}</style>"
            b"<img src='/stream'>")

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a): pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/stats"):
                with shared.lock: body = json.dumps(shared.stats, default=float).encode()
                return self._send(body, "application/json")
            if not self.path.startswith("/stream"):
                return self._send(PAGE, "text/html; charset=utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
            try:
                while True:
                    with shared.lock: jpg = shared.jpg
                    if jpg:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    print(f"  HUD: http://127.0.0.1:{port}/   (stats: /stats.json)", flush=True)
    return t


# ---- protocol cues: the CANONICAL table, imported so the three places cannot drift apart ------------------------
from .p1_rgbd_arm import PROTOCOL_60S as PROTOCOL      # noqa: E402  (name, t0, t1, operator hint)


def segment_at(t_s: float):
    for name, a, b, hint in PROTOCOL:
        if a <= t_s < b: return name, a, b, hint
    return None


def banner_for(t_s: float, total: float = 60.0) -> str:
    seg = segment_at(t_s)
    if seg is None: return f"done ({t_s:.0f}s)"
    name, a, b, _ = seg
    nxt = segment_at(float(b))
    return f"{t_s:5.1f}/{total:.0f}s  [{name}]  {b - t_s:4.1f}s left" + (f"  ->  {nxt[0]}" if nxt else "  -> end")


# ---- runner -------------------------------------------------------------------------------------------------------
def run(source, poc: RgbdArmPocCfg, head_rgbd, *, out: Path, port: int | None = None, protocol: bool = False,
        window_s: float = 3.0, max_frames: int | None = None, show_window: bool = False,
        exposure_note: dict | None = None, recycle_every: int = 0, no_frame_timeout_s: float = 20.0,
        session_frames: int = 5000) -> dict:
    """Bounded session, because the macOS MediaPipe GPU graph cannot run indefinitely on this machine.

    Measured (mediapipe 1.0.1, 2026-09-11): the GPU graph leaks ~3.4 MB per frame and the process aborts with
    CoreVideo `kCVReturnAllocationFailed` near frame 7500 — about four minutes at 30 Hz.

    Rebuilding the landmarker bounds that (`recycle_every`), and it works: 9000 frames at 60 fps with RSS sawtoothing
    between 1.3 and 3.4 GB. But **not under sudo** — and the live camera needs sudo on macOS. There, creating the
    second GL/Metal context hangs the process at the first rebuild (observed: a second `GL version` line, then
    silence), the same root-vs-WindowServer restriction that stops cv2 from opening a window. So `recycle_every`
    defaults to 0 and the session is bounded instead: at `session_frames` the tool stops cleanly and says to restart
    it. An instrument that ends and tells you why beats one that freezes."""
    import cv2
    import dataclasses as dc
    hr = dc.replace(head_rgbd, provider="head_rgbd", side=poc.side)
    hand_provider = hr.build_provider(recycle_every=recycle_every)
    provider = poc.build_pose_provider(hand_provider=hand_provider, hand_pose_cfg=hr.hand_pose)
    est = provider.est
    stats = LiveStats(window_s, poc.rate_hz)
    shared = _Shared()
    if port: serve(shared, port)
    t0 = None
    n = 0
    # A camera that stops delivering must say so. `cam.read()` returns None on a timeout, and simply `continue`-ing
    # spins forever with no output and an empty /stats.json — which is indistinguishable from a hung program.
    empty = 0
    empty_t0 = time.monotonic()
    warned: set[float] = set()
    for frame in source:
        if frame is None:
            empty += 1
            waited = time.monotonic() - empty_t0
            for mark in (2.0, 6.0):
                if waited >= mark and mark not in warned:
                    warned.add(mark)
                    print(f"  ! no frames for {waited:.0f}s ({empty} timeouts). The camera is open but silent — "
                          f"usually the device is left claimed after a crash. Unplug/replug it and check with "
                          f"`orbbec_recorder.py preview --secs 3`.", flush=True)
            if waited >= no_frame_timeout_s:
                raise RuntimeError(f"no frames from the camera for {waited:.0f}s ({empty} timeouts) — giving up "
                                   f"rather than spinning silently. Replug the Orbbec and retry.")
            continue
        empty = 0; empty_t0 = time.monotonic(); warned.clear()
        t = int(frame.timestamp_ns)
        if t0 is None: t0 = t
        t_s = (t - t0) / 1e9
        c0 = time.monotonic_ns()
        wp = provider.push_image(t, getattr(frame, "frame_index", n), frame)
        proc_ms = (time.monotonic_ns() - c0) / 1e6
        palm = est.last_palm
        stats.push(t_s, wp, palm, proc_ms)
        hand_est = est.last_hand.estimate if est.last_hand is not None else None
        recycles = getattr(hand_provider._lm, "recycles", 0)
        hud = render(frame, hand_est, palm, wp, stats, poc, banner=banner_for(t_s) if protocol else "",
                     window_s=window_s, recycles=recycles)
        ok, buf = cv2.imencode(".jpg", hud, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.write_bytes(buf.tobytes())
            with shared.lock:
                shared.jpg = buf.tobytes()
                shared.stats = dict(stats.to_dict(), recycles=recycles, **(exposure_note or {}))
        if show_window:
            cv2.imshow("RGB-D arm HUD", hud)
            if cv2.waitKey(1) & 0xFF == 27: break
        n += 1
        if max_frames and n >= max_frames: break
        if not max_frames and session_frames and n >= session_frames:
            print(f"\n  session limit: {n} frames ({n / max(poc.rate_hz, 1):.0f}s). Stopping before the macOS "
                  f"MediaPipe GPU graph runs out of memory (it aborts near frame 7500). Nothing is wrong — "
                  f"restart the HUD to keep going.", flush=True)
            break
    return dict(stats.to_dict(), recycles=getattr(hand_provider._lm, "recycles", 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live HUD for the RGB-D arm POC pose source (pre-flight, not a take)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episode", help="replay a recorded RGB-D episode (works without the camera)")
    src.add_argument("--live", action="store_true", help="live fixed Orbbec (macOS: needs sudo)")
    ap.add_argument("--out", default="/tmp/rgbd_arm_hud.jpg")
    ap.add_argument("--serve", type=int, default=None, metavar="PORT", help="also stream as MJPEG on localhost")
    ap.add_argument("--protocol", action="store_true", help="overlay the 60 s protocol segment timer (dry run only)")
    ap.add_argument("--window", type=float, default=3.0, help="rolling window in seconds for jitter (default 3)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--show-window", action="store_true", help="also open a cv2 window (usually fails under sudo)")
    ap.add_argument("--color-auto-exposure", choices=("true", "false"), default=None,
                    help="live only: false = fix the exposure. Use the SAME value the take will use.")
    ap.add_argument("--color-exposure", type=int, default=None, help="device units, from the recorder's preview")
    ap.add_argument("--color-gain", type=int, default=None)
    ap.add_argument("--no-frame-timeout", type=float, default=20.0, metavar="S",
                    help="give up (with a diagnosis) after this long with no camera frames")
    ap.add_argument("--session-frames", type=int, default=5000, metavar="N",
                    help="stop cleanly after N frames (~%(default)s = 2.8 min at 30 Hz). The macOS MediaPipe GPU "
                         "graph leaks ~3.4 MB/frame and aborts near frame 7500.")
    ap.add_argument("--recycle-every", type=int, default=0, metavar="N",
                    help="rebuild the landmarker every N frames to bound the leak. Works WITHOUT sudo (9000 frames "
                         "verified); under sudo the second GL context hangs the process, so leave it at 0 for --live.")
    a = ap.parse_args(argv)

    cfg = load_teleop_cfg()
    applied = {}
    if a.live:
        source, applied = _live_source(cfg, _exposure(a))
    else:
        source = RecordedRgbdSource(a.episode)
    out_path = Path(a.out)
    viewer = write_viewer(out_path)
    print(f"  HUD image: {out_path}", flush=True)
    print(f"  WATCH IT:  open {viewer}      <- run this in another terminal (not under sudo)", flush=True)
    s = run(source, cfg.rgbd_arm_poc, cfg.head_rgbd, out=out_path, port=a.serve, protocol=a.protocol,
            window_s=a.window, max_frames=a.frames, show_window=a.show_window, exposure_note=applied,
            recycle_every=a.recycle_every, no_frame_timeout_s=a.no_frame_timeout,
            session_frames=a.session_frames)
    print(json.dumps(dict(s, **applied), indent=1, default=float))
    return 0


def _exposure(a) -> ColorExposure:
    auto = None if a.color_auto_exposure is None else (a.color_auto_exposure == "true")
    return ColorExposure(auto=auto, exposure=a.color_exposure, gain=a.color_gain)


def _live_source(cfg, exposure: ColorExposure):
    """Live camera, with the colour exposure the take will run at.

    Measured on a Gemini 336: reopening the pipeline turns AE back on, so a pre-flight that does not set the exposure
    itself runs in a DIFFERENT camera mode from the recording — and its jitter number then predicts nothing."""
    cam = OrbbecHeadCamera(fps=int(cfg.rgbd_arm_poc.rate_hz), exposure=exposure).open()
    applied = dict(cam.applied_exposure)
    print(f"  color exposure: {applied.get('color_control')} auto={applied.get('color_auto_exposure')} "
          f"exposure={applied.get('color_exposure')} gain={applied.get('color_gain')}", flush=True)
    if applied.get("color_auto_exposure") is not False:
        print("  ! auto-exposure — the take will run fixed, so this pre-flight is a different camera mode.\n"
              "    pass --color-auto-exposure false [--color-exposure V] with the value the take will use.", flush=True)
    def gen():
        while True: yield cam.read()
    return gen(), applied


if __name__ == "__main__":
    raise SystemExit(main())
