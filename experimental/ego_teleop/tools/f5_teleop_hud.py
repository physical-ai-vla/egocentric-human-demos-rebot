"""Live HUD for the multi-sensor wrist teleoperation stack — the three branches, side by side, while they run.

    .venv/bin/python -m ego_teleop.tools.f5_teleop_hud --synthetic --stage virtual --protocol   # no hardware
    .venv/bin/python -m ego_teleop.tools.f5_teleop_hud --episode EP --vi-poses POSES.parquet --stage fused
    sudo .venv/bin/python -m ego_teleop.tools.f5_teleop_hud --live --stage virtual --protocol   # the real rig
    ~/xvla-mac/bin/python -m ego_teleop.tools.f5_teleop_hud --m0 --axes x --signs + --step-m 0.01  # robot safety panel

`--m0` serves a different page: the reBot safety panel (tools/hud_robot.py). That one does have ENABLE/HOLD/E-STOP, but
ENABLE only starts the fixed M0 sequence through ArmSafetyPipeline -> HttpRebotClient -> OutboundGuard; the panel
builds no command of its own. Everything below describes the fusion page.

What it shows, in one page (http://127.0.0.1:8713 by default):

    chest RGB-D  ──► p_rgbd (mapped into the VI world W)  ─┐
    wrist VI     ──► p_vi   (Arducam + IMU, local pose)    ├──► the three traces, per axis, ON ONE PLOT
    fusion       ──► p_fused = p_vi + C                    ─┘

**That overlay is the point of the tool.** Spec P5 is "record RGB-D and Arducam+IMU side by side BEFORE fusing, so
you can see which sensor is good on which axis" — a table in a report tells you that afterwards; this tells you
while the operator is still standing in front of the camera and can re-run the window. Everything else on the page
exists to explain a bad overlay: the alignment block (is `T_W_D` fitted at all, is the lever arm observable yet),
the branch lamps (which sensor dropped), the correction trace (how much of the gap the anchor is actually taking
out), the latency strip (§25: a 30 Hz tracker with a 400 ms backlog is a failure, and FPS alone will not show it).

It is an OBSERVER, never a second teleoperation system:

  * the pose the page draws is the pose the robot got — it comes out of `f4_fusion.run`'s own tick loop through
    `tick_hook`, so this cannot drift from what the report computes, and there is no second copy of the pipeline;
  * the gate table is `f4_fusion.report` run on the rows so far, not a HUD-local re-implementation;
  * it never writes to the provider, the coordinator or the robot. `--stage real` is refused here on purpose: watch
    a real arm with your eyes and the e-stop, not through a browser tab. Use `f4_fusion --stage real`.

The page is served over HTTP rather than drawn in a cv2 window because the Orbbec needs sudo on macOS and a root
process generally cannot talk to WindowServer — the same reason `rgbd_arm_hud.py` serves instead of showing."""
from __future__ import annotations
import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import load_teleop_cfg
from . import f4_fusion as F
from .p1_rgbd_arm import PROTOCOL_60S

TRACE_S = 20.0               # how much history the rolling plots hold
REPORT_EVERY_S = 0.5         # how often the gate table is recomputed from the accumulated rows
DEFAULT_PORT = 8713          # rgbd_arm_hud uses 8712; the two are often wanted at once
MAX_PLOT_POINTS = 600        # points per series in /state.json — the plots are ~1400 px wide and this is polled at
                             # ~16 Hz, so sending every tick would be bandwidth nobody can see


# ---- what the page reads ------------------------------------------------------------------------------------------
class HudState:
    """The tick loop writes, the HTTP handler reads. One lock, held only for the copy — a slow browser must never
    be able to stall the fusion, so the reader takes a snapshot and the writer never waits on a client."""

    def __init__(self, *, meta: dict, trace_s: float = TRACE_S, rate_hz: float = 50.0) -> None:
        n = max(int(trace_s * rate_hz), 64)
        self.lock = threading.Lock()
        self.meta = dict(meta)
        self.row: dict = {}
        self.stats: dict = {}
        self.gate: dict = {}
        self.report_headline: dict = {}
        self.finished = False
        self.note = ""
        self.ticks = 0
        self.t: deque[float] = deque(maxlen=n)
        # one deque per drawn series. Missing samples are NaN, never dropped: a gap in the chest branch has to look
        # like a gap on the plot, not like a straight line across the dropout.
        self.series: dict[str, deque] = {k: deque(maxlen=n) for k in (
            "fused_x", "fused_y", "fused_z", "vi_x", "vi_y", "vi_z", "rgbd_x", "rgbd_y", "rgbd_z",
            "correction_mm", "residual_mm", "residual_corrected_mm",
            "fusion_ms", "pose_age_ms", "step_mm",
            "cap_to_send_ms", "net_ms", "gpu_ms", "motion_to_pose_ms")}
        self.jpeg: dict[str, bytes] = {}
        self._last_p: np.ndarray | None = None
        self.jumps = 0
        # `engaged` is true on the single tick that fired ENGAGE, so reading it raw shows "not engaged" for the
        # whole rest of the take. The page wants the latched fact.
        self.ever_engaged = False
        self.measured_rate_hz = float("nan")
        # THE first-hardware-run gate (spec section 3): when the backend announces a map correction, the MAP pose is
        # allowed to move and T_local_control is not. Both steps are measured on the same tick so the page states
        # the gate as a number instead of implying it.
        self.map_updates = 0
        self.gate_max_local_step_mm = 0.0      # worst |dT_local_control| on a map-update tick  <- the gate
        self.gate_last = {}
        self._prev_local: np.ndarray | None = None
        self._prev_map: np.ndarray | None = None

    @staticmethod
    def _f(row, key) -> float:
        v = row.get(key)
        try: v = float(v)
        except (TypeError, ValueError): return float("nan")
        return v

    def push(self, row: dict, stats: dict) -> None:
        with self.lock:
            self.ticks += 1
            self.ever_engaged = self.ever_engaged or bool(row.get("engaged"))
            self.row = dict(row)
            self.stats = dict(stats)
            self.t.append(float(row.get("s", 0.0)))
            for a in "xyz":
                self.series[f"fused_{a}"].append(self._f(row, a) * 1000.0)
                self.series[f"vi_{a}"].append(self._f(row, f"vi_{a}") * 1000.0)
                # the anchor is drawn where it lands in the VI world (rgbd_w_*), not in the chest camera frame:
                # the raw chest-frame number is in a different frame and overlaying it would compare nothing
                self.series[f"rgbd_{a}"].append(self._f(row, f"rgbd_w_{a}") * 1000.0)
            self.series["correction_mm"].append(self._f(row, "correction_m") * 1000.0)
            self.series["residual_mm"].append(self._f(row, "residual_m") * 1000.0)
            self.series["residual_corrected_mm"].append(self._f(row, "residual_corrected_m") * 1000.0)
            self.series["fusion_ms"].append(self._f(row, "fusion_ms"))
            self.series["pose_age_ms"].append(self._f(row, "pose_age_ms"))
            self.series["step_mm"].append(self._f(row, "step_mm"))
            cap, net, gpu = (self._f(row, "vi_cap_to_send_ms"), self._f(row, "vi_net_ms"), self._f(row, "vi_server_ms"))
            self.series["cap_to_send_ms"].append(cap)
            self.series["net_ms"].append(net)
            self.series["gpu_ms"].append(gpu)
            # motion -> fused pose: the four stages plus how stale the pose was when this tick consumed it. This is
            # the number the operator feels; the stages are what you fix when it is too big (section 25).
            self.series["motion_to_pose_ms"].append(cap + net + gpu + self._f(row, "pose_age_ms"))
            loc = np.array([self._f(row, f"vi_{a}") for a in "xyz"])
            mp = np.array([self._f(row, f"vi_map_{a}") for a in "xyz"])
            if row.get("vi_map_update"):
                self.map_updates += 1
                d_loc = float(np.linalg.norm(loc - self._prev_local) * 1000.0) if (
                    self._prev_local is not None and np.isfinite(loc).all()) else float("nan")
                d_map = float(np.linalg.norm(mp - self._prev_map) * 1000.0) if (
                    self._prev_map is not None and np.isfinite(mp).all()) else float("nan")
                if np.isfinite(d_loc): self.gate_max_local_step_mm = max(self.gate_max_local_step_mm, d_loc)
                self.gate_last = dict(s=float(row.get("s", 0.0)), d_local_mm=d_loc, d_map_mm=d_map,
                                      d_visual_raw_mm=self._f(row, "d_visual_raw_m") * 1000.0,
                                      announced_mm=self._f(row, "vi_map_update_m") * 1000.0,
                                      absorbed_mm=self._f(row, "map_correction_m") * 1000.0)
            if np.isfinite(loc).all(): self._prev_local = loc
            if np.isfinite(mp).all(): self._prev_map = mp
            p = np.array([self._f(row, "x"), self._f(row, "y"), self._f(row, "z")])
            if np.isfinite(p).all() and row.get("tracking_health") != "TRACKING_LOST":
                if self._last_p is not None and np.linalg.norm(p - self._last_p) * 1000.0 > F.JUMP_MM: self.jumps += 1
                self._last_p = p
            else:
                self._last_p = None                  # never measure a jump across a gap

    def set_report(self, rep: dict) -> None:
        with self.lock:
            self.gate = rep.get("gate", {})
            self.report_headline = rep.get("headline", {})
            self.measured_rate_hz = float(rep.get("rate_hz", float("nan")))

    def snapshot(self) -> dict:
        """Hold the lock only for C-level copies; do every float check, round and dict build outside it.

        This is not micro-optimisation. The writer is the fusion tick loop and it re-acquires this lock on every
        tick; Python locks are not fair, so a reader that holds one while formatting ~16k floats gets starved by it
        and the page times out (measured, with a tight-loop writer, before the split). A HUD is allowed to miss a
        frame; it is not allowed to slow the thing it is watching, and it is not allowed to stop updating."""
        with self.lock:
            meta, row, stats, gate = self.meta, dict(self.row), dict(self.stats), self.gate
            headline, ticks, finished = dict(self.report_headline), self.ticks, self.finished
            engaged, rate, note, jumps = self.ever_engaged, self.measured_rate_hz, self.note, self.jumps
            map_gate = dict(map_updates=self.map_updates, max_local_step_mm=round(self.gate_max_local_step_mm, 3),
                            last=dict(self.gate_last))
            t = list(self.t)                                    # deque -> list is a C-level copy
            series = {k: list(v) for k, v in self.series.items()}
            frames = sorted(self.jpeg)
        clean = lambda d: {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}
        # ceil, not floor, or 1000 points with a 600 limit decimates by 1 and nothing is saved. Decimated from the
        # END so the NEWEST sample is always on the plot — on a live page the last point is the whole point.
        step = max(-(-len(t) // MAX_PLOT_POINTS), 1)
        thin = lambda v: v[::-1][::step][::-1]
        enc = lambda v: [None if not np.isfinite(x) else round(x, 4) for x in thin(v)]
        return dict(meta=meta, row=clean(row), stats=stats, gate=gate, headline=clean(headline), ticks=ticks,
                    finished=finished, ever_engaged=engaged, note=note, jumps=jumps, t=thin(t), map_gate=map_gate,
                    measured_rate_hz=None if not np.isfinite(rate) else round(rate, 2),
                    series={k: enc(v) for k, v in series.items()},
                    frames=frames, protocol=[dict(name=n, t0=a, t1=b, hint=h) for n, a, b, h in PROTOCOL_60S],
                    wall=time.time())

    def set_frame(self, name: str, img) -> None:
        """Encode a camera frame for /frame/<name>.jpg. Small and cheap: the page is for judging tracking, not focus."""
        if img is None: return
        try:
            import cv2
            a = np.asarray(img)
            if a.ndim == 2: a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            if a.dtype != np.uint8: a = np.clip(a, 0, 255).astype(np.uint8)
            h, w = a.shape[:2]
            if w > 480: a = cv2.resize(a, (480, max(int(h * 480 / w), 1)))
            ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                with self.lock: self.jpeg[name] = buf.tobytes()
        except Exception:
            pass                                      # a HUD that cannot draw a thumbnail must not stop the fusion


# ---- the tick hook ---------------------------------------------------------------------------------------------
class HudHook:
    """Called by `f4_fusion.run` / `run_live` after each fusion tick. Publishes, recomputes the gate table on a slow
    timer, and (replay only) paces the loop to wall clock so the page shows the take at the speed it was recorded."""

    def __init__(self, state: HudState, cfg, mode: str, stage: str, segments: dict, *, pace: bool, speed: float = 1.0) -> None:
        self.st, self.cfg, self.mode, self.stage, self.segments = state, cfg, mode, stage, segments
        self.pace, self.speed = pace, max(float(speed), 1e-3)
        self.rows: list[dict] = []
        self._t_wall0: float | None = None
        self._next_report = 0.0
        self._frame_at = 0.0

    def __call__(self, row: dict, prov, tick_state: dict) -> None:
        self.rows.append(row)
        self.st.push(row, prov.stats())
        s = float(row.get("s", 0.0))
        now = time.monotonic()
        if now >= self._frame_at:                      # thumbnails at ~5 Hz: JPEG encoding is not free
            self._frame_at = now + 0.2
            f = tick_state.get("last_rgbd")
            if f is not None: self.st.set_frame("chest", getattr(f, "color_bgr", None))
            w = tick_state.get("last_wrist")
            if w is not None: self.st.set_frame("wrist", w)
        if s >= self._next_report and len(self.rows) > 4:
            self._next_report = s + REPORT_EVERY_S
            try:
                df = pd.DataFrame(self.rows); df.attrs["stats"] = prov.stats()
                self.st.set_report(F.report(df, self.cfg, self.mode, self.stage, self.segments))
            except Exception as exc:                   # a metric that cannot be computed yet is not a run-stopper
                self.st.note = f"report pending: {exc}"
        if self.pace:
            if self._t_wall0 is None: self._t_wall0 = now
            slack = s / self.speed - (now - self._t_wall0)
            if slack > 0: time.sleep(min(slack, 0.5))


# ---- server ------------------------------------------------------------------------------------------------------
def serve(state: HudState, port: int, *, host: str = "127.0.0.1"):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    page = PAGE.encode()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a): pass

        def _send(self, body: bytes, ctype: str, *, cache: bool = False):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if not cache: self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/state.json":
                return self._send(json.dumps(state.snapshot(), default=str).encode(), "application/json")
            if p.startswith("/frame/"):
                name = p[len("/frame/"):].removesuffix(".jpg")
                with state.lock: jpg = state.jpeg.get(name)
                if jpg is None:
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                return self._send(jpg, "image/jpeg")
            return self._send(page, "text/html; charset=utf-8")

    srv = ThreadingHTTPServer((host, port), H)
    threading.Thread(target=srv.serve_forever, name="f5-hud", daemon=True).start()
    return srv


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>reBot one-arm visual teleoperation — live</title>
<style>
:root{--bg:#0d1215;--card:#141c21;--line:#22313a;--fg:#d7e3e8;--dim:#7b929c;
      --ok:#4cc38a;--warn:#e5a53a;--bad:#ff6f61;--rgbd:#59a8ff;--vi:#c58cff;--fused:#4cc38a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid var(--line)}
h1{font-size:13px;margin:0;letter-spacing:.06em;text-transform:uppercase}
.dim{color:var(--dim)} .grow{flex:1}
#banner{font-size:15px;color:#8de8b4}
main{display:grid;gap:10px;padding:10px 14px 24px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:9px 11px;min-width:0}
.card h2{font-size:10.5px;margin:0 0 7px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
.wide{grid-column:1/-1}
.lamps{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.lamp{border:1px solid var(--line);border-radius:5px;padding:7px 9px;background:#101a1f}
.lamp b{display:block;font-size:15px;letter-spacing:.02em}
.lamp small{color:var(--dim);display:block;font-size:10px}
table{width:100%;border-collapse:collapse}
td{padding:1.5px 0;vertical-align:baseline} td:first-child{color:var(--dim);padding-right:8px;white-space:nowrap}
td.v{text-align:right;font-variant-numeric:tabular-nums}
canvas{width:100%;display:block;border-radius:4px;background:#0b1114}
.legend{display:flex;gap:11px;margin:1px 0 5px;font-size:10px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:2px;vertical-align:middle;margin-right:3px}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.pill{padding:1px 6px;border-radius:9px;font-size:10px;border:1px solid currentColor}
#gates td:first-child{white-space:normal}
.imgs{display:flex;gap:8px;flex-wrap:wrap} .imgs figure{margin:0} .imgs img{max-width:230px;border-radius:4px;display:block}
.imgs figcaption{color:var(--dim);font-size:10px;padding-top:2px}
footer{padding:0 14px 20px;color:var(--dim)}
</style>
<header>
  <h1>reBot one-arm visual teleoperation</h1>
  <span id="meta" class="dim"></span>
  <span class="grow"></span>
  <span id="banner"></span>
  <span id="clock" class="dim"></span>
  <span id="link" class="pill">connecting</span>
</header>
<main>
  <section class="card wide">
    <div class="lamps">
      <div class="lamp"><small>chest RGB-D (anchor)</small><b id="l-rgbd">--</b><small id="l-rgbd-s"></small></div>
      <div class="lamp"><small>wrist Arducam + IMU (VI)</small><b id="l-vi">--</b><small id="l-vi-s"></small></div>
      <div class="lamp"><small>fused state</small><b id="l-fused">--</b><small id="l-fused-s"></small></div>
      <div class="lamp"><small>teleop / arm</small><b id="l-teleop">--</b><small id="l-teleop-s"></small></div>
    </div>
  </section>

  <section class="card wide">
    <h2>position, per axis — chest RGB-D vs wrist VI vs fused &nbsp;(spec P5: look here before trusting the fusion)</h2>
    <div class="legend">
      <span style="color:var(--rgbd)"><i style="background:var(--rgbd)"></i>chest RGB-D, mapped into W</span>
      <span style="color:var(--vi)"><i style="background:var(--vi)"></i>wrist VI (local)</span>
      <span style="color:var(--fused)"><i style="background:var(--fused)"></i>fused = VI + C &nbsp;(what the arm got)</span>
      <span class="dim">mm, each axis auto-scaled; gaps are dropouts, not straight lines</span>
    </div>
    <canvas id="c-x" height="120"></canvas>
    <canvas id="c-y" height="120"></canvas>
    <canvas id="c-z" height="120"></canvas>
  </section>

  <section class="card">
    <h2>fused pose (what the robot consumes)</h2>
    <table id="pose"></table>
  </section>

  <section class="card">
    <h2>anchor alignment &nbsp;T_W_D + lever arm r</h2>
    <table id="align"></table>
  </section>

  <section class="card">
    <h2>correction C(t) and residual</h2>
    <div class="legend">
      <span style="color:#59a8ff"><i style="background:#59a8ff"></i>|e| drift the anchor sees</span>
      <span style="color:#4cc38a"><i style="background:#4cc38a"></i>|e-C| still in the robot's pose</span>
      <span style="color:#e5a53a"><i style="background:#e5a53a"></i>|C| applied</span>
    </div>
    <canvas id="c-corr" height="110"></canvas>
    <table id="corr"></table>
  </section>

  <section class="card">
    <h2>latency chain &nbsp;(&sect;25: not FPS alone)</h2>
    <div class="legend">
      <span style="color:#59a8ff"><i style="background:#59a8ff"></i>capture&rarr;send</span>
      <span style="color:#e5a53a"><i style="background:#e5a53a"></i>network RTT</span>
      <span style="color:#ff6f61"><i style="background:#ff6f61"></i>GPU inference</span>
      <span style="color:#c58cff"><i style="background:#c58cff"></i>pose age</span>
      <span style="color:#4cc38a"><i style="background:#4cc38a"></i>motion&rarr;fused pose</span>
    </div>
    <canvas id="c-lat" height="110"></canvas>
    <table id="lat"></table>
  </section>

  <section class="card" id="gate-card">
    <h2>map-correction gate &nbsp;<span class="dim">&mdash; the map may move, T_local_control may not</span></h2>
    <table id="mapgate"></table>
  </section>

  <section class="card">
    <h2>teleop / arm command</h2>
    <table id="teleop"></table>
  </section>

  <section class="card wide" id="cams-card" hidden>
    <h2>cameras</h2>
    <div class="imgs" id="cams"></div>
  </section>

  <section class="card wide">
    <h2>gates &nbsp;<span class="dim">— f4_fusion.report on the rows so far. Thresholds are pre-measurement starting points.</span></h2>
    <table id="gates"></table>
  </section>
</main>
<footer id="note"></footer>
<script>
const $ = id => document.getElementById(id);
const COL = {rgbd:"#59a8ff", vi:"#c58cff", fused:"#4cc38a"};
const HCOL = s => /OK$/.test(s||"") ? "var(--ok)" : /DEGRADED/.test(s||"") ? "var(--warn)"
                : /LOST|HOLD/.test(s||"") ? "var(--bad)" : "var(--dim)";
const num = (v, d=1, suf="") => (v===null||v===undefined||Number.isNaN(v)) ? "--" : (+v).toFixed(d)+suf;

function rows(el, pairs){
  el.innerHTML = pairs.map(([k,v,cls]) =>
    `<tr><td>${k}</td><td class="v ${cls||""}">${v}</td></tr>`).join("");
}

function plot(cv, t, series, opts){
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = +cv.getAttribute("height");
  if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)) { cv.width = w*dpr; cv.height = h*dpr; }
  const g = cv.getContext("2d");
  g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  g.fillStyle = "#0b1114"; g.fillRect(0,0,w,h);
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.v) if (v !== null) { if (v<lo) lo=v; if (v>hi) hi=v; }
  if (!isFinite(lo)) { g.fillStyle="#44585f"; g.font="11px monospace"; g.fillText((opts.label||"")+"  waiting for data", 6, 15); return; }
  if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const pad = Math.max((hi-lo)*0.12, 1e-3); lo -= pad; hi += pad;
  const X = i => 4 + i/(Math.max(t.length-1,1)) * (w-8);
  const Y = v => h-14 - (v-lo)/(hi-lo) * (h-22);
  if (lo < 0 && hi > 0) { g.strokeStyle="#1d2a31"; g.beginPath(); g.moveTo(0,Y(0)); g.lineTo(w,Y(0)); g.stroke(); }
  for (const s of series){
    g.strokeStyle = s.c; g.lineWidth = s.w || 1.2; g.beginPath();
    let pen = false;
    s.v.forEach((v,i) => { if (v===null) { pen=false; return; }
      const x=X(i), y=Y(v); if(!pen){g.moveTo(x,y);pen=true;} else g.lineTo(x,y); });
    g.stroke();
  }
  g.fillStyle = "#6f8791"; g.font = "10px ui-monospace,monospace";
  g.fillText(opts.label||"", 5, 11);
  g.fillText(hi.toFixed(opts.d??1), w-46, 11);
  g.fillText(lo.toFixed(opts.d??1), w-46, h-4);
}

let stale = 0;
async function tick(){
  let s;
  try { s = await (await fetch("/state.json", {cache:"no-store"})).json(); }
  catch(e){ stale++; $("link").textContent = "no server"; $("link").className="pill bad"; return setTimeout(tick, 700); }
  stale = 0;
  const r = s.row || {}, st = s.stats || {}, al = st.alignment || {};
  $("link").textContent = s.finished ? "finished" : "live";
  $("link").className = "pill " + (s.finished ? "warn" : "ok");
  $("meta").textContent = `${s.meta.stage} / ${s.meta.mode} / ${s.meta.side}  @${s.meta.rate_hz}Hz  backend=${s.meta.vi_backend}  src=${s.meta.source}`;
  $("clock").textContent = `t=${num(r.s,1)}s   ${s.ticks} ticks`;

  const seg = (s.protocol||[]).find(p => r.s >= p.t0 && r.s < p.t1);
  $("banner").textContent = (s.meta.protocol && seg) ? `${seg.name.toUpperCase()}  ${Math.ceil(seg.t1-r.s)}s  —  ${seg.hint}` : "";

  const lamp = (id, txt, sub) => { const e=$(id); e.textContent = txt||"--"; e.style.color = HCOL(txt); $(id+"-s").textContent = sub||""; };
  lamp("l-rgbd", r.rgbd_health || (s.meta.mode==="vi_only" ? "not used (ablation B)" : "--"),
       `${st.rgbd_samples??0} samples  ${r.anchored ? "anchored" : "not anchored"}`);
  lamp("l-vi", r.vi_health, `${st.vi_samples??0} samples  pairs ${st.pairs??0}`);
  lamp("l-fused", r.fusion_state, r.fusion_reason || "");
  lamp("l-teleop", r.state || (s.meta.stage==="compare" ? "observer only" : "--"),
       [s.ever_engaged?"ENGAGED":"not engaged", r.clutch_state?"CLUTCH (arm frozen)":"", r.hold_reason||"", r.safety||""].filter(Boolean).join("  "));

  const t = s.t, S = s.series;
  for (const a of ["x","y","z"])
    plot($("c-"+a), t, [{v:S["rgbd_"+a], c:COL.rgbd}, {v:S["vi_"+a], c:COL.vi}, {v:S["fused_"+a], c:COL.fused, w:1.7}],
         {label:a.toUpperCase()+" mm"});
  plot($("c-corr"), t, [{v:S.residual_mm, c:"#59a8ff"}, {v:S.residual_corrected_mm, c:"#4cc38a", w:1.7}, {v:S.correction_mm, c:"#e5a53a"}],
       {label:"mm", zero:true});
  plot($("c-lat"), t, [{v:S.cap_to_send_ms, c:"#59a8ff"}, {v:S.net_ms, c:"#e5a53a"}, {v:S.gpu_ms, c:"#ff6f61"},
                       {v:S.pose_age_ms, c:"#c58cff"}, {v:S.motion_to_pose_ms, c:"#4cc38a", w:1.7}],
       {label:"ms", zero:true, d:1});

  rows($("pose"), [
    ["x", num(r.x*1000,1," mm")], ["y", num(r.y*1000,1," mm")], ["z", num(r.z*1000,1," mm")],
    ["speed", num(Math.hypot(r.vx||0,r.vy||0,r.vz||0)*1000,0," mm/s")],
    ["health", r.tracking_health||"--", /OK/.test(r.tracking_health||"")?"ok":/DEGRADED/.test(r.tracking_health||"")?"warn":"bad"],
    ["held (no new sample)", r.held ? "YES" : "no", r.held?"bad":""],
    ["stable gate", r.stable ? "STABLE" : "not stable", r.stable?"ok":"warn"],
    ["catastrophic jumps >30mm", s.jumps, s.jumps?"bad":"ok"],
    ["map corrections absorbed", `${st.local_pose?.map_absorbed ?? 0} abs / ${st.local_pose?.map_refused ?? 0} refused`],
    ["map correction cumulative", num((st.local_pose?.map_correction_cum_m ?? 0)*1000,1," mm")],
  ]);

  rows($("align"), [
    ["aligned", al.aligned ? "YES" : "no", al.aligned?"ok":"warn"],
    ["reason", al.reason || "--"],
    ["fit RMS", num((al.rms_m??NaN)*1000,1," mm")],
    ["pairs / span", `${al.n_pairs??0} / ${num((al.span_m??NaN)*1000,0," mm")}`],
    ["rotation span", num(al.rotation_span_deg,1,"&deg;")],
    ["lever arm r", (al.lever_arm_m||[]).map(v=>num(v*1000,1)).join(", ")+" mm"],
    ["r sigma", num((al.lever_arm_sigma_m??NaN)*1000,2," mm")],
    ["r observable", al.lever_arm_observable ? "YES" : "no — rotate the wrist without translating", al.lever_arm_observable?"ok":"warn"],
    ["fits / re-alignments", `${al.n_fits ?? 0} / ${al.n_realign ?? 0}`],
  ]);

  rows($("corr"), [
    ["|e| drift anchor sees", num(r.residual_m*1000,1," mm")],
    ["|e-C| left in the pose", num(r.residual_corrected_m*1000,1," mm")],
    ["|C| applied", num(r.correction_m*1000,1," mm")],
    ["pending (deferred to clutch)", num(r.correction_pending_m*1000,1," mm")],
    ["max rate", num((s.meta.max_correction_rate_m_s??0)*1000,1," mm/s — this is arm velocity")],
  ]);

  const p95 = a => { const v=(a||[]).filter(x=>x!==null).sort((x,y)=>x-y); return v.length?v[Math.floor(v.length*0.95)]:NaN; };
  const med = a => { const v=(a||[]).filter(x=>x!==null).sort((x,y)=>x-y); return v.length?v[Math.floor(v.length*0.5)]:NaN; };
  const m2p_med = med(S.motion_to_pose_ms), m2p_p95 = p95(S.motion_to_pose_ms);
  rows($("lat"), [
    ["capture &rarr; send p95", num(p95(S.cap_to_send_ms),1," ms")],
    ["network RTT p95", num(p95(S.net_ms),1," ms")],
    ["GPU inference p95", num(p95(S.gpu_ms),1," ms")],
    ["pose age p95", num(p95(S.pose_age_ms),1," ms")],
    ["<b>motion &rarr; fused median</b>", num(m2p_med,1," ms"), m2p_med>100?"bad":"ok"],
    ["<b>motion &rarr; fused p95</b>", num(m2p_p95,1," ms"), m2p_p95>150?"bad":"ok"],
    ["fusion compute p95", num(p95(S.fusion_ms),2," ms")],
    ["tick rate measured", num(s.measured_rate_hz,1,` Hz  (nominal ${s.meta.rate_hz})`),
     (s.measured_rate_hz!==null && s.measured_rate_hz < s.meta.rate_hz*0.9) ? "bad" : ""],
    ["exec latency", num(r.exec_latency_ms,1," ms")],
  ]);

  rows($("teleop"), [
    ["coordinator state", r.state || "--"],
    ["engaged / clutch", `${s.ever_engaged?"yes":"no"} / ${r.clutch_state?"CLUTCH":"open"}`],
    ["TCP x", num(r.tcp_x*1000,1," mm")], ["TCP y", num(r.tcp_y*1000,1," mm")], ["TCP z", num(r.tcp_z*1000,1," mm")],
    ["command step", num(r.step_mm,2," mm"), (r.step_mm>10)?"bad":""],
    ["workspace clamped", r.clamped ? "CLAMPED" : "no", r.clamped?"warn":""],
    ["safety", r.safety || "-"],
    ["sent to arm", r.arm_sent ? "YES" : "no (virtual)", r.arm_sent?"bad":"dim"],
  ]);

  const mg = s.map_gate || {}, gl = mg.last || {};
  rows($("mapgate"), [
    ["map updates announced", mg.map_updates ?? 0],
    ["<b>max |&Delta;T_local_control| on one</b>", num(mg.max_local_step_mm,3," mm"),
     (mg.map_updates && mg.max_local_step_mm > 1.0) ? "bad" : (mg.map_updates ? "ok" : "")],
    ["&mdash; last event at", gl.s===undefined ? "none yet" : num(gl.s,1," s")],
    ["&nbsp;&nbsp;&Delta;T_visual_raw (pre-opt)", num(gl.d_visual_raw_mm,2," mm")],
    ["&nbsp;&nbsp;&Delta;T_map (post-opt)", num(gl.d_map_mm,2," mm")],
    ["&nbsp;&nbsp;&Delta;T_local_control", num(gl.d_local_mm,3," mm"), (gl.d_local_mm>1.0)?"bad":""],
    ["&nbsp;&nbsp;announced by server", num(gl.announced_mm,2," mm")],
    ["&nbsp;&nbsp;absorbed by continuity", num(gl.absorbed_mm,2," mm")],
    // == null catches both undefined and a null column: a replay row HAS the key and leaves it null, so an
    // ===undefined test reported "yes" for a take with no backend behind it at all.
    ["backend reports visual_raw", ((s.row||{}).d_visual_raw_m == null) ? "no (replay / no backend)" : "yes"],
  ]);

  const g = s.gate || {};
  $("gates").innerHTML = Object.keys(g).length ? Object.entries(g).map(([k,v]) =>
    `<tr><td>${k}</td><td class="v">${num(v.value,2)}</td><td class="v dim">thr ${num(v.threshold,2)}</td>`+
    `<td class="v ${v.passed?"ok":"bad"}">${v.passed?"PASS":"FAIL"}${v.note?" ("+v.note+")":""}</td></tr>`).join("")
    : `<tr><td class="dim">filling the first window…</td></tr>`;

  if ((s.frames||[]).length){
    $("cams-card").hidden = false;
    const want = s.frames.join(",");
    if ($("cams").dataset.have !== want)
      $("cams").innerHTML = s.frames.map(n => `<figure><img id="img-${n}"><figcaption>${n}</figcaption></figure>`).join("");
    $("cams").dataset.have = want;
    for (const n of s.frames){ const im=$("img-"+n); if(im) im.src = `/frame/${n}.jpg?t=${Date.now()}`; }
  }
  $("note").textContent = [s.note, s.meta.note].filter(Boolean).join("  |  ");
  setTimeout(tick, s.finished ? 1000 : 60);
}
tick();
</script></html>"""


# ---- driving the pipeline ----------------------------------------------------------------------------------------
def run_hud(args) -> int:
    import dataclasses as dc
    teleop = load_teleop_cfg(args.config_dir) if args.config_dir else load_teleop_cfg()
    cfg = teleop.fused_wrist
    mode = args.mode or cfg.mode
    segments = F.parse_segments(args.segment)
    if args.protocol: segments = {n: (a, b) for n, a, b, _ in PROTOCOL_60S} | segments

    state = HudState(meta=dict(stage=args.stage, mode=mode, side=cfg.side, rate_hz=cfg.rate_hz,
                               vi_backend=cfg.vi_backend, protocol=bool(args.protocol),
                               source="live" if args.live else ("synthetic" if args.synthetic else "episode"),
                               max_correction_rate_m_s=cfg.provider.fusion.max_correction_rate_m_s, note=""),
                     rate_hz=cfg.rate_hz)
    hook = HudHook(state, cfg, mode, args.stage, segments, pace=not args.live, speed=args.speed)
    srv = serve(state, args.port, host=args.host)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  HUD: {url}    (raw state: {url}state.json)\n", flush=True)
    if args.open_browser:
        import webbrowser; webbrowser.open(url)

    rc = 0
    try:
        if args.live:
            rc = F.run_live(args, teleop, segments, tick_hook=hook)
        else:
            events, rgbd_provider, vi_provider, note = F.build_events(args, cfg, teleop)
            state.meta["note"] = note
            df = F.run(events=events, cfg=cfg, mode=mode, stage=args.stage, rgbd_provider=rgbd_provider,
                       vi_provider=vi_provider, engage_at_s=args.engage_at, clutch_at_s=args.clutch_at,
                       release_at_s=args.release_at, tick_hook=hook)
            rep = F.report(df, cfg, mode, args.stage, segments)
            if note: rep["note"] = note
            state.set_report(rep)
            F.print_report(rep)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(dict(reports={mode: rep}), indent=1, default=str))
                print(f"\nreport -> {args.out}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        state.finished = True
    if args.hold:
        print(f"\n  take finished — the page stays up at {url}   (ctrl-C to stop)", flush=True)
        try:
            while True: time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    srv.shutdown()
    return rc


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--m0" in argv:                        # the robot safety panel (M0 jog behind ENABLE/HOLD/E-STOP): tools/hud_robot.py
        from .hud_robot import main as robot_main
        return robot_main(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("compare", "fused", "virtual"), default="virtual",
                    help="same stages as f4_fusion. `real` is deliberately absent: see the module docstring")
    ap.add_argument("--mode", choices=F.ABLATIONS, default=None, help="ablation arm (default: the config's mode)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="generated 60 s take — plumbing/demo, never evidence")
    src.add_argument("--episode", type=Path, help="recorded take with head/ RGB-D frames (needs --vi-poses)")
    src.add_argument("--live", action="store_true", help="the three real sensors (macOS: needs sudo for the Orbbec)")
    ap.add_argument("--vi-poses", type=Path, help="wrist VI trajectory parquet (m1_vio output)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to read the page from a phone on the same network")
    ap.add_argument("--open", dest="open_browser", action="store_true", help="open the page in a browser")
    ap.add_argument("--speed", type=float, default=1.0, help="replay pacing multiplier (live ignores it)")
    ap.add_argument("--hold", action="store_true", help="keep serving after the take ends")
    ap.add_argument("--protocol", action="store_true", help="the 60 s protocol windows + the operator banner")
    ap.add_argument("--segment", action="append", default=[], metavar="name:t0:t1")
    ap.add_argument("--engage-at", type=float, default=3.0)
    ap.add_argument("--clutch-at", type=float, default=None)
    ap.add_argument("--release-at", type=float, default=None)
    ap.add_argument("--live-seconds", type=float, default=60.0)
    ap.add_argument("--hardware", default="handumi_v1", help="--live: collector profile for the WRIST camera + IMU")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--synthetic-frames", type=int, default=None)
    ap.add_argument("--record", type=Path, help="--live: write the fused stream into this episode directory")
    ap.add_argument("--out", type=Path, help="write the final report json here")
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--allow-identity-mount", action="store_true")
    args = ap.parse_args(argv)
    if args.episode and not args.vi_poses:
        raise SystemExit("--episode needs --vi-poses PARQUET (the wrist VI trajectory m1_vio produced)")
    args.ablation = False
    args.i_am_at_the_robot = False
    return run_hud(args)


if __name__ == "__main__":
    raise SystemExit(main())
