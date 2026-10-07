"""Robot safety panel for f5_teleop_hud (`--m0`): the reBot's state at a glance, and the M0 jog behind four buttons.

    ~/xvla-mac/bin/python -m ego_teleop.tools.f5_teleop_hud --m0 --axes x --signs + --step-m 0.01            # dry run
    ~/xvla-mac/bin/python -m ego_teleop.tools.f5_teleop_hud --m0 --axes x --step-m 0.01 --execute --i-am-at-the-robot

It is a monitoring and control SURFACE, not a control path:
  * OBSERVE ONLY (the default) reads /observe and /motor_states. Nothing is ever sent.
  * ENABLE runs `m0_eef_jog.run` — the configured sequence, fixed at launch (axes/signs/step), through
    ArmSafetyPipeline -> HttpRebotClient (IK, joint guards, leader conversion) -> OutboundGuard -> robot_service.
    The page has no button that moves the robot freely, and no code here builds an /execute_step vector.
    ENABLE is refused unless every pre-flight gate passes (motors, fresh observation, IK, joint bounds, service).
    Without --execute, ENABLE runs the same sequence as a dry run (guarded, logged, never POSTed).
  * HOLD stops the command stream at the next tick (the follower keeps its last setpoint).
  * E-STOP stops the stream AND posts robot_service /estop. Clearing the robot's E-STOP is left to the operator.
  * CLUTCH / RECENTER belong to teleoperation (M1+) and are shown disabled here.
Every abort reason — measured stop condition, guard refusal, motor status change, operator HOLD — is shown as a banner."""
from __future__ import annotations
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from ..robot.rebot_client import HttpRebotClient, JointLimits, SIDE_SLICE, GRIPPER_INDEX, unwrap_grip, grip_hold_command
from . import m0_eef_jog as M0

ARM_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")


class RobotPanel:
    """Owns the M0 runner thread and the read-only monitor. The HTTP handler only calls enable/hold/estop/snapshot."""

    def __init__(self, *, client: HttpRebotClient, guard: M0.OutboundGuard, base_http, axes="x", signs=(+1,), step=0.01,
                 hz=30.0, max_lin_vel=0.02, dwell_s=3.0, out_root: Path = Path("runs"), monitor_hz: float = 4.0) -> None:
        self.client, self.guard, self.base = client, guard, base_http
        self.cfg = dict(side=client.side, axes=axes, signs="".join("+" if s > 0 else "-" for s in signs), step_mm=step * 1e3,
                        hz=hz, max_lin_vel_mm_s=max_lin_vel * 1e3, dwell_s=dwell_s, live=guard.live)
        self._run_kw = dict(axes=axes, signs=signs, step=step, hz=hz, max_lin_vel=max_lin_vel, dwell_s=dwell_s)
        self.out_root = out_root; self.monitor_dt = 1.0 / monitor_hz
        self.lock = threading.Lock(); self.stop_ev = threading.Event(); self._quit = threading.Event()
        self.worker: threading.Thread | None = None
        self.s = dict(mode="OBSERVE", banner="", motors={}, joints_deg=[], obs_age_ms=None, t_obs=None, service=dict(ok=False, error=""),
                      robot_estop=None, tcp=None, target=None, cmd=None, move=None, progress=0.0, live_metrics={}, moves=[],
                      checks={}, gates={}, loop_hz=None, run_dir=None)
        self._move_rows: list = []; self._tick_t: list = []; self._grip0 = None

    # ---- monitor (read-only) -----------------------------------------------------------------------------------
    def _read_robot(self) -> None:
        try:
            st = self.client.observe(); m = self.base("GET", "/motor_states", None); stat = self.base("GET", "/status", None)
            motors = {a: {k: (v.get("status") if isinstance(v, dict) else None) for k, v in m.get(a, {}).items()} for a in ("left_arm", "right_arm")}
            with self.lock:
                self.s.update(motors=motors, joints_deg=[round(float(v), 2) for v in st.joints_deg_all], t_obs=time.time(),
                              tcp=[round(float(v), 4) for v in st.T_RB_RE[:3, 3]], service=dict(ok=True, error=""),
                              robot_estop=bool(stat.get("estop")))
        except Exception as exc:
            with self.lock: self.s["service"] = dict(ok=False, error=str(exc)[:160])

    def monitor_loop(self) -> None:
        while not self._quit.is_set():
            if self.worker is None or not self.worker.is_alive(): self._read_robot()
            time.sleep(self.monitor_dt)

    # ---- pre-flight --------------------------------------------------------------------------------------------
    def preflight(self) -> dict:
        self._read_robot()
        with self.lock: s = json.loads(json.dumps(self.s, default=str))
        g = {}
        g["service"] = (s["service"]["ok"] and s["robot_estop"] is False, "connected, E-STOP clear" if s["service"]["ok"] else s["service"]["error"])
        bad = {f"{a}.{k}": v for a, ms in s["motors"].items() for k, v in ms.items() if k != "gripper" and v != 1}
        g["motors"] = (bool(s["motors"]) and not bad, "all arm motors status 1" if not bad else f"not enabled: {bad}")
        age = (time.time() - s["t_obs"]) * 1e3 if s["t_obs"] else None
        g["observation"] = (age is not None and age < 1000.0, f"age {age:.0f} ms" if age is not None else "none")
        try:
            q = self.client._last.q_rad; lim = self.client.limits or JointLimits(); why = lim.check(q, q)
            g["joint_bounds"] = (not why, "inside the C30 teleop range" if not why else ",".join(why))
        except Exception as exc:
            g["joint_bounds"] = (False, str(exc)[:120])
        g["ik"] = (self.client.ik is not None and s["tcp"] is not None, "FK/IK loaded" if self.client.ik is not None else "no solver")
        with self.lock: self.s["gates"] = {k: dict(ok=bool(v[0]), detail=v[1]) for k, v in g.items()}
        return self.s["gates"]

    # ---- controls ----------------------------------------------------------------------------------------------
    def enable(self) -> tuple[bool, str]:
        if self.worker is not None and self.worker.is_alive(): return False, "already running"
        gates = self.preflight()
        failed = [k for k, v in gates.items() if not v["ok"]]
        if failed: return False, f"pre-flight failed: {failed}"
        self.stop_ev.clear(); self._move_rows = []; self._tick_t = []
        self._grip0 = np.asarray([unwrap_grip(self.client._last.joints_deg_all[i]) for i in GRIPPER_INDEX.values()])
        run_dir = self.out_root / f"m0hud_{time.strftime('%Y%m%d_%H%M%S')}_{self.client.side}{'_LIVE' if self.guard.live else '_dry'}"
        with self.lock: self.s.update(mode="EXECUTING" if self.guard.live else "DRY_RUN", banner="", moves=[], run_dir=str(run_dir))
        self.worker = threading.Thread(target=self._work, args=(run_dir,), daemon=True); self.worker.start()
        return True, "started"

    def _work(self, run_dir: Path) -> None:
        try:
            summary = M0.run(self.client, self.guard, out=run_dir, show=0, rules=M0.StopRules(), watch_http=self.base,
                             on_tick=self._on_tick, stop=self.stop_ev, on_move=self._on_move, **self._run_kw)
            last = summary[-1] if summary else {}
            with self.lock:
                if last.get("move") in ("ABORT", "GUARD", "INTERRUPTED"):
                    self.s.update(mode="ESTOP" if self.s["mode"] == "ESTOP" else ("HOLD" if "operator HOLD" in last.get("error", "") else "ABORTED"),
                                  banner=last.get("error", last.get("move")))
                else:
                    self.s.update(mode="DONE", banner="sequence complete" + ("" if all(m.get("failed", 1) == 0 for m in summary) else " -- with refused ticks"))
        except Exception as exc:                                  # pre-flight watch refusal or anything unexpected
            with self.lock: self.s.update(mode="ABORTED", banner=f"{type(exc).__name__}: {exc}")

    def _on_tick(self, r: dict, ctx: dict) -> None:
        now = time.time(); self._tick_t.append(now); self._tick_t = self._tick_t[-60:]
        if not self._move_rows or self._move_rows[-1]["move"] != r["move"]: self._move_rows = []
        self._move_rows.append(r)
        d = ctx["dist_mm"]; mv = self._move_rows
        def first(pred):
            hit = [x["el_s"] for x in mv if pred(x)]
            return round(hit[0] * 1e3) if hit else None
        st = ctx["st"]
        g = [round(unwrap_grip(st.joints_deg_all[i]), 1) for i in GRIPPER_INDEX.values()]
        jd = None if self._grip0 is None else float(np.max(np.abs(np.asarray(g) - self._grip0)))
        lm = dict(t2mm_ms=first(lambda x: x["along_mm"] >= 2.0), t50_ms=first(lambda x: d > 0 and x["along_mm"] >= 0.5 * d),
                  t90_ms=first(lambda x: d > 0 and x["along_mm"] >= 0.9 * d), along_mm=round(r["along_mm"], 2), commanded_mm=round(d, 2),
                  err_to_target_mm=round(r["err_to_target_mm"], 2), off_axis_mm=round(r["off_axis_mm"], 2),
                  other_arm_deg=round(r["other_arm_max_deg"], 3), jaw_raw=g, jaw_drift=None if jd is None else round(jd, 1), ik_mm=None if r["ik_mm"] is None else round(r["ik_mm"], 2),
                  ok=r["ok"], err=r["err"])
        hz = (len(self._tick_t) - 1) / (self._tick_t[-1] - self._tick_t[0]) if len(self._tick_t) > 2 else None
        with self.lock:
            self.s.update(move=ctx["move"], target=[round(v, 4) for v in ctx["target"]], cmd=[round(r["cmd_x"], 4), round(r["cmd_y"], 4), round(r["cmd_z"], 4)],
                          tcp=[round(r["meas_x"], 4), round(r["meas_y"], 4), round(r["meas_z"], 4)], live_metrics=lm,
                          progress=float(np.clip(r["along_mm"] / d, 0, 1.2)) if d > 0 else 1.0,
                          joints_deg=[round(float(v), 2) for v in st.joints_deg_all], t_obs=now, loop_hz=None if hz is None else round(hz, 1))

    def _on_move(self, summary: dict) -> None:
        with self.lock: self.s["moves"] = self.s["moves"] + [summary]

    def hold(self) -> str:
        self.stop_ev.set()
        with self.lock:
            if self.worker is None or not self.worker.is_alive(): self.s.update(mode="OBSERVE", banner="")
        return "hold requested"

    def estop(self) -> tuple[bool, str]:
        """Stop our own stream first, then robot_service /estop — which freezes the arms where they are and refuses
        every later /execute_step from ANY client (CLI tools, other UIs), until /clear_estop. Retried once."""
        self.stop_ev.set(); err = ""
        for _ in range(2):
            try:
                r = self.base("POST", "/estop", {})
                if r.get("ok"): break
                err = str(r)
            except Exception as exc:
                err = str(exc)[:160]
        else:
            with self.lock: self.s.update(mode="ESTOP", banner=f"E-STOP REQUEST FAILED ({err}) -- CUT ROBOT POWER")
            return False, f"robot_service /estop failed: {err}"
        with self.lock: self.s.update(mode="ESTOP", banner="E-STOP: robot_service frozen at the current pose; every command from any client is refused")
        self._read_robot()
        return True, "E-STOP engaged"

    def clear_estop(self) -> tuple[bool, str]:
        if self.worker is not None and self.worker.is_alive(): return False, "a run is still stopping"
        try: r = self.base("POST", "/clear_estop", {})
        except Exception as exc: return False, str(exc)[:160]
        self._read_robot()
        with self.lock: self.s.update(mode="OBSERVE", banner="")
        return bool(r.get("ok")), "E-STOP cleared -- OBSERVE ONLY"

    def snapshot(self) -> dict:
        with self.lock: s = json.loads(json.dumps(self.s, default=str))
        s["obs_age_ms"] = round((time.time() - s["t_obs"]) * 1e3) if s["t_obs"] else None
        s["cfg"] = self.cfg; s["running"] = bool(self.worker is not None and self.worker.is_alive())
        j = s["joints_deg"]
        if len(j) == 14:
            s["jaws"] = {side: dict(raw=j[i], hold_cmd=round(grip_hold_command(j[i]), 2)) for side, i in GRIPPER_INDEX.items()}
            s["arms"] = {side: j[sl] for side, sl in SIDE_SLICE.items()}
        lm = s.get("live_metrics") or {}
        s["checks"] = {
            "inactive arm drift": _grade(lm.get("other_arm_deg"), 0.25, 0.5, "deg"),
            "jaw drift": _grade(lm.get("jaw_drift"), 1.5, 3.0, "raw"),
            "TCP wrong direction": _grade(None if lm.get("along_mm") is None else max(0.0, -lm["along_mm"]), 2.0, 4.0, "mm"),
            "overshoot": _grade(None if lm.get("along_mm") is None or not lm.get("commanded_mm") else max(0.0, lm["along_mm"] - lm["commanded_mm"]), 2.5, 5.0, "mm"),
            "off-axis": _grade(lm.get("off_axis_mm"), 3.0, 6.0, "mm"),
            "IK residual": _grade(lm.get("ik_mm"), 4.0, 8.0, "mm"),
            "service": dict(level="ok" if s["service"]["ok"] else "abort", text="OK" if s["service"]["ok"] else s["service"]["error"]),
            "motor status": _motor_grade(s["motors"]),
        }
        if s["mode"] in ("ABORTED", "ESTOP") and s["banner"]:
            for k in s["checks"]:
                if k.split()[0].lower() in s["banner"].lower(): s["checks"][k]["level"] = "abort"
        return s

    def close(self) -> None:
        self._quit.set(); self.stop_ev.set()


def _grade(v, warn, abort, unit) -> dict:
    if v is None: return dict(level="idle", text="—")
    return dict(level="abort" if v > abort else ("warn" if v > warn else "ok"), text=f"{v:.2f} {unit}")


def _motor_grade(motors: dict) -> dict:
    bad = {f"{a}.{k}": v for a, ms in motors.items() for k, v in ms.items() if k != "gripper" and v != 1}
    if not motors: return dict(level="idle", text="—")
    return dict(level="abort" if bad else "ok", text="ALL ARM MOTORS NORMAL" if not bad else ", ".join(f"{k}={v}" for k, v in bad.items()))


def serve(panel: RobotPanel, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"): return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/state.json": return self._send(json.dumps(panel.snapshot()).encode(), "application/json")
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            act = {"/api/enable": lambda: panel.enable(), "/api/hold": lambda: (True, panel.hold()),
                   "/api/estop": lambda: panel.estop(), "/api/clear_estop": lambda: panel.clear_estop(), "/api/preflight": lambda: (True, panel.preflight())}.get(self.path)
            if act is None: return self._send(b"not found", "text/plain", 404)
            ok, msg = act(); self._send(json.dumps(dict(ok=ok, msg=msg)).encode(), "application/json")

    srv = ThreadingHTTPServer((host, port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def build(args) -> RobotPanel:
    from ..robot.rebot_client import PyrokiIkSolver
    ik = PyrokiIkSolver.for_rebot_b601(args.side)
    base = HttpRebotClient(args.robot, args.side, None, timeout_s=2.0)._requests_http
    guard = M0.OutboundGuard(base, args.side, live=args.execute, max_step_rad=np.radians(args.max_step_deg))
    client = HttpRebotClient(args.robot, args.side, ik, limits=JointLimits(max_step_rad=np.radians(args.max_step_deg)), http=guard, timeout_s=2.0)
    signs = tuple(+1 if c == "+" else -1 for c in args.signs)
    return RobotPanel(client=client, guard=guard, base_http=base, axes=args.axes, signs=signs, step=args.step_m, hz=args.hz,
                      max_lin_vel=args.max_lin_vel, dwell_s=args.dwell_s)


def run_panel(args) -> int:
    if args.execute and not args.i_am_at_the_robot: raise SystemExit("--execute needs --i-am-at-the-robot (hand on the e-stop)")
    if args.step_m > 0.05: raise SystemExit("--step-m above 5 cm is not an M0 jog")
    panel = build(args)
    threading.Thread(target=panel.monitor_loop, daemon=True).start()
    srv = serve(panel, args.port, args.host)
    url = f"http://{args.host}:{args.port}/"
    print(f"[panel] {url}   side={args.side} sequence={args.axes}{args.signs} {args.step_m*1e3:.0f} mm   "
          f"{'LIVE -- ENABLE sends to the robot' if args.execute else 'DRY RUN -- nothing is ever sent'}", flush=True)
    if args.open_browser:
        import webbrowser; webbrowser.open(url)
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        panel.hold(); panel.close(); srv.shutdown()
    return 0


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>reBot Safety Panel</title><style>
:root{--bg:#0f1115;--card:#171a21;--line:#2a2f3a;--fg:#e6e8ec;--dim:#8b93a3;--ok:#2fbf71;--warn:#e0a43a;--bad:#e5484d;--info:#4c8dff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,system-ui,sans-serif;padding:16px 16px 96px}
h1{font-size:18px;margin:0}header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px}
.pill{padding:6px 14px;border-radius:999px;font-weight:700;letter-spacing:.04em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.card h2{font-size:12px;color:var(--dim);margin:0 0 8px;text-transform:uppercase;letter-spacing:.08em}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td{padding:3px 4px;border-bottom:1px solid var(--line)}td.r{text-align:right}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
.ok{color:var(--ok)}.warn{color:var(--warn)}.abort{color:var(--bad);font-weight:700}.idle{color:var(--dim)}
#banner{display:none;border:3px solid var(--bad);background:#2a1215;border-radius:10px;padding:16px;margin-bottom:12px;text-align:center}
#banner b{display:block;font-size:22px;color:var(--bad);margin-bottom:6px}
.bar{height:14px;background:#0b0d11;border:1px solid var(--line);border-radius:7px;overflow:hidden}.bar i{display:block;height:100%;background:var(--info)}
.mono{font-family:ui-monospace,Menlo,monospace}.big{font-size:20px;font-weight:700}
footer{position:fixed;left:0;right:0;bottom:0;background:#0b0d11;border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;font-weight:700;border:0;border-radius:8px;padding:12px 16px;cursor:pointer;color:#fff;background:#2a2f3a}
button:disabled{opacity:.35;cursor:not-allowed}#b-enable{background:#1f6f43}#b-hold{background:#8a6414}#b-estop{background:var(--bad);margin-left:auto;min-width:220px;font-size:20px;padding:14px 20px}#b-clear{background:#3a2a2a}
#msg{align-self:center;color:var(--dim)}
</style></head><body>
<header><h1>reBot Safety Panel <span class="idle" id="cfg"></span></h1><span class="pill" id="mode">…</span></header>
<div id="banner"><b id="btitle">TELEOP ABORTED</b><div id="btext"></div><div class="idle">Command stream stopped · last safe target retained by the follower</div></div>
<div class="grid">
 <div class="card"><h2>Robot</h2><table id="robot"></table></div>
 <div class="card"><h2>Motors (status 1 = enabled)</h2><table id="motors"></table></div>
 <div class="card"><h2>M0 move</h2><div class="big" id="move">—</div><div class="bar" style="margin:8px 0"><i id="prog" style="width:0%"></i></div><table id="m0"></table></div>
 <div class="card"><h2>Safety</h2><table id="checks"></table></div>
 <div class="card"><h2>Pre-flight gates (ENABLE needs all)</h2><table id="gates"></table></div>
 <div class="card"><h2>Completed moves</h2><table id="moves"></table></div>
</div>
<footer><button id="b-observe" disabled title="default mode: read-only">OBSERVE ONLY</button><button id="b-enable">ENABLE</button>
<button id="b-clutch" disabled title="teleoperation (M1+)">CLUTCH / RECENTER</button><button id="b-hold">HOLD</button>
<span id="msg"></span><button id="b-clear" disabled>CLEAR E-STOP</button><button id="b-estop" title="Space or Esc">E-STOP<br><small>Space / Esc</small></button></footer>
<script>
const $=id=>document.getElementById(id), f=(v,d=1)=>v==null?"—":(+v).toFixed(d), arr=a=>a?a.map(v=>f(v,3)).join(", "):"—";
const MODE={OBSERVE:["OBSERVE ONLY","#2a2f3a"],DRY_RUN:["DRY RUN · nothing sent","#1d4ed8"],EXECUTING:["EXECUTING","#1f6f43"],
 HOLD:["HOLD","#8a6414"],ABORTED:["ABORTED","#e5484d"],ESTOP:["E-STOP","#e5484d"],DONE:["DONE","#2a2f3a"]};
const M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_yaw","wrist_roll","gripper"];
function rows(el,list){el.innerHTML=list.map(r=>"<tr>"+r.map((c,i)=>`<td class="${i?'r':''}">${c}</td>`).join("")+"</tr>").join("")}
async function post(p){const r=await fetch(p,{method:"POST"});const j=await r.json();$("msg").textContent=(j.ok?"":"refused: ")+(typeof j.msg=="string"?j.msg:"gates updated")}
$("b-enable").onclick=()=>post("/api/enable");$("b-hold").onclick=()=>post("/api/hold");
$("b-estop").onclick=()=>post("/api/estop");
$("b-clear").onclick=()=>{if(confirm("Clear the robot E-STOP? Only if the arm is safe and the workspace is clear."))post("/api/clear_estop")};
document.addEventListener("keydown",e=>{if(e.code=="Space"||e.code=="Escape"){e.preventDefault();post("/api/estop")}});
async function tick(){let s;try{s=await (await fetch("/state.json")).json()}catch(e){$("mode").textContent="PANEL OFFLINE";return}
 const m=MODE[s.mode]||[s.mode,"#2a2f3a"];$("mode").textContent=m[0];$("mode").style.background=m[1];
 $("cfg").textContent=` · ${s.cfg.side} arm · ${s.cfg.axes}${s.cfg.signs} ${f(s.cfg.step_mm,0)} mm · ${s.cfg.live?"LIVE":"dry run"}`;
 if(s.robot_estop&&s.mode!="ESTOP"){s.mode="ESTOP";s.banner="robot_service E-STOP is ENGAGED (set from somewhere else): every command is refused";$("mode").textContent="E-STOP";$("mode").style.background="#e5484d"}
 const bad=["ABORTED","ESTOP","HOLD"].includes(s.mode)&&s.banner;$("banner").style.display=bad?"block":"none";
 $("btitle").textContent=s.mode=="HOLD"?"HOLD":s.mode=="ESTOP"?"E-STOP":"TELEOP ABORTED";$("btext").textContent=s.banner||"";
 const other=s.cfg.side=="right"?"left":"right";
 rows($("robot"),[["service",s.service.ok?'<span class="ok">connected</span>':`<span class="abort">${s.service.error||"down"}</span>`],
  ["robot E-STOP",s.robot_estop?'<span class="abort">ENGAGED</span>':'<span class="ok">clear</span>'],
  ["/observe age",s.obs_age_ms==null?"—":s.obs_age_ms+" ms"],["active arm",s.cfg.side+" (moves)"],["inactive arm",other+" (held)"],
  [s.cfg.side+" q (deg)",s.arms?s.arms[s.cfg.side].map(v=>f(v)).join(" "):"—"],[other+" q (deg)",s.arms?s.arms[other].map(v=>f(v)).join(" "):"—"],
  ["jaw raw L / R",s.jaws?`${f(s.jaws.left.raw)} / ${f(s.jaws.right.raw)}`:"—"],["jaw hold cmd L / R",s.jaws?`${s.jaws.left.hold_cmd} / ${s.jaws.right.hold_cmd}`:"—"],
  ["TCP measured (m)",arr(s.tcp)],["TCP commanded (m)",arr(s.cmd)],["target (m)",arr(s.target)],["loop",s.loop_hz==null?"—":s.loop_hz+" Hz"]]);
 const mot=[];M.forEach(n=>{const c=["left_arm","right_arm"].map(a=>{const v=(s.motors[a]||{})[n];const cls=v==null?"idle":(v==1?"ok":(n=="gripper"&&v==0?"warn":"abort"));
  return `<span class="${cls}"><span class="dot" style="background:currentColor"></span>${v==null?"—":v}</span>`});mot.push([n,...c])});
 $("motors").innerHTML="<tr><td></td><td class='r idle'>left</td><td class='r idle'>right</td></tr>"+mot.map(r=>"<tr>"+r.map((c,i)=>`<td class="${i?'r':''}">${c}</td>`).join("")+"</tr>").join("");
 const lm=s.live_metrics||{};$("move").textContent=s.move?`${s.move}  (${f(lm.commanded_mm,0)} mm)`:"—";$("prog").style.width=Math.min(100,100*(s.progress||0))+"%";
 rows($("m0"),[["along / commanded",`${f(lm.along_mm)} / ${f(lm.commanded_mm)} mm`],["t2mm",lm.t2mm_ms==null?"—":lm.t2mm_ms+" ms"],
  ["t50",lm.t50_ms==null?"—":lm.t50_ms+" ms"],["t90",lm.t90_ms==null?"—":lm.t90_ms+" ms"],["error to target",f(lm.err_to_target_mm)+" mm"],
  ["last tick",lm.ok==null?"—":(lm.ok?'<span class="ok">sent</span>':`<span class="warn">refused ${lm.err||""}</span>`)]]);
 const sym={ok:"✓",warn:"⚠",abort:"■",idle:"·"};
 rows($("checks"),Object.entries(s.checks).map(([k,v])=>[k,`<span class="${v.level}">${sym[v.level]} ${v.text}</span>`]));
 const g=Object.entries(s.gates||{});rows($("gates"),g.length?g.map(([k,v])=>[k,`<span class="${v.ok?'ok':'abort'}">${v.ok?"✓":"■"} ${v.detail}</span>`]):[["press ENABLE to run the checks",""]]);
 rows($("moves"),(s.moves||[]).map(x=>[x.move,`${f(x.final_along_mm)}/${f(x.commanded_mm,0)} mm · t50 ${x.t_50_s??"—"} s · t90 ${x.t_90_s??"—"} s · fail ${x.failed}`]));
 $("b-enable").disabled=s.running||!!s.robot_estop;$("b-clear").disabled=!s.robot_estop||s.running;$("b-hold").textContent=s.running?"HOLD":"HOLD / ACK";
}
setInterval(tick,250);tick();
</script></body></html>"""


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m0", action="store_true", help="(accepted so f5_teleop_hud can forward its argv unchanged)")
    ap.add_argument("--side", choices=tuple(SIDE_SLICE), default="right")
    ap.add_argument("--axes", default="x")
    ap.add_argument("--signs", default="+-")
    ap.add_argument("--step-m", type=float, default=0.01)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--max-lin-vel", type=float, default=0.02)
    ap.add_argument("--dwell-s", type=float, default=3.0)
    ap.add_argument("--max-step-deg", type=float, default=2.9)
    ap.add_argument("--robot", default="http://127.0.0.1:8020")
    ap.add_argument("--port", type=int, default=8713)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", dest="open_browser", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--i-am-at-the-robot", action="store_true")
    args = ap.parse_args(argv)
    if set(args.axes) - set("xyz") or set(args.signs) - set("+-"): ap.error("--axes from xyz, --signs from +-")
    return run_panel(args)


if __name__ == "__main__":
    raise SystemExit(main())
