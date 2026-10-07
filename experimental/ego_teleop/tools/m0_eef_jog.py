"""M0: manual EEF offset -> continuity IK -> q target -> reBot, one axis at a time, through the production arm branch.

    HOME+X, HOME, HOME-X, HOME, ... Y, Z      (absolute targets around the start pose, robot base frame RB, default 2 cm)

Every tick: target -> ArmSafetyPipeline (velocity-limited ramp, workspace = a box around the START pose so it can never
pull the first command somewhere else) -> HttpRebotClient.command_ee_pose (IK seeded with the previous command, joint
bounds + per-command step guard, follower->leader conversion) -> /execute_step. The same objects the teleop coordinator
drives, so M0 tests the path M1-M5 will use, not a side path.

OutboundGuard sits between the client and the wire and refuses (does not send) any vector whose other arm or jaws differ
from the hold of the latest observation, or whose active arm moves more than --max-step-deg per command. In --dry-run
(the default) /observe is really read — the numbers are about the real robot pose — and nothing is POSTed.

It also measures what M0 is for: command -> actual TCP lag and reach per axis (the Damiao follower needs repeated sends
to realise a setpoint, measured 2026-09-22), off-axis motion, other-arm drift, IK residual and loop rate.

    ~/xvla-mac/bin/python -m ego_teleop.tools.m0_eef_jog --side right                                   # dry run
    ~/xvla-mac/bin/python -m ego_teleop.tools.m0_eef_jog --side right --axes x --execute --i-am-at-the-robot
Needs handumi (pyroki) for IK -> ~/xvla-mac/bin/python, not ego_collector/.venv."""
from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from ..robot.rebot_client import (HttpRebotClient, JointLimits, PyrokiIkSolver, SIDE_SLICE, GRIPPER_INDEX, hold_action,
                                  FLIP_IDX, unwrap_grip, grip_hold_command, leader_action)
from ..robot.safety import ArmSafetyPipeline, SafetyConfig, WorkspaceBox
from ..transforms.se3 import T_to_pose7

AXES = {"x": 0, "y": 1, "z": 2}


class GuardViolation(RuntimeError):
    pass


class AbortRun(RuntimeError):
    """A measured stop condition fired: nothing further is sent; the follower holds its last setpoint."""


@dataclass
class StopRules:
    """Measured (not commanded) stop conditions, checked every tick against the robot's own readback."""
    other_arm_deg: float = 0.5        # inactive arm joint change vs run start (encoder noise is ~0.01 deg)
    grip_raw: float = 3.0             # either jaw raw change vs run start (counts; 6 counts = 1 cmd unit)
    reverse_mm: float = 4.0           # TCP moving against the commanded direction
    overshoot_mm: float = 5.0         # TCP past the target along the commanded direction
    off_axis_mm: float = 6.0          # TCP motion perpendicular to the commanded direction
    ik_mm: float = 8.0                # IK residual spike

    def check(self, *, other_deg, grip_delta, along_mm, dist_mm, off_mm, ik_mm, err) -> str:
        if err.startswith("http:"): return f"robot_service error: {err}"
        if other_deg > self.other_arm_deg: return f"inactive arm moved {other_deg:.2f} deg"
        if grip_delta > self.grip_raw: return f"jaw moved {grip_delta:.1f} raw counts"
        if dist_mm > 0 and along_mm < -self.reverse_mm: return f"TCP moved {along_mm:.1f} mm AGAINST the command"
        if dist_mm > 0 and along_mm > dist_mm + self.overshoot_mm: return f"TCP overshot: {along_mm:.1f} mm along a {dist_mm:.1f} mm command"
        if off_mm > self.off_axis_mm: return f"off-axis motion {off_mm:.1f} mm"
        if ik_mm is not None and ik_mm > self.ik_mm: return f"IK residual {ik_mm:.1f} mm"
        return ""


@dataclass
class OutboundGuard:
    """Wraps the client's http(). GET passes through; POST /execute_step is checked, logged, and sent only if `live`."""
    http: object
    side: str
    live: bool
    max_step_rad: float
    sent: list = field(default_factory=list)
    _hold: list | None = None
    _prev: list | None = None

    def set_observation(self, joints_deg_all) -> None:
        self._hold = hold_action(joints_deg_all)
        if self._prev is None: self._prev = list(self._hold)

    def __call__(self, method, path, payload):
        if path != "/execute_step": return self.http(method, path, payload)
        a = list(payload["action"]); other = "left" if self.side == "right" else "right"
        if self._hold is None: raise GuardViolation("no observation before the first command")
        keep = list(range(14))[SIDE_SLICE[other]] + list(GRIPPER_INDEX.values())
        bad = [i for i in keep if abs(a[i] - self._hold[i]) > 1e-9]
        if bad: raise GuardViolation(f"indices {bad} differ from the hold of the observation: {[a[i] for i in bad]} vs {[self._hold[i] for i in bad]}")
        act = list(range(14))[SIDE_SLICE[self.side]]
        step = max(abs(a[i] - self._prev[i]) for i in act)
        if step > self.max_step_rad: raise GuardViolation(f"active-arm step {np.degrees(step):.2f} deg > {np.degrees(self.max_step_rad):.2f}")
        self.sent.append(a); self._prev = a
        if not self.live: return {"ok": True, "sent_deg": None, "dry_run": True}
        return self.http(method, path, payload)


class MotorWatch:
    """robot_service /motor_states watchdog. Baseline at start; every arm motor must be status 1 (enabled) or the run is
    refused. Afterwards ANY motor whose status differs from its baseline (arm or jaw) aborts the command stream."""
    ARM_SIDES = ("left_arm", "right_arm")

    def __init__(self, http, every_n: int = 5) -> None:
        self.http, self.every_n, self._n = http, max(1, every_n), 0
        self.base = self._read(); self.last = self.base
        bad = {f"{a}.{m}": st for a, ms in self.base.items() for m, st in ms.items() if m != "gripper" and st != 1}
        if bad: raise AbortRun(f"pre-flight: arm motors not enabled (status != 1): {bad}")

    def _read(self) -> dict:
        d = self.http("GET", "/motor_states", None)
        return {a: {m: (v.get("status") if isinstance(v, dict) else None) for m, v in d.get(a, {}).items()} for a in self.ARM_SIDES}

    def tick(self, force: bool = False) -> None:
        self._n += 1
        if not force and self._n % self.every_n: return
        now = self._read(); self.last = now
        changed = {f"{a}.{m}": f"{self.base[a].get(m)}->{st}" for a, ms in now.items() for m, st in ms.items() if st != self.base[a].get(m)}
        if changed: raise AbortRun(f"motor status changed: {changed}")


def _moves(axes: str, step: float, signs=(+1, -1)) -> list[tuple[str, np.ndarray]]:
    """(name, offset from HOME). Every out-move is followed by a return to HOME itself, never a relative step back,
    so follower lag cannot accumulate into drift."""
    out = []
    for ax in axes:
        for sgn in signs:
            d = np.zeros(3); d[AXES[ax]] = sgn * step
            out.append((f"{'+' if sgn > 0 else '-'}{ax.upper()}", d)); out.append((f"home<{'+' if sgn > 0 else '-'}{ax.upper()}", np.zeros(3)))
    return out


def run_move(client: HttpRebotClient, guard: OutboundGuard, safety: ArmSafetyPipeline, name: str, target: np.ndarray, *,
             hz: float, max_lin_vel: float, dwell_s: float, rows: list, t_origin: int, other0: np.ndarray, grip0: np.ndarray,
             rules: StopRules | None = None, watch: MotorWatch | None = None, on_tick=None, stop=None) -> dict:
    """Ramp from the last COMMANDED pose to the absolute `target` (safety limiter), then keep sending it for dwell_s."""
    st = client.observe(); guard.set_observation(st.joints_deg_all)
    p_cmd0 = safety.last_command[:3, 3]; p_meas0 = st.T_RB_RE[:3, 3].copy()
    delta = target[:3, 3] - p_cmd0; dist = float(np.linalg.norm(delta))
    u = delta / dist if dist > 1e-9 else np.zeros(3)
    dt = 1.0 / hz; ramp_s = dist / max_lin_vel; t_end = ramp_s + dwell_s
    other = "left" if client.side == "right" else "right"
    t_start = time.monotonic_ns(); n_fail = 0; first_err = ""; tick_ms = []; ik_mm = []; mv = []
    while True:
        t_tick = time.monotonic_ns(); el = (t_tick - t_start) / 1e9
        if el > t_end: break
        t_req = time.monotonic_ns(); st = client.observe(); t_resp = time.monotonic_ns(); guard.set_observation(st.joints_deg_all)
        v = safety.check(target, dt_s=dt, robot_obs_age_ms=(time.monotonic_ns() - st.timestamp_ns) / 1e6)
        rep = client.command_ee_pose(v.T_cmd, t_tick) if v.ok else None
        if rep is None or not rep.ok:
            n_fail += 1; first_err = first_err or (",".join(v.reasons) if rep is None else rep.error)
        if rep is not None and rep.ik_error_m is not None: ik_mm.append(rep.ik_error_m * 1e3)
        meas = st.T_RB_RE[:3, 3]; disp = meas - p_meas0
        r = dict(move=name, t_s=round((t_tick - t_origin) / 1e9, 4), el_s=round(el, 4),
                 cmd_x=v.T_cmd[0, 3], cmd_y=v.T_cmd[1, 3], cmd_z=v.T_cmd[2, 3], meas_x=meas[0], meas_y=meas[1], meas_z=meas[2],
                 err_to_target_mm=float(np.linalg.norm(meas - target[:3, 3])) * 1e3,
                 along_mm=float(disp @ u) * 1e3, off_axis_mm=float(np.linalg.norm(disp - (disp @ u) * u)) * 1e3,
                 other_arm_max_deg=float(np.max(np.abs(np.asarray(st.joints_deg_all[SIDE_SLICE[other]]) - other0))),
                 ok=bool(rep is not None and rep.ok), err="" if rep is None else rep.error,
                 ik_mm=None if rep is None or rep.ik_error_m is None else rep.ik_error_m * 1e3,
                 action=json.dumps([round(x, 6) for x in rep.action]) if rep is not None and rep.action else "")
        # high-rate diagnostics: observed vs commanded joints, FK of both, observe timing, and per-joint "stale" flags
        # (a joint that repeats its previous reading EXACTLY while the arm is being driven was probably not refreshed:
        # get_observation polls the CAN feedback once and falls back to each motor's last cached state)
        q_obs = np.asarray(st.joints_deg_all[SIDE_SLICE[client.side]], np.float64)
        prev = mv[-1] if mv else None
        for i in range(6):
            r[f"qobs{i+1}"] = round(float(q_obs[i]), 4)
            r[f"qcmd{i+1}"] = None if rep is None or rep.q_cmd_rad is None else round(float(np.degrees(rep.q_cmd_rad[i])), 4)
            r[f"stale{i+1}"] = bool(prev is not None and prev[f"qobs{i+1}"] == r[f"qobs{i+1}"])
        if rep is not None and rep.q_cmd_rad is not None:
            fc = client.ik.fk(rep.q_cmd_rad)[:3, 3]; r.update(fkcmd_x=fc[0], fkcmd_y=fc[1], fkcmd_z=fc[2])
        else:
            r.update(fkcmd_x=None, fkcmd_y=None, fkcmd_z=None)
        r.update(t_req_s=round((t_req - t_origin) / 1e9, 4), obs_ms=round((t_resp - t_req) / 1e6, 2),
                 motors="" if watch is None else json.dumps(watch.last, sort_keys=True))
        rows.append(r); mv.append(r)
        if on_tick is not None: on_tick(r, dict(move=name, target=target[:3, 3].tolist(), dist_mm=dist * 1e3, t_end_s=t_end, st=st, rep=rep))
        if stop is not None and stop.is_set(): raise AbortRun(f"{name} at {el:.2f} s: operator HOLD")
        if watch is not None: watch.tick()
        if rules is not None:
            gd = float(np.max(np.abs([unwrap_grip(st.joints_deg_all[g]) for g in GRIPPER_INDEX.values()] - grip0)))
            why = rules.check(other_deg=r["other_arm_max_deg"], grip_delta=gd, along_mm=r["along_mm"], dist_mm=dist * 1e3,
                              off_mm=r["off_axis_mm"], ik_mm=r["ik_mm"], err=r["err"] or "")
            if why: raise AbortRun(f"{name} at {el:.2f} s: {why}")
        tick_ms.append((time.monotonic_ns() - t_tick) / 1e6)
        sleep = dt - (time.monotonic_ns() - t_tick) / 1e9
        if sleep > 0: time.sleep(sleep)
    step_mm = dist * 1e3
    def t_reach(frac):
        hit = [r["el_s"] for r in mv if step_mm > 0 and r["along_mm"] >= frac * step_mm]
        return round(hit[0], 3) if hit else None
    t_move = [r["el_s"] for r in mv if r["along_mm"] >= 2.0]
    return dict(move=name, ticks=len(mv), failed=n_fail, first_error=first_err, commanded_mm=round(step_mm, 2),
                final_along_mm=round(mv[-1]["along_mm"], 2), final_off_axis_mm=round(mv[-1]["off_axis_mm"], 2),
                final_err_to_target_mm=round(mv[-1]["err_to_target_mm"], 2),
                max_other_arm_deg=round(max(r["other_arm_max_deg"] for r in mv), 3),
                t_first_2mm_s=round(t_move[0], 3) if t_move else None, t_50_s=t_reach(0.5), t_90_s=t_reach(0.9),
                cmd_ramp_s=round(ramp_s, 2), ik_mm_max=round(max(ik_mm), 2) if ik_mm else None,
                loop_hz=round(1000.0 / float(np.mean(tick_ms)), 1) if tick_ms else None,
                tick_ms_p95=round(float(np.percentile(tick_ms, 95)), 1) if tick_ms else None)


def run(client: HttpRebotClient, guard: OutboundGuard, *, axes="xyz", step=0.02, hz=30.0, max_lin_vel=0.02, dwell_s=3.0,
        out: Path | None = None, show=3, rules: StopRules | None = None, signs=(+1, -1), watch_http=None, on_tick=None, stop=None,
        on_move=None) -> list[dict]:
    rows: list = []; t0 = time.monotonic_ns(); summary = []
    st = client.observe(); guard.set_observation(st.joints_deg_all); client.reset_seed()
    T_home = st.T_RB_RE.copy(); other = "left" if client.side == "right" else "right"
    # warm the IK up BEFORE the first command: pyroki/jax compiles on the first solve (~0.9 s measured), which
    # otherwise shows up as a 0.9 s gap between the first and second command and reads as follower dead time
    t_w = time.monotonic_ns(); client.ik.ik(st.q_rad.copy(), T_home); client.ik.ik(st.q_rad.copy(), T_home)
    print(f"[M0] IK warm-up {(time.monotonic_ns() - t_w) / 1e6:.0f} ms (nothing sent)")
    other0 = np.asarray(st.joints_deg_all[SIDE_SLICE[other]], np.float64)
    grip0 = np.asarray([unwrap_grip(st.joints_deg_all[g]) for g in GRIPPER_INDEX.values()])
    pad = step + 0.01                           # workspace = a box around HOME: contains the start by construction
    box = WorkspaceBox(tuple(T_home[:3, 3] - pad), tuple(T_home[:3, 3] + pad))
    safety = ArmSafetyPipeline(SafetyConfig(workspace=box, workspace_mode="hold", max_lin_vel_m_s=max_lin_vel,
                                            max_ang_vel_deg_s=10.0, max_lin_acc_m_s2=5.0, robot_obs_max_age_ms=1000.0))
    # acc limit deliberately non-binding: CartesianLimiter has no deceleration planning and overshoots by v^2/2a
    safety.reset(T_home)
    print(f"[M0] side={client.side} live={guard.live} HOME TCP (FK, RB) = {np.round(T_home[:3, 3], 4).tolist()} m")
    print(f"[M0] hold vector (leader frame, what 'do nothing' looks like on the wire):\n     {np.round(hold_action(st.joints_deg_all), 5).tolist()}")
    try:
        watch = MotorWatch(watch_http) if watch_http is not None else None
        for name, d in _moves(axes, step, signs):
            n0 = len(guard.sent); target = T_home.copy(); target[:3, 3] += d
            s = run_move(client, guard, safety, name, target, hz=hz, max_lin_vel=max_lin_vel, dwell_s=dwell_s, rows=rows,
                         t_origin=t0, other0=other0, grip0=grip0, rules=rules, watch=watch, on_tick=on_tick, stop=stop)
            summary.append(s); print(f"[M0] {json.dumps(s)}")
            if on_move is not None: on_move(s)
            for a in guard.sent[n0:n0 + show]: print(f"     out {np.round(a, 5).tolist()}")
    except GuardViolation as exc:
        print(f"[M0] GUARD REFUSED, nothing further sent: {exc}"); summary.append(dict(move="GUARD", error=str(exc)))
    except AbortRun as exc:
        print(f"[M0] STOP CONDITION, nothing further sent: {exc}"); summary.append(dict(move="ABORT", error=str(exc)))
    except KeyboardInterrupt:
        print("[M0] interrupted; the robot holds its last setpoint"); summary.append(dict(move="INTERRUPTED"))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(out / "ticks.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (out / "summary.json").write_text(json.dumps(dict(live=guard.live, side=client.side, flip_idx=list(FLIP_IDX),
                                                          home_tcp=T_home[:3, 3].tolist(), moves=summary), indent=1))
        print(f"[M0] wrote {out}")
    return summary


def observe_check(client: HttpRebotClient, seconds: float = 1.0, hz: float = 20.0) -> dict:
    """Read-only sanity: which slots are which, encoder noise, jaw raws, FK TCP. Sends nothing."""
    J = []
    for _ in range(max(2, int(seconds * hz))):
        J.append(client.observe().joints_deg_all); time.sleep(1.0 / hz)
    J = np.asarray(J, np.float64); st = client._last
    rep = dict(left_arm_deg=np.round(J[-1, 0:6], 2).tolist(), right_arm_deg=np.round(J[-1, 7:13], 2).tolist(),
               jaw_raw_LR=[round(J[-1, 6], 1), round(J[-1, 13], 1)], jaw_hold_cmd_LR=[round(grip_hold_command(J[-1, g]), 3) for g in (6, 13)],
               noise_ptp_deg=np.round(np.ptp(J, axis=0), 3).tolist(), tcp_fk_m=np.round(st.T_RB_RE[:3, 3], 4).tolist(),
               hold_vector_leader=np.round(hold_action(J[-1]), 5).tolist())
    for k, v in rep.items(): print(f"[check] {k}: {v}")
    return rep


def goto_joints(client: HttpRebotClient, q6_deg, *, substep_rad=0.01, hz=20.0, tol_deg=1.5, dwell_timeout_s=8.0,
                rules: StopRules | None = None, watch_http=None) -> dict:
    """Joint-space move of the ACTIVE arm only, same pattern as mac_v4_smoke_ui._goto (ramp <= substep_rad per send,
    then keep sending the target until within tol_deg or timeout) — but the other arm and both jaws are held at their
    observation and every vector passes the OutboundGuard. No IK involved."""
    watch = MotorWatch(watch_http, every_n=4) if watch_http is not None else None
    st = client.observe(); guard = client._http; guard.set_observation(st.joints_deg_all)
    sl = SIDE_SLICE[client.side]; other = SIDE_SLICE["left" if client.side == "right" else "right"]
    q0 = np.radians(np.asarray(st.joints_deg_all[sl], np.float64)); q1 = np.radians(np.asarray(q6_deg, np.float64))
    other0 = np.asarray(st.joints_deg_all[other], np.float64)
    grip0 = np.asarray([unwrap_grip(st.joints_deg_all[g]) for g in GRIPPER_INDEX.values()])
    n = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / substep_rad)))
    def send(q6):
        s2 = client.observe(); guard.set_observation(s2.joints_deg_all)
        od = float(np.max(np.abs(np.asarray(s2.joints_deg_all[other]) - other0)))
        gd = float(np.max(np.abs([unwrap_grip(s2.joints_deg_all[g]) for g in GRIPPER_INDEX.values()] - grip0)))
        if rules is not None and od > rules.other_arm_deg: raise AbortRun(f"goto: inactive arm moved {od:.2f} deg")
        if rules is not None and gd > rules.grip_raw: raise AbortRun(f"goto: jaw moved {gd:.1f} raw counts")
        if watch is not None: watch.tick()
        res = guard("POST", "/execute_step", {"action": client.build_action(q6, s2)})
        if not res.get("ok", False): raise AbortRun(f"goto: robot_service refused: {res}")
        return s2
    t0 = time.time()
    for k in range(1, n + 1):
        send(q0 + (q1 - q0) * k / n); time.sleep(1.0 / hz)
    err = float("nan"); reached = False; t_dw = time.time()
    while time.time() - t_dw < dwell_timeout_s:
        s2 = send(q1); time.sleep(1.0 / hz)
        err = float(np.max(np.abs(np.asarray(s2.joints_deg_all[sl]) - np.degrees(q1))))
        if err < tol_deg: reached = True; break
    st = client.observe()
    rep = dict(substeps=n, reached=reached, joint_err_deg=round(err, 2), seconds=round(time.time() - t0, 1),
               arm_deg=np.round(st.joints_deg_all[sl], 2).tolist(), tcp_fk_m=np.round(st.T_RB_RE[:3, 3], 4).tolist())
    print(f"[goto] {rep}")
    return rep


MOTOR_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")


def read_fast(base_http) -> tuple[list[float], dict]:
    """robot_service /joints_fast -> (joints_deg_all in the /observe 14-slot layout, raw per-motor dict)."""
    d = base_http("GET", "/joints_fast", None); m = d["motors"]; j = []
    for side in ("left_arm", "right_arm"):
        for name in MOTOR_ORDER:
            e = m[side][name]
            if e is None: raise AbortRun(f"/joints_fast: no state for {side}.{name}")
            j.append(float(e["pos"]))
    return j, d


def joint_step_test(client: HttpRebotClient, base_http, joint: int, steps_deg, *, dwell_s=3.0, hz=30.0,
                    rules: StopRules | None = None, out: Path | None = None) -> list[dict]:
    """Joint-space step response of ONE active-arm joint (1..6), no IK, no Cartesian path: for each step d in steps_deg,
    command q_home[joint] + d (held constant, re-sent at hz) for dwell_s, then q_home for dwell_s. Other joints of the
    active arm are commanded at HOME, the other arm and both jaws are held at their observation, every vector passes
    the OutboundGuard, and MotorWatch aborts on any status change. Logs pos/vel/torque of all six joints per tick."""
    guard = client._http; sl = SIDE_SLICE[client.side]; side_key = f"{client.side}_arm"
    j0, _ = read_fast(base_http); guard.set_observation(j0)
    q_home = np.radians(np.asarray(j0[sl], np.float64)); k = joint - 1
    other = SIDE_SLICE["left" if client.side == "right" else "right"]
    other0 = np.asarray(j0[other]); grip0 = np.asarray([unwrap_grip(j0[g]) for g in GRIPPER_INDEX.values()])
    watch = MotorWatch(base_http, every_n=3)
    rows: list = []; summary = []; t_origin = time.monotonic_ns()
    def hold_at(q6, label, d_cmd):
        t_s = time.monotonic_ns(); mv = []
        while (time.monotonic_ns() - t_s) / 1e9 < dwell_s:
            t_tick = time.monotonic_ns()
            j, raw = read_fast(base_http); guard.set_observation(j)
            od = float(np.max(np.abs(np.asarray(j[other]) - other0)))
            gd = float(np.max(np.abs([unwrap_grip(j[g]) for g in GRIPPER_INDEX.values()] - grip0)))
            if rules is not None and od > rules.other_arm_deg: raise AbortRun(f"{label}: inactive arm moved {od:.2f} deg")
            if rules is not None and gd > rules.grip_raw: raise AbortRun(f"{label}: jaw moved {gd:.1f} raw counts")
            watch.tick()
            st = client._last if client._last is not None else None
            from types import SimpleNamespace
            res = guard("POST", "/execute_step", {"action": client.build_action(q6, SimpleNamespace(joints_deg_all=j))})
            if not res.get("ok", False): raise AbortRun(f"{label}: robot_service refused: {res}")
            mj = raw["motors"][side_key]
            r = dict(step=label, t_s=round((t_tick - t_origin) / 1e9, 4), el_s=round((t_tick - t_s) / 1e9, 4), d_cmd_deg=d_cmd,
                     q_cmd_deg=round(float(np.degrees(q6[k])), 4))
            for i, name in enumerate(MOTOR_ORDER[:6]):
                e = mj[name]; r[f"pos{i+1}"] = e["pos"]; r[f"vel{i+1}"] = e["vel"]; r[f"torq{i+1}"] = e["torq"]
                r[f"fresh{i+1}"] = e["fresh"]; r[f"status{i+1}"] = e["status"]
            r["read_ms"] = round((raw["t1"] - raw["t0"]) * 1e3, 2)
            rows.append(r); mv.append(r)
            sleep = 1.0 / hz - (time.monotonic_ns() - t_tick) / 1e9
            if sleep > 0: time.sleep(sleep)
        p0 = mv[0][f"pos{joint}"]; target = float(np.degrees(q6[k])); span = target - p0
        def t_at(frac):
            hit = [x["el_s"] for x in mv if abs(span) > 1e-6 and (x[f"pos{joint}"] - p0) / span >= frac]
            return round(hit[0], 3) if hit else None
        moved = [x["el_s"] for x in mv if abs(x[f"pos{joint}"] - p0) > 0.05]
        s = dict(step=label, commanded_change_deg=round(span, 3), start_pos=round(p0, 3), end_pos=round(mv[-1][f"pos{joint}"], 3),
                 ss_error_deg=round(target - mv[-1][f"pos{joint}"], 3), t_first_0p05deg_s=moved[0] if moved else None,
                 t50_s=t_at(0.5), t90_s=t_at(0.9), torq_start=round(mv[0][f"torq{joint}"], 3), torq_end=round(mv[-1][f"torq{joint}"], 3),
                 torq_peak=round(max(abs(x[f"torq{joint}"]) for x in mv), 3), ticks=len(mv),
                 read_ms_p95=round(float(np.percentile([x["read_ms"] for x in mv], 95)), 1))
        summary.append(s); print(f"[step] {json.dumps(s)}")
    try:
        for d in steps_deg:
            q = q_home.copy(); q[k] += np.radians(d)
            hold_at(q, f"j{joint}{d:+.2f}", d)
            hold_at(q_home.copy(), f"j{joint}{d:+.2f}->home", 0.0)
    except (AbortRun, GuardViolation) as exc:
        print(f"[step] STOPPED, nothing further sent: {exc}"); summary.append(dict(step="ABORT", error=str(exc)))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(out / "ticks.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (out / "summary.json").write_text(json.dumps(dict(side=client.side, joint=joint, steps=list(steps_deg), dwell_s=dwell_s,
                                                          home_deg=np.degrees(q_home).tolist(), results=summary), indent=1))
        print(f"[step] wrote {out}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=tuple(SIDE_SLICE), default="right")
    ap.add_argument("--axes", default="xyz")
    ap.add_argument("--step-m", type=float, default=0.02)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--max-lin-vel", type=float, default=0.02, help="m/s ramp speed (2 cm/s default)")
    ap.add_argument("--dwell-s", type=float, default=3.0, help="keep sending the final target this long (follower lag)")
    ap.add_argument("--max-step-deg", type=float, default=2.9, help="guard: active-arm joint change per command")
    ap.add_argument("--robot", default="http://127.0.0.1:8020")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--assume-joints-deg", default=None,
                    help="dry run only: 14 comma-separated /observe values to plan from instead of reading the robot")
    ap.add_argument("--signs", default="+-", help="which directions per axis: '+-' (default), '+' or '-'")
    ap.add_argument("--check", action="store_true", help="read-only sanity check (slots, noise, jaws, FK) and exit")
    ap.add_argument("--goto-deg", default=None, help="6 comma-separated active-arm joint targets (follower deg): ramped "
                    "joint move, other arm + jaws held, then exit")
    ap.add_argument("--goto-substep-rad", type=float, default=0.01, help="joint ramp per send (0.01 rad @ 20 Hz = 0.2 rad/s)")
    ap.add_argument("--step-joint", type=int, default=None, help="joint-space step response of this active-arm joint (1..6)")
    ap.add_argument("--step-list", default="0.25,-0.25,0.5,-0.5,1.0,-1.0,2.0,-2.0", help="degrees, each followed by a return to HOME")
    ap.add_argument("--no-stop-rules", action="store_true", help="disable the measured stop conditions (dry-run analysis only)")
    ap.add_argument("--execute", action="store_true", help="really POST /execute_step (default: dry run)")
    ap.add_argument("--i-am-at-the-robot", action="store_true")
    args = ap.parse_args(argv)
    if set(args.axes) - set(AXES): ap.error("--axes takes letters from xyz")
    if args.execute and not args.i_am_at_the_robot: ap.error("--execute needs --i-am-at-the-robot (hand on the e-stop)")
    if args.step_m > 0.05: ap.error("--step-m above 5 cm is not an M0 jog")
    ik = PyrokiIkSolver.for_rebot_b601(args.side)
    base = HttpRebotClient(args.robot, args.side, None, timeout_s=2.0)._requests_http
    if args.assume_joints_deg:
        if args.execute: ap.error("--assume-joints-deg is a planning check; it cannot be combined with --execute")
        fixed = [float(x) for x in args.assume_joints_deg.split(",")]
        if len(fixed) != 14: ap.error("--assume-joints-deg needs 14 values")
        base = lambda m, path, payload: {"joints_deg": fixed} if path == "/observe" else {"ok": True}
    guard = OutboundGuard(base, args.side, live=args.execute, max_step_rad=np.radians(args.max_step_deg))
    client = HttpRebotClient(args.robot, args.side, ik, limits=JointLimits(max_step_rad=np.radians(args.max_step_deg)),
                             http=guard, timeout_s=2.0)
    rules = None if args.no_stop_rules else StopRules()
    if args.no_stop_rules and args.execute: ap.error("--no-stop-rules cannot be combined with --execute")
    if args.check:
        observe_check(client)
        try: MotorWatch(base); print("[check] motors: every arm motor status 1")
        except AbortRun as exc: print(f"[check] motors: NOT READY -- {exc}"); return 1
        return 0
    if args.goto_deg:
        q6 = [float(x) for x in args.goto_deg.split(",")]
        if len(q6) != 6: ap.error("--goto-deg needs 6 values")
        if args.goto_substep_rad > 0.02: ap.error("--goto-substep-rad above 0.02 is faster than the smoke UI ramp")
        try: rep = goto_joints(client, q6, rules=rules, substep_rad=args.goto_substep_rad,
                               watch_http=base if args.execute else None)
        except (AbortRun, GuardViolation) as exc: print(f"[goto] STOPPED, nothing further sent: {exc}"); return 1
        except Exception as exc:          # e.g. robot_service E-STOP -> HTTP 400 on /execute_step
            print(f"[goto] STOPPED by robot_service ({type(exc).__name__}: {str(exc)[:160]}), nothing further sent"); return 1
        return 0 if rep["reached"] else 1
    if args.step_joint is not None:
        steps = [float(x) for x in args.step_list.split(",")]
        if not 1 <= args.step_joint <= 6: ap.error("--step-joint is 1..6")
        if max(abs(x) for x in steps) > 2.5: ap.error("joint steps above 2.5 deg are not part of this characterization")
        if not args.execute: ap.error("--step-joint needs --execute --i-am-at-the-robot (there is no meaningful dry run of a step response)")
        so = args.out or Path("runs") / f"m0_jstep_{time.strftime('%Y%m%d_%H%M%S')}_{args.side}_j{args.step_joint}"
        r = joint_step_test(client, base, args.step_joint, steps, dwell_s=args.dwell_s, hz=args.hz, rules=rules, out=so)
        return 0 if r and r[-1].get("step") != "ABORT" else 1
    signs = tuple(+1 if c == "+" else -1 for c in args.signs)
    out = args.out or Path("runs") / f"m0_{time.strftime('%Y%m%d_%H%M%S')}_{args.side}{'_LIVE' if args.execute else '_dry'}"
    s = run(client, guard, axes=args.axes, step=args.step_m, hz=args.hz, max_lin_vel=args.max_lin_vel, dwell_s=args.dwell_s, out=out, rules=rules, signs=signs,
            watch_http=base if (args.execute or not args.assume_joints_deg) else None)
    return 0 if all(m.get("failed", 1) == 0 for m in s) else 1


if __name__ == "__main__":
    raise SystemExit(main())
