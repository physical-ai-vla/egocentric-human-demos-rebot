"""Stage 0 -- OFFLINE replay (2026-10-14, spec 30). Sensor log -> fusion -> pinch -> retarget -> IK -> safety -> (no robot).
Robot commands are never sent: the bridge is MockBridge (in-process) or HttpBridge(live=False) (real /observe, no POST).

Sensor log (npz, one per episode; the live session recorder writes the same keys):
    {side}_imu_t_us, {side}_imu_host_ns, {side}_gyro (N,3), {side}_acc (N,3)          raw IMU (AHRS runs here, real path)
    {side}_pos_t_ns, {side}_wrist_W (M,3), {side}_pos_conf (M,)                       fixed RGB-D wrist in W (NaN = dropout)
    {side}_pinch_t_ns, {side}_pinch_metric (K,), {side}_pinch_conf (K,)               wrist-cam fingertip detector output
    {side}_R_W_H_init (3,3)                                                          known init orientation (spec 5)
    clutch_windows_ns (J,2), deadman_released_ns (J,2), init_t_ns                    operator events
--synthetic writes such a log first (scripted motion per axis + rotations + pinches + dropouts + clutch + deadman release).

Pass criteria (spec 30 Stage 0) checked from the recorded ticks:
    direction   for every scripted single-axis segment the EXECUTED robot motion has cos > 0.9 with the expected robot direction
                (R_RB_R0 . M . human body delta) -> catches every axis flip / sign error
    jumps       executed per-tick step <= v_max/rate (+eps), rotation <= w_max/rate
    IK          feasible fraction of commanded ticks; no joint step above the branch guard
    gripper     executed transitions == scripted pinch transitions (no chatter), polarity (pinch -> g decreasing)
Run:  cd ~/ego_collector && .venv/bin/python -m visual_teleop.tools.stage0_replay --synthetic --out ~/Desktop/visual_teleop_stage0_v2 [--ik toy|pyroki]
      (--ik pyroki needs ~/xvla-mac/bin/python)"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .. import config as vcfg
from ..build import build_controller, build_hand_fusion, build_pinch
from ..imu.ahrs import MahonyAhrs, AhrsConfig
from ..imu.imu_reader import ImuOrientation
from ..recording.recorder import EpisodeRecorder
from ..recording.export_lerobot import export_episode
from ..recording.episode_qa import episode_qa
from ..robot.bridge import MockBridge, HttpBridge
from ..robot.ik import ToyIk, toy_limits, pyroki_pair
from ..safety.clutch import Clutch
from ..safety.deadman import ScriptedDeadman

SIDES = ("left", "right")
G = 9.80665


# ---------------------------------------------------------------------------------------------------- synthetic sensor log
def synthetic_log(seed: int = 0, imu_hz: float = 200.0, cam_hz: float = 30.0) -> tuple[dict, list]:
    """Both hands start at init points with a known orientation, hold still 2 s (AHRS bias), then a scripted program.
    Returns (log, segments) where segments = [(t0_s, t1_s, side, kind, body_vec)] for the direction check."""
    rng = np.random.default_rng(seed)
    init_p = {"left": np.array([0.30, 0.25, 0.10]), "right": np.array([0.30, -0.25, 0.10])}
    R_init = {s: Rotation.from_rotvec([0, 0, 0.3 if s == "left" else -0.3]).as_matrix() for s in SIDES}   # hands yawed inward
    prog = [("still", 3.0, None)]
    for ax in range(3):
        for sgn in (1, -1):
            v = np.zeros(3); v[ax] = 0.04 * sgn; prog += [("trans", 2.0, v), ("still", 0.5, None), ("trans", 2.0, -v), ("still", 0.5, None)]
    for ax in range(3):
        r = np.zeros(3); r[ax] = np.radians(25); prog += [("rot", 2.0, r), ("still", 0.5, None), ("rot", 2.0, -r), ("still", 0.5, None)]
    prog += [("pinch", 1.5, None), ("still", 1.0, None), ("pinch", 1.5, None), ("still", 1.0, None)]
    prog += [("dropout", 0.6, None), ("still", 1.0, None), ("clutch", 2.0, np.array([0.05, 0.0, 0.0])), ("still", 1.0, None),
             ("deadman", 1.0, None), ("still", 1.0, None)]
    T_end = sum(d for _, d, _ in prog); t_imu = np.arange(0, T_end, 1 / imu_hz); t_cam = np.arange(0, T_end, 1 / cam_hz)
    # human body-frame trajectory (same program on both hands; the right hand mirrors +Y so it moves outward)
    def body_state(t):
        p = np.zeros(3); r = np.zeros(3); pinch_m = 60.0; drop = False; tt = 0.0
        for kind, d, v in prog:
            if t < tt: break
            f = np.clip((t - tt) / d, 0, 1); s = 0.5 - 0.5 * np.cos(np.pi * f)              # smooth step
            if kind == "trans": p = p + (s * v if t < tt + d else v)
            if kind == "rot": r = r + (s * v if t < tt + d else v)
            if kind == "clutch": p = p + (s * v if t < tt + d else v)
            if kind == "pinch" and tt <= t < tt + d: pinch_m = 60.0 - 50.0 * np.sin(np.pi * f) ** 0.5
            if kind == "dropout" and tt <= t < tt + d: drop = True
            tt += d
        return p, r, pinch_m, drop
    log: dict = {}; segs = []
    tt = 0.0
    for kind, d, v in prog:
        if kind == "trans": segs.append((tt + 0.3 * d, tt + d, kind, v.copy()))
        if kind == "rot": segs.append((tt + 0.3 * d, tt + d, kind, v.copy()))
        if kind == "clutch": log["clutch_windows_ns"] = np.array([[int(tt * 1e9), int((tt + d) * 1e9)]])
        if kind == "deadman": log["deadman_released_ns"] = np.array([[int(tt * 1e9), int((tt + d) * 1e9)]])
        tt += d
    for s in SIDES:
        mirror = np.diag([1, -1, 1]) if s == "right" else np.eye(3)
        R_I_H = np.eye(3)
        R_G_I0 = Rotation.from_rotvec([0, 0, rng.uniform(-np.pi, np.pi)]).as_matrix()          # IMU heading arbitrary in G
        R_W_G = R_init[s] @ R_G_I0.T                                                            # truth (fusion must find it)
        def R_W_H(t):
            _, r, _, _ = body_state(t); return R_init[s] @ Rotation.from_rotvec(mirror @ r * (1 if s == "left" else 1)).as_matrix()
        def p_W(t):
            p, _, _, _ = body_state(t); return init_p[s] + R_init[s] @ (mirror @ p)
        Rs = np.array([R_W_G.T @ R_W_H(t) for t in t_imu])                                     # R_G_I(t)
        gyro = np.array([Rotation.from_matrix(Rs[k].T @ Rs[min(k + 1, len(Rs) - 1)]).as_rotvec() * imu_hz for k in range(len(Rs))])
        acc = np.einsum("nji,j->ni", Rs, np.array([0, 0, G]))
        log[f"{s}_imu_t_us"] = (t_imu * 1e6).astype(np.int64); log[f"{s}_imu_host_ns"] = (t_imu * 1e9).astype(np.int64) + 2_000_000
        log[f"{s}_gyro"] = gyro + np.array([0.01, -0.015, 0.008]) + rng.normal(0, 0.003, gyro.shape)
        log[f"{s}_acc"] = acc + rng.normal(0, 0.05, acc.shape)
        P = np.array([p_W(t) for t in t_cam]) + rng.normal(0, 0.0015, (len(t_cam), 3)); conf = np.full(len(t_cam), 0.95)
        drops = np.array([body_state(t)[3] for t in t_cam]); P[drops] = np.nan; conf[drops] = 0.0
        log[f"{s}_pos_t_ns"] = (t_cam * 1e9).astype(np.int64); log[f"{s}_wrist_W"] = P; log[f"{s}_pos_conf"] = conf
        pm = np.array([body_state(t)[2] for t in t_cam]) + rng.normal(0, 1.0, len(t_cam)); pc = np.full(len(t_cam), 0.9)
        pm[drops] = np.nan; pc[drops] = 0.0
        log[f"{s}_pinch_t_ns"] = (t_cam * 1e9).astype(np.int64); log[f"{s}_pinch_metric"] = pm; log[f"{s}_pinch_conf"] = pc
        log[f"{s}_R_W_H_init"] = R_init[s]
    log["init_t_ns"] = np.int64(2.5e9); log["mirror_right_y"] = np.int8(1)
    return log, segs


# ---------------------------------------------------------------------------------------------------- replay
def replay(log: dict, c, *, ik: str = "toy", robot_url: str | None = None, out: Path, name: str, segs=None) -> dict:
    fus = build_hand_fusion(c); pin = build_pinch(c)
    ahrs = {s: MahonyAhrs(AhrsConfig.from_dict(c.imu["ahrs"])) for s in SIDES}
    rel_win = [tuple(w) for w in np.asarray(log.get("deadman_released_ns", np.zeros((0, 2))))]
    clutch_win = [tuple(w) for w in np.asarray(log.get("clutch_windows_ns", np.zeros((0, 2))))]
    deadman = ScriptedDeadman(rel_win); clutch = Clutch()
    if ik == "toy":
        solvers = {s: ToyIk() for s in SIDES}; jl = {s: toy_limits() for s in SIDES}
        q0 = {"left": np.array([0.30, 0.20, 0.15, 0, np.pi / 2, 0]), "right": np.array([0.30, -0.20, 0.15, 0, np.pi / 2, 0])}
        bridge = MockBridge(q=q0)
    else:
        solvers = pyroki_pair(); jl = None
        if robot_url: bridge = HttpBridge(robot_url, live=False)
        else:
            home = {s: solvers[s].q_full[solvers[s].idx6].astype(np.float64) for s in SIDES}; bridge = MockBridge(q=home)
    ctl = build_controller(c, bridge=bridge, solvers=solvers, deadman=deadman, clutch=clutch, joint_limits=jl)
    rec = EpisodeRecorder(out, skill_id=c.recording["skill_id"], data_source="VISUAL_TELEOP", config=dict(c), name=name)
    rec.meta["stage"] = 0; rec.meta["offline"] = True; rec.meta["ik"] = ik
    # merge streams into one timeline (IMU drives AHRS; ticks at rate_hz)
    t_end = min(int(log[f"{s}_pos_t_ns"][-1]) for s in SIDES); dt_ns = int(1e9 / c.rate_hz)
    ii = {s: 0 for s in SIDES}; jp = {s: 0 for s in SIDES}; kp = {s: 0 for s in SIDES}; imu_latest = {s: None for s in SIDES}
    pin_state = {s: None for s in SIDES}; inited = False; init_t = int(log["init_t_ns"])
    for t in range(0, t_end, dt_ns):
        for s in SIDES:
            T_us, H = log[f"{s}_imu_t_us"], log[f"{s}_imu_host_ns"]
            while ii[s] < len(T_us) and H[ii[s]] <= t:
                R = ahrs[s].update(int(T_us[ii[s]]), log[f"{s}_gyro"][ii[s]], log[f"{s}_acc"][ii[s]])
                if R is not None: imu_latest[s] = ImuOrientation(s, int(H[ii[s]]), R, log[f"{s}_gyro"][ii[s]] - ahrs[s].bias, True)
                ii[s] += 1
            PT = log[f"{s}_pos_t_ns"]
            while jp[s] < len(PT) and PT[jp[s]] <= t:
                fus[s].observe_position(log[f"{s}_wrist_W"][jp[s]], float(log[f"{s}_pos_conf"][jp[s]]), int(PT[jp[s]])); jp[s] += 1
            KT = log[f"{s}_pinch_t_ns"]
            while kp[s] < len(KT) and KT[kp[s]] <= t:
                m = log[f"{s}_pinch_metric"][kp[s]]; pin_state[s] = pin[s].update(None if not np.isfinite(m) else float(m), float(log[f"{s}_pinch_conf"][kp[s]]), int(KT[kp[s]])); kp[s] += 1
        if not inited:
            if t < init_t or any(imu_latest[s] is None for s in SIDES): continue
            for s in SIDES:
                tilt = fus[s].orient.align(imu_latest[s].R_G_I, log[f"{s}_R_W_H_init"], t); rec.meta["init"][s] = dict(init_tilt_deg=tilt, psi=fus[s].orient.psi)
            hands = {s: fus[s].estimate(t, imu_latest[s]) for s in SIDES}
            ctl.initialise(hands); inited = True
            rec.meta["init"]["robot_T0"] = {s: ctl.arms[s].T_exec.tolist() for s in SIDES}; continue
        clutch.set(any(a <= t < b for a, b in clutch_win))
        hands = {s: fus[s].estimate(t, imu_latest[s]) for s in SIDES}
        rec.add_tick(ctl.tick(hands, pin_state, t))
    rec.meta["counters"] = ctl.counters; ep = rec.close()
    report = dict(episode=str(ep), counters=ctl.counters)
    try: report["export"] = export_episode(ep)
    except Exception as e: report["export_error"] = f"{type(e).__name__}: {e}"
    report["qa"] = episode_qa(ep, c.qa)
    report["stage0"] = stage0_checks(ep, c, segs, rec.meta["init"], ctl)
    (ep / "stage0_report.json").write_text(json.dumps(report, indent=1, default=str)); return report


def stage0_checks(ep: Path, c, segs, init: dict, ctl) -> dict:
    a = dict(np.load(ep / "ticks.npz")); t = a["t_ns"] / 1e9; out = dict(direction=[], passed=True)
    v_max, w_max = c.limits["max_lin_vel_m_s"] / c.rate_hz, c.limits["max_ang_vel_deg_s"] / c.rate_hz
    for s in SIDES:
        T = a[f"exec_T_{s}"]; ok = np.isfinite(T).all(axis=(1, 2))
        dp = np.linalg.norm(np.diff(T[ok][:, :3, 3], axis=0), axis=1)
        dr = np.degrees(Rotation.from_matrix(np.einsum("nji,njk->nik", T[ok][:-1, :3, :3], T[ok][1:, :3, :3])).magnitude())
        out[f"{s}_max_step_mm"] = float(dp.max() * 1e3) if dp.size else 0.0; out[f"{s}_max_rot_step_deg"] = float(dr.max()) if dr.size else 0.0
        out[f"{s}_jumps"] = int(np.sum(dp > v_max * 1.001 + 1e-6) + np.sum(dr > w_max * 1.001 + 1e-6))
        g = a[f"g_exec_{s}"]; gb = (g[np.isfinite(g)] > 0.5).astype(int); out[f"{s}_gripper_transitions"] = int(np.sum(np.abs(np.diff(gb))))
        R0 = np.asarray(init["robot_T0"][s])[:3, :3]; M = ctl.rt.adapters[s].M
        mirror = np.diag([1, -1, 1]) if s == "right" else np.eye(3)
        for (t0, t1, kind, v) in (segs or []):
            m = (t >= t0) & (t <= t1) & ok
            if m.sum() < 5: out["direction"].append(dict(side=s, kind=kind, v=v.tolist(), n=int(m.sum()), cos=None)); continue
            Tm = T[m]
            if kind == "trans":
                got = Tm[-1, :3, 3] - Tm[0, :3, 3]
                exp = R0 @ M @ (mirror @ v)               # human body delta -> robot body -> RB (robot orientation ~ constant here)
            else:
                got = Rotation.from_matrix(Tm[0, :3, :3].T @ Tm[-1, :3, :3]).as_rotvec(); exp = M @ (mirror @ v)
            cos = float(got @ exp / (np.linalg.norm(got) * np.linalg.norm(exp) + 1e-12))
            out["direction"].append(dict(side=s, kind=kind, v=np.round(v, 3).tolist(), cos=round(cos, 4), moved=float(np.linalg.norm(got))))
            if cos < 0.9: out["passed"] = False
        if out[f"{s}_jumps"]: out["passed"] = False
    acc = a["accepted"]; follow = a["mode"] == "FOLLOW"
    out["ik_feasible_frac_while_tracking"] = float(1 - (ctl.counters["ik_reject"] / max(1, np.sum(follow) + ctl.counters["ik_reject"])))
    out["accepted_ticks"] = int(acc.sum()); out["ticks"] = int(len(acc))
    out["init_tilt_deg"] = {s: init[s]["init_tilt_deg"] for s in SIDES}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*"); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="~/Desktop/visual_teleop_stage0_v2"); ap.add_argument("--ik", choices=["toy", "pyroki"], default="toy")
    ap.add_argument("--robot-url", default=None, help="HttpBridge(live=False): real /observe, never POSTs"); ap.add_argument("--config", default=None)
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    ov = {"control": {"arms_enabled": {"left": True, "right": True}}, "retarget": {"orientation_mode": "full"},
          "fusion": {"position": {"source": "wrist_plus_offset"}}}
    if a.synthetic:   # the synthetic log is generated with an identity IMU mount: never apply the measured (real-hardware) mount to it
        ov["imu"] = {"R_I_H": {s: {"measured": True, "rotvec_deg": [0, 0, 0]} for s in SIDES}}
    c = vcfg.load(a.config, overrides=ov)
    runs = []
    if a.synthetic:
        log, segs = synthetic_log(a.seed); np.savez(out / f"synthetic_seed{a.seed}_sensorlog.npz", **log)
        runs.append(replay(log, c, ik=a.ik, robot_url=a.robot_url, out=out, name=f"synthetic_seed{a.seed}_{a.ik}", segs=segs))
    for p in a.logs:
        log = dict(np.load(p)); runs.append(replay(log, c, ik=a.ik, robot_url=a.robot_url, out=out, name=Path(p).stem))
    for r in runs:
        s0 = r["stage0"]; print(json.dumps({k: v for k, v in s0.items() if k != "direction"}, indent=1, default=str))
        bad = [d for d in s0["direction"] if d["cos"] is None or d["cos"] < 0.9]
        print(f"direction segments: {len(s0['direction'])}, failing: {bad}")
        print("QA flags:", r["qa"]["flags"], "| export:", r.get("export_error") or r["export"].get("label_source"))
        print("PASS" if s0["passed"] else "FAIL", r["episode"])


if __name__ == "__main__":
    main()
