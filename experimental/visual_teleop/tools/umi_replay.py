"""Replay UMI translation strokes on the robot (2026-10-20, GPT plan step 6: B = robot commanded, C = robot executed). Translation only:
target = READY TCP + (p_B(t) - p_B(hold before the stroke)), orientation and jaw fixed at READY, time stretched by 1/--speed (default 0.5x).
10-20 offline plan: 15 cm back / down from READY hits joint limits and 40+ deg/s, so by default each stroke is CENTRED on READY (start =
READY - half the stroke's main displacement; --no-center = start at READY). Parity uses relative motion only, so this does not change it.
Input = scripts/umi_tcp_traj.py output of an --proto axes take (needs p_B, i.e. robot_cam_v001.json) + its segments.json.
Every step is IK-planned before anything moves (IK error, bridge joint limits, TCP >= --min-z, joint speed <= --max-deg-s); dry-run prints
the plan, --go streams it at --hz through HttpBridge.send (never a hand POST) and returns to READY between strokes.
    ~/xvla-mac/bin/python -m visual_teleop.tools.umi_replay datasets/tag_rig/<take> --hand 오른손 [--axis X] [--rep 1]          # 계획만
    ~/xvla-mac/bin/python -m visual_teleop.tools.umi_replay datasets/tag_rig/<take> --hand 오른손 --go                          # 실제 (사람이 옆에 있을 때만)
-> datasets/replay/<ts>/: meta.json (segments, speed, READY TCP) + seg_XX.npz (src_t_ns, t_cmd_ns, q_cmd, p_cmd, t_obs_ns, q_obs, p_obs)
then: .venv/bin/python scripts/replay_parity.py datasets/tag_rig/<take> datasets/replay/<ts>"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from ..robot.ik import ArmIk
from ..robot.ik_pink import PinkIkSolver, PinkParams
from .robot_moves import READY_Q_DEG, ramp_to, say, unpark

ROOT = Path(__file__).resolve().parents[2]
SIDE = {"오른손": "right", "왼손": "left"}


def load(take: Path, side: str):
    d = take if (take / f"umi_traj_{side}.npz").exists() else take.parent / (take.name + "_derived")
    z = np.load(d / f"umi_traj_{side}.npz", allow_pickle=True)
    if "p_B" not in z: sys.exit("궤적에 로봇 좌표(p_B)가 없음 -- robot_cam_solve --write 후 umi_tcp_traj 다시")
    return z["t_ns"].astype(np.int64), z["p_B"], json.loads((d / "segments.json").read_text())


def stroke_path(t, p, sg, hz, speed, smooth):
    """UMI relative displacement over [hold_pre mid, hold_post mid] on a uniform robot-time grid (src time = robot time * speed)"""
    a = (sg["hold_pre"][0] + sg["hold_pre"][1]) // 2; b = (sg["hold_post"][0] + sg["hold_post"][1]) // 2 if sg["hold_post"] else sg["back"][1]
    m = (t >= a) & (t <= b)
    if m.sum() < 10: return None, None, "UMI 프레임 부족"
    tt, pp = t[m], p[m]; gap = np.diff(tt).max() / 1e9 if len(tt) > 1 else 9
    h0 = (t >= sg["hold_pre"][0]) & (t <= sg["hold_pre"][1]); p0 = np.median(p[h0], 0) if h0.sum() >= 3 else pp[0]
    src = np.arange(a, b, int(1e9 * speed / hz)); P = np.stack([np.interp(src, tt, pp[:, j]) for j in range(3)], 1) - p0
    if smooth > 1:
        k = np.ones(smooth) / smooth; P = np.stack([np.convolve(np.pad(P[:, j], smooth // 2, mode="edge"), k, "valid")[:len(P)] for j in range(3)], 1)
    return src, P, (f"끊김 {gap:.2f}s" if gap > 0.4 else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--hand", default="오른손", choices=list(SIDE)); ap.add_argument("--axis", choices=["X", "Y", "Z"]); ap.add_argument("--rep", type=int)
    ap.add_argument("--speed", type=float, default=0.5); ap.add_argument("--hz", type=float, default=30.0); ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--max-deg-s", type=float, default=30.0); ap.add_argument("--min-z", type=float, default=0.10); ap.add_argument("--obs-every", type=int, default=3)
    ap.add_argument("--lag-ms", type=float, nargs=6, help="관절별 지연 보상 (ms, 명령을 그만큼 앞당김)"); ap.add_argument("--no-center", action="store_true"); ap.add_argument("--go", action="store_true"); ap.add_argument("--offline", action="store_true", help="로봇 없이 READY 에서 계획만")
    a = ap.parse_args(); c = vcfg.load(); s = SIDE[a.hand]
    if a.offline and a.go: sys.exit("--offline 과 --go 는 같이 못 씀")
    if s not in READY_Q_DEG: sys.exit(f"{s} 팔의 READY 자세가 아직 없음 (robot_moves.READY_Q_DEG)")
    t, p, segs = load(Path(a.take), s)
    segs = [g for g in segs if g["hand"] == a.hand and (a.axis is None or g["axis"] == a.axis) and (a.rep is None or g["rep"] == a.rep)]
    if not segs: sys.exit("선택한 구간 없음")
    br = None if a.offline else HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"])
    try:
        qr = np.radians(READY_Q_DEG[s])
        if a.offline:
            from ego_teleop.robot.rebot_client import SIDE_SLICE
            jda = np.zeros(14); jda[SIDE_SLICE[s]] = READY_Q_DEG[s]; q_now = qr
        else:
            ob = br.observe()
            if ob.estop: sys.exit("로봇 상태가 E-STOP/미연결/읽기 지연입니다 (설치 전 점검은 --offline)")
            jda, q_now = ob.joints_deg_all, ob.q[s]
        if np.degrees(np.abs(q_now - qr)).max() > 5: sys.exit("팔이 READY 자세가 아님 -- 먼저 visual_teleop.tools.ready_pose --go")
        pk = PinkIkSolver(s, PinkParams(**c.ik["pink"])); pk.sync(jda)
        free = PinkIkSolver(s, PinkParams(max_joint_delta_deg=180, iters=200)); free.sync(jda)
        L = ArmIk.from_cfg(c, pk).gate.cfg.limits; lo, hi = np.asarray(L.lower_rad), np.asarray(L.upper_rad); T_r = pk.fk(qr)
        plans, bad = [], []
        for j, sg in enumerate(segs):
            src, P, note = stroke_path(t, p, sg, a.hz, a.speed, a.smooth)
            if src is None: bad.append((j, note)); continue
            i_ax = "XYZ".index(sg["axis"]); off = np.zeros(3)
            if not a.no_center: off[i_ax] = -0.5 * P[np.argmax(np.abs(P[:, i_ax])), i_ax]
            T0 = T_r.copy(); T0[:3, 3] += off; q, pe0, _ = free.ik(qr, T0)          # stroke start (ramped to before streaming)
            Q, worst, err = [], 0.0, pe0
            for x in P + off:
                T = T_r.copy(); T[:3, 3] += x; q, pe, re = free.ik(q, T); Q.append(q); err = max(err, pe)
            Q = np.array(Q); Qs = Q
            if a.lag_ms:      # 10-20 X replay: j2 led ~-100 ms, j3 lagged ~+200 ms -> Z path error up to 45 mm; send q(t + lag_j) per joint
                tg = np.arange(len(Q)) / a.hz; Qs = np.stack([np.interp(tg + a.lag_ms[k] / 1000, tg, Q[:, k]) for k in range(6)], 1)
            vel = np.degrees(np.abs(np.diff(Qs, axis=0))).max() * a.hz if len(Q) > 1 else 0
            zmin = float(min(T_r[2, 3] + P[:, 2] + off[2])); lim = bool(np.any(Q < lo) or np.any(Q > hi))
            ok = err < 0.003 and not lim and zmin >= a.min_z and vel <= a.max_deg_s
            d = P[np.argmax(np.abs(P[:, "XYZ".index(sg["axis"])]))]
            print(f"  {j:2d} {sg['hand']} {sg['axis']} {sg['dir']:5s} {sg['rep']}회: {len(P) / a.hz:4.1f}s, 최대 변위 {np.round(d * 100, 1)} cm, IK 최대오차 {err * 1e3:.1f} mm, "
                  f"관절속도 최대 {vel:.0f}°/s, 최저 z {zmin:.3f}{'  관절한계 밖' if lim else ''} {note} {'OK' if ok else '<- 불가'}")
            if not ok: bad.append((j, "IK/한계/높이/속도")); continue
            plans.append(dict(seg=sg, src=src, P=P, Q=Q, Qs=Qs, off=off))
        print(f"구간 {len(plans)}/{len(segs)}개 실행 가능, 총 {sum(len(x['P']) for x in plans) / a.hz / 60:.1f}분 ({a.speed}배속)")
        if bad: print("제외:", bad)
        if not a.go: print("계획만 출력함 (--go 로 실제 이동)"); return
        out = ROOT / "datasets/replay" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps(dict(take=str(a.take), hand=a.hand, side=s, speed=a.speed, hz=a.hz, smooth=a.smooth, T_ready=T_r.tolist(),
                                                       segments=[x["seg"] for x in plans]), ensure_ascii=False, indent=1))
        say("유미 동작 재생을 시작합니다"); unpark(br, s); ob = br.observe(); g = {x: ob.g(x) for x in ("left", "right")}; other = "left" if s == "right" else "right"
        for j, x in enumerate(plans):
            ramp_to(br, s, x["Qs"][0], 10.0, g=g); time.sleep(0.5); say(f"{x['seg']['dir']} {x['seg']['rep']}회")
            q_oth = br.observe().q[other]; tc, to, qo = [], [], []; t0 = time.monotonic()
            for k, q in enumerate(x["Qs"]):
                r = br.send({s: q, other: q_oth}, g); tc.append(time.monotonic_ns())
                if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
                if k % a.obs_every == 0:
                    o = br.observe()
                    if o.estop: br.estop(); sys.exit("E-STOP/읽기 지연 -- 정지")
                    to.append(o.t_ns if getattr(o, "t_ns", None) else time.monotonic_ns()); qo.append(o.q[s].copy())
                time.sleep(max(0.0, t0 + (k + 1) / a.hz - time.monotonic()))
            for _ in range(int(a.hz)):                       # 1 s tail: executed motion settles
                o = br.observe(); to.append(o.t_ns if getattr(o, "t_ns", None) else time.monotonic_ns()); qo.append(o.q[s].copy()); time.sleep(1 / a.hz)
            qo = np.array(qo)
            np.savez(out / f"seg_{j:02d}.npz", src_t_ns=x["src"], t_cmd_ns=np.array(tc), q_cmd=x["Q"], p_cmd=np.array([pk.fk(q)[:3, 3] for q in x["Q"]]) - T_r[:3, 3] - x["off"], start_offset=x["off"], q_send=x["Qs"], lag_ms=np.array(a.lag_ms or [0] * 6),
                     t_obs_ns=np.array(to), q_obs=qo, p_obs=np.array([pk.fk(q)[:3, 3] for q in qo]) - T_r[:3, 3] - x["off"])
            ramp_to(br, s, qr, 10.0, g=g)
        say("재생 끝났습니다"); print(f"저장 {out}\n다음: .venv/bin/python scripts/replay_parity.py {a.take} {out}")
    finally:
        if br is not None: br.close()


if __name__ == "__main__":
    main()
