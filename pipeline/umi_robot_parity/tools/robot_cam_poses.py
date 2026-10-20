"""Robot base <-> global Orbbec calibration, robot side (2026-10-20, GPT plan step 5: tag 220 on the right gripper, 25-40 poses with
rotation excitation, held-out check). Moves the arm through N poses around READY (position box + 10-25 deg tilts), holds each still, and
logs the observed joints + FK (<side>_tcp, same URDF as ik_pink) with host monotonic_ns hold windows. The Orbbec is recorded by a
SEPARATE sudo process (same monotonic clock), the solver pairs them:
    1) sudo .venv/bin/python -m visual_teleop.tools.tag_rig_record --proto robotcal --secs 900     (미리보기 http://localhost:8747/)
    2) ~/xvla-mac/bin/python -m visual_teleop.tools.robot_cam_poses --side right                    # 계획만 (로봇 안 움직임)
       ~/xvla-mac/bin/python -m visual_teleop.tools.robot_cam_poses --side right --go               # 실제 이동 (사람이 옆에 있을 때만)
    3) (끝나면 1번 창에서 Ctrl+C) -> .venv/bin/python scripts/robot_cam_solve.py datasets/tag_rig/<take> datasets/robot_cam/<run>
The arm must already be at READY (visual_teleop.tools.ready_pose --go). Poses are IK-checked (error, bridge joint limits, TCP >= --min-z
on the whole joint path) before anything moves; while recording, /tags of the recorder says whether the tag was seen at each hold
(poses where it was not are kept in the log but flagged)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from ..robot.ik import ArmIk
from ..robot.ik_pink import PinkIkSolver, PinkParams
from .robot_moves import READY_Q_DEG, move_secs, ramp_to, say, unpark

ROOT = Path(__file__).resolve().parents[2]


def plan_poses(pk, free, q_ready, n, box, rot_deg, min_z, lo, hi, seed=0):
    rng = np.random.default_rng(seed); T_r = pk.fk(q_ready); out, tries = [], 0
    while len(out) < n and tries < n * 40:
        tries += 1
        dp = rng.uniform(-1, 1, 3) * box
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = np.radians(rng.uniform(*rot_deg)) * (len(out) % 4 != 0)   # every 4th: pure translation
        T = T_r.copy(); T[:3, 3] += dp; T[:3, :3] = R.from_rotvec(ax * ang).as_matrix() @ T_r[:3, :3]
        if T[2, 3] < min_z: continue
        q, pe, re = free.ik(q_ready, T)
        if pe > 0.003 or re > 1.5 or np.any(q < lo) or np.any(q > hi): continue
        out.append(dict(q=q, T=T, dp=dp, ang_deg=float(np.degrees(ang))))
    order, left, cur = [], list(range(len(out))), q_ready        # greedy nearest neighbour in joint space (short, slow moves)
    while left:
        j = min(left, key=lambda i: np.abs(out[i]["q"] - cur).max()); order.append(out[j]); left.remove(j); cur = out[j]["q"]
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="right", choices=["right", "left"]); ap.add_argument("--go", action="store_true")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--box-cm", type=float, nargs=3, default=[8, 8, 6]); ap.add_argument("--rot-deg", type=float, nargs=2, default=[10, 25])
    ap.add_argument("--hold", type=float, default=2.0); ap.add_argument("--settle", type=float, default=1.2); ap.add_argument("--deg-s", type=float, default=12.0)
    ap.add_argument("--min-z", type=float, default=0.10); ap.add_argument("--tag", type=int, default=220); ap.add_argument("--tags-url", default="http://localhost:8747/tags")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--offline", action="store_true", help="로봇 없이 READY 에서 계획만 (설치 전 점검)")
    a = ap.parse_args(); c = vcfg.load(); s = a.side
    if a.offline and a.go: sys.exit("--offline 과 --go 는 같이 못 씀")
    if s not in READY_Q_DEG: sys.exit(f"{s} 팔의 READY 자세가 아직 없음 (robot_moves.READY_Q_DEG)")
    br = HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"])
    try:
        qr = np.radians(READY_Q_DEG[s])
        if a.offline:
            from ego_teleop.robot.rebot_client import SIDE_SLICE
            jda = np.zeros(14); jda[SIDE_SLICE[s]] = READY_Q_DEG[s]; q_now = qr.copy(); print("[오프라인] 로봇 없이 READY 에서 계획만")
        else:
            ob = br.observe()
            if ob.estop: sys.exit("로봇 상태가 E-STOP/미연결/읽기 지연입니다 -- 브리지 확인 후 다시 (설치 전 점검은 --offline)")
            jda = ob.joints_deg_all; q_now = ob.q[s]
        pk = PinkIkSolver(s, PinkParams(**c.ik["pink"])); pk.sync(jda)
        free = PinkIkSolver(s, PinkParams(max_joint_delta_deg=180, iters=400)); free.sync(jda)
        L = ArmIk.from_cfg(c, pk).gate.cfg.limits; lo, hi = np.asarray(L.lower_rad), np.asarray(L.upper_rad)
        off = np.degrees(np.abs(q_now - qr)).max()
        print(f"현재 관절이 READY와 최대 {off:.1f}도 차이")
        if off > 5: sys.exit("팔이 READY 자세가 아님 -- 먼저 visual_teleop.tools.ready_pose --go")
        P = plan_poses(pk, free, qr, a.n, np.array(a.box_cm) / 100, a.rot_deg, a.min_z, lo, hi, a.seed)
        if len(P) < a.n: print(f"[주의] 유효 포즈 {len(P)}/{a.n} -- --box-cm / --rot-deg 를 줄이세요")
        prev, tot, bad = qr, 0.0, []
        for i, p in enumerate(P):
            zmin = min(pk.fk(prev + u * (p["q"] - prev))[2, 3] for u in np.linspace(0, 1, 21))
            if zmin < a.min_z: bad.append(i)
            tot += move_secs(prev, p["q"], a.deg_s) + a.settle + a.hold; prev = p["q"]
            print(f"  {i:2d}: dP {np.round(p['dp'] * 100, 1)} cm  기울기 {p['ang_deg']:4.1f}°  관절(도) {np.round(np.degrees(p['q']), 1)}  경로최저 z {zmin:.3f}")
        tot += move_secs(prev, qr, a.deg_s)
        print(f"포즈 {len(P)}개, 예상 {tot / 60:.1f}분 ({a.deg_s:.0f}도/초, 정지 {a.settle + a.hold:.1f}초)")
        if bad: sys.exit(f"경로가 너무 낮은 포즈 {bad} -- 이동 안 함")
        if not a.go: print("계획만 출력함 (--go 로 실제 이동, 먼저 tag_rig_record --proto robotcal 켜기)"); return
        import requests
        try: requests.get(a.tags_url, timeout=1).json(); has_tags = True
        except Exception: has_tags = False; print(f"[주의] {a.tags_url} 응답 없음 -- 녹화기가 꺼져 있으면 캘리브레이션이 안 됩니다")
        out = ROOT / "datasets/robot_cam" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(dict(side=s, tag=a.tag, q_ready=qr.tolist(), args=vars(a), poses=[dict(q=p["q"].tolist(), T=p["T"].tolist()) for p in P]), indent=1))
        say("로봇 캘리브레이션 동작을 시작합니다"); unpark(br, s); log = open(out / "poses.jsonl", "w")
        for i, p in enumerate(P):
            ramp_to(br, s, p["q"], a.deg_s); time.sleep(a.settle)
            Q, seen, side_px, t0 = [], 0, [], time.monotonic_ns(); tn = 0
            while time.monotonic_ns() - t0 < a.hold * 1e9:
                o = br.observe()
                if o.estop: br.estop(); sys.exit("E-STOP/읽기 지연 -- 정지")
                Q.append(o.q[s].copy())
                if has_tags:
                    try:
                        j = requests.get(a.tags_url, timeout=0.3).json(); tn += 1
                        if a.tag in j.get("ids", []): seen += 1; side_px.append(j["side_px"].get(str(a.tag), 0))
                    except Exception: pass
                time.sleep(0.1)
            t1 = time.monotonic_ns(); Q = np.array(Q); qm = Q.mean(0)
            rec = dict(i=i, t_hold0_ns=t0, t_hold1_ns=t1, q_cmd=p["q"].tolist(), q_obs=qm.tolist(), q_obs_std_deg=float(np.degrees(Q.std(0)).max()),
                       T_B_E=pk.fk(qm).tolist(), T_B_E_cmd=p["T"].tolist(), tag_seen=(seen / tn) if tn else None, tag_side_px=float(np.median(side_px)) if side_px else None)
            log.write(json.dumps(rec) + "\n"); log.flush()
            err = np.degrees(np.abs(qm - p["q"])).max()
            print(f"  {i:2d}/{len(P)}: 관절 추종 오차 {err:.2f}°, 정지 중 흔들림 {rec['q_obs_std_deg']:.2f}°, 태그 {a.tag} " +
                  (f"{rec['tag_seen'] * 100:.0f}% 보임 ({rec['tag_side_px'] or 0:.0f}px)" if rec["tag_seen"] is not None else "확인 불가"), flush=True)
        ramp_to(br, s, qr, a.deg_s); log.close(); say("끝났습니다. 녹화 창에서 컨트롤 C")
        print(f"저장 {out}\n다음: .venv/bin/python scripts/robot_cam_solve.py datasets/tag_rig/<녹화폴더> {out}")
    finally: br.close()


if __name__ == "__main__":
    main()
