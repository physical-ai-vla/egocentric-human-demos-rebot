"""Move one arm from the folded rest pose to a teleop READY pose (2026-10-15). Teleop from rest is impossible: rest is next to the
joint2 limit (near-singular) and j3 sits past the bridge's limit margin, so every IK step is refused (pyroki and Pink alike).

READY = a FIXED joint pose (READY_Q_DEG, TCP ~ (0.45, -0.30, 0.28) for the right arm). 10-15 bug: it used to be "current TCP +
offset", which pushed an already-extended arm further out. --relative keeps the old behaviour (offset from the CURRENT TCP).
The joint-space path is checked first: TCP must stay >= --min-z the whole way. Sent as a joint-space ramp at <= --deg-s through HttpBridge.send (the only sanctioned 14-vector path; never a hand
POST). Jaw kept at its measured value; the other arm gets its observed joints. Aborts on E-STOP / stale reads / refusal.
    ~/xvla-mac/bin/python -m visual_teleop.tools.ready_pose --side right            # 계획만 출력 (로봇 안 움직임)
    ~/xvla-mac/bin/python -m visual_teleop.tools.ready_pose --side right --go       # 실제 이동"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
import numpy as np
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from ..robot.ik import ArmIk
from ..robot.ik_pink import PinkIkSolver, PinkParams

READY_Q_DEG = {"right": [1.8, -56.8, -32.5, -19.9, -1.3, 5.8]}       # measured 10-15: TCP (0.449, -0.295, 0.283), teleop IK unblocked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="right", choices=["right", "left"]); ap.add_argument("--go", action="store_true")
    ap.add_argument("--forward", type=float, default=0.12); ap.add_argument("--up", type=float, default=0.12)
    ap.add_argument("--deg-s", type=float, default=15.0); ap.add_argument("--relative", action="store_true"); ap.add_argument("--min-z", type=float, default=0.10); ap.add_argument("--hz", type=float, default=30.0)
    a = ap.parse_args(); c = vcfg.load(); s = a.side
    br = HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"])
    try:
        ob = br.observe()
        if ob.estop: sys.exit("로봇 상태가 E-STOP/미연결/읽기 지연입니다 -- 브리지 확인 후 다시")
        pk = PinkIkSolver(s, PinkParams(**c.ik["pink"])); pk.sync(ob.joints_deg_all)
        q0 = ob.q[s].copy(); T0 = pk.fk(q0); T = T0.copy(); T[:3, 3] += [a.forward, 0.0, a.up]
        free = PinkIkSolver(s, PinkParams(max_joint_delta_deg=180, iters=400)); free.sync(ob.joints_deg_all)
        seed = np.radians([0, -30, -40, 30, 0, 0]) * (1 if s == "right" else np.array([1, 1, 1, 1, -1, -1]))
        if a.relative or s not in READY_Q_DEG: qr, pe, re = free.ik(seed, T)
        else: qr = np.radians(READY_Q_DEG[s]); pe, re = 0.0, 0.0
        zmin = min(pk.fk(q0 + u * (qr - q0))[2, 3] for u in np.linspace(0, 1, 41))
        print(f"경로 중 TCP 최저 높이 {zmin:.3f} m (하한 {a.min_z:.2f})")
        if zmin < a.min_z: sys.exit("경로가 너무 낮게 지나감 -- 이동 안 함")
        L = ArmIk.from_cfg(c, pk).gate.cfg.limits; lo, hi = np.asarray(L.lower_rad), np.asarray(L.upper_rad)
        print(f"현재 관절(도) {np.round(np.degrees(q0), 1)}  TCP {np.round(T0[:3, 3], 3)}")
        print(f"READY 관절(도) {np.round(np.degrees(qr), 1)}  TCP {np.round(pk.fk(qr)[:3, 3], 3)}  IK 오차 {pe * 1e3:.1f} mm / {re:.1f}도")
        if pe > 0.005 or re > 2: sys.exit("READY 자세 IK가 수렴하지 않음 -- 오프셋을 줄여서 다시")
        if np.any(qr < lo) or np.any(qr > hi): sys.exit(f"READY 자세가 브리지 관절 한계 밖: {np.round(np.degrees(qr), 1)}")
        d = qr - q0; n = max(2, int(np.ceil(np.degrees(np.abs(d)).max() / a.deg_s * a.hz)))
        print(f"최대 관절 이동 {np.degrees(np.abs(d)).max():.1f}도, {n / a.hz:.1f}초 ({a.deg_s:.0f}도/초 이하)")
        if not a.go: print("계획만 출력함 (--go 로 실제 이동)"); return
        try: subprocess.run(["say", "-v", "Yuna", "로봇 팔이 준비 자세로 움직입니다"], timeout=10)
        except Exception: pass
        st = br.status()
        if s in (st.get("parked") or []):          # 10-15: the first /execute_step on a parked arm unparks (torque on) and took > 0.2 s
            r = br.s.post(br.base_url + "/torque", json={"arm": s, "on": True}, timeout=5.0).json(); print("토크 켬:", r.get("arms")); time.sleep(0.5)
        br.timeout_s = max(br.timeout_s, 0.5)
        ob = br.observe(); q0 = ob.q[s].copy(); d = qr - q0
        if ob.estop: sys.exit("E-STOP 상태 -- 해제 후 다시")
        g = {x: ob.g(x) for x in ("left", "right")}; other = "left" if s == "right" else "right"
        t0 = time.monotonic()
        for k in range(1, n + 1):
            u = k / n; u = u * u * (3 - 2 * u)                                       # smoothstep: soft start / stop
            q = {s: q0 + u * d, other: ob.q[other]}
            r = br.send(q, g)
            if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
            if k % int(a.hz) == 0:
                o = br.observe()
                if o.estop: br.estop(); sys.exit("이동 중 E-STOP/읽기 지연 -- 정지")
            time.sleep(max(0.0, t0 + k / a.hz - time.monotonic()))
        time.sleep(1.0); o = br.observe(); err = np.degrees(np.abs(o.q[s] - qr)).max()
        print(f"도착: 관절 오차 최대 {err:.1f}도, TCP {np.round(pk.fk(o.q[s])[:3, 3], 3)}")
    finally: br.close()


if __name__ == "__main__":
    main()
