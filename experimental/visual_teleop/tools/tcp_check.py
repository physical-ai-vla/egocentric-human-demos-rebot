"""Robot TCP location check (2026-10-15, user: "TCP가 그리퍼 끝이 아니라 wrist인 것 같아").
Rotates the arm about its IK TCP (URDF <side>_tcp) with the TCP POSITION held: +-deg about the TCP y axis (jaw axis, nod) and then
about the TCP x axis, slowly, through Pink + the feasibility gate + HttpBridge.send (single writer: live_teleop must be OFF).
Watch the real jaw tip: it stays put if the URDF TCP is at the tip; it draws an arc of radius ~ (tip - TCP) otherwise.
    ~/xvla-mac/bin/python -m visual_teleop.tools.tcp_check --side right            # plan only
    ~/xvla-mac/bin/python -m visual_teleop.tools.tcp_check --side right --go"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from ..robot.ik import ArmIk
from ..robot.ik_pink import PinkIkSolver, PinkParams


def say(s: str) -> None:
    print(s, flush=True)
    try: subprocess.run(["say", "-v", "Yuna", s], timeout=10)
    except Exception: pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="right"); ap.add_argument("--deg", type=float, default=10.0); ap.add_argument("--period-s", type=float, default=4.0)
    ap.add_argument("--hz", type=float, default=30.0); ap.add_argument("--go", action="store_true")
    ap.add_argument("--axes", default="y", help="y = nod about the jaw axis (first axis), x = the other; 'yx' = both")
    ap.add_argument("--video-index", type=int, default=None, help="OpenCV index of a camera to film the jaw tip (C922 = 3)")
    a = ap.parse_args(); c = vcfg.load(); s = a.side; other = "left" if s == "right" else "right"
    br = HttpBridge(c.robot["base_url"], live=a.go, timeout_s=max(0.5, c.robot["request_timeout_s"]))
    try:
        ob = br.observe()
        if ob.estop: sys.exit("브리지 E-STOP/미연결/읽기 지연")
        if s in (br.status().get("parked") or []): sys.exit("팔이 parking(토크 꺼짐) 상태 -- ready_pose 먼저")
        pk = PinkIkSolver(s, PinkParams(**c.ik["pink"])); pk.sync(ob.joints_deg_all); arm = ArmIk.from_cfg(c, pk)
        q = ob.q[s].copy(); T0 = pk.fk(q); g = {x: ob.g(x) for x in ("left", "right")}
        print(f"시작 TCP {np.round(T0[:3, 3], 3)}  ±{a.deg:.0f}도, 축마다 {a.period_s:.0f}초")
        # plan everything first (refuse before moving if any step is infeasible)
        plan = []
        for ax in [{"x": 0, "y": 1}[ch] for ch in a.axes]:                         # TCP y (nod about the jaw axis), then TCP x
            n = int(a.period_s * a.hz)
            for k in range(n + 1):
                ang = np.radians(a.deg) * np.sin(2 * np.pi * k / n)
                T = T0.copy(); T[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(np.eye(3)[ax] * ang).as_matrix()
                v = arm.solve(q, T)
                if not v.ok: sys.exit(f"계획 단계에서 IK 거부 (축 {'yx'[ax]}, {np.degrees(ang):.1f}도): {v.reasons} -- 각도를 줄이세요")
                q = v.q; plan.append((ax, q.copy()))
        drift = max(np.linalg.norm(pk.fk(qq)[:3, 3] - T0[:3, 3]) for _, qq in plan) * 1000
        print(f"계획 {len(plan)}스텝, IK 상 TCP 위치 흔들림 최대 {drift:.1f} mm")
        if not a.go: print("계획만 출력함 (--go 로 실제 실행)"); return
        vid = None
        if a.video_index is not None:
            import cv2, threading
            from pathlib import Path
            cap = cv2.VideoCapture(a.video_index)
            if cap.isOpened():
                vp = Path(__file__).resolve().parents[2] / "datasets" / "tcp_check" / time.strftime("tcp_%H%M%S.mp4"); vp.parent.mkdir(parents=True, exist_ok=True)
                stop_v = threading.Event(); frames = []
                def grab():
                    while not stop_v.is_set():
                        ok, f = cap.read()
                        if ok: frames.append(f)
                th = threading.Thread(target=grab, daemon=True); th.start(); vid = (cap, stop_v, frames, vp, th)
            else: print("카메라를 열 수 없음 -- 영상 없이 진행")
        say("집게 끝을 보세요. 지금부터 끝을 중심으로 끄덕입니다.")
        t0 = time.monotonic(); cur = None; log = []
        try:
            for i, (ax, qq) in enumerate(plan):
                if ax != cur:
                    cur = ax
                    if ax == 0: say("이번에는 다른 축으로 돌립니다.")
                    t0 = time.monotonic() - i / a.hz
                r = br.send({s: qq, other: ob.q[other]}, g)
                if not r.accepted: br.estop(); sys.exit(f"브리지 거부: {r.error} -- E-STOP 보냄")
                if i % 3 == 0:                                                    # 10 Hz: measured vs commanded TCP
                    o = br.observe()
                    if o.estop: br.estop(); sys.exit("E-STOP/읽기 지연 -- 정지")
                    Tm, Tc = pk.fk(o.q[s]), pk.fk(qq); log.append((i / a.hz, Tc[:3, 3], Tm[:3, 3], np.degrees(np.abs(o.q[s] - qq)).max()))
                    if not log[:-1]: pm0 = Tm[:3, 3].copy()                       # measured start (static gravity sag ~1 cm is normal)
                    if np.linalg.norm(Tm[:3, 3] - pm0) > 0.02:
                        print(f"측정 TCP가 시작 위치에서 {np.linalg.norm(Tm[:3, 3] - pm0) * 1000:.0f} mm 벗어남 -- 여기서 멈춤 (현재 자세 유지)"); break
                time.sleep(max(0.0, t0 + i / a.hz - time.monotonic()))
        finally:
            if vid is not None:
                import cv2
                cap, stop_v, frames, vp, th = vid; stop_v.set(); th.join(1.0); cap.release()
                if frames:
                    h, w = frames[0].shape[:2]; vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
                    for f in frames: vw.write(f)
                    vw.release(); print("영상 저장:", vp)
        time.sleep(0.8); o = br.observe()
        if log:
            C = np.array([x[1] for x in log]); M = np.array([x[2] for x in log])
            print(f"측정 TCP의 시작 대비 흔들림 최대 {np.linalg.norm(M - M[0], axis=1).max() * 1000:.1f} mm (정적 처짐 제외) | "
                  f"명령 TCP 위치 흔들림 최대 {np.linalg.norm(C - T0[:3, 3], axis=1).max() * 1000:.1f} mm | "
                  f"측정 TCP 위치 흔들림 최대 {np.linalg.norm(M - T0[:3, 3], axis=1).max() * 1000:.1f} mm | "
                  f"측정-명령 차이 최대 {np.linalg.norm(M - C, axis=1).max() * 1000:.1f} mm | 관절 추종 오차 최대 {max(x[3] for x in log):.1f}도")
        print(f"끝: 측정 TCP {np.round(pk.fk(o.q[s])[:3, 3], 3)} (시작 {np.round(T0[:3, 3], 3)})")
        say("끝났습니다. 집게 끝이 제자리였는지, 원을 그렸는지 알려주세요.")
    finally: br.close()


if __name__ == "__main__":
    main()
