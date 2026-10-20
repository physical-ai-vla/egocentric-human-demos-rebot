"""Orientation-only probe (2026-10-15): roll/pitch/yaw looked frozen on the robot. No robot involved.
Needs the sensor hub running and already INIT-aligned; live_teleop must be OFF (this binds the hand port).
20 s with voice cues: 5 s still, 5 s roll +-30 (about the jaw/tool axis), 5 s pitch +-30, 5 s yaw +-30. Logs per frame:
    raw IMU quaternion (hub recording pose_log.npz, R_G_I)    fused UMI TCP rotation R_W_H (hub hand message)
    relative R0^T R(t) (R0 = first still frame, fixed)        robot-side target rotation = RelativeRetargeter with the YAML
and prints, per phase, the peak rotation (deg) about each axis, so a dead stage is visible at a glance.
    .venv/bin/python -m visual_teleop.tools.orient_probe [--config visual_teleop/config/teleop_kpt.yaml]"""
from __future__ import annotations
import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .. import config as vcfg
from ..link import Publisher

PHASES = [("정지", "가만히 들고 계세요"), ("롤", "집게 축을 중심으로 좌우로 비트세요, 30도"),
          ("피치", "집게 끝을 위아래로 끄덕이세요, 30도"), ("요", "집게 끝을 좌우로 돌리세요, 30도")]


def peak_axes(R0: np.ndarray, Rs: list, frame: str) -> np.ndarray:
    """Max |rotvec component| (deg) of the rotation since R0, in the body (R0^T R) or world (R R0^T) frame."""
    v = [Rotation.from_matrix(R0.T @ R if frame == "body" else R @ R0.T).as_rotvec() for R in Rs]
    return np.degrees(np.abs(np.array(v)).max(0)) if v else np.full(3, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None); ap.add_argument("--phase-s", type=float, default=5.0)
    a = ap.parse_args(); c = vcfg.load(a.config)
    out = Path(__file__).resolve().parents[2] / "datasets" / "orient_probe" / time.strftime("%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rx.bind(("127.0.0.1", c.link["hand_port"])); rx.settimeout(0.05)
    cmd = Publisher(c.link["cmd_port"]); cmd.send("cmd", cmd="rec_start", dir=str(out)); time.sleep(0.3)
    rows = []; t0 = time.monotonic(); ph = -1
    while (el := time.monotonic() - t0) < a.phase_s * len(PHASES):
        k = int(el // a.phase_s)
        if k != ph:
            ph = k; print(f"[{PHASES[k][0]}] {PHASES[k][1]}", flush=True)
            subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"]); subprocess.Popen(["say", "-v", "Yuna", PHASES[k][1]])
        try: m = json.loads(rx.recvfrom(65535)[0])
        except socket.timeout: continue
        if m.get("kind") == "hand" and m.get("side") == "right" and m.get("T_W_H") is not None:
            rows.append((ph, int(m["t_ns"]), np.asarray(m["T_W_H"], np.float64), m.get("orientation_source"), m.get("status")))
    cmd.send("cmd", cmd="rec_stop"); time.sleep(1.0); rx.close()
    if len(rows) < 50: print(f"허브 자세 메시지 {len(rows)}개뿐 -- 허브 INIT(정렬)이 됐는지, live_teleop이 꺼졌는지 확인"); return
    np.savez(out / "probe.npz", phase=np.array([r[0] for r in rows]), t_ns=np.array([r[1] for r in rows]), T_W_H=np.array([r[2] for r in rows]))
    # raw IMU from the hub recording (written by root; readable)
    imu = None
    try:
        P = np.load(out / "pose_log.npz"); sel = P["is_right"]; imu = (P["t_ns"][sel], P["imu_quat_xyzw"][sel])
    except Exception as e: print("pose_log.npz 없음:", e)
    from ..build import build_controller  # noqa: F401  (import check only)
    from ..retargeting.frame_adapter import FrameAdapter
    from ..retargeting.motion_scaling import MotionScaling
    from ..retargeting.relative_se3 import RelativeRetargeter
    rt = RelativeRetargeter({s: FrameAdapter.from_cfg(c.retarget["adapter"][s]) for s in ("left", "right")}, MotionScaling.from_cfg(c.retarget),
                            orientation_mode=c.retarget["orientation_mode"], translation_frame=c.retarget.get("translation_frame", "body"),
                            pivot_human_m=float(c.retarget.get("pivot_human_m", 0.0)), pivot_robot_m=float(c.retarget.get("pivot_robot_m", 0.0)),
                            rotation_frame=c.retarget.get("rotation_frame", "body"))
    T0 = rows[0][2]; rt.anchor("right", T0, np.eye(4))
    print(f"\n프레임 {len(rows)}개, 방향 소스 {rows[0][3]}, 저장 {out}")
    print("단계별 최대 회전(도) [x, y, z]  -- body = UMI TCP 축(z = 집게 방향), world = W 축(z = 위)")
    for k, (name, _) in enumerate(PHASES):
        Rs = [r[2][:3, :3] for r in rows if r[0] == k]
        tg = []
        for r in rows:
            if r[0] != k: continue
            try: tg.append(rt.target("right", r[2])[0][:3, :3])
            except Exception: pass
        line = f"  {name:3s} TCP body {np.round(peak_axes(T0[:3, :3], Rs, 'body'), 1)}  world {np.round(peak_axes(T0[:3, :3], Rs, 'world'), 1)}  로봇목표 {np.round(peak_axes(np.eye(3), tg, 'body'), 1)}"
        if imu is not None:
            tk = np.array([r[1] for r in rows if r[0] == k])
            if len(tk):
                m = (imu[0] >= tk.min()) & (imu[0] <= tk.max()); q = imu[1][m]; q = q[np.all(np.isfinite(q), 1)]
                if len(q): Ri = Rotation.from_quat(q).as_matrix(); Rq0 = Rotation.from_quat(imu[1][np.all(np.isfinite(imu[1]), 1)][0]).as_matrix()
                if len(q): line += f"  원시IMU {np.round(peak_axes(Rq0, list(Ri), 'body'), 1)}"
        print(line)


if __name__ == "__main__":
    main()
