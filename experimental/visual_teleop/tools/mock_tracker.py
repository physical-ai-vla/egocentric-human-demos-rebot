"""Mock 6DoF tracker (2026-10-15): emits the tracker_pose UDP protocol so the tracker path can be tested before hardware.
    .venv/bin/python -m visual_teleop.tools.mock_tracker [--hz 120] [--preset steamvr_y_up] [--seconds 0]
Motion: slow circle 5 cm + 15 deg wrist roll oscillation around a fixed start, in the chosen tracker-world convention.
--faults: every 10 s cycles through stale (pause 150 ms), conf 0.3 for 0.5 s, 5 % packet loss for 1 s, a 20 cm jump (one sample),
an invalid quaternion (one sample) -- to exercise TRACKING -> DEGRADED -> LOST and back.
The tracker sits on the UMI with a ground-truth body->TCP offset (TRUE_T_TB_TCP below), so calibrations can be checked:
--motion pivot   jaw tip fixed at one point, UMI rocks +-35 deg / twists -> calib_tracker pivot must recover TRUE r (RMS ~ --noise-mm).
--noise-mm       gaussian position noise per sample."""
from __future__ import annotations
import argparse, json, socket, time
import numpy as np
from scipy.spatial.transform import Rotation
from ..vision.tracker6dof import PRESETS, make_T

TRUE_T_TB_TCP = make_T(Rotation.from_euler("xyz", [10, -20, 90], degrees=True).as_matrix(), [0.03, -0.02, 0.11])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--hz", type=float, default=120); ap.add_argument("--port", type=int, default=47813)
    ap.add_argument("--preset", default="steamvr_y_up"); ap.add_argument("--seconds", type=float, default=0.0); ap.add_argument("--faults", action="store_true")
    ap.add_argument("--motion", choices=["teleop", "pivot"], default="teleop"); ap.add_argument("--noise-mm", type=float, default=0.0)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); Rw = PRESETS[a.preset].T          # W -> tracker world
    t0 = time.monotonic(); n = 0
    while a.seconds <= 0 or time.monotonic() - t0 < a.seconds:
        t = time.monotonic() - t0
        if a.motion == "pivot":   # TCP (jaw tip) fixed, orientation sweeps a cone + twist
            p_W = np.array([0.4, 0.0, 0.05]); R_W = Rotation.from_euler("xyz", [35 * np.sin(0.9 * t), 180 + 35 * np.sin(0.6 * t + 1), 40 * np.sin(0.4 * t)], degrees=True).as_matrix()
        else:
            p_W = np.array([0.4 + 0.05 * np.cos(0.5 * t), 0.05 * np.sin(0.5 * t), 0.3]); R_W = Rotation.from_rotvec([np.radians(15) * np.sin(0.7 * t), np.pi / 2, 0]).as_matrix()
        T_W_TB = make_T(R_W, p_W) @ np.linalg.inv(TRUE_T_TB_TCP)                     # tracker body pose from the TCP pose
        p_W, R_W = T_W_TB[:3, 3] + rng.normal(0, a.noise_mm / 1000, 3), T_W_TB[:3, :3]
        q = Rotation.from_matrix(Rw @ R_W).as_quat(); conf = 1.0; send = True; ph = t % 10.0
        if a.faults:
            if 2.0 <= ph < 2.15: send = False                                  # stale 150 ms
            elif 4.0 <= ph < 4.5: conf = 0.3                                   # confidence drop
            elif 6.0 <= ph < 7.0: send = rng.random() > 0.05                   # 5 % packet loss
            elif 8.0 <= ph < 8.0 + 1.0 / a.hz: p_W = p_W + [0.2, 0, 0]         # one 20 cm jump
            elif 9.0 <= ph < 9.0 + 1.0 / a.hz: q = q * 3.0                     # invalid quaternion
        n += 1
        if send:
            msg = dict(kind="tracker_pose", side="right", seq=n, t_src_ns=None, t_send_ns=time.monotonic_ns(), p=(Rw @ p_W).tolist(), q_xyzw=list(q), valid=True, conf=conf)
            s.sendto(json.dumps(msg).encode(), ("127.0.0.1", a.port))
        time.sleep(max(0.0, n / a.hz - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
