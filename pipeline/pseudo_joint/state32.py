"""[2026-09-23] Track C C6: the ONE function that lays out observation.state s32. Used by both the C0 compatibility
test (fed R150's own poses/joints, must reproduce r150_umi_v4_3cam_s32 exactly) and the HandUMI exporter.

Per arm, left (robot0) then right:  [prev_rel pos3 + rot6d (first two ROWS of R), width_m, q6 rad]  -> 16, x2 = 32
    prev_rel[t] = inv(T_t) @ T_{t-1}, T in the DATASET TCP frame; t = 0 uses itself (identity), prev_valid = 0
Rows are float32. The caller decides which rows are stored (Track B drops the last frame, which has no t+1).
"""
import pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path.home() / "holobrain-mac-model/umi_pkg"))
from umi.common.pose_util import mat_to_pose10d          # same function Track B's converter uses

PER_ARM = 16
FIELDS = ([f"L_prev_rel_pos{i}" for i in range(3)] + [f"L_prev_rel_rot6d{i}" for i in range(6)] + ["L_width_m"]
          + [f"L_q{i}_rad" for i in range(6)])
FIELDS = FIELDS + [f.replace("L_", "R_", 1) for f in FIELDS]


def prev_rel(T):
    prev = np.concatenate([T[:1], T[:-1]], axis=0)
    return np.linalg.inv(T) @ prev


def build_state32(T_ds, width_m, q_rad):
    """T_ds: [T_left (n,4,4), T_right (n,4,4)] dataset-frame TCP; width_m: [(n,), (n,)]; q_rad: [(n,6), (n,6)]."""
    n = len(T_ds[0])
    out = np.zeros((n, 2 * PER_ARM), np.float32)
    for r in range(2):
        o = r * PER_ARM
        out[:, o:o + 9] = mat_to_pose10d(prev_rel(T_ds[r]))
        out[:, o + 9] = width_m[r]
        out[:, o + 10:o + 16] = q_rad[r]
    prev_valid = np.ones((n, 1), np.float32); prev_valid[0] = 0.0
    return out, prev_valid
