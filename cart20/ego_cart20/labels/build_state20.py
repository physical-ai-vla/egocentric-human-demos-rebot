"""state20 builders.

MAIN (state.npy, design decision D3) -- v4 RELCART20 task-anchor state, identical semantics and layout to the reBot FT data:
    per arm  T_rel = inv(T_task_start) @ T(t)
    state20 = [L xyz3 rot6d6 | R xyz3 rot6d6 | gL | gR]                         grippers at dims 18, 19
DIAGNOSTIC (state_prevrel.npy) -- spec v2 previous-relative state:
    per arm  T_rel = inv(T(t)) @ T(t - UMI_DT)          (T(t - UMI_DT) interpolated on the raw track)
    state20 = [L xyz3 rot6d6 gL | R xyz3 rot6d6 gR]                             grippers at dims 9, 19
"""
import numpy as np
from ..config import PR_GRIP, PR_POSE, ST_GRIP, ST_POSE
from ..geometry.transforms import pose9, relative


def build_state20_relcart20(T_left_anchor, T_left_curr, left_gripper_curr, T_right_anchor, T_right_curr, right_gripper_curr):
    """batched over leading axes of T_*_curr; returns float32 [..., 20]"""
    gl = np.asarray(left_gripper_curr, np.float64); out = np.zeros(gl.shape + (20,))
    out[..., ST_POSE["left"]] = pose9(relative(T_left_anchor, T_left_curr))
    out[..., ST_POSE["right"]] = pose9(relative(T_right_anchor, T_right_curr))
    out[..., ST_GRIP["left"]] = gl; out[..., ST_GRIP["right"]] = right_gripper_curr
    return out.astype(np.float32)


def build_state20_prevrel(T_left_prev, T_left_curr, left_gripper_curr, T_right_prev, T_right_curr, right_gripper_curr):
    """spec v2 API (diagnostic only).  Convention: T_rel = inv(T_curr) @ T_prev"""
    gl = np.asarray(left_gripper_curr, np.float64); out = np.zeros(gl.shape + (20,))
    out[..., PR_POSE["left"]] = pose9(relative(T_left_curr, T_left_prev))
    out[..., PR_POSE["right"]] = pose9(relative(T_right_curr, T_right_prev))
    out[..., PR_GRIP["left"]] = gl; out[..., PR_GRIP["right"]] = right_gripper_curr
    return out.astype(np.float32)


build_state20 = build_state20_relcart20     # the default state of this pipeline
