"""CART20 horizon labels.  Current-anchor relative, never sequential:

    A_k = inv(T(t)) @ T(t + k * UMI_DT),   k = 1..16,   both arms on the SAME physical instants
    CART20_k = [L xyz3 rot6d6 g(t+k dt) | R xyz3 rot6d6 g(t+k dt)]

T(t + k dt) and g(t + k dt) are interpolated on the RAW tracks (PoseTrack.sample).  A chunk is valid only if every one
of the 16 targets of both arms is valid -- no tail padding, no terminal-pose repetition.
"""
import numpy as np
from ..config import ACT_GRIP, ACT_POS, ACT_ROT, CART_DIM, HORIZON, UMI_DT
from ..geometry.transforms import pose9, relative


def cart20_from_poses(T_left_t, T_left_future, g_left_future, T_right_t, T_right_future, g_right_future):
    """pure array form: T_*_t (4,4), T_*_future (H,4,4), g_*_future (H,) -> float32 [H, 20]"""
    H = len(T_left_future); out = np.zeros((H, CART_DIM))
    for side, T0, Tf, g in (("left", T_left_t, T_left_future, g_left_future), ("right", T_right_t, T_right_future, g_right_future)):
        p9 = pose9(relative(np.asarray(T0)[None], Tf))
        out[:, ACT_POS[side]] = p9[:, :3]; out[:, ACT_ROT[side]] = p9[:, 3:]; out[:, ACT_GRIP[side]] = g
    return out.astype(np.float32)


def target_times(t, horizon=HORIZON, dt=UMI_DT):
    return t + np.arange(1, horizon + 1) * dt


def build_cart20_chunk(left_track, right_track, t, horizon=HORIZON, dt=UMI_DT, T_left_t=None, T_right_t=None):
    """-> (cart20 float32 [horizon, 20], ok bool).  T_*_t: the current pose if already sampled (else sampled at t)."""
    tk = target_times(t, horizon, dt); fut = {}; ok = True
    for side, tr, T0 in (("left", left_track, T_left_t), ("right", right_track, T_right_t)):
        if T0 is None:
            T0s, _, ok0 = tr.sample([t]); T0 = T0s[0]; ok &= bool(ok0[0])
        Tf, gf, okf = tr.sample(tk); ok &= bool(okf.all()); fut[side] = (T0, Tf, gf)
    c = cart20_from_poses(*fut["left"], *fut["right"])
    return c, ok
