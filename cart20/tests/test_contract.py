"""Synthetic contract tests (spec 27-31 + design decisions D1-D3).  Run: python -m pytest tests -q   (or python tests/test_contract.py)"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ego_cart20.config import AUX, ST_GRIP, UMI_DT
from ego_cart20.convert import convert_episode
from ego_cart20.geometry.rotation6d import (matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_matrix,
                                            rotation_6d_to_matrix, slerp)
from ego_cart20.geometry.transforms import make_T, pose9_to_T
from ego_cart20.io.raw_episode_loader import RawArm, RawEpisode
from ego_cart20.labels.build_cart20 import build_cart20_chunk
from ego_cart20.labels.build_state20 import build_state20_prevrel, build_state20_relcart20
from ego_cart20.labels.pack_action32 import check_action32, pack_cart20_to_action32
from ego_cart20.preprocessing.resample import PoseTrack

RNG = np.random.default_rng(0)


def rand_R(n):
    q = RNG.normal(size=(n, 4)); return quaternion_to_matrix(q / np.linalg.norm(q, axis=1, keepdims=True))


def track_x(fn, t_end=5.0, hz=30.0, g=None):
    """translation-only raw track x = fn(t)"""
    t = np.arange(0, t_end, 1 / hz); T = np.tile(np.eye(4), (len(t), 1, 1)); T[:, 0, 3] = fn(t)
    return PoseTrack(t, T, np.ones(len(t), bool), np.zeros(len(t)) if g is None else g(t))


def test_rotation_roundtrip():
    R = rand_R(500); R2 = rotation_6d_to_matrix(matrix_to_rotation_6d(R))
    assert np.abs(R2 - R).max() < 1e-12
    assert np.abs(np.swapaxes(R2, -1, -2) @ R2 - np.eye(3)).max() < 1e-12 and np.abs(np.linalg.det(R2) - 1).max() < 1e-12
    assert np.abs(quaternion_to_matrix(matrix_to_quaternion(R)) - R).max() < 1e-12
    assert np.allclose(matrix_to_rotation_6d(np.eye(3)), [1, 0, 0, 0, 1, 0])              # v4: first two ROWS
    M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]); assert np.allclose(matrix_to_rotation_6d(M), [0, -1, 0, 1, 0, 0])


def test_slerp_midpoint():
    R = rand_R(2); q = matrix_to_quaternion(R); qm = slerp(q[0], q[1], 0.5)
    Rm = quaternion_to_matrix(qm); d0 = Rm.T @ R[0]; d1 = Rm.T @ R[1]
    a = lambda M: np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)); assert abs(a(d0) - a(d1)) < 1e-9


def test_current_anchor_not_sequential():
    """spec 28 at UMI_DT spacing: x(t) = 0.01 * t / UMI_DT -> A_k.x = 0.01 k (sequential delta would give 0.01 always)"""
    L = track_x(lambda t: 0.01 * t / UMI_DT); R = track_x(lambda t: 0 * t)
    c, ok = build_cart20_chunk(L, R, 1.0); assert ok and c.shape == (16, 20)
    assert np.allclose(c[:, 0], 0.01 * np.arange(1, 17), atol=1e-9), c[:, 0]
    assert np.allclose(c[[0, 1, 3, 7, 15], 0], [0.01, 0.02, 0.04, 0.08, 0.16], atol=1e-9)


def test_target_dt_is_umi_dt_not_row_spacing():
    """D1: k-step = 50.05 ms on the raw track, not 1/15 s; x = t (m) -> A16.x = 16 * 0.05005"""
    L = track_x(lambda t: t); R = track_x(lambda t: 0 * t); c, ok = build_cart20_chunk(L, R, 1.0)
    assert ok and abs(c[15, 0] - 16 * UMI_DT) < 1e-9 and abs(c[15, 0] - 16 / 15) > 0.2


def test_rotated_anchor_frame():
    """the target is expressed in the CURRENT pose frame: inv(T_t) T_{t+k}"""
    Rt = rand_R(1)[0]; Tt = make_T(Rt, [0.3, -0.2, 0.1]); Tf = make_T(Rt, Tt[:3, 3] + Rt @ np.array([0.05, 0, 0]))
    from ego_cart20.labels.build_cart20 import cart20_from_poses
    c = cart20_from_poses(Tt, np.tile(Tf, (16, 1, 1)), np.zeros(16), np.eye(4), np.tile(np.eye(4), (16, 1, 1)), np.zeros(16))
    assert np.allclose(c[0, 0:3], [0.05, 0, 0], atol=1e-6) and np.allclose(c[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)


def test_state_prevrel_direction():
    """spec 29: t-dt x = 0.09, t x = 0.10 -> state x = -0.01 (inv(T_t) T_{t-dt})"""
    Tp = make_T(np.eye(3), [0.09, 0, 0]); Tc = make_T(np.eye(3), [0.10, 0, 0])
    s = build_state20_prevrel(Tp, Tc, 0.3, Tc, Tc, 0.4); assert abs(s[0] + 0.01) < 1e-7 and s[9] == np.float32(0.3) and s[19] == np.float32(0.4)


def test_state_relcart20_layout():
    """D3: [L pose9 | R pose9 | gL gR]; anchor row = identity"""
    A = make_T(rand_R(1)[0], [0.1, 0.2, 0.3]); s = build_state20_relcart20(A, A, 0.25, A, A, 0.75)
    assert np.allclose(s[:9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6) and np.allclose(s[9:18], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    assert s[ST_GRIP["left"]] == np.float32(0.25) and s[ST_GRIP["right"]] == np.float32(0.75)


def test_gripper_future_interpolated():
    """g_action[k] = g(t + k dt), continuous, not the current g copied"""
    L = track_x(lambda t: 0 * t, g=lambda t: np.clip(t / 4, 0, 1)); R = track_x(lambda t: 0 * t, g=lambda t: 1 - np.clip(t / 4, 0, 1))
    c, ok = build_cart20_chunk(L, R, 1.0)
    assert np.allclose(c[:, 9], (1.0 + np.arange(1, 17) * UMI_DT) / 4, atol=1e-6) and np.allclose(c[:, 19], 1 - c[:, 9], atol=1e-6)


def test_pack_and_aux_zero():
    c = RNG.normal(size=(16, 20)).astype(np.float32); a = pack_cart20_to_action32(c)
    assert a.shape == (16, 32) and np.array_equal(a[:, :20], c) and np.count_nonzero(a[:, AUX]) == 0; check_action32(a, c)
    bad = a.copy(); bad[3, 25] = 1e-6
    try: check_action32(bad, c); raise AssertionError("non-zero AUX12 accepted")
    except ValueError: pass
    bad[3, 25] = np.nan
    try: check_action32(bad, c); raise AssertionError("NaN AUX12 accepted")
    except ValueError: pass


def test_no_tail_padding_and_gap():
    """rows exist only where all 16 targets are real; a raw gap kills the rows whose window spans it"""
    t = np.arange(0, 4, 1 / 30); n = len(t); T = np.tile(np.eye(4), (n, 1, 1)); T[:, 0, 3] = t
    v = np.ones(n, bool); v[60:66] = False                                             # 0.2 s hole at 2.0-2.17 s
    arm = lambda: RawArm(t.copy(), T.copy(), v.copy(), np.full(n, 0.5))
    cams = {"head": (t.copy(), np.arange(n))}
    raw = RawEpisode("synthetic", dict(task="t", instruction="i", stack_order="RBP"), {"left": arm(), "right": arm()}, cams)
    ep = convert_episode(raw)
    tr = ep["timestamps"][ep["train_rows"]]
    assert np.all(tr + 16 * UMI_DT <= t[-1] + 1e-9)                                    # no tail padding
    assert not np.any((tr < 2.2) & (tr + 16 * UMI_DT > 1.96))                         # no window across the hole
    assert ep["action"].shape == (len(tr), 16, 32) and np.count_nonzero(ep["action"][..., 20:]) == 0
    assert np.allclose(ep["state"][0, :18], [0, 0, 0, 1, 0, 0, 0, 1, 0] * 2, atol=1e-6)   # task-start anchor = identity
    assert ep["state_prev_valid"][0] == 0 and ep["state_prev_valid"][5] == 1
    r = ep["train_rows"][10]; t_r = ep["timestamps"][r]
    assert np.allclose(ep["cart20"][10, :, 0], np.arange(1, 17) * UMI_DT, atol=1e-6)    # x = t
    assert abs(ep["state"][r, 0] - (t_r - ep["timestamps"][0])) < 1e-5


def test_discontinuity_filter():
    """v2b: one-way jump -> remainder invalid; jump-and-return spike -> only the spike samples invalid; smooth fast motion kept"""
    from ego_cart20.preprocessing.pose_filter import discontinuity_filter
    t = np.arange(0, 4, 1 / 30); n = len(t)
    def track(x):
        T = np.tile(np.eye(4), (n, 1, 1)); T[:, 0, 3] = x; return T
    x = 0.3 * t                                                                    # 0.3 m/s real motion
    v, ev = discontinuity_filter(t, track(x), np.ones(n, bool)); assert v.all() and not ev
    xs = x.copy(); xs[60:63] += 0.2                                                # spike: +200 mm for 3 samples, then back
    v, ev = discontinuity_filter(t, track(xs), np.ones(n, bool))
    assert [e["kind"] for e in ev] == ["spike"] and not v[60:63].any() and v[:60].all() and v[63:].all()
    xo = x.copy(); xo[80:] += 0.25                                                 # one-way jump of 250 mm
    v, ev = discontinuity_filter(t, track(xo), np.ones(n, bool))
    assert ev[-1]["kind"] == "one_way" and v[:80].all() and not v[80:].any()
    fast = 1.4 * t                                                                 # 1.4 m/s = 93 mm / row, below the 100 mm limit
    v, ev = discontinuity_filter(t, track(fast), np.ones(n, bool)); assert v.all()


if __name__ == "__main__":
    fs = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fs: f(); print("PASS", f.__name__)
    print(f"ALL {len(fs)} PASS")
