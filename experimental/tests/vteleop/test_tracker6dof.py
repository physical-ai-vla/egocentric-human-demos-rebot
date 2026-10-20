"""Dedicated 6DoF tracker source (2026-10-15): protocol checks, TRACKING -> DEGRADED -> LOST transitions under the mock's
fault set (100 ms stale, confidence drop, packet loss, 20 cm jump, invalid quaternion), and the three calibrations."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from visual_teleop.vision.tracker6dof import (PRESETS, Tracker6DoFProvider, make_T, pivot_calibration, table_frame_from_points,
                                              R_TB_TCP_from_gravity)

MS = 1_000_000


def msg(seq, p=(0.1, 0.2, 0.3), q=(0, 0, 0, 1), conf=1.0, valid=True, **kw):
    return dict(kind="tracker_pose", side="right", seq=seq, p=list(p), q_xyzw=list(q), conf=conf, valid=valid, **kw)


@pytest.fixture
def tr():
    return Tracker6DoFProvider(0, stale_warn_ms=50, hold_ms=300, min_conf=0.5, warn_conf=0.8, v_max_m_s=2.0, margin_m=0.03)


def q_at(tr, t): return tr.estimate("right", t)[1]


def test_no_data_is_init(tr): assert q_at(tr, 0) == "INIT"


def test_fresh_valid_is_tracking_and_pose_composes():
    T_W_TW = make_T(PRESETS["steamvr_y_up"], [0.0, 0.0, 0.5]); T_tcp = make_T(np.eye(3), [0, 0, 0.1])
    tr = Tracker6DoFProvider(0, T_W_TW, T_tcp)
    tr.ingest(msg(1, p=(0.2, 0.3, -0.1)), t_rx_ns=0)
    e, q = tr.estimate("right", 5 * MS)
    assert q == "TRACKING"
    np.testing.assert_allclose(e.T_W_H, T_W_TW @ make_T(np.eye(3), [0.2, 0.3, -0.1]) @ T_tcp)
    np.testing.assert_allclose(e.T_W_H[:3, 3], [0.2, 0.0, 0.8], atol=1e-12)   # tip (0.2, 0.3, 0.0) in y-up world -> W (0.2, 0, 0.3) + 0.5 z


def test_stale_100ms_degraded_then_lost(tr):
    tr.ingest(msg(1), t_rx_ns=0)
    assert q_at(tr, 20 * MS) == "TRACKING"
    assert q_at(tr, 100 * MS) == "DEGRADED"                                # 100 ms stale: warned, still commandable
    assert q_at(tr, 301 * MS) == "LOST"
    tr.ingest(msg(2), t_rx_ns=310 * MS); assert q_at(tr, 311 * MS) == "TRACKING"   # recovery (controller adds the re-entry debounce)


def test_confidence_drop(tr):
    tr.ingest(msg(1, conf=0.7), t_rx_ns=0); assert q_at(tr, MS) == "DEGRADED"
    tr.ingest(msg(2, conf=0.3), t_rx_ns=8 * MS); assert q_at(tr, 9 * MS) == "LOST"
    tr.ingest(msg(3, conf=0.95), t_rx_ns=16 * MS); assert q_at(tr, 17 * MS) == "TRACKING"


def test_packet_loss_counted_reorder_dropped(tr):
    for k, seq in enumerate([1, 2, 5, 6, 4, 7]): tr.ingest(msg(seq), t_rx_ns=k * 8 * MS)
    assert tr.stats["lost_seq"] == 2 and tr.stats["reordered"] == 1 and tr.latest_raw("right").seq == 7
    assert q_at(tr, 41 * MS) == "TRACKING"


def test_sudden_20cm_jump_rejected_held_then_lost(tr):
    tr.ingest(msg(1, p=(0.1, 0.2, 0.3)), t_rx_ns=0)
    tr.ingest(msg(2, p=(0.3, 0.2, 0.3)), t_rx_ns=8 * MS)                    # 20 cm in 8 ms
    e, q = tr.estimate("right", 9 * MS)
    assert q == "DEGRADED" and tr.stats["jump_rejected"] == 1 and "tracker_jump" in e.reasons[0]
    np.testing.assert_allclose(e.T_W_H[:3, 3], [0.1, 0.2, 0.3])           # last ACCEPTED pose held, the jump never passes
    assert q_at(tr, 301 * MS) == "LOST"
    tr.ingest(msg(3, p=(0.105, 0.2, 0.3)), t_rx_ns=16 * MS); assert q_at(tr, 17 * MS) == "TRACKING"   # back near the true pose


def test_jump_gate_allows_fast_real_motion_after_gap(tr):
    tr.ingest(msg(1, p=(0.1, 0.2, 0.3)), t_rx_ns=0)
    tr.ingest(msg(2, p=(0.3, 0.2, 0.3)), t_rx_ns=200 * MS)                  # 20 cm in 200 ms = 1 m/s: real
    assert q_at(tr, 201 * MS) == "TRACKING"


@pytest.mark.parametrize("q", [(0, 0, 0, 3.0), (0, 0, 0, 0), (np.nan, 0, 0, 1)])
def test_invalid_quaternion(tr, q):
    tr.ingest(msg(1), t_rx_ns=0); tr.ingest(msg(2, q=q), t_rx_ns=8 * MS)
    e, qual = tr.estimate("right", 9 * MS)
    assert qual == "DEGRADED" and e.reasons[0].startswith("tracker_bad_quat") and tr.stats["invalid"] == 1
    assert q_at(tr, 400 * MS) == "LOST"


def test_slightly_unnormalised_quaternion_is_normalised(tr):
    tr.ingest(msg(1, q=(0, 0, 0, 1.04)), t_rx_ns=0)
    e, q = tr.estimate("right", MS); assert q == "TRACKING"; np.testing.assert_allclose(e.T_W_H[:3, :3], np.eye(3), atol=1e-12)


def test_invalid_flag_without_history_is_lost(tr):
    tr.ingest(msg(1, valid=False), t_rx_ns=0); assert q_at(tr, MS) == "LOST"


def test_timestamps_logged(tr):
    tr.log = []
    tr.ingest(msg(1, t_ns=111, t_send_ns=222), t_rx_ns=333)
    tp = tr.latest_raw("right"); assert (tp.t_src_ns, tp.t_send_ns, tp.t_ns) == (111, 222, 333)
    assert tr.log[0][:4] == (333, 1, 111, 222)


def test_from_cfg_explicit_extrinsic_wins_over_preset():
    T = make_T(Rotation.from_euler("z", 30, degrees=True).as_matrix(), [1, 2, 3])
    tc = dict(port=1, world_preset="steamvr_y_up", T_W_TW=dict(measured=True, matrix=T.tolist()), T_TB_TCP=dict(matrix=np.eye(4).tolist()),
              stale_warn_ms=40, hold_ms=250, min_conf=0.4, warn_conf=0.9)
    tr = Tracker6DoFProvider.from_cfg(tc); np.testing.assert_allclose(tr.T_W_TW, T); assert (tr.stale_warn_ms, tr.hold_ms) == (40, 250)
    tc["T_W_TW"]["measured"] = False; np.testing.assert_allclose(Tracker6DoFProvider.from_cfg(tc).T_W_TW[:3, :3], PRESETS["steamvr_y_up"])


def test_pivot_recovers_offset_with_rms():
    rng = np.random.default_rng(1); r = np.array([0.03, -0.02, 0.11]); pp = np.array([0.4, 0.1, 0.0]); Ts = []
    for _ in range(200):
        R = Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix(); Ts.append(make_T(R, pp - R @ r + rng.normal(0, 0.0005, 3)))
    t, p, rms = pivot_calibration(Ts)
    np.testing.assert_allclose(t, r, atol=1e-3); np.testing.assert_allclose(p, pp, atol=1e-3); assert 0.0003 < rms < 0.0015


def test_table_frame_from_points():
    T_TW_W = make_T(PRESETS["steamvr_y_up"].T @ Rotation.from_euler("z", 25, degrees=True).as_matrix(), [0.5, -0.2, 1.0])   # truth
    pts_W = [np.array([0, 0, 0.0]), np.array([0.2, 0.01, 0]), np.array([0.02, 0.2, 0]), np.array([0.15, 0.15, 0])]
    P = [(T_TW_W @ np.r_[x, 1])[:3] for x in pts_W]
    T_W_TW, rms = table_frame_from_points(P[0], P[1], P[2], P[3:])
    assert rms < 1e-9
    for x, pt in zip(pts_W, P): assert np.allclose((T_W_TW @ np.r_[pt, 1])[2], 0, atol=1e-9)              # table at z = 0
    np.testing.assert_allclose(T_W_TW[:3, :3] @ T_TW_W[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-9)        # up preserved
    x_W = T_W_TW[:3, :3] @ (P[1] - P[0]); assert x_W[0] > 0.19 and abs(x_W[1]) < 1e-9                    # P0->P1 = +x


def test_mount_from_gravity_recovers_R_TB_TCP():
    R_true = Rotation.from_euler("xyz", [10, -20, 90], degrees=True).as_matrix()
    # A: TCP x = gravity (down), z = forward;  B: TCP z = down
    R_W_TCP_A = np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]]); R_W_TCP_B = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R, ang = R_TB_TCP_from_gravity(R_W_TCP_A @ R_true.T, R_W_TCP_B @ R_true.T)
    assert abs(ang - 90) < 1e-6; np.testing.assert_allclose(R, R_true, atol=1e-9)
