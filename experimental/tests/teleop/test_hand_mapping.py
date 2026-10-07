import numpy as np
import pytest
from ego_collector.hands3d.aero import ACTUATION_LOWER_DEG, ACTUATION_UPPER_DEG, COMPACT_UPPER_DEG
from ego_teleop.tracking.interfaces import AERO_CHANNELS, HandHealth, HandFeatures
from ego_teleop.retarget.hand_features import features_from_landmarks, HandTrackingSupervisor, HandSupervisorConfig, normalize_landmarks
from ego_teleop.retarget.aero_retarget import AeroRetargeter, ChannelMap


def synth_hand(flex_deg=0.0, thumb_abd_deg=30.0, thumb_flex_deg=5.0, scale=0.09):
    """Synthetic 21-landmark hand in metres: fingers along +y, curled by `flex_deg` at MCP/PIP/DIP; thumb abducted from index."""
    J = np.zeros((21, 3)); seg = scale / 3
    bases = {"index": (-scale / 6, 5), "middle": (0.0, 9), "ring": (scale / 6, 13), "pinky": (scale / 3, 17)}
    for x, i0 in bases.values():
        p = np.array([x, scale, 0.0]); J[i0] = p; d = np.array([0.0, 1.0, 0.0]); ang = 0.0
        for k in range(1, 4):
            ang += np.radians(flex_deg); d = np.array([0.0, np.cos(ang), -np.sin(ang)]); p = p + seg * d; J[i0 + k] = p
    a = np.radians(thumb_abd_deg); cmc = np.array([-scale / 3, scale * 0.22, 0.0]); J[1] = cmc
    d = np.array([-np.sin(a), np.cos(a), 0.0]); p = cmc; ang = 0.0
    for k in (2, 3, 4):
        ang += np.radians(thumb_flex_deg); rot = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
        p = p + seg * (rot @ d); J[k] = p
    return J + np.array([0.3, -0.1, 0.5])       # arbitrary offset: absolute XYZ must not matter


def test_features_are_wrist_relative_scale_invariant_and_monotonic():
    f_open = features_from_landmarks(0, synth_hand(0.0), 0.9, side="right")
    f_half = features_from_landmarks(1, synth_hand(35.0), 0.9)
    f_closed = features_from_landmarks(2, synth_hand(85.0), 0.9)
    assert f_open.health == HandHealth.OK and f_open.u7.shape == (7,)
    assert np.allclose(f_open.landmarks_wrist_rel[0], 0)                       # wrist at origin
    big = features_from_landmarks(3, synth_hand(35.0, scale=0.18), 0.9)
    assert np.allclose(big.u7, f_half.u7, atol=1e-6)                          # scale invariant
    for i in (3, 4, 5, 6):                                                    # fingers flex monotonically
        assert f_open.u7[i] < f_half.u7[i] < f_closed.u7[i]
    assert f_closed.u7[3] > 0.9 and f_open.u7[3] < 0.1


def test_degenerate_landmarks_rejected():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((21, 3)))


def test_hand_supervisor_ok_hold_lost():
    sup = HandTrackingSupervisor(HandSupervisorConfig(hold_ms=300, fresh_ms=60))
    f = features_from_landmarks(1_000_000_000, synth_hand(10), 0.9)
    assert sup.update(f, f.timestamp_ns).health == HandHealth.OK
    h = sup.update(None, f.timestamp_ns + 200_000_000); assert h.health == HandHealth.HOLD and np.allclose(h.u7, f.u7)
    l = sup.update(None, f.timestamp_ns + 400_000_000); assert l.health == HandHealth.LOST and l.u7 is None
    assert sup.current(f.timestamp_ns + 30_000_000).health == HandHealth.OK
    assert sup.current(f.timestamp_ns + 100_000_000).health == HandHealth.HOLD
    low = HandFeatures(2_000_000_000, f.landmarks_wrist_rel, f.u7, 0.2, HandHealth.OK)
    assert sup.update(low, low.timestamp_ns).health != HandHealth.OK        # low confidence not accepted as fresh


def make_calibrated(**kw):
    rt = AeroRetargeter([ChannelMap(n, robot_max_deg=COMPACT_UPPER_DEG[i], deadband=0.0, lpf_alpha=1.0, max_rate_deg_s=1e9, **kw) for i, n in enumerate(AERO_CHANNELS)])
    rt.calibrate_open(np.array([0.2, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1]))
    rt.calibrate_closed(np.array([0.3, 0.5, 0.9, 0.95, 0.95, 0.95, 0.95]))
    rt.calibrate_pinch(np.array([0.8, 0.6, 0.5, 0.6, 0.2, 0.2, 0.2]))
    rt.finalize_calibration(); return rt


def test_calibration_maps_open_to_min_closed_to_max_and_pinch_defines_thumb_abduction():
    rt = make_calibrated()
    o = rt.retarget(0, [0.2, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1]); assert np.allclose(o.compact_deg, 0.0)
    c = rt.retarget(1, [0.8, 0.6, 0.9, 0.95, 0.95, 0.95, 0.95]); assert np.allclose(c.compact_deg, COMPACT_UPPER_DEG)
    assert rt.ch[0].human_max == 0.8 and rt.ch[1].human_max == 0.6 and rt.ch[2].human_max == 0.9   # thumb abd/flex from pinch, curl from fist
    assert np.all(c.actuations_deg >= np.array(ACTUATION_LOWER_DEG) - 1e-9) and np.all(c.actuations_deg <= np.array(ACTUATION_UPPER_DEG) + 1e-9)


def test_dead_channel_refused():
    rt = AeroRetargeter(); rt.calibrate_open(np.full(7, 0.2)); rt.calibrate_closed(np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.25]))
    with pytest.raises(RuntimeError, match="pinky"):
        rt.finalize_calibration()


def test_filters_deadband_lpf_rate_limit_and_sign():
    rt = AeroRetargeter([ChannelMap(n, robot_max_deg=90.0, deadband=0.05, lpf_alpha=0.5, max_rate_deg_s=100.0, sign=(-1 if i == 6 else 1)) for i, n in enumerate(AERO_CHANNELS)])
    a = rt.retarget(0, np.zeros(7)); assert np.allclose(a.compact_deg[:6], 0) and np.isclose(a.compact_deg[6], 90)   # sign flip on pinky
    b = rt.retarget(33_000_000, np.full(7, 0.03)); assert np.allclose(b.normalized[:6], 0)                       # inside deadband
    c = rt.retarget(66_000_000, np.full(7, 1.0)); assert np.allclose(c.normalized[:6], 0.5)                      # LPF half-way
    assert np.allclose(c.compact_deg[:6], 100.0 * 0.033, atol=1e-6)                                              # rate limited: 100 deg/s * 33 ms
    d = rt.retarget(2_066_000_000, np.full(7, 1.0)); assert np.all(d.compact_deg[[0, 2, 3, 4, 5]] > 60) and d.compact_deg[1] == 55   # converges; ch1 hard-clamped at Aero limit


def test_hold_and_relaxed_and_roundtrip(tmp_path):
    rt = make_calibrated()
    rt.retarget(0, [0.25, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5]); h = rt.hold(1); assert h.source == "hold" and h.u7 is None
    r = rt.relaxed(2); assert r.source == "relaxed" and np.allclose(r.normalized, rt.relaxed_n)
    p = tmp_path / "aero_retarget.json"; rt.save(p); rt2 = AeroRetargeter.load(p)
    assert [c.human_max for c in rt2.ch] == [c.human_max for c in rt.ch] and rt2.to_dict()["order"] == list(AERO_CHANNELS)
