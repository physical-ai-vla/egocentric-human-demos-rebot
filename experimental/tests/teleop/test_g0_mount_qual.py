"""G0-0 mount qualification: windows, aperture scoring, hysteresis, mask, extrinsic-free rotation check, verdict."""
import numpy as np
import pandas as pd
import pytest

from ego_teleop.tools import g0_mount_qual as G

S = 1_000_000_000


def _grip_take(open_v=1.0, half_v=0.6, pinch_v=0.2, noise=0.02, fps=30, seed=0):
    rng = np.random.default_rng(seed)
    wins, t, rows = [], 0, []
    for k in range(3):
        for st, v in (("open", open_v), ("half", half_v), ("pinch", pinch_v)):
            wins.append(G.Window(st, t, t + 4 * S))
            for i in range(4 * fps):
                ti = t + int(i * S / fps)
                # the first 0.5 s is the posture change: values ramp from the previous state
                rows.append(dict(t_ns=ti, A_px=v * 100 + rng.normal(0, noise * 100), B_ratio=v * 2 + rng.normal(0, noise * 2),
                                 C_world=(v + rng.normal(0, 4 * noise)) * 0.05, detected=True))
            t += 4 * S
    df = pd.DataFrame(rows)
    st, held, win = G.label_frames(df.t_ns.to_numpy(), wins)
    return df.assign(state=st, held=held, win=win), wins


def test_windows_from_events_run_to_next_start_and_stop():
    ev = [dict(t_ns=10, kind="g0_segment", detail=dict(name="open")), dict(t_ns=5, kind="other", detail={}),
          dict(t_ns=30, kind="g0_segment", detail=dict(name="pinch"))]
    w = G.windows_from_events(ev, 50)
    assert [(x.name, x.t0_ns, x.t1_ns) for x in w] == [("open", 10, 30), ("pinch", 30, 50)]


def test_label_frames_skips_the_settle_time():
    w = [G.Window("open", 0, 4 * S)]
    t = np.array([0, int(0.5 * S), int(0.9 * S), 3 * S])
    name, held, _ = G.label_frames(t, w)
    assert list(name) == ["open"] * 4 and list(held) == [False, False, True, True]


def test_miss_runs_measured_to_next_good_frame():
    t = np.arange(10) * (S // 30)
    ok = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1], bool)
    (r,) = G.miss_runs_ms(ok, t)
    assert r == pytest.approx(4 * 1000 / 30, rel=0.02)       # 3 missing frames = 4 intervals of hold


def test_hysteresis_holds_inside_the_band():
    x = np.array([1.0, 0.5, 0.2, 0.5, 0.45, 0.9, np.nan, 0.5])
    assert list(G.hysteresis(x, 0.35, 0.55)) == ["open", "open", "close", "close", "close", "open", "open", "open"]


def test_clean_signal_separates_and_noisy_one_is_ranked_below():
    df, _ = _grip_take()
    m = {s: G.signal_metrics(df, s) for s in G.SIGNALS}
    assert m["A_px"]["ok"] and m["A_px"]["monotonic_half"] and m["A_px"]["margin"] > 0.6 and m["A_px"]["chatter"] == 0
    assert m["A_px"]["close_th"] < m["A_px"]["open_th"]
    assert m["C_world"]["margin"] < m["A_px"]["margin"]          # C was given 4x the noise
    assert G.pick_signal(m) in ("A_px", "B_ratio")


def test_inverted_signal_is_not_ok():
    df, _ = _grip_take(open_v=0.2, pinch_v=1.0)
    m = G.signal_metrics(df, "A_px")
    assert not m["ok"] and "not above" in m["why"]


def test_hand_gate_reports_each_failure():
    df, _ = _grip_take()
    best = G.signal_metrics(df, "A_px")
    view = dict(detect_grip=0.80, detect_pinch=0.5, miss_run_grip_ms_max=900.0, tips_in_view=0.99)
    ok, fails = G.hand_gate(view, best)
    assert not ok and len(fails) == 3
    ok, fails = G.hand_gate(dict(view, detect_grip=0.99, detect_pinch=0.99, miss_run_grip_ms_max=50.0), best)
    assert ok, fails


def test_static_region_mask_finds_the_camera_fixed_blob():
    rng = np.random.default_rng(1)
    d = rng.integers(10, 60, size=(60, 90, 160)).astype(np.uint8)      # environment changes while the camera moves
    d[:, 40:, :50] = rng.integers(0, 2, size=(60, 50, 50))             # hand corner: camera-fixed, nearly unchanged
    m = G.static_region_mask(d)
    assert m[60:, 10:40].all() and not m[:30, 80:].any()


def test_build_mask_unions_and_dilates():
    st = np.zeros((10, 20), bool); st[0:2, 0:2] = True
    occ = np.zeros((10, 20), np.float32); occ[8:, 18:] = 0.5
    m = G.build_mask(st, occ, (200, 100), dilate_frac=0.0)
    assert m[5, 5] and m[95, 195] and not m[50, 100]


def _texture(h=360, w=640, seed=3):
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2.0)
    return cv2.equalizeHist(img)


def test_pair_rotation_recovers_a_pure_rotation_angle():
    import cv2
    from scipy.spatial.transform import Rotation
    K = np.array([[400.0, 0, 320], [0, 400, 180], [0, 0, 1]])
    a = _texture(); R = Rotation.from_euler("xyz", [1.0, 2.0, 0.5], degrees=True).as_matrix()
    Hm = K @ R @ np.linalg.inv(K)
    b = cv2.warpPerspective(a, Hm, (a.shape[1], a.shape[0]))
    region = np.ones_like(a, bool); region[:, :20] = region[:, -20:] = False
    r = G.pair_rotation(a, b, K, region)
    assert r["inliers"] > 50 and r["grid_cells"] >= 12
    # pure rotation makes E degenerate in translation; the ANGLE is still what the gyro would report
    assert r["angle_deg"] == pytest.approx(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())), abs=0.6)


def test_gyro_angles_and_agreement():
    t_f = np.arange(0, 2 * S, S // 30)
    t_i = np.arange(0, 2 * S, S // 400)
    w = np.zeros((len(t_i), 3)); w[:, 2] = np.radians(60) * np.sin(2 * np.pi * t_i / S)
    g = G.gyro_angles(t_f, t_i, w, np.zeros(3))
    ag = G.gyro_agreement(g * 1.0, g)
    assert ag["r"] == pytest.approx(1.0) and ag["slope"] == pytest.approx(1.0)
    assert np.nanmax(g) == pytest.approx(2.0, rel=0.1)                 # 60 deg/s peak over 1/30 s


def test_vi_gate_and_verdict_matrix():
    vi = dict(env_ratio=0.7, masked_motion=dict(inliers_p50=150, inliers_p10=60, grid_cells_p50=10), gyro_masked=dict(r=0.9, slope=1.0))
    assert G.vi_gate(vi) == (True, [])
    bad = dict(vi, env_ratio=0.3, gyro_masked=dict(r=0.5, slope=0.6))
    ok, f = G.vi_gate(bad); assert not ok and len(f) == 3
    assert G.verdict(True, True).startswith("SHARE") and G.verdict(True, False).startswith("SPLIT")
    assert G.verdict(False, True).startswith("GAP") and G.verdict(False, False).startswith("RE-AIM")
    assert G.verdict(None, True).startswith("INCOMPLETE")


def test_protocol_has_three_grip_cycles_and_all_motion_windows():
    names = [p[0] for p in G.PROTOCOL]
    assert names.count("open") == names.count("half") == names.count("pinch") == 3
    assert set(G.MOTION_WINDOWS) <= set(names)
    assert 80 <= sum(p[1] for p in G.PROTOCOL) <= 120


def test_static_region_mask_ignores_still_stretches():
    rng = np.random.default_rng(2)
    move = rng.integers(10, 60, size=(40, 90, 160)).astype(np.uint8); move[:, 40:, :50] = 0
    still = rng.integers(0, 2, size=(60, 90, 160)).astype(np.uint8)     # camera not moving: nothing changes anywhere
    m = G.static_region_mask(np.concatenate([still, move]))
    assert m[60:, 10:40].all() and m.mean() < 0.3
