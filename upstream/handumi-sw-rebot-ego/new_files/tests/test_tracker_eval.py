from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from handumi.eval.tracker_eval import (
    CSV_HEADER,
    CaptureBuilder,
    Thresholds,
    TrackerCapture,
    left_right_agreement,
    return_to_point,
    static_jitter,
    table_z,
)


def _pose(p, rotvec_deg=(0.0, 0.0, 0.0)):
    q = Rotation.from_rotvec(np.radians(rotvec_deg)).as_quat()
    return np.concatenate([np.asarray(p, dtype=float), q])


def _build(samples):
    """samples: list of (t_s, mark, left_pose7, right_pose7, tracked_l, tracked_r)."""
    b = CaptureBuilder()
    for t_s, mark, lp, rp, tl, tr in samples:
        b.add(
            t_ns=int(t_s * 1e9),
            mark=mark,
            streaming=True,
            clock_synced=True,
            left_tracked=tl,
            right_tracked=tr,
            left_controller=lp,
            right_controller=rp,
            left_tcp=lp,
            right_tcp=rp,
        )
    return b.build()


def test_csv_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    samples = []
    for i in range(50):
        samples.append((i / 60, 1 if 10 <= i < 20 else 0, _pose(rng.normal(size=3)), _pose(rng.normal(size=3)), True, i % 7 != 0))
    cap = _build(samples)
    path = tmp_path / "cap.csv"
    cap.write_csv(path)
    back = TrackerCapture.read_csv(path)
    assert len(back) == 50
    assert back.mark_ids() == [1]
    np.testing.assert_allclose(back.tcp["left"], cap.tcp["left"], atol=1e-9)
    assert (back.tracked["right"] == cap.tracked["right"]).all()
    assert len(CSV_HEADER) == 6 + 4 * 7


def test_static_jitter_pass_and_fail():
    rng = np.random.default_rng(1)
    good, bad = [], []
    for i in range(600):
        t = i / 60
        lp = _pose([0.1, 0.2, 0.0] + rng.normal(scale=0.001, size=3), rng.normal(scale=0.2, size=3))
        rp = _pose([0.3, 0.2, 0.0] + rng.normal(scale=0.010, size=3), rng.normal(scale=3.0, size=3))
        good.append((t, 0, lp, lp, True, True))
        bad.append((t, 0, rp, rp, True, True))
    rep = static_jitter(_build(good))
    assert rep.status == "PASS"
    assert rep.sides["left"].translation_rms_mm < 3.0
    assert rep.sides["left"].rotation_rms_deg < 0.6
    assert abs(rep.sample_rate_hz - 60.0) < 0.5
    rep_bad = static_jitter(_build(bad))
    assert rep_bad.status == "FAIL"
    assert rep_bad.sides["right"].translation_rms_mm > 10.0


def test_static_jitter_quaternion_sign_flip_is_harmless():
    # Same orientation, alternating q / -q must not create fake rotation noise.
    base = _pose([0, 0, 0], (10.0, 20.0, 30.0))
    samples = []
    for i in range(120):
        p = base.copy()
        if i % 2:
            p[3:] *= -1
        samples.append((i / 60, 0, p, p, True, True))
    rep = static_jitter(_build(samples))
    assert rep.sides["left"].rotation_rms_deg < 1e-6


def test_return_to_point_uses_marks_and_median():
    rng = np.random.default_rng(2)
    A = np.array([0.10, 0.20, 0.0])
    samples = []
    t = 0.0
    mark = 0
    for visit, offset in enumerate([[0, 0, 0], [0.004, 0, 0], [0, 0.006, 0], [0.002, 0.002, 0.012]]):
        mark += 1
        for _ in range(30):  # 0.5 s mark window
            p = _pose(A + offset + rng.normal(scale=0.0005, size=3))
            samples.append((t, mark, p, p, True, True))
            t += 1 / 60
        for _ in range(60):  # excursion away from A (unmarked)
            p = _pose(A + [0.3, 0.1, 0.1] + rng.normal(scale=0.01, size=3))
            samples.append((t, 0, p, p, True, True))
            t += 1 / 60
    rep = return_to_point(_build(samples), side="left")
    assert rep.reference_mark == 1
    assert rep.marks == [2, 3, 4]
    assert abs(rep.errors_mm[0] - 4.0) < 0.6
    assert abs(rep.errors_mm[1] - 6.0) < 0.6
    assert abs(rep.errors_mm[2] - np.linalg.norm([2, 2, 12])) < 0.8
    assert rep.status == "WARN"  # 12.3 mm > 10 mm gate but < 20 mm
    assert return_to_point(_build(samples), side="left", thresholds=Thresholds(return_to_point_max_mm=15)).status == "PASS"


def test_agreement_simultaneous_and_sequential_and_untracked():
    X = np.array([0.0, 0.3, 0.0])
    samples = []
    t = 0.0
    # mark 1: both tips on X, 3 mm apart
    for _ in range(20):
        samples.append((t, 1, _pose(X), _pose(X + [0.003, 0, 0]), True, True)); t += 1 / 60
    # mark 2: right tip untracked
    for _ in range(20):
        samples.append((t, 2, _pose(X), _pose(X + [0.5, 0, 0]), True, False)); t += 1 / 60
    cap = _build(samples)
    rep = left_right_agreement(cap, mode="simultaneous")
    assert rep.separations_mm == [3.0] or abs(rep.separations_mm[0] - 3.0) < 1e-6
    assert rep.pairs[1]["skipped"] is True
    assert rep.status == "PASS"

    # sequential: mark 1 = left on X, mark 2 = right on X (+8 mm)
    seq = []
    t = 0.0
    for _ in range(20):
        seq.append((t, 1, _pose(X), _pose([1, 1, 1]), True, True)); t += 1 / 60
    for _ in range(20):
        seq.append((t, 2, _pose([1, 1, 1]), _pose(X + [0, 0.008, 0]), True, True)); t += 1 / 60
    rep2 = left_right_agreement(_build(seq), mode="sequential")
    assert abs(rep2.max_mm - 8.0) < 1e-6
    assert rep2.status == "PASS"


def test_table_z_grades_marked_tip_height():
    samples = []
    t = 0.0
    for z_mm, mark in ((2.0, 1), (-4.0, 2), (30.0, 3)):
        for _ in range(10):
            p = _pose([0.1, 0.1, z_mm / 1000])
            samples.append((t, mark, p, p, True, True)); t += 1 / 60
    rep = table_z(_build(samples))
    assert rep.status == "FAIL"  # 30 mm >= 2 * 15 mm
    assert [round(v) for v in rep.z_mm["left"]] == [2, -4, 30]


def test_thresholds_from_rig_ignores_unknown_keys():
    th = Thresholds.from_rig({"tracker_eval": {"static_translation_rms_mm": 3, "bogus": 1}})
    assert th.static_translation_rms_mm == 3.0
    assert th.return_to_point_max_mm == 10.0
