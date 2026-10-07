"""Motion QA records the distribution and rejects only catastrophe.

The temptation this guards against is inventing pass/fail thresholds before any real take exists. Motion QA must report
what the rig produced and refuse only the handful of ways a take carries no information at all — wrong camera, wrong
format, an imaging preset that did not apply, frame-loss bursts, timestamp discontinuities, feature collapse, massive
blur, depth collapse."""
import numpy as np
import pytest
from handumi_collector.tools.motion_qa import CATASTROPHIC, check_provenance, check_wrist


def _dev(name="FisheyeCamLeft", matches=1, w=1920, h=1080, preset_ok=True):
    return {"left_wrist": dict(
        index=1, actual=dict(width=float(w), height=float(h), fps=30.0),
        identity=dict(match_name="FisheyeCamLeft", matches=matches, listing_name=name,
                      observed_serials=["UC684"], strict=True),
        uvc_controls=[dict(control="exposure-time-abs", requested=20, readback="20" if preset_ok else "156",
                           applied=preset_ok, reason="" if preset_ok else "read back '156', asked for 20")])}


EXPECT = {"left_wrist": dict(width=1920, height=1080, fps=30)}


def test_clean_provenance_raises_nothing():
    fails = []
    rep = check_provenance(dict(devices=_dev()), EXPECT, fails)
    assert fails == []
    assert rep["left_wrist"]["uvc_controls"]["exposure-time-abs"] == "20"


def test_wrong_camera_is_catastrophic():
    fails = []
    check_provenance(dict(devices=_dev(matches=2)), EXPECT, fails)
    assert any("wrong camera" in f and "ordinal" in f for f in fails)
    fails = []
    check_provenance(dict(devices=_dev(name="Arducam 1080P Low Light")), EXPECT, fails)
    assert any("wrong camera" in f and "profile names" in f for f in fails)


def test_format_mismatch_and_unfrozen_imaging_are_catastrophic():
    fails = []
    check_provenance(dict(devices=_dev(w=1280, h=720)), EXPECT, fails)
    assert any("format mismatch" in f and "1280x720" in f for f in fails)
    fails = []
    check_provenance(dict(devices=_dev(preset_ok=False)), EXPECT, fails)
    assert any("imaging not frozen" in f and "exposure-time-abs" in f for f in fails)


def test_catastrophic_floors_are_floors_not_quality_gates():
    """Every number in this dict must be a level below which a take is worthless — not a target to hit."""
    assert set(CATASTROPHIC) == {"min_corners_median", "min_static_blur_var", "max_consecutive_drops",
                                 "max_gap_frame_periods", "min_depth_coverage"}
    assert CATASTROPHIC["min_corners_median"] < 100          # a real take detects hundreds
    assert CATASTROPHIC["min_depth_coverage"] < 0.5          # the Orbbec measured 88-96% on real takes


def test_feature_collapse_and_blur_are_detected(monkeypatch, tmp_path):
    import handumi_collector.tools.wrist_exposure_qa as wq
    rows = [dict(t_ns=i, brightness=40.0, blur_var=5.0, corners=8, survived=0.2, flow_px=0.3, bucket="static")
            for i in range(30)]
    monkeypatch.setattr(wq, "analyse", lambda *a, **k: dict(episode="e", stream="left_wrist", n_frames=len(rows), rows=rows))
    fails = []
    out = check_wrist(tmp_path, "left_wrist", fails, limit=None)
    assert any("feature collapse" in f for f in fails) and any("massive blur" in f for f in fails)
    assert out["buckets"]["static"]["n"] == 30 and out["brightness"]["mean"] == pytest.approx(40.0)
