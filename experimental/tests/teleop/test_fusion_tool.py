"""The multi-sensor ladder tool on its synthetic take (ego_teleop/tools/f4_fusion.py).

The synthetic take is never evidence about the hardware — it has a known answer, which is what makes it a test:
the chest camera sees the truth, the wrist VI drifts away from it after the alignment windows, and the fused arm of
the ablation must be the one that ends up back at the start."""
from __future__ import annotations
import json
import numpy as np
import pytest
from ego_teleop.tools.f4_fusion import ABLATIONS, main, synthetic_take


def test_the_synthetic_take_is_continuous():
    """A teleporting fixture would show up as a catastrophic jump and the metric would be measuring the fixture."""
    vi, rgbd, truth, _ = synthetic_take()
    P = truth[:, 1:]
    assert (np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000.0).max() < 25.0
    assert len(vi) == len(rgbd) == len(truth)


@pytest.mark.parametrize("stage", ["compare", "fused", "virtual"])
def test_every_offline_stage_runs_and_reports(tmp_path, stage):
    out = tmp_path / f"{stage}.json"
    assert main(["--stage", stage, "--synthetic", "--protocol", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())["reports"]["fused"]
    assert rep["stage"] == stage and rep["ticks"] > 100
    assert rep["alignment"]["aligned"] and rep["alignment"]["lever_arm_observable"]
    assert np.isfinite(rep["headline"]["return_to_start_error_mm"])
    if stage == "virtual":
        assert rep["engaged"] and rep["command"]["step_mm"]["p95"] > 0
        assert all(v["passed"] for v in rep["gate"].values()), \
            {k: v for k, v in rep["gate"].items() if not v["passed"]}


def test_the_ablation_shows_what_each_sensor_contributes(tmp_path):
    out = tmp_path / "abl.json"
    assert main(["--stage", "virtual", "--synthetic", "--protocol", "--ablation", "--out", str(out)]) == 0
    reps = json.loads(out.read_text())["reports"]
    assert set(reps) == set(ABLATIONS)
    ret = {m: reps[m]["return_to_start"]["position_error_mm"] for m in ABLATIONS}
    leak = {m: reps[m]["rotation_only_leak"]["displacement_mm"] for m in ABLATIONS}
    # B (wrist VI alone) drifts; the chest anchor is what removes that drift
    assert ret["fused"] < 0.5 * ret["vi_only"], ret
    # A (chest RGB-D alone) tracks the palm centroid, so a wrist roll moves its "arm pose"; fusion suppresses that
    assert leak["fused"] < 0.5 * leak["rgbd_only"], leak


def test_the_real_stage_refuses_to_start_by_accident():
    with pytest.raises(SystemExit) as e:
        main(["--stage", "real", "--synthetic"])
    assert "i-am-at-the-robot" in str(e.value)


def test_a_replay_without_the_wrist_camera_calibration_is_refused(tmp_path):
    """T_H_C identity would fabricate translation from every wrist rotation (spec section 8) — never a default."""
    with pytest.raises(SystemExit) as e:
        main(["--stage", "compare", "--episode", str(tmp_path), "--vi-poses", str(tmp_path / "none.parquet")])
    assert "wrist_camera" in str(e.value) or "identity" in str(e.value)
