"""Spec 31-34: recorder keeps executed vs requested separately, 20-D export from the EXECUTED trajectory, QA flags."""
import json
import numpy as np
from visual_teleop.recording.recorder import EpisodeRecorder
from visual_teleop.recording.export_lerobot import export_episode
from visual_teleop.recording.episode_qa import episode_qa
from visual_teleop import config as vcfg
from .conftest import T_of, hand, pinch, DT_NS


def test_record_export_qa_end_to_end(rig, tmp_path):
    ctl, br = rig()
    rec = EpisodeRecorder(tmp_path, skill_id="UNFOLD_ALIGN", name="ep")
    for k in range(90):
        p = pinch(30 <= k < 60)                                                     # close for 1 s, then open
        rec.add_tick(ctl.tick({s: hand(s, T_of((0.0005 * k, 0, 0))) for s in ("left", "right")}, {"left": p, "right": p}, k * DT_NS))
    d = rec.close()
    a = np.load(d / "ticks.npz"); meta = json.loads((d / "metadata.json").read_text())
    assert meta["data_source"] == "VISUAL_TELEOP" and meta["skill_id"] == "UNFOLD_ALIGN" and len(a["t_ns"]) == 90
    assert np.all(np.diff(a["t_ns"]) > 0)                                           # monotonic
    assert np.isfinite(a["obs_T_left"]).all() and np.isfinite(a["human_T_left"]).all()
    m = export_episode(d, horizon=4)
    c = np.load(d / "contract.npz")
    assert m["label_source"] == "obs_T" and c["action20"].shape[1] == 20
    g = c["action20"][:, 18]
    assert g.min() < 0.2 and g.max() > 0.9                                          # gripper label from the robot, 0 = closed
    dx = c["action20"][:, 0:3]; assert np.all(np.isfinite(dx))
    q = episode_qa(d, vcfg.load().qa)
    assert q["sides"]["left"]["gripper_transitions"] == 2 and "left:gripper_polarity" not in q["flags"]


def test_export_equals_shared_contract_encoder(rig, tmp_path):
    """At the contract clock (20 Hz) the exported chunk must equal Agent A's encode_action_chunk on the same executed poses."""
    ctl, br = rig(); rec = EpisodeRecorder(tmp_path, skill_id="UNFOLD_ALIGN", name="ep20")
    for k in range(60):
        p = pinch(20 <= k < 40)
        rec.add_tick(ctl.tick({s: hand(s, T_of((0.0004 * k, 0.0002 * k, 0), (0, 0, 0.2 * k))) for s in ("left", "right")},
                              {"left": p, "right": p}, k * 50_000_000))
    m = export_episode(rec.close(), horizon=8)
    assert m["contract_version"].startswith("laundry-contract") and m["dt_s"] == 0.05
    assert m["shared_encoder_max_abs_diff"] is not None and m["shared_encoder_max_abs_diff"] < 1e-6
