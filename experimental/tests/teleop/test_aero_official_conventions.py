"""A0 regression tests: pin the official TetherIA conventions our Aero branch is built on.

These fail if the upstream stack changes (or if someone "tidies" our transcription), which is the point — the
alternative is a silent behaviour change on a real hand. When the checked-out upstream repo is not present the
cross-checks skip, but the *self-consistency* checks always run.

Upstream: ~/aero-hand/aero-hand-open (docs/ego_teleop/AERO_A0_AUDIT.md records what was read, and when)."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pytest
from ego_teleop.hand3d import aero_mocap as M
from ego_teleop.retarget.aero_backends import OFFICIAL_SCALE_FACTORS, DexPilotAeroRetargeter, DexPilotConfig

UPSTREAM = Path.home() / "aero-hand" / "aero-hand-open"
ROS_SRC = UPSTREAM / "ros2" / "src"
pytestmark_upstream = pytest.mark.skipif(not ROS_SRC.exists(), reason="upstream aero-hand-open checkout not present")


# ---- self-consistent conventions (always run) -------------------------------------------------------------------
def test_mocap25_expansion_duplicates_the_wrist_into_every_finger_cmc():
    J = np.arange(63, dtype=float).reshape(21, 3)
    k = M.expand_to_mocap25(J)
    assert k.shape == (25, 3)
    for cmc in (5, 10, 15, 20):                      # index/middle/ring/pinky CMC slots
        assert np.array_equal(k[cmc], J[0])
    assert np.array_equal(k[1], J[1]) and np.array_equal(k[4], J[4])        # thumb passes straight through
    for slot, mp in zip(M.MOCAP25_TIPS, M.MP_TIPS):
        assert np.array_equal(k[slot], J[mp])
    assert len(M.MOCAP25_NAMES) == 25 == len(M.MEDIAPIPE_TO_MOCAP25)


def test_palm_frame_matches_the_official_formula():
    """Reference implementation transcribed inline from webcam_mocap.process_landmarks."""
    rng = np.random.default_rng(7)
    for side in ("right", "left"):
        lm = rng.normal(size=(21, 3))
        x = lm[5] - lm[13]; x = x / np.linalg.norm(x)
        if side == "left": x = -x
        z = lm[9] - lm[0]; z = z / np.linalg.norm(z)
        y = np.cross(z, x); y = y / np.linalg.norm(y)
        x = np.cross(y, z); x = x / np.linalg.norm(x)
        expected = (lm - lm[0]) @ np.array([x, y, z]).T
        assert np.allclose(M.to_palm_local(lm, side), expected)


def test_palm_frame_is_a_proper_rotation_for_both_hands():
    rng = np.random.default_rng(3)
    for side in ("right", "left"):
        R, t = M.palm_frame(rng.normal(size=(21, 3)), side)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_palm_local_is_invariant_to_camera_pose():
    """The property the whole head-camera design rests on (spec section 10): moving the camera must not change the
    articulation the retargeter sees."""
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(11)
    lm = rng.normal(size=(21, 3)) * 0.05
    base = M.to_palm_local(lm, "right")
    for seed in range(5):
        R = Rot.random(random_state=seed).as_matrix(); t = rng.normal(size=3)
        moved = (R @ lm.T).T + t
        assert np.allclose(M.to_palm_local(moved, "right"), base, atol=1e-9)


def test_degenerate_hand_raises_instead_of_returning_a_wrong_frame():
    lm = np.zeros((21, 3))
    with pytest.raises(ValueError):
        M.palm_frame(lm, "right")


def test_handedness_is_swapped_for_a_non_selfie_camera():
    # the head camera looks outward: MediaPipe's selfie-mirrored label must be inverted (A0 audit)
    assert M.handedness_for_image("Right", selfie_mirrored=True) == "right"
    assert M.handedness_for_image("Right", selfie_mirrored=False) == "left"
    with pytest.raises(ValueError): M.handedness_for_image("either", selfie_mirrored=False)


def test_aero_joint_order_and_limits():
    assert M.AERO_JOINT_NAMES[:4] == ("thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip")
    assert M.AERO_JOINT_NAMES[4::3][:4] == ("index_mcp_flex", "middle_mcp_flex", "ring_mcp_flex", "pinky_mcp_flex")
    assert M.AERO_JOINT_UPPER_DEG[:4] == (100.0, 55.0, 90.0, 90.0)
    assert set(M.AERO_JOINT_UPPER_DEG[4:]) == {90.0} and set(M.AERO_JOINT_LOWER_DEG) == {0.0}


def test_scale_factors_match_the_official_table():
    assert [s for s, _ in OFFICIAL_SCALE_FACTORS] == [1.5, 2.0, 3.5, 1.15, 1.15, 1.15, 1.2]
    assert [round(np.degrees(o)) for _, o in OFFICIAL_SCALE_FACTORS] == [0, -60, 0, -10, -10, -10, -10]


# ---- cross-checks against the checked-out upstream source --------------------------------------------------------
@pytestmark_upstream
def test_handmocap_msg_keypoint_order_unchanged():
    txt = (ROS_SRC / "aero_hand_open_msgs" / "msg" / "HandMocap.msg").read_text()
    order = [m.group(2) for m in re.finditer(r"^#\s*(\d+)\s*-\s*(\w+)\s*$", txt, re.M)]
    assert tuple(order) == M.MOCAP25_NAMES


@pytestmark_upstream
def test_jointcontrol_msg_joint_order_unchanged():
    txt = (ROS_SRC / "aero_hand_open_msgs" / "msg" / "JointControl.msg").read_text()
    order = [m.group(2) for m in re.finditer(r"^#\s*(\d+)\s*-\s*(\w+)\s*$", txt, re.M)]
    assert tuple(order) == M.AERO_JOINT_NAMES


@pytestmark_upstream
def test_sdk_constants_unchanged():
    txt = (UPSTREAM / "sdk" / "src" / "aero_open_sdk" / "aero_hand_constants.py").read_text()
    upper = [float(v) for v in re.search(r"joint_upper_limits.*?\((.*?)\)", txt, re.S).group(1).replace("\n", "").split(",") if v.strip()]
    lower = [float(v) for v in re.search(r"joint_lower_limits.*?\((.*?)\)", txt, re.S).group(1).replace("\n", "").split(",") if v.strip()]
    assert tuple(upper) == M.AERO_JOINT_UPPER_DEG and tuple(lower) == M.AERO_JOINT_LOWER_DEG


@pytestmark_upstream
def test_dexpilot_config_matches_the_official_node():
    node = (ROS_SRC / "aero_hand_open_retargeting" / "aero_hand_open_retargeting" / "dex_retargeting_node.py").read_text()
    ours = DexPilotAeroRetargeter.make_config(type("_", (), {"cfg": DexPilotConfig(side="right")})())
    assert ours["wrist_link_name"] == "right_base_link"
    assert ours["finger_tip_link_names"] == [f"right_{f}_tip_link" for f in ("thumb", "index", "middle", "ring", "pinky")]
    assert ours["scaling_factor"] == 1.2 and ours["low_pass_alpha"] == 0.9
    block = node[node.index('elif retargeting_method == "dexpilot"'):]
    rows = re.search(r"target_link_human_indices\":\s*\[\s*\[(.*?)\],\s*\[(.*?)\]", block, re.S)
    upstream = [[int(v) for v in rows.group(i).replace("\n", "").split(",") if v.strip()] for i in (1, 2)]
    assert ours["target_link_human_indices"] == upstream
    # the two undocumented hacks are still there, with the same numbers
    assert "data += np.array([0.0, 0.01, 0.0])" in node
    assert "if data[24][2] > 0.12" in node and "data[24][2] += 0.02" in node
    assert DexPilotConfig().base_offset_m == (0.0, 0.01, 0.0) and DexPilotConfig().pinky_lift == (0.12, 0.02)


@pytestmark_upstream
def test_webcam_mocap_still_uses_world_landmarks_and_a_mirrored_frame():
    """If upstream stops mirroring, our `selfie_mirrored` handedness rule needs revisiting — catch it here."""
    txt = (ROS_SRC / "webcam_mocap" / "webcam_mocap" / "webcam_mocap.py").read_text()
    assert "cv2.flip(frame_bgr, 1)" in txt
    assert "multi_hand_world_landmarks" in txt
    assert "[-lm.x, lm.y, lm.z]" in txt
