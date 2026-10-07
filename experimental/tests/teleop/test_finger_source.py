import pytest
from ego_teleop.retarget.hand_features import FingerSource, LiveHandTracker


def test_live_tracker_refuses_unverified_finger_source():
    with pytest.raises(RuntimeError, match="not decided"):
        LiveHandTracker("right")                       # default source = UNVERIFIED


def test_rectifier_only_for_fisheye_source():
    with pytest.raises(ValueError):
        LiveHandTracker("right", source=FingerSource.DEDICATED_CAMERA, rectifier=object())


def test_config_defaults_encode_the_agreed_layout():
    from ego_teleop.config import load_teleop_cfg
    cfg = load_teleop_cfg()
    assert cfg.wrists == ["left", "right"] and cfg.finger_source == "unverified" and cfg.head_camera_role == "observation_only"
