import numpy as np
import pytest
from handumi_collector.pose.estimator import TrackingState
from ego_teleop.tracking.interfaces import WristPose, TrackingHealth
from ego_teleop.transforms.frames import FrameMapperConfig, HumanRobotFrameMapper
from ego_teleop.retarget.arm_relative_se3 import RelativeSE3Retargeter, ArmRetargetState
from .conftest import T_of


def wp(T, health=TrackingHealth.OK, t=0):
    return WristPose.from_T(t, T, health=health)


def test_disengaged_returns_current_pose():
    rt = RelativeSE3Retargeter(HumanRobotFrameMapper())
    cur = T_of((0.3, 0, 0.2))
    out = rt.update(wp(T_of()), cur)
    assert out.held and np.allclose(out.T_RB_RE_target, cur)


def test_relative_translation_in_anchor_frame_with_scale():
    rt = RelativeSE3Retargeter(HumanRobotFrameMapper(FrameMapperConfig(translation_scale=(0.5, 0.5, 0.5))))
    H0 = T_of((5.0, 5.0, 1.0), (0, 0, 90))          # arbitrary VIO world origin/orientation must not matter
    R0 = T_of((0.3, 0.0, 0.2))
    rt.engage(wp(H0), R0)
    out0 = rt.update(wp(H0), R0)
    assert np.allclose(out0.T_RB_RE_target, R0) and not out0.held
    H1 = H0 @ T_of((0.10, 0, 0))                     # human moves 10 cm along its own wrist x
    out1 = rt.update(wp(H1), R0)
    assert np.allclose(out1.T_RB_RE_target, R0 @ T_of((0.05, 0, 0)))   # robot moves 5 cm along EEF x
    assert np.allclose(out1.delta_H[:3, 3], [0.10, 0, 0])


def test_clutch_freezes_and_release_reanchors_without_jump():
    rt = RelativeSE3Retargeter(HumanRobotFrameMapper()); H0, R0 = T_of(), T_of((0.3, 0, 0.2))
    rt.engage(wp(H0), R0)
    rt.update(wp(H0 @ T_of((0.05, 0, 0))), R0)
    rt.clutch(); assert rt.state == ArmRetargetState.CLUTCHED
    frozen = rt.update(wp(H0 @ T_of((0.5, 0.5, 0.5))), R0)
    assert frozen.held and frozen.hold_reason == "clutch" and np.allclose(frozen.T_RB_RE_target, R0 @ T_of((0.05, 0, 0)))
    H_new, R_now = T_of((0.9, 0.9, 0.9), (0, 45, 0)), R0 @ T_of((0.05, 0, 0))
    rt.release(wp(H_new), R_now)
    out = rt.update(wp(H_new), R_now)
    assert not out.held and np.allclose(out.T_RB_RE_target, R_now)     # zero jump at release


def test_tracking_lost_holds_last_target_and_degraded_slows():
    rt = RelativeSE3Retargeter(HumanRobotFrameMapper(), degraded_speed_factor=0.25); H0, R0 = T_of(), T_of()
    rt.engage(wp(H0), R0)
    a = rt.update(wp(H0 @ T_of((0.1, 0, 0))), R0)
    lost = rt.update(wp(H0 @ T_of((0.9, 0, 0)), TrackingHealth.LOST), R0)
    assert lost.held and lost.hold_reason == "tracking_lost" and lost.speed_factor == 0.0 and np.allclose(lost.T_RB_RE_target, a.T_RB_RE_target)
    none = rt.update(None, R0); assert none.held
    deg = rt.update(wp(H0 @ T_of((0.2, 0, 0)), TrackingHealth.DEGRADED), R0)
    assert not deg.held and deg.speed_factor == 0.25


def test_engage_requires_valid_tracking():
    rt = RelativeSE3Retargeter(HumanRobotFrameMapper())
    with pytest.raises(RuntimeError):
        rt.engage(wp(T_of(), TrackingHealth.LOST), T_of())


def test_action6_is_local_delta_between_consecutive_targets():
    rt = RelativeSE3Retargeter(HumanRobotFrameMapper()); rt.engage(wp(T_of()), T_of((0.3, 0, 0.2), (0, 0, 90)))
    prev = rt.update(wp(T_of()), T_of()).T_RB_RE_target
    cur = rt.update(wp(T_of((0.02, 0.01, 0))), T_of())
    a = cur.action6(prev)
    assert np.allclose(a[:3], [0.02, 0.01, 0]) and np.allclose(a[3:], 0)
