"""Wire the teleop stack from the YAML (2026-10-14). The only place config values become objects."""
from __future__ import annotations
from .config import Cfg, SIDES, R_I_H
from .fusion.filters import TransactionalPoseFilter
from .fusion.wrist_pose_fusion import HandFusion
from .gripper.gripper_mapper import GripperMapper
from .gripper.pinch_detector import PinchDetector, SigmoidPinch
from .retargeting.frame_adapter import FrameAdapter
from .retargeting.motion_scaling import MotionScaling
from .retargeting.relative_se3 import RelativeRetargeter
from .robot.dual_arm_controller import ControllerConfig, DualArmController
from .robot.ik import ArmIk
from .safety.clutch import Clutch
from .safety.collision import InterArmCheck
from .safety.deadman import EStopLatch
from .safety.limits import ArmLimits


def build_controller(c: Cfg, *, bridge, solvers: dict, deadman, clutch: Clutch | None = None, estop: EStopLatch | None = None,
                     joint_limits: dict | None = None) -> DualArmController:
    rt = RelativeRetargeter({s: FrameAdapter.from_cfg(c.retarget["adapter"][s]) for s in SIDES}, MotionScaling.from_cfg(c.retarget),
                            orientation_mode=c.retarget["orientation_mode"], translation_frame=c.retarget.get("translation_frame", "body"),
                            pivot_human_m=float(c.retarget.get("pivot_human_m", 0.0)), pivot_robot_m=float(c.retarget.get("pivot_robot_m", 0.0)),
                            rotation_axis_scale=tuple(float(v) for v in c.retarget.get("rotation_axis_scale", [1.0, 1.0, 1.0])),
                            rotation_heading=str(c.retarget.get("rotation_heading", "base")),
                            rotation_frame=c.retarget.get("rotation_frame", "body"))
    arms = {s: (ArmLimits.from_cfg(c, s), TransactionalPoseFilter(**c.limits["filter"]), GripperMapper.from_cfg(c.gripper),
                ArmIk.from_cfg(c, solvers[s], (joint_limits or {}).get(s))) for s in SIDES}
    k = c.control
    cc = ControllerConfig(float(c.rate_hz), dict(k["arms_enabled"]), dict(k["grippers_enabled"]), bool(k["hold_both_on_any"]),
                          bool(c.operator["estop"]["call_bridge_estop"]), float(k["warn_speed_factor"]))
    return DualArmController(bridge=bridge, retarget=rt, arms=arms, collision=InterArmCheck.from_cfg(c.collision), estop=estop or EStopLatch(),
                             deadman=deadman, clutch=clutch or Clutch(), cfg=cc)


def build_hand_fusion(c: Cfg) -> dict:
    return {s: HandFusion.from_cfg(s, c, R_I_H(c, s)[0]) for s in SIDES}


def build_pinch(c: Cfg) -> dict:
    cls = SigmoidPinch if c.gripper.get("mode", "binary") == "sigmoid" else PinchDetector
    return {s: cls.from_cfg(c.gripper, c.tracking["min_fingertip_conf"]) for s in SIDES}
