import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ego_teleop.transforms.se3 import make_T
from visual_teleop import config as vcfg
from visual_teleop.build import build_controller
from visual_teleop.fusion.wrist_pose_fusion import HandEstimate, HandStatus
from visual_teleop.gripper.pinch_detector import PinchState
from visual_teleop.robot.bridge import MockBridge
from visual_teleop.robot.ik import ToyIk, toy_limits
from visual_teleop.safety.deadman import ScriptedDeadman, HeartbeatDeadman

DT_NS = int(1e9 / 30)


def T_of(xyz=(0, 0, 0), rotvec_deg=(0, 0, 0)):
    return make_T(Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix(), xyz)


def hand(side, T, t_ns=0, status=HandStatus.TRACKING):
    return HandEstimate(side, t_ns, status, [], None if T is None else np.asarray(T, float), None, None, 1.0, "imu", 0.0)


def lost(side, t_ns=0):
    return HandEstimate(side, t_ns, HandStatus.HOLD, ["position_lost"], None, None, None, 0.0, "none", 1e9)


def pinch(p, status="OK"):
    return PinchState(p, status, 10.0 if p else 60.0, 0.0)


# toy robot: q = [x y z rotvec]; start poses inside the workspace, arms apart in y
Q0 = {"left": np.array([0.30, 0.20, 0.15, 0, 0, 0.0]), "right": np.array([0.30, -0.20, 0.15, 0, 0, 0.0])}


@pytest.fixture
def cfg():
    return vcfg.load(overrides={"retarget": {"translation_frame": "body"}, "control": {"arms_enabled": {"left": True, "right": True}},
                                "limits": {"filter": {"alpha_pos": 1.0, "alpha_rot": 1.0}, "max_lin_acc_m_s2": 100.0}})


@pytest.fixture
def rig(cfg):
    """controller + mock robot + toy IK, initialised with both hands at the origin."""
    def make(c=None, deadman=None, unreachable=None):
        c = c or cfg
        br = MockBridge(q={s: Q0[s].copy() for s in Q0})
        dm = deadman or ScriptedDeadman()
        ctl = build_controller(c, bridge=br, solvers={s: ToyIk(unreachable) for s in Q0}, deadman=dm,
                               joint_limits={s: toy_limits() for s in Q0})
        ctl.initialise({s: hand(s, np.eye(4)) for s in Q0})
        return ctl, br
    return make
