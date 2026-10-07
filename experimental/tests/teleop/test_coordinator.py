import json
import numpy as np
import pytest
import pandas as pd
from handumi_collector.pose.backends.mock import MockBackend
from ego_teleop.tracking.interfaces import WristPose, TrackingHealth, HandHealth, HandFeatures, AERO_CHANNELS
from ego_collector.hands3d.aero import COMPACT_UPPER_DEG
UP = np.minimum(90.0, COMPACT_UPPER_DEG)   # Aero hard clamp (thumb_cmc_flexion max 55)
def exp(n): return np.minimum(n * 90.0, COMPACT_UPPER_DEG)   # test channels: robot_max 90 deg, then the Aero clamp
from ego_teleop.tracking.wrist_pose_provider import EstimatorWristPoseProvider, TrackingSupervisor, TrackingSupervisorConfig
from ego_teleop.transforms.frames import HumanRobotFrameMapper
from ego_teleop.transforms.se3 import make_T
from ego_teleop.retarget.arm_relative_se3 import RelativeSE3Retargeter
from ego_teleop.retarget.aero_retarget import AeroRetargeter, ChannelMap
from ego_teleop.robot.safety import ArmSafetyPipeline, SafetyConfig, WorkspaceBox
from ego_teleop.robot.rebot_client import MockRebotController
from ego_teleop.robot.aero_client import MockAeroClient
from ego_teleop.robot.coordinator import TeleopStateMachine, TeleopState, Readiness, TeleopCoordinator, CoordinatorConfig
from ego_teleop.recorder.episode_logger import TeleopEpisodeLogger
from .conftest import T_of

NS = 1_000_000_000


class ScriptedWrist:
    """WristPoseProvider driven by the test: set_pose(T, health)."""
    def __init__(self): self.T = np.eye(4); self.health = TrackingHealth.OK; self.t = 0
    def set(self, T=None, health=None, t=None):
        if T is not None: self.T = T
        if health is not None: self.health = health
        if t is not None: self.t = t
    def get_pose(self, now_ns=None): return WristPose.from_T(self.t, self.T, health=self.health)


class ScriptedHand:
    def __init__(self): self.u7 = np.zeros(7); self.health = HandHealth.OK
    def get_features(self, now_ns=None):
        return HandFeatures(now_ns or 0, np.zeros((21, 3)), self.u7.copy() if self.health != HandHealth.LOST else None, 0.9, self.health)


def ready(): return Readiness(True, True, True, True, True, True, True)


def test_state_machine_legal_path_and_guards():
    sm = TeleopStateMachine(); r = Readiness()
    with pytest.raises(RuntimeError): sm.fire("start")
    with pytest.raises(RuntimeError): sm.fire("sensors_ready", r)          # camera/imu not online
    r.camera_online = r.imu_online = True; assert sm.fire("sensors_ready", r) == TeleopState.SENSOR_READY
    r.calibration_loaded = True; sm.fire("calibrated", r)
    with pytest.raises(RuntimeError): sm.fire("robot_ready", r)            # HOMING/robot only after VIO_STABLE
    sm.fire("vio_start", r); assert sm.state == TeleopState.VIO_INITIALIZING
    ok, why = sm.can("vio_stable", r); assert not ok and "vio_stable" in why
    r.tracking_valid = r.vio_stable = True; sm.fire("vio_stable", r)
    r.robot_observed = r.aero_homed = True; sm.fire("robot_ready", r)
    sm.fire("arm", r); sm.fire("start", r)
    sm.fire("clutch"); assert sm.state == TeleopState.CLUTCHED; sm.fire("release", r); assert sm.state == TeleopState.TELEOP
    sm.fire("estop"); assert sm.state == TeleopState.ESTOP
    with pytest.raises(RuntimeError): sm.fire("start", r)
    sm.fire("clear_estop", r); assert sm.state == TeleopState.ROBOT_READY
    assert [h[2] for h in sm.history][:4] == ["sensors_ready", "calibrated", "vio_start", "vio_stable"]


def build(clutch_freezes_hand=False):
    wrist, hand = ScriptedWrist(), ScriptedHand()
    robot = MockRebotController(T_of((0.3, 0.0, 0.2))); aero = MockAeroClient(lag=1.0); aero.home()
    rt = AeroRetargeter([ChannelMap(n, robot_max_deg=90, deadband=0, lpf_alpha=1, max_rate_deg_s=1e9) for n in AERO_CHANNELS])
    safety = ArmSafetyPipeline(SafetyConfig(workspace=WorkspaceBox((0, -0.5, 0), (0.6, 0.5, 0.5)), max_lin_vel_m_s=100, max_ang_vel_deg_s=1e4, max_lin_acc_m_s2=1e6))
    co = TeleopCoordinator(wrist=wrist, hand=hand, arm_retargeter=RelativeSE3Retargeter(HumanRobotFrameMapper()), aero_retargeter=rt, safety=safety,
                           robot=robot, aero=aero, cfg=CoordinatorConfig(clutch_freezes_hand=clutch_freezes_hand), readiness=ready())
    for ev in ("sensors_ready", "calibrated", "vio_start", "vio_stable", "robot_ready", "arm", "start"): co.fire(ev, t_ns=0)
    return co, wrist, hand, robot, aero


def test_no_commands_before_teleop():
    wrist, hand = ScriptedWrist(), ScriptedHand(); robot = MockRebotController(); aero = MockAeroClient()
    co = TeleopCoordinator(wrist=wrist, hand=hand, arm_retargeter=RelativeSE3Retargeter(HumanRobotFrameMapper()), aero_retargeter=AeroRetargeter(),
                           safety=ArmSafetyPipeline(), robot=robot, aero=aero, readiness=ready())
    assert co.tick(0) is None and robot.commands == [] and aero.commands == []


def test_full_loop_arm_follows_hand_follows_clutch_lost_estop():
    co, wrist, hand, robot, aero = build()
    T0 = robot.T.copy()
    wrist.set(T_of((0.05, 0, 0)), t=NS // 30); hand.u7 = np.full(7, 0.5)
    c = co.tick(NS // 30)
    assert c.arm_sent and np.allclose(robot.T[:3, 3], T0[:3, 3] + [0.05, 0, 0]) and np.allclose(c.arm_action6[:3], [0.05, 0, 0])
    assert c.hand_sent and np.allclose(c.hand_target7_deg, exp(0.5)) and np.allclose(aero.pos, exp(0.5))
    assert c.arm_exec.execution_timestamp_ns >= 0 and c.hand_exec_ns is not None and c.wrist_health == TrackingHealth.OK
    assert np.allclose(c.delta_H6[:3], [0.05, 0, 0])
    # clutch: arm frozen, hand still moves (default)
    co.fire("clutch", t_ns=2 * NS // 30); wrist.set(T_of((0.5, 0.5, 0)), t=2 * NS // 30); hand.u7 = np.full(7, 1.0)
    c = co.tick(2 * NS // 30)
    assert c.state == TeleopState.CLUTCHED and c.clutch and not c.arm_sent and np.allclose(robot.T[:3, 3], T0[:3, 3] + [0.05, 0, 0])
    assert c.hand_sent and np.allclose(c.hand_target7_deg, exp(1.0))
    # release re-anchors: no jump
    co.fire("release", t_ns=3 * NS // 30); c = co.tick(3 * NS // 30)
    assert c.recenter and np.allclose(robot.T[:3, 3], T0[:3, 3] + [0.05, 0, 0])   # zero delta => target == current pose
    # tracking lost: arm holds, hand keeps working
    wrist.set(T_of((0.9, 0.5, 0)), health=TrackingHealth.LOST, t=4 * NS // 30); hand.u7 = np.full(7, 0.2)
    n_before = len(robot.commands); c = co.tick(4 * NS // 30)
    assert c.wrist_health == TrackingHealth.LOST and c.hold_reason == "tracking_lost" and len(robot.commands) == n_before and np.allclose(c.hand_target7_deg, exp(0.2))
    # hand lost: relaxed pose
    hand.health = HandHealth.LOST; c = co.tick(5 * NS // 30); assert c.hand_health == HandHealth.LOST and np.allclose(c.hand_target7_deg, exp(co.aero_rt.relaxed_n))
    # estop
    co.fire("estop", t_ns=6 * NS // 30); assert robot.estops == 1 and co.tick(6 * NS // 30) is None


def test_release_with_zero_motion_is_sent_and_keeps_pose():
    co, wrist, hand, robot, aero = build()
    wrist.set(T_of((0.02, 0, 0)), t=1); c = co.tick(1); assert c.arm_sent
    T_after = robot.T.copy(); co.fire("clutch", t_ns=2); co.fire("release", t_ns=3); c = co.tick(3)
    assert np.allclose(robot.T, T_after) and c.recenter


def test_clutch_can_freeze_hand_too():
    co, wrist, hand, robot, aero = build(clutch_freezes_hand=True)
    hand.u7 = np.full(7, 0.5); co.tick(1); co.fire("clutch", t_ns=2); hand.u7 = np.full(7, 1.0); c = co.tick(2)
    assert np.allclose(c.hand_target7_deg, exp(0.5)) and c.hand_u7 is None


def test_arm_failure_reported_not_hidden():
    co, wrist, hand, robot, aero = build(); robot.fail_every = 1
    wrist.set(T_of((0.01, 0, 0)), t=1); c = co.tick(1)
    assert not c.arm_sent and c.arm_exec is not None and c.arm_exec.error == "mock_fail"


def test_episode_logger_layout_and_no_overwrite(tmp_path):
    co, wrist, hand, robot, aero = build()
    ep = tmp_path / "episode_000001"; log = TeleopEpisodeLogger(ep, metadata=dict(side="right"))
    log.add_calibration("aero_retarget", co.aero_rt.to_dict()); log.add_event("start", 0, state="TELEOP")
    for k in range(1, 6):
        wrist.set(T_of((0.01 * k, 0, 0)), t=k * NS // 30); c = co.tick(k * NS // 30)
        log.add_wrist_pose('right', wrist.get_pose()); log.add_wrist_pose('left', wrist.get_pose()); log.add_finger_state(hand.get_features(k)); log.add_command(c); log.add_rebot_state(robot.observe()); log.add_aero_state(aero.observe())
    assert (ep / ".incomplete").exists()
    meta = log.close(status="KEEP")
    assert not (ep / ".incomplete").exists() and meta["row_counts"]["rebot_command"] == 5
    for f in ("raw/wrist_pose_live_left", "raw/wrist_pose_live_right", "raw/finger_state", "raw/events", "robot/rebot_state", "robot/rebot_command", "robot/aero_state", "robot/aero_command"):
        assert (ep / f"{f}.parquet").exists(), f
    assert not any("head" in p.name for p in ep.rglob("*pose*"))          # head C922 has no pose stream anywhere
    assert set(pd.read_parquet(ep / "raw/wrist_pose_live_left.parquet")["side"]) == {"left"}
    with pytest.raises(ValueError):
        log2 = TeleopEpisodeLogger(tmp_path / "ep2"); log2.add_wrist_pose("head", wrist.get_pose())
    df = pd.read_parquet(ep / "robot/rebot_command.parquet")
    assert list(df["arm_dx"].round(3)) == [0.01] * 5 and df["arm_exec_ns"].notna().all() and df["hand_exec_ns"].notna().all()
    assert json.loads((ep / "calibration/aero_retarget.json").read_text())["order"] == list(AERO_CHANNELS)
    assert json.loads((ep / "metadata.json").read_text())["status"] == "KEEP"
    with pytest.raises(FileExistsError):
        TeleopEpisodeLogger(ep)


def test_estimator_provider_plugs_into_coordinator():
    """End-to-end with the pose package's MockBackend as the 'VIO': arm follows, LOST window holds."""
    be = MockBackend(lambda t: make_T(None, [0.1 * t / NS, 0, 0]), init_frames=1, lost_windows_ns=[(NS, int(1.4 * NS))])
    prov = EstimatorWristPoseProvider(be, T_H_C=np.eye(4), supervisor=TrackingSupervisor(TrackingSupervisorConfig(recover_after_n_ok=1)))
    robot = MockRebotController(T_of((0.3, 0, 0.2))); aero = MockAeroClient(); aero.home()
    safety = ArmSafetyPipeline(SafetyConfig(workspace=WorkspaceBox((0, -1, 0), (2, 1, 1)), max_lin_vel_m_s=100, max_ang_vel_deg_s=1e4, max_lin_acc_m_s2=1e6))
    co = TeleopCoordinator(wrist=prov, hand=None, arm_retargeter=RelativeSE3Retargeter(HumanRobotFrameMapper()), aero_retargeter=AeroRetargeter(), safety=safety,
                           robot=robot, aero=None, cfg=CoordinatorConfig(hand_enabled=False), readiness=ready())
    for i, t in enumerate(range(0, NS // 10, NS // 30)): prov.push_image(t, i, None)
    for ev in ("sensors_ready", "calibrated", "vio_start", "vio_stable", "robot_ready", "arm", "start"): co.fire(ev, t_ns=NS // 10)
    x0 = robot.T[0, 3]; held = 0
    for i, t in enumerate(range(NS // 10, 2 * NS, NS // 30), start=10):
        prov.push_image(t, i, None); c = co.tick(t)
        if c.wrist_health == TrackingHealth.LOST: held += 1
    assert held >= 10 and np.isclose(robot.T[0, 3] - x0, 0.1 * (2 - 0.1) - 0.0, atol=0.02)   # ends where the wrist is (relative), lost window held in between
