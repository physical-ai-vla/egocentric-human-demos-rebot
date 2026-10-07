"""RGB-D-only arm POC (V0/V1): the fixed-camera palm pose source and its trip through the EXISTING arm stack.

What these tests are actually pinning:
  * the arm-root pose is graded separately from the hand — a hand with unusable fingertips still drives the arm;
  * the palm origin survives one bad depth sample, and refuses to invent a pose when it cannot;
  * 3-DoF really is 3-DoF (identity rotation out, so the robot holds its ENGAGE orientation by construction);
  * the pose source plugs into `RelativeSE3Retargeter` / `TeleopCoordinator` with no change to either;
  * it never writes, or is written as, a `wrist_pose_live_*` (OpenVINS wrist VIO) stream.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from ego_teleop.hand3d import aero_mocap as M
from ego_teleop.hand3d.interfaces import HandPoseEstimate, HandPoseHealth
from ego_teleop.hand3d.palm_pose import (POSE_SOURCE, ArmPoseHealth, PalmGeometryMonitor, PalmPoseConfig,
                                         palm_origin_camera, palm_pose_from_estimate, palm_spans_m)
from ego_teleop.hand3d.supervisor import HandPoseSupervisor, HandPoseSupervisorConfig
from ego_teleop.recorder.episode_logger import EPISODE_LAYOUT, TeleopEpisodeLogger
from ego_teleop.retarget.arm_relative_se3 import RelativeSE3Retargeter
from ego_teleop.robot.coordinator import CoordinatorConfig, Readiness, TeleopCoordinator
from ego_teleop.robot.rebot_client import MockRebotController
from ego_teleop.robot.safety import ArmSafetyPipeline, SafetyConfig, WorkspaceBox
from ego_teleop.tracking.interfaces import TrackingHealth
from ego_teleop.tracking.rgbd_hand_pose import STREAM_PREFIX, build_rgbd_hand_pose_provider
from ego_teleop.tracking.wrist_pose_provider import TrackingSupervisorConfig
from ego_teleop.transforms.frames import FrameMapperConfig, HumanRobotFrameMapper
from ego_teleop.tools import p1_rgbd_arm as P
from .conftest import T_of

NS = 1_000_000_000
# A plausible right hand in its own palm-local frame (metres): wrist at the origin, MCPs across the palm, tips beyond.
_LOCAL = np.zeros((21, 3))
_LOCAL[M.MIDDLE_MCP] = (0.000, 0.0, 0.090)
_LOCAL[M.INDEX_MCP] = (0.020, 0.0, 0.085)
_LOCAL[M.RING_MCP] = (-0.020, 0.0, 0.082)
_LOCAL[M.PINKY_MCP] = (-0.040, 0.0, 0.075)
for _tip, _mcp in ((4, M.INDEX_MCP), (8, M.INDEX_MCP), (12, M.MIDDLE_MCP), (16, M.RING_MCP), (20, M.PINKY_MCP)):
    _LOCAL[_tip] = _LOCAL[_mcp] + (0.0, 0.01, 0.070)
for _j in range(1, 21):
    if not _LOCAL[_j].any(): _LOCAL[_j] = _LOCAL[M.MIDDLE_MCP] * (_j / 21.0)


def hand(t_ns=0, xyz=(0.0, 0.0, 0.5), rotvec_deg=(0, 0, 0), *, invalid=(), scale=1.0, side="right",
         health=HandPoseHealth.OK) -> HandPoseEstimate:
    """A synthetic METRIC hand at a known camera-frame pose. `invalid` marks landmarks whose depth was not measured."""
    R = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    cam = (R @ (_LOCAL * scale).T).T + np.asarray(xyz, np.float64)
    valid = np.ones(21, bool); valid[list(invalid)] = False
    return HandPoseEstimate(int(t_ns), side, np.zeros((21, 2)), cam, M.to_palm_local(cam, side), valid,
                            cam[:, 2].copy(), np.full(21, 0.9), 0.9, health, "head_rgbd", metric=True,
                            palm_scale_m=M.palm_scale_m(cam), filled=~valid, pose_done_ns=int(t_ns) + 5_000_000)


def status_of(est, cfg=None):
    """One estimate through the hand supervisor (the ADVISORY input to the arm branch)."""
    sup = HandPoseSupervisor(cfg or HandPoseSupervisorConfig(recover_stable_ms=0.0))
    return sup.update(est, est.timestamp_ns)


def arm_pose(est, cfg=None, *, advise=True):
    cfg = cfg or PalmPoseConfig()
    return palm_pose_from_estimate(est, cfg, hand_status=status_of(est) if advise else None)


# ---- palm origin ------------------------------------------------------------------------------------------------
def test_palm_centroid_uses_only_really_measured_landmarks():
    cfg = PalmPoseConfig()
    est = hand(xyz=(0.1, 0.0, 0.5), invalid=(M.PINKY_MCP,))
    p, n, spread, why = palm_origin_camera(est, cfg)
    assert why == "" and n == 4                      # the filled pinky MCP is present in the array but never counted
    expected = np.median(est.landmarks_camera_3d[[M.WRIST, M.INDEX_MCP, M.MIDDLE_MCP, M.RING_MCP]], axis=0)
    assert np.allclose(p, expected)
    assert arm_pose(est).health is ArmPoseHealth.DEGRADED     # an incomplete palm is declared, never silently used


def test_palm_centroid_rejects_a_depth_outlier_and_keeps_the_pose():
    cfg = PalmPoseConfig()
    est = hand(xyz=(0.0, 0.0, 0.5))
    clean, _, _, _ = palm_origin_camera(est, cfg)
    est.landmarks_camera_3d[M.RING_MCP, 2] += 0.25    # one flying pixel / background mix
    p, n, spread, why = palm_origin_camera(est, cfg)
    assert why == "" and n == 5
    assert np.linalg.norm(p - clean) < 0.005          # a 25 cm outlier moves the median by under 5 mm ...
    assert spread > 0.2                               # ... and is still reported as evidence


def test_palm_origin_refuses_rather_than_guesses_when_the_palm_is_gone():
    cfg = PalmPoseConfig()
    est = hand(invalid=(M.WRIST, M.INDEX_MCP, M.MIDDLE_MCP))
    p, n, _, why = palm_origin_camera(est, cfg)
    assert p is None and n == 2 and "palm_landmarks" in why
    pose = arm_pose(est, cfg)
    assert pose.health is ArmPoseHealth.LOST and pose.position_m is None


def test_wrist_origin_mode_reproduces_the_canonical_landmark():
    est = hand(xyz=(0.02, -0.03, 0.44))
    p, n, _, why = palm_origin_camera(est, PalmPoseConfig(origin="wrist"))
    assert why == "" and n == 1 and np.allclose(p, est.landmarks_camera_3d[M.WRIST])


# ---- arm health is not hand health ------------------------------------------------------------------------------
def test_missing_fingertips_do_not_lose_the_arm_pose():
    """The whole reason `ArmPoseHealth` exists: the Aero branch needs fingertips, the arm needs a palm."""
    est = hand(invalid=(4, 8, 12, 16, 20))
    assert status_of(est).health is HandPoseHealth.LOST   # the Aero branch cannot use this hand at all ...
    pose = arm_pose(est, PalmPoseConfig(hand_degraded_is_arm_degraded=False))
    assert pose.health is ArmPoseHealth.OK               # ... and the arm branch is fine: the palm is complete
    assert pose.n_palm_landmarks == 5


def test_hand_degraded_downgrades_but_does_not_drop_the_arm_pose():
    est = hand(invalid=(4, 8, 12, 16, 20))
    pose = arm_pose(est)
    assert pose.health is ArmPoseHealth.DEGRADED and pose.position_m is not None


def test_thin_palm_evidence_is_degraded_not_lost():
    est = hand(invalid=(M.PINKY_MCP, M.RING_MCP))
    pose = arm_pose(est, PalmPoseConfig(hand_degraded_is_arm_degraded=False))
    assert pose.health is ArmPoseHealth.DEGRADED and pose.n_palm_landmarks == 3
    assert pose.position_m is not None                   # degraded still commands, just more slowly


def test_out_of_band_depth_is_lost():
    cfg = PalmPoseConfig(min_origin_depth_m=0.2, max_origin_depth_m=1.0)
    assert arm_pose(hand(xyz=(0, 0, 1.4)), cfg).health is ArmPoseHealth.LOST
    assert arm_pose(hand(xyz=(0, 0, 0.1)), cfg).health is ArmPoseHealth.LOST


# ---- 3-DoF vs 6-DoF ----------------------------------------------------------------------------------------------
def test_three_dof_emits_identity_rotation_but_still_measures_the_palm_basis():
    est = hand(rotvec_deg=(0, 35, 0))
    pose = arm_pose(est, PalmPoseConfig(orientation="fixed"))
    T = pose.T_camera_palm("fixed")
    assert np.allclose(T[:3, :3], np.eye(3))          # robot keeps its ENGAGE orientation, by construction
    assert pose.R_palm is not None                    # ... and V3/V4 still get their measurement


def test_six_dof_uses_the_canonical_aero_palm_basis_not_a_new_one():
    est = hand(rotvec_deg=(10, -20, 5))
    pose = arm_pose(est, PalmPoseConfig(orientation="palm"))
    R_canonical, _ = M.palm_frame(est.landmarks_camera_3d, "right")
    assert np.allclose(pose.R_palm, R_canonical)
    assert np.allclose(pose.T_camera_palm("palm")[:3, :3], R_canonical)


def test_six_dof_refuses_a_basis_built_on_invented_landmarks():
    est = hand(invalid=(M.RING_MCP,))
    assert arm_pose(est, PalmPoseConfig(orientation="palm")).health is ArmPoseHealth.LOST
    assert arm_pose(est, PalmPoseConfig(orientation="fixed")).position_m is not None


# ---- geometry sanity --------------------------------------------------------------------------------------------
def test_geometry_monitor_is_measurement_only_by_default():
    mon = PalmGeometryMonitor(PalmPoseConfig(scale_reference_frames=2))
    poses = [mon.update(arm_pose(hand(scale=s))) for s in (1.0, 1.0, 2.5)]
    assert mon.rejects == 0 and all(p.health is not ArmPoseHealth.LOST for p in poses)
    assert mon.reference_palm_scale_m == pytest.approx(poses[0].palm_scale_m, rel=1e-6)
    assert poses[-1].palm_scale_ratio > 2.0           # the evidence is recorded, the frame is not thrown away


def test_configured_scale_jump_threshold_rejects_impossible_geometry():
    cfg = PalmPoseConfig(scale_reference_frames=2, max_palm_scale_jump=0.3)
    mon = PalmGeometryMonitor(cfg)
    for s in (1.0, 1.0):
        mon.update(arm_pose(hand(scale=s), cfg))
    bad = mon.update(arm_pose(hand(scale=2.5), cfg))
    assert bad.health is ArmPoseHealth.LOST and "palm_scale_jump" in bad.reason and bad.position_m is None
    assert mon.rejects == 1


def test_spans_are_nan_unless_both_endpoints_were_really_measured():
    assert all(np.isfinite(palm_spans_m(hand())))
    assert np.isnan(palm_spans_m(hand(invalid=(M.PINKY_MCP,)))[0])


# ---- the provider contract --------------------------------------------------------------------------------------
class FakeHandProvider:
    """Stands in for `RgbdHandPoseProvider`: the estimator is fed hand estimates, not pixels, in these tests."""
    source = "head_rgbd"
    def __init__(self, estimates): self.q = list(estimates)
    def get_hand_pose(self, frame): return self.q.pop(0)


def provider_over(estimates, *, palm=None, tracking=None):
    p = build_rgbd_hand_pose_provider(hand_provider=FakeHandProvider(estimates), palm=palm,
                                      tracking=tracking or TrackingSupervisorConfig(recover_after_n_ok=1, min_features_ok=0),
                                      hand_pose_cfg=HandPoseSupervisorConfig(recover_stable_ms=0.0))
    return p


def feed(provider, estimates):
    return [provider.est.ingest_hand(e.timestamp_ns, i, e) and provider.ingest_estimate(provider.est._last)
            for i, e in enumerate(estimates)]


def test_provider_emits_a_wrist_pose_with_its_own_identity_and_evidence():
    ests = [hand(t_ns=i * NS // 30, xyz=(0.01 * i, 0, 0.5)) for i in range(4)]
    prov = provider_over(ests)
    poses = feed(prov, ests)
    wp = poses[-1]
    assert wp.source == "rgbd_hand_palm" and wp.health is TrackingHealth.OK
    assert wp.extra["pose_source"] == POSE_SOURCE and wp.extra["imu_used"] is False
    assert wp.extra["camera_mode"] == "fixed" and wp.extra["arm_pose_health"] == "ARM_POSE_OK"
    assert np.allclose(wp.T()[:3, :3], np.eye(3))                 # 3-DoF default
    assert wp.position_xyz_m[0] == pytest.approx(poses[0].position_xyz_m[0] + 0.03, abs=1e-6)


def test_lost_palm_holds_the_last_position_and_never_extrapolates():
    ests = [hand(t_ns=i * NS // 30, xyz=(0.01 * i, 0, 0.5)) for i in range(4)]
    ests.append(hand(t_ns=4 * NS // 30, invalid=tuple(range(21))))          # hand gone
    prov = provider_over(ests)
    poses = feed(prov, ests)
    assert poses[-1].health is TrackingHealth.LOST
    assert np.allclose(poses[-1].position_xyz_m, poses[-2].position_xyz_m)  # held, not integrated forward
    assert np.allclose(poses[-1].linear_velocity_xyz, 0)


def test_shared_tracking_supervisor_rejects_a_teleport():
    ests = [hand(t_ns=0, xyz=(0, 0, 0.5)), hand(t_ns=NS // 30, xyz=(0, 0, 0.5)),
            hand(t_ns=2 * NS // 30, xyz=(0.9, 0, 0.5))]
    prov = provider_over(ests, tracking=TrackingSupervisorConfig(max_jump_m=0.10, recover_after_n_ok=1, min_features_ok=0))
    assert feed(prov, ests)[-1].health is TrackingHealth.LOST


# ---- through the existing arm stack ------------------------------------------------------------------------------
def test_three_dof_relative_motion_reaches_the_robot_through_the_unchanged_retargeter():
    """Hand +10 cm along camera x -> robot moves scale * (mapped axis), orientation untouched."""
    mapper = HumanRobotFrameMapper(FrameMapperConfig(translation_axis_map=("z", "x", "y"), translation_sign=(1, -1, -1),
                                                    translation_scale=(0.4, 0.4, 0.4)))
    ests = [hand(t_ns=i * NS // 30, xyz=(0.10 * i, 0, 0.5)) for i in range(3)]
    prov = provider_over(ests)
    poses = feed(prov, ests)
    rt = RelativeSE3Retargeter(mapper)
    R0 = T_of((0.35, 0.0, 0.25), (0, 0, 30))
    rt.engage(poses[0], R0)
    out = rt.update(poses[2], R0)
    assert not out.held
    assert np.allclose(out.delta_H[:3, 3], [0.20, 0, 0])            # raw human motion, camera axes
    assert np.allclose(out.delta_R[:3, 3], [0.0, -0.08, 0.0])       # mapped: camera +x -> robot -y, x0.4
    assert np.allclose(out.delta_R[:3, :3], np.eye(3))              # 3-DoF: no rotation reaches the robot
    assert np.allclose(out.T_RB_RE_target[:3, :3], R0[:3, :3])      # robot keeps its ENGAGE orientation


def test_v1_virtual_stack_holds_on_loss_and_freezes_on_clutch():
    """The coordinator, safety pipeline and retargeter run unchanged on the RGB-D source; nothing is commanded."""
    ests = ([hand(t_ns=i * NS // 30, xyz=(0.01 * i, 0, 0.5)) for i in range(8)]
            + [hand(t_ns=8 * NS // 30, invalid=tuple(range(21)))]
            + [hand(t_ns=i * NS // 30, xyz=(0.01 * i, 0, 0.5)) for i in range(9, 14)])
    prov = provider_over(ests)
    robot = MockRebotController(T0=T_of((0.35, 0, 0.25)))
    safety = ArmSafetyPipeline(SafetyConfig(workspace=WorkspaceBox((0.2, -0.3, 0.1), (0.5, 0.3, 0.4)),
                                            max_lin_vel_m_s=0.2, max_lin_acc_m_s2=10.0))
    from ego_teleop.retarget.aero_retarget import AeroRetargeter
    co = TeleopCoordinator(wrist=prov, hand=None, arm_retargeter=RelativeSE3Retargeter(HumanRobotFrameMapper()),
                           aero_retargeter=AeroRetargeter(), safety=safety, robot=robot, aero=None,
                           cfg=CoordinatorConfig(hand_enabled=False, arm_enabled=False),
                           readiness=Readiness(True, True, True, True, True, True, True))
    cmds = []
    for i, e in enumerate(ests):
        prov.est.ingest_hand(e.timestamp_ns, i, e); prov.ingest_estimate(prov.est._last)
        if i == 2:
            for ev in ("sensors_ready", "calibrated", "vio_start", "vio_stable", "robot_ready", "arm", "start"):
                co.fire(ev, t_ns=e.timestamp_ns)
        if i == 11:
            co.fire("clutch", t_ns=e.timestamp_ns); frozen_target = co.arm_rt.last_target
        c = co.tick(e.timestamp_ns)
        if c is not None: cmds.append(c)
    assert robot.commands == []                                   # V1 never touches the robot
    lost = [c for c in cmds if c.hold_reason == "tracking_lost"]
    assert lost and all(not c.arm_sent for c in cmds)
    # clutch freezes the RETARGETER target; the safety limiter is still allowed to finish travelling to it, which is
    # why the frozen quantity is the target and not the commanded pose
    clutched = [c for c in cmds if c.hold_reason == "clutch"]
    assert clutched and np.allclose(co.arm_rt.last_target, frozen_target)
    moved = np.linalg.norm(clutched[-1].arm_target_xyz - clutched[0].arm_target_xyz)
    assert moved <= np.linalg.norm(frozen_target[:3, 3] - clutched[0].arm_target_xyz) + 1e-9


def test_anchor_workspace_is_a_small_box_around_the_engage_tcp_and_never_wider():
    from ego_teleop.config import RgbdArmPocCfg
    poc = RgbdArmPocCfg(safety=SafetyConfig(workspace=WorkspaceBox((0.0, -0.5, 0.0), (0.8, 0.5, 0.6))),
                        workspace_half_extent_m=(0.1, 0.1, 0.1))
    box = poc.anchor_workspace(T_of((0.4, 0.0, 0.05)))
    assert np.allclose(box.min_xyz, (0.3, -0.1, 0.0)) and np.allclose(box.max_xyz, (0.5, 0.1, 0.15))
    with pytest.raises(RuntimeError):
        poc.anchor_workspace(T_of((2.0, 0.0, 0.2)))               # anchor outside the standing limits


def test_head_mounted_camera_is_refused_by_config():
    from ego_teleop.config import RgbdArmPocCfg
    with pytest.raises(ValueError, match="camera_mode"):
        RgbdArmPocCfg(camera_mode="head_mounted")


# ---- storage ------------------------------------------------------------------------------------------------------
def test_poc_poses_never_land_in_the_wrist_vio_stream(tmp_path):
    ests = [hand(t_ns=i * NS // 30, xyz=(0.01 * i, 0, 0.5)) for i in range(3)]
    prov = provider_over(ests)
    lg = TeleopEpisodeLogger(tmp_path / "ep", metadata=dict(pose_source=POSE_SOURCE, imu_used=False))
    for wp in feed(prov, ests): lg.add_rgbd_hand_pose("right", wp)
    lg.add_rgbd_relative_pose("right", dict(t_ns=0, tcp_x=0.3, tcp_y=0.0, tcp_z=0.2))
    meta = lg.close()
    assert f"{STREAM_PREFIX}_right" in EPISODE_LAYOUT["raw"]
    df = pd.read_parquet(tmp_path / "ep/raw/rgbd_hand_pose_live_right.parquet")
    assert len(df) == 3 and set(df["arm_pose_health"]) == {"ARM_POSE_OK"} and "raw_qw" in df
    assert pd.read_parquet(tmp_path / "ep/raw/wrist_pose_live_right.parquet").empty   # the VIO stream stays untouched
    prov_meta = meta["pose_streams"][f"{STREAM_PREFIX}_right"]
    assert prov_meta["pose_source"] == "fixed_rgbd_hand" and prov_meta["imu_used"] is False
    assert prov_meta["causal"] is True and prov_meta["source_imu"] == ""
    assert json.loads((tmp_path / "ep/metadata.json").read_text())["pose_streams"] == meta["pose_streams"]
    assert len(pd.read_parquet(tmp_path / "ep/raw/rgbd_relative_pose_live_right.parquet")) == 1


# ---- the three headline metrics -------------------------------------------------------------------------------------
def _v0_frame(t_s, xyz, health=TrackingHealth.OK.value, v=(0, 0, 0)):
    return dict(t_ns=int(t_s * 1e9), x=xyz[0], y=xyz[1], z=xyz[2], vx=v[0], vy=v[1], vz=v[2],
                tracking_health=health, arm_pose_health="ARM_POSE_OK", arm_pose_reason="")


def _trial(jitter_m=0.0, drift_m=0.0, seed=0):
    """Still 0-3 s, a 3 cm excursion, still again 6-9 s with `drift_m` of residual error."""
    rng = np.random.default_rng(seed)
    rows = [_v0_frame(t, (jitter_m * rng.standard_normal(), 0.0, 0.5 + jitter_m * rng.standard_normal()))
            for t in np.arange(0, 3, 1 / 30)]
    rows += [_v0_frame(t, (0.03, 0, 0.5), v=(0.3, 0, 0)) for t in np.arange(3, 6, 1 / 30)]
    rows += [_v0_frame(t, (drift_m + jitter_m * rng.standard_normal(), 0.0, 0.5)) for t in np.arange(6, 9, 1 / 30)]
    return pd.DataFrame(rows)


def test_still_windows_are_found_without_labels():
    w = P.still_windows(_trial())
    assert len(w) == 2 and w[0][0] == pytest.approx(0.0) and w[1][1] == pytest.approx(8.97, abs=0.05)


def test_stationary_jitter_and_return_to_start_are_measured():
    df = _trial(jitter_m=0.002, drift_m=0.010)
    w = P.still_windows(df)
    stat = P.stationary_metrics(P._window(df, w[0]))
    assert 1.0 < stat["jitter_mm"]["p95"] < 8.0                    # ~2 mm of injected noise
    ret = P._return_to_start(df, w[0], w[-1])
    assert ret["position_error_mm"] == pytest.approx(10.0, abs=1.5)


def test_tracking_loss_rate_and_dropout_shape():
    df = _trial()
    df.loc[40:49, "tracking_health"] = TrackingHealth.LOST.value
    d = P.dropout_metrics(df)
    assert d["loss_rate"] == pytest.approx(10 / len(df), abs=1e-6)
    assert d["events"] == 1 and d["longest_s"] == pytest.approx(10 / 30, abs=0.02)  # to the next good frame


def test_segment_table_reports_the_axis_mapping_without_applying_it():
    df = _trial()
    mapper = HumanRobotFrameMapper(FrameMapperConfig(translation_axis_map=("z", "x", "y"), translation_sign=(1, -1, -1),
                                                    translation_scale=(0.4, 0.4, 0.4)))
    seg = P.segment_directions(df, {"right": (3.0, 5.9)}, mapper)["right"]
    assert seg["camera_displacement_mm"]["x"] == pytest.approx(0.0, abs=1.0)   # constant within the window
    df2 = pd.concat([_trial(), pd.DataFrame([_v0_frame(t, (0.10, 0, 0.5)) for t in np.arange(9, 11, 1 / 30)])])
    seg2 = P.segment_directions(df2, {"right": (6.0, 11.0)}, mapper)["right"]
    assert seg2["camera_displacement_mm"]["x"] > 50
    assert seg2["robot_displacement_mm"]["y"] < -20      # camera +x maps to robot -y at 0.4 scale


# ---- live HUD statistics (same metric definitions the report uses, so the HUD cannot flatter a take) -------------
def _hud_push(stats, t_s, xyz, *, health=TrackingHealth.OK, R=None, n_palm=5):
    from ego_teleop.hand3d.palm_pose import PalmPose
    from ego_teleop.tracking.interfaces import WristPose
    lost = health is TrackingHealth.LOST
    palm = PalmPose(ArmPoseHealth.LOST if lost else ArmPoseHealth.OK, "",
                    None if lost else np.asarray(xyz, np.float64), R, n_palm)
    wp = WristPose.from_T(int(t_s * 1e9), T_of(xyz), health=health)
    stats.push(t_s, wp, palm, 3.0)


def _hud_stats(window_s=3.0):
    from ego_teleop.tools.rgbd_arm_hud import LiveStats
    return LiveStats(window_s, 30.0)


def test_hud_reports_stationary_jitter_only_while_the_window_is_still():
    rng = np.random.default_rng(1)
    st = _hud_stats()
    for i in range(90):                                   # 3 s of 2 mm noise, hand still
        _hud_push(st, i / 30, (0.002 * rng.standard_normal(), 0.0, 0.5 + 0.002 * rng.standard_normal()))
    assert st.still and st.drift_mm < 10.0                # noise does not make a still hand look like motion ...
    assert st.peak_speed_mm_s > 20.0                       # ... even though its instantaneous speed is large
    rms, p95 = st.jitter_mm()
    assert 1.0 < rms < 5.0 and p95 < 10.0                 # would pass the provisional V2 gate
    for i in range(90, 150):                              # then move at ~0.3 m/s
        _hud_push(st, i / 30, (0.01 * (i - 90), 0.0, 0.5))
    assert not st.still                                    # jitter is withheld rather than reported from motion


def test_hud_counts_a_catastrophic_jump_and_never_across_a_gap():
    st = _hud_stats()
    for i in range(5): _hud_push(st, i / 30, (0.0, 0.0, 0.5))
    _hud_push(st, 5 / 30, (0.050, 0.0, 0.5))              # 50 mm in one frame
    assert st.jumps == 1 and st.jump_max_mm == pytest.approx(50.0, abs=0.5)
    _hud_push(st, 6 / 30, (0.050, 0.0, 0.5), health=TrackingHealth.LOST)
    _hud_push(st, 7 / 30, (0.400, 0.0, 0.5))              # reappears far away: a gap, not a 350 mm human motion
    assert st.jumps == 1 and st.jump_max_mm == pytest.approx(50.0, abs=0.5)


def test_hud_measures_lost_runs_including_one_still_in_progress():
    st = _hud_stats()
    for i in range(3): _hud_push(st, i / 30, (0, 0, 0.5))
    for i in range(3, 8): _hud_push(st, i / 30, (0, 0, 0.5), health=TrackingHealth.LOST)
    cur, mx = st.lost_ms()
    assert cur == pytest.approx(4 / 30 * 1000, abs=1) and mx == cur >= 0     # ongoing outage is not reported as 0
    _hud_push(st, 8 / 30, (0, 0, 0.5))
    cur, mx = st.lost_ms()
    assert cur == 0.0 and mx == pytest.approx(5 / 30 * 1000, abs=1) and len(st.lost_runs_ms) == 1


def test_hud_duty_counts_every_frame_and_orientation_is_measured_in_3dof():
    st = _hud_stats()
    for i in range(10): _hud_push(st, i / 30, (0, 0, 0.5), R=np.eye(3))
    for i in range(10, 20): _hud_push(st, i / 30, (0, 0, 0.5), health=TrackingHealth.LOST)
    assert st.duty == pytest.approx(0.5)
    for i, ang in enumerate(np.linspace(0, 6, 20)):        # 6 deg of wobble while "stationary"
        _hud_push(st, (20 + i) / 30, (0, 0, 0.5), R=Rotation.from_euler("x", ang, degrees=True).as_matrix())
    assert 1.0 < st.orientation_jitter_deg() < 6.0
    assert st.orientation_trace_deg().size > 5             # the raw series, not just the percentile


def test_hud_protocol_banner_tracks_the_segment_boundaries():
    from ego_teleop.tools.rgbd_arm_hud import PROTOCOL, banner_for, segment_at
    assert [x[0] for x in PROTOCOL][:2] == ["stationary", "slow_xyz"]
    assert segment_at(0.0)[0] == "stationary" and segment_at(25.0)[0] == "palm_rotation_only"
    assert segment_at(59.9)[0] == "return_stationary" and segment_at(60.0) is None
    assert "stationary" in banner_for(1.0) and "slow_xyz" in banner_for(1.0)     # current + next


def test_protocol_segment_names_are_the_same_in_all_three_places():
    """The recorder says these names out loud and writes them into `--segment`; the report finds the stationary and
    return windows BY NAME. A rename in one place NaNs a headline number instead of erroring, so it is pinned here.

    `~/orbbec_recorder.py` is a standalone script in the user's home (its own venv, needs the Orbbec SDK), so this
    skips rather than fails when it is not present."""
    import importlib.util
    from ego_teleop.tools.p1_rgbd_arm import PROTOCOL_60S, SEGMENT_ARGS
    from ego_teleop.tools.rgbd_arm_hud import PROTOCOL
    assert list(PROTOCOL) is not None and list(PROTOCOL) == list(PROTOCOL_60S)   # HUD imports it, cannot drift
    rec = Path("/Users/jeonghwanlee/orbbec_recorder.py")
    if not rec.exists(): pytest.skip("orbbec_recorder.py not present")
    spec = importlib.util.spec_from_file_location("orec_under_test", rec)
    try:
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    except Exception as e:
        pytest.skip(f"recorder not importable here: {e}")
    assert list(m.POC_PROTOCOL) == list(PROTOCOL_60S)
    assert m.poc_segment_args() == SEGMENT_ARGS


def test_report_finds_the_return_window_from_the_recorder_segment_names():
    """The exact `--segment` string the recorder prints must produce a return-to-start number, not a NaN."""
    from ego_teleop.tools.p1_rgbd_arm import PROTOCOL_60S
    df = _trial(jitter_m=0.001, drift_m=0.012)
    df["t_ns"] = (np.linspace(0, 60, len(df)) * 1e9).astype("int64")        # relabel onto a 0-60 s timeline
    segs = {n: (float(a), float(b)) for n, a, b, _ in PROTOCOL_60S}
    w = P._window(df, segs["stationary"]), P._window(df, segs["return_stationary"])
    assert len(w[0]) > 3 and len(w[1]) > 3
    starts = [n for n in segs if n.startswith(("stationary", "static", "still"))]
    ends = [n for n in segs if n.startswith("return")]
    assert starts == ["stationary"] and ends == ["return_stationary"]
    ret = P._return_to_start(df, segs[starts[0]], segs[ends[-1]])
    assert np.isfinite(ret["position_error_mm"])


# ---- V0 post-hoc metrics (the numbers the V2 decision is made from) ---------------------------------------------
from ego_teleop.tools import rgbd_arm_metrics as Mx


def _take(n=1800, rate=30.0):
    """A synthetic 60 s take on the canonical protocol timeline, all frames good and stationary."""
    t = np.arange(n) / rate
    return pd.DataFrame(dict(t_ns=(t * 1e9).astype("int64"), x=0.0, y=0.0, z=0.5,
                             tracking_health=TrackingHealth.OK.value, arm_pose_health=Mx.ARM_OK,
                             arm_pose_reason="", raw_qx=0.0, raw_qy=0.0, raw_qz=0.0, raw_qw=1.0))


def test_a_run_is_measured_to_the_next_good_frame():
    df = _take(20)
    mask = np.zeros(20, bool); mask[5:15] = True                  # 10 frames lost at 30 Hz
    r = Mx.runs_ms(df, mask)
    assert len(r) == 1 and r[0] == pytest.approx(10 / 30 * 1000, abs=1)   # 333 ms of hold, not 300


def test_a_run_still_open_at_the_end_of_the_take_is_counted():
    df = _take(20)
    mask = np.zeros(20, bool); mask[15:] = True
    r = Mx.runs_ms(df, mask)
    assert len(r) == 1 and r[0] == pytest.approx(4 / 30 * 1000, abs=1)


def test_jump_is_not_measured_across_a_tracking_gap():
    df = _take(10)
    df.loc[5, "tracking_health"] = TrackingHealth.LOST.value
    df.loc[6:, "x"] = 0.35                                        # reappears 35 cm away
    j = Mx.jumps(df)
    assert j["count"] == 0 and j["max_mm"] < 1.0                  # a loss, already counted as a loss
    df2 = _take(10); df2.loc[5:, "x"] = 0.05                      # a genuine 50 mm one-frame step
    j2 = Mx.jumps(df2)
    assert j2["count"] == 1 and j2["max_mm"] == pytest.approx(50.0, abs=0.5)
    assert j2["over"][0]["t_s"] == pytest.approx(5 / 30, abs=0.02)


def test_jitter_is_withheld_when_the_window_moved():
    rng = np.random.default_rng(4)
    still = _take(90); still["x"] = 0.002 * rng.standard_normal(90); still["z"] = 0.5 + 0.002 * rng.standard_normal(90)
    j = Mx.jitter(still)
    assert j["still"] and 1.0 < j["rms_mm"] < 5.0
    moving = _take(90); moving["x"] = np.linspace(0, 0.30, 90)
    jm = Mx.jitter(moving)
    assert not jm["still"] and jm["drift_mm"] > 10.0
    m = Mx.take_metrics(moving, {"stationary": (0.0, 3.0)})
    assert not np.isfinite(m["stationary_xyz_rms_mm"])            # the gate reads "not measured", never a motion number


def test_translation_leak_is_reported_against_its_rotation_stimulus():
    df = _take(300)
    ang = np.linspace(-45, 45, 300).reshape(-1, 1)        # scipy wants (N, n_axes), not (N,)
    q = Rotation.from_euler("x", ang, degrees=True).as_quat()
    df[["raw_qx", "raw_qy", "raw_qz", "raw_qw"]] = q
    df["x"] = 0.02 * np.sin(np.linspace(0, 3, 300))               # 20 mm of wander while rotating ~90 deg
    leak = Mx.translation_leak(df)
    assert 10.0 < leak["p95_mm"] < 30.0 and leak["rotation_deg"] > 60.0
    assert leak["mm_per_10deg"] < 5.0                              # small leak per unit rotation = origin is fine
    tight = _take(300)
    tight[["raw_qx", "raw_qy", "raw_qz", "raw_qw"]] = Rotation.from_euler(
        "x", np.linspace(-3, 3, 300).reshape(-1, 1), degrees=True).as_quat()
    tight["x"] = 0.02 * np.sin(np.linspace(0, 3, 300))             # the SAME 20 mm under a 6 deg wobble
    lt = Mx.translation_leak(tight)
    assert lt["p95_mm"] == pytest.approx(leak["p95_mm"], rel=1e-9)  # identical millimetres ...
    assert lt["mm_per_10deg"] > 10 * leak["mm_per_10deg"]           # ... and an order-of-magnitude worse verdict


def test_gate_levels_follow_the_provisional_thresholds_and_never_default_to_green():
    g = Mx.gate_verdict(dict(arm_pose_valid_duty=0.97, stationary_xyz_rms_mm=4.0, stationary_xyz_p95_mm=9.0,
                             return_to_start_translation_mm=12.0, catastrophic_jump_count=0, long_tracking_loss_count=0))
    assert {v["level"] for v in g.values()} == {"green"}
    y = Mx.gate_verdict(dict(arm_pose_valid_duty=0.92, stationary_xyz_rms_mm=7.0, stationary_xyz_p95_mm=15.0,
                             return_to_start_translation_mm=20.0, catastrophic_jump_count=1, long_tracking_loss_count=1))
    assert {v["level"] for v in y.values()} == {"yellow"}
    r = Mx.gate_verdict(dict(arm_pose_valid_duty=0.5, stationary_xyz_rms_mm=30.0, stationary_xyz_p95_mm=40.0,
                             return_to_start_translation_mm=90.0, catastrophic_jump_count=5, long_tracking_loss_count=9))
    assert {v["level"] for v in r.values()} == {"red"}
    n = Mx.gate_verdict({})
    assert {v["level"] for v in n.values()} == {"none"}            # unmeasured is not a pass


def test_the_worst_take_decides_the_gate():
    good = dict(gate=Mx.gate_verdict(dict(arm_pose_valid_duty=0.99, stationary_xyz_rms_mm=2.0, stationary_xyz_p95_mm=4.0,
                                          return_to_start_translation_mm=5.0, catastrophic_jump_count=0, long_tracking_loss_count=0)))
    bad = dict(gate=Mx.gate_verdict(dict(arm_pose_valid_duty=0.99, stationary_xyz_rms_mm=2.0, stationary_xyz_p95_mm=4.0,
                                         return_to_start_translation_mm=5.0, catastrophic_jump_count=3, long_tracking_loss_count=0)))
    agg = Mx.aggregate([good, good, bad])
    assert agg["verdict"] == "red" and agg["gate"]["catastrophic_jump_count"]["level"] == "red"
    assert Mx.aggregate([good, good, good])["verdict"] == "green"


def test_take_metrics_runs_end_to_end_on_the_canonical_segments():
    from ego_teleop.tools.p1_rgbd_arm import PROTOCOL_60S
    rng = np.random.default_rng(7)
    df = _take(1800)
    df["x"] = 0.0015 * rng.standard_normal(1800)
    df.loc[300:600, "x"] = np.linspace(0, 0.1, 301)               # slow_xyz: out to 10 cm ...
    df.loc[600:1200, "x"] = 0.1
    df.loc[1200:1500, "x"] = np.linspace(0.1, 0.0, 301)           # ... and continuously back (no teleport)
    df.loc[1500:, "x"] = 0.001 * rng.standard_normal(300)         # return + stationary
    segs = {n: (float(a), float(b)) for n, a, b, _ in PROTOCOL_60S}
    m = Mx.take_metrics(df, segs, mapper=HumanRobotFrameMapper(), name="t1")
    assert m["arm_pose_valid_duty"] == 1.0 and m["catastrophic_jump_count"] == 0
    assert m["stationary"]["still"] and m["stationary_xyz_rms_mm"] < 5.0
    assert m["return_to_start_translation_mm"] < 5.0
    assert set(m["segments"]) == set(segs) and "translation_leak" in m["segments"]["palm_rotation_only"]
    assert m["segments"]["slow_xyz"]["displacement"]["camera_mm"]["x"] > 50
    assert m["gate"]["arm_pose_valid_duty"]["level"] == "green"


# ---- fixed colour exposure (experimental control; measured 2026-09-11: reopening the pipeline re-enables AE) -----
class _FakeDevice:
    """The Orbbec property API, with the real Gemini 336 ranges measured on hardware."""
    def __init__(self, *, gain_rw=True): self.auto = True; self.exp = 156; self.gain = 16; self.rw_gain = gain_rw; self.writes = []
    def is_property_supported(self, prop, perm):
        from pyorbbecsdk import OBPropertyID
        return self.rw_gain or prop is not OBPropertyID.OB_PROP_COLOR_GAIN_INT
    def get_bool_property(self, p): return self.auto
    def get_int_property(self, p):
        from pyorbbecsdk import OBPropertyID
        return self.exp if p is OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT else self.gain
    def set_bool_property(self, p, v): self.auto = bool(v); self.writes.append(("ae", bool(v)))
    def set_int_property(self, p, v):
        from pyorbbecsdk import OBPropertyID
        if p is OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT: self.exp = int(v); self.writes.append(("exp", int(v)))
        else: self.gain = int(v); self.writes.append(("gain", int(v)))


def test_exposure_untouched_by_default():
    from ego_teleop.hand3d.head_camera import ColorExposure, apply_color_exposure
    pytest.importorskip("pyorbbecsdk")
    d = _FakeDevice()
    for cfg in (None, ColorExposure()):
        a = apply_color_exposure(d, cfg)
        assert d.writes == [] and a["color_control"] == "untouched" and a["color_auto_exposure"] is True


def test_exposure_is_fixed_and_read_back_from_the_device():
    from ego_teleop.hand3d.head_camera import ColorExposure, apply_color_exposure
    pytest.importorskip("pyorbbecsdk")
    d = _FakeDevice()
    a = apply_color_exposure(d, ColorExposure(auto=False, exposure=120, gain=16))
    assert d.writes == [("ae", False), ("exp", 120), ("gain", 16)]
    assert a["color_control"] == "manual" and a["color_auto_exposure"] is False and a["color_exposure"] == 120


def test_exposure_without_an_explicit_auto_flag_is_refused():
    from ego_teleop.hand3d.head_camera import ColorExposure, apply_color_exposure
    pytest.importorskip("pyorbbecsdk")
    with pytest.raises(ValueError, match="auto"):
        apply_color_exposure(_FakeDevice(), ColorExposure(exposure=120))      # AE on -> the device ignores the value
    with pytest.raises(ValueError, match="auto=True"):
        apply_color_exposure(_FakeDevice(), ColorExposure(auto=True, exposure=120))


def test_gain_is_skipped_when_unsupported_but_exposure_still_gets_fixed():
    from ego_teleop.hand3d.head_camera import ColorExposure, apply_color_exposure
    pytest.importorskip("pyorbbecsdk")
    d = _FakeDevice(gain_rw=False)
    a = apply_color_exposure(d, ColorExposure(auto=False, exposure=100, gain=24))
    assert d.writes == [("ae", False), ("exp", 100)] and a["color_exposure"] == 100
    assert a["color_gain"] == 16                                              # the ACTUAL value, not the request


# ---- bounded MediaPipe leak (macOS GPU graph leaks ~3.4 MB/frame; measured 2026-09-11) --------------------------
def test_recycle_schedule_is_off_by_default_and_counts_from_construction():
    """The decision is a pure method, so it is testable without constructing a landmarker (which needs the GPU)."""
    from ego_collector.hands.mediapipe_tracker import HandLandmarker
    lm = HandLandmarker.__new__(HandLandmarker)          # no __init__: this tests the schedule, not MediaPipe
    lm.recycle_every, lm.recycles, lm._frames = 0, 0, 0
    for i in range(2000):
        lm._frames = i
        assert not lm.due_for_recycle()                  # default off: an existing caller is unchanged
    lm.recycle_every = 300
    due = [i for i in range(1200) if (setattr(lm, "_frames", i) or lm.due_for_recycle())]
    assert due == [300, 600, 900]                        # never on frame 0, then every N
