"""Live MASt3R wrist frontend: the adapter, not the tracker (docs/ego_teleop/MAST3R_LIVE_PROTOCOL.md).

There is no GPU here and no MASt3R-Fusion, so what is testable is exactly what this repo owns — and it happens to
be where the failures that would hurt a real arm live:

  * the wire format cannot drift between the two halves, because the server ships in this repo and the client in
    the same commit but they run on different machines and are deployed by scp;
  * a busy server must cost DROPPED FRAMES, never a growing queue of stale poses (spec sections 17, 25);
  * a returned pose must carry ITS OWN timestamp, so a stalled link ages into DEGRADED/LOST instead of looking
    fresh;
  * a global/loop-closure correction must arrive FLAGGED, so `LocalPoseContinuity` absorbs it and the arm does not
    jump (section 3) — and an unflagged jump must NOT be absorbed;
  * a missing server, a refused handshake and a dropped link must all produce LOST / BackendUnavailable, never an
    invented pose;
  * fisheye rectification must be the same call the offline EuRoC export makes, or live and replay are two cameras.
"""
from __future__ import annotations
import importlib.util
import socket
import time
from pathlib import Path
import numpy as np
import pytest
from handumi_collector.pose.estimator import BackendUnavailable, TrackingState

from ego_teleop.tracking.backends.mast3r_live import EchoPoseServer, Mast3rLiveBackend
from ego_teleop.tracking.backends import mast3r_live as client
from ego_teleop.tracking.interfaces import TrackingHealth
from ego_teleop.tracking.local_pose import LocalPoseConfig, LocalPoseContinuity
from ego_teleop.tracking.wrist_pose_provider import EstimatorWristPoseProvider, TrackingSupervisor, TrackingSupervisorConfig
from ego_teleop.transforms.se3 import make_T

REPO = Path(__file__).resolve().parents[2]
INTR = dict(model="pinhole", fx=300.0, fy=300.0, cx=320.0, cy=240.0, width=640, height=480)
IMG = np.zeros((480, 640, 3), np.uint8)


def server_module():
    """The GPU-side script, imported as a module. It must import with no CUDA and no mast3r_fusion present."""
    spec = importlib.util.spec_from_file_location("mast3r_live_server", REPO / "scripts/mast3r_live_server.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def connected(**kw):
    srv = EchoPoseServer(**kw)
    b = Mast3rLiveBackend(host=srv.host, port=srv.port)
    b.initialize(intrinsics=INTR, T_camera_imu=np.eye(4), image_size=(640, 480))
    return b, srv


def pump(b, target: int, timeout_s: float = 3.0) -> bool:
    end = time.monotonic() + timeout_s
    while b.counters["poses"] < target and time.monotonic() < end: time.sleep(0.002)
    return b.counters["poses"] >= target


# ---- the two halves are one protocol -----------------------------------------------------------------------
def test_client_and_server_agree_on_the_wire_format():
    s = server_module()
    assert (s.MAGIC, s.PROTOCOL_VERSION, s.HEAD.format) == (client.MAGIC, client.PROTOCOL_VERSION, client.HEAD.format)


def test_the_server_script_imports_without_cuda_or_mast3r():
    assert server_module().main(["--dry-run"]) == 0


def test_a_version_mismatch_is_refused_not_tolerated():
    srv = EchoPoseServer(version=client.PROTOCOL_VERSION + 1)
    with pytest.raises(BackendUnavailable, match="protocol mismatch"):
        Mast3rLiveBackend(host=srv.host, port=srv.port).initialize(intrinsics=INTR)
    srv.close()


# ---- nothing is ever faked ----------------------------------------------------------------------------------
def test_no_server_is_backend_unavailable_not_a_fake_pose():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()   # a port with nothing on it
    with pytest.raises(BackendUnavailable, match="no MASt3R live server"):
        Mast3rLiveBackend(host="127.0.0.1", port=port, connect_timeout_s=0.5).initialize(intrinsics=INTR)


def test_missing_intrinsics_is_refused():
    with pytest.raises(BackendUnavailable, match="intrinsics"):
        Mast3rLiveBackend().initialize(intrinsics=None)


def test_a_dropped_link_makes_every_estimate_lost():
    b, srv = connected(drop_after=2)
    for i in range(6):
        est = b.push_image(i * 33_000_000, i, IMG)
        time.sleep(0.03)
    assert b.push_image(9 * 33_000_000, 9, IMG).tracking_state is TrackingState.LOST
    assert b.get_quality()["error"]
    b.close(); srv.close()


# ---- latest frame, never a queue (sections 17, 25) -----------------------------------------------------------
def test_a_busy_server_costs_dropped_frames_not_a_backlog():
    b, srv = connected(delay_s=0.15)
    for i in range(12):
        b.push_image(i * 33_000_000, i, IMG)
        time.sleep(0.01)                                   # 100 Hz of offers against a 6.7 Hz server
    q = b.get_quality()
    assert q["frames_dropped_busy"] >= 6, q                # most offers are dropped ...
    assert q["frames_sent"] <= 4, q                        # ... and only a couple ever go on the wire
    b.close(); srv.close()


def test_a_stale_pose_keeps_its_own_timestamp_and_ages_into_lost():
    """The failure §25 is written against: a backlog would keep the pose looking fresh while the hand moved on."""
    b, srv = connected()
    assert pump(b, 1) or b.push_image(0, 0, IMG) is not None
    b.push_image(0, 0, IMG); pump(b, 1)
    est = b.get_pose()
    assert est is not None and est.timestamp_ns == 0        # the pose is stamped at ITS frame, not at "now"

    prov = EstimatorWristPoseProvider(b, T_H_C=np.eye(4),
                                      supervisor=TrackingSupervisor(TrackingSupervisorConfig(
                                          degraded_after_ms=80, lost_after_ms=250, recover_after_n_ok=1)))
    prov.ingest_estimate(est)
    assert prov.get_pose(int(0.05e9) // 1000 * 1000).health is TrackingHealth.OK
    assert prov.get_pose(int(0.1e9)).health is TrackingHealth.DEGRADED
    assert prov.get_pose(int(0.5e9)).health is TrackingHealth.LOST
    b.close(); srv.close()


# ---- global corrections are announced, and only announced ones are absorbed (section 3) ----------------------
def test_a_flagged_map_update_is_absorbed_and_does_not_move_the_local_pose():
    b, srv = connected(map_update_every=3)
    for i in range(9):
        b.push_image(i * 33_000_000, i, IMG); pump(b, i + 1)
    assert b.counters["map_updates"] >= 2, b.get_quality()

    b.close(); srv.close()

    # A hand that is not moving, and a global BA that suddenly re-seats the whole map by 12 cm. The MAP pose jumps;
    # the pose the robot consumes must not, because nothing about the hand changed.
    local = LocalPoseContinuity(LocalPoseConfig(absorb_flagged_only=True))
    p_local = []
    for i in range(8):
        shift = np.array([0.12, -0.04, 0.03]) if i >= 4 else np.zeros(3)     # the re-optimisation lands at i == 4
        T = make_T(np.eye(3), shift)
        p_local.append(local.update(T, extra=dict(map_update=True, map_update_m=0.13) if i == 4 else {})[:3, 3].copy())
    steps = np.linalg.norm(np.diff(np.asarray(p_local), axis=0), axis=1)
    assert steps.max() < 1e-9, steps                  # the arm never saw the loop closure
    assert local.stats()["map_absorbed"] == 1
    assert local.stats()["map_correction_cum_m"] == pytest.approx(np.linalg.norm([0.12, -0.04, 0.03]), rel=1e-6)


def test_an_unflagged_jump_is_not_absorbed_so_a_tracking_failure_stays_visible():
    local = LocalPoseContinuity(LocalPoseConfig(absorb_flagged_only=True))
    local.update(make_T(np.eye(3), [0.0, 0.0, 0.0]), extra={})
    out = local.update(make_T(np.eye(3), [0.4, 0.0, 0.0]), extra={})      # a 40 cm step, announced by nobody
    assert np.linalg.norm(out[:3, 3]) > 0.3          # it survives, so TrackingSupervisor.max_jump_m can grade it
    assert local.stats()["map_absorbed"] == 0


# ---- fisheye rectification is the export's call, once ---------------------------------------------------------
def test_the_client_rectifies_kannala_brandt_and_tells_the_server_pinhole():
    from ego_teleop.transforms.fisheye import make_rectifier
    K = np.array([[791.28, 0, 956.9], [0, 793.15, 523.48], [0, 0, 1.0]])
    D = np.array([-0.0522, 0.0287, -0.016, 0.0034])
    srv = EchoPoseServer()
    b = Mast3rLiveBackend(host=srv.host, port=srv.port)
    b.initialize(intrinsics=dict(model="kannala_brandt", K=K.tolist(), D=D.tolist(),
                                 image_size=[1920, 1080], downscale=2))
    ref = make_rectifier(K, D, (1920, 1080), downscale=2)
    assert b.rect.size == ref.size == (960, 540)
    assert np.allclose(b.rect.K_new, ref.K_new)
    r = b.rect.intrinsics()
    assert 0 < r["cx"] < r["width"] and 0 < r["cy"] < r["height"], r     # a doubly-applied downscale lands outside
    b.close(); srv.close()


def test_downscale_is_applied_exactly_once():
    from ego_teleop.transforms.fisheye import make_rectifier
    K = np.array([[800.0, 0, 960.0], [0, 800.0, 540.0], [0, 0, 1.0]]); D = np.zeros(4)
    a = make_rectifier(K, D, (1920, 1080), downscale=1)
    b = make_rectifier(K, D, (1920, 1080), downscale=2)
    assert a.size == (1920, 1080) and b.size == (960, 540)
    assert b.K_new[0, 0] == pytest.approx(a.K_new[0, 0] / 2, rel=1e-6)


def test_a_frame_of_the_wrong_size_is_refused_rather_than_silently_remapped():
    K = np.array([[800.0, 0, 960.0], [0, 800.0, 540.0], [0, 0, 1.0]])
    srv = EchoPoseServer(); b = Mast3rLiveBackend(host=srv.host, port=srv.port)
    b.initialize(intrinsics=dict(model="kannala_brandt", K=K.tolist(), D=np.zeros(4).tolist(),
                                 image_size=[1920, 1080], downscale=2))
    est = b.push_image(0, 0, np.zeros((480, 640, 3), np.uint8))
    assert est.tracking_state is TrackingState.LOST
    assert "rectifier was built for" in b.get_quality()["error"]
    b.close(); srv.close()


# ---- the appendable IMU pool keeps the offline pool's contract -------------------------------------------------
def test_the_live_imu_pool_matches_the_offline_pool_units_and_invariants():
    pool = server_module().AppendableImuPool()
    for k in range(200): pool.append(k * 0.0024, (0.1, 0.2, 0.3), (0.0, 0.0, 9.81))
    rec = pool.get_records(0.005, 0.02)
    assert len(rec) >= 2 and rec[0][0] == 0.005 and rec[-1][1] == pytest.approx(0.02)
    assert rec[0][2][0] == pytest.approx(0.1 * 180.0 / np.pi)        # IMUPool hands the factor graph DEGREES/s
    assert rec[0][2][3:].tolist() == [0.0, 0.0, 9.81]                # accel stays m/s^2


def test_the_imu_pool_refuses_a_range_it_does_not_cover_instead_of_truncating():
    pool = server_module().AppendableImuPool()
    for k in range(20): pool.append(k * 0.0024, (0.0, 0.0, 0.0), (0.0, 0.0, 9.81))
    with pytest.raises(LookupError):
        pool.get_records(0.001, 5.0)


def test_the_imu_pool_drops_non_monotonic_samples_rather_than_raising_mid_session():
    pool = server_module().AppendableImuPool()
    pool.append(1.0, (0, 0, 0), (0, 0, 9.81))
    pool.append(0.5, (0, 0, 0), (0, 0, 9.81))       # a late/duplicated packet
    pool.append(1.5, (0, 0, 0), (0, 0, 9.81))
    assert pool.dropped_nonmonotonic == 1 and len(pool.time) == 2


# ---- the config layer hands a live backend its camera ----------------------------------------------------------
def test_the_live_branch_initializes_the_estimator_with_the_versioned_calibration():
    """Before this existed the live rig built a backend and never called initialize(), so OpenVINS would have run
    on default intrinsics and mast3r_live would have had no rectifier — a systematic error nothing downstream sees."""
    from ego_teleop.config import load_teleop_cfg
    cfg = load_teleop_cfg().fused_wrist
    srv = EchoPoseServer(); b = Mast3rLiveBackend(host=srv.host, port=srv.port)
    intr = cfg.initialize_estimator(b, downscale=2)
    assert intr["model"] == "kannala_brandt"
    assert intr["image_size"] == [1920, 1080]        # FULL resolution: the factor is passed, not pre-applied
    assert intr["downscale"] == 2 and intr["width"] == 960
    assert b.rect is not None and b.rect.size == (960, 540)
    b.close(); srv.close()


# ---- the first-hardware-run gate, over the WHOLE path ----------------------------------------------------------
# These two bugs were each individually silent and together made the gate unmeasurable. They are regression-tested
# through backend -> EstimatorWristPoseProvider -> FusedWristPoseProvider, not against LocalPoseContinuity alone,
# because that is the layer both of them lived in.
def _vi_provider(max_jump_m=0.15):
    from ego_teleop.tracking.wrist_pose_provider import TrackingSupervisorConfig as C
    est = type("E", (), {"info": type("i", (), {"name": "probe"}), "get_quality": lambda s: {}})()
    return EstimatorWristPoseProvider(est, T_H_C=np.eye(4),
                                      supervisor=TrackingSupervisor(C(recover_after_n_ok=1, max_jump_m=max_jump_m)))


def test_the_backends_extra_survives_into_the_wrist_pose():
    """It did not. `ingest_estimate` rebuilt `extra` from two fields, so `map_update` was set, sent, parsed and
    then dropped one layer above `LocalPoseContinuity` — the only code that reads it."""
    from handumi_collector.pose.estimator import PoseEstimate
    wp = _vi_provider().ingest_estimate(
        PoseEstimate(1000, 0, make_T(np.eye(3), [0.1, 0, 0]), TrackingState.TRACKING,
                     extra=dict(map_update=True, map_update_m=0.08, backend="mast3r_live")))
    assert wp.extra.get("map_update") is True
    assert wp.extra.get("map_update_m") == 0.08
    assert "num_features" in wp.extra                      # the provider's own fields are still there


def test_an_announced_map_update_is_not_graded_as_a_tracking_jump():
    """It was. A loop closure bigger than `max_jump_m` was graded LOST, `_push_vi` returned early, and the absorber
    never saw it — leaving the correction permanently in the commanded trajectory."""
    from handumi_collector.pose.estimator import PoseEstimate
    p = _vi_provider(max_jump_m=0.15)
    p.ingest_estimate(PoseEstimate(0, 0, make_T(np.eye(3), [0, 0, 0]), TrackingState.TRACKING))
    flagged = p.ingest_estimate(PoseEstimate(33_000_000, 1, make_T(np.eye(3), [0.40, 0, 0]),
                                             TrackingState.TRACKING, extra=dict(map_update=True, map_update_m=0.40)))
    assert flagged.health is TrackingHealth.OK, "an announced correction must reach the absorber"

    q = _vi_provider(max_jump_m=0.15)                      # ... and an UNANNOUNCED one must still be caught
    q.ingest_estimate(PoseEstimate(0, 0, make_T(np.eye(3), [0, 0, 0]), TrackingState.TRACKING))
    unflagged = q.ingest_estimate(PoseEstimate(33_000_000, 1, make_T(np.eye(3), [0.40, 0, 0]), TrackingState.TRACKING))
    assert unflagged.health is TrackingHealth.LOST


def test_the_map_may_move_and_the_control_pose_may_not():
    """THE gate for the first hardware run, end to end: a 40 cm global re-seat moves T_map by 400 mm and
    T_local_control by nothing, while ordinary hand motion passes through both untouched."""
    from handumi_collector.pose.estimator import PoseEstimate
    from ego_teleop.tracking.fused_wrist import FusedWristConfig, FusedWristPoseProvider
    vi, fused = _vi_provider(), FusedWristPoseProvider(FusedWristConfig(mode="vi_only"))
    local, mapp = [], []
    for i in range(8):
        x = 0.001 * i + (0.40 if i >= 4 else 0.0)          # the re-optimisation lands at frame 4
        extra = dict(map_update=True, map_update_m=0.40, visual_raw_x=0.001 * i) if i == 4 else {}
        vi_wp = vi.ingest_estimate(PoseEstimate(i * 33_000_000, i, make_T(np.eye(3), [x, 0, 0]),
                                                TrackingState.TRACKING, extra=extra))
        fused.push_vi(vi_wp)
        out = fused.get_pose(i * 33_000_000)
        local.append(out.position_xyz_m[0]); mapp.append(x)

    d_local = np.abs(np.diff(local)) * 1000.0
    d_map = np.abs(np.diff(mapp)) * 1000.0
    # 400 mm of re-optimisation plus the 1 mm the hand actually moved on that frame
    assert d_map[3] == pytest.approx(401.0, abs=0.1), "the map is supposed to move"
    assert d_local[3] < 0.001, f"T_local_control jumped {d_local[3]:.3f} mm at the map update"
    assert np.allclose(np.delete(d_local, 3), 1.0, atol=1e-6), "ordinary motion must pass through unchanged"
    assert fused.stats()["local_pose"]["map_absorbed"] == 1


def test_the_hud_reports_the_gate_as_a_number():
    from ego_teleop.tools import f5_teleop_hud as H
    st = H.HudState(meta=dict(stage="virtual", mode="vi_only", side="right", rate_hz=50.0), rate_hz=50.0)
    for i in range(6):
        row = dict(s=i * 0.02, x=0.001 * i, y=0.0, z=0.0, vi_x=0.001 * i, vi_y=0.0, vi_z=0.0,
                   vi_map_x=0.001 * i + (0.4 if i >= 3 else 0.0), vi_map_y=0.0, vi_map_z=0.0,
                   tracking_health="TRACKING_OK")
        if i == 3: row.update(vi_map_update=True, vi_map_update_m=0.4, map_correction_m=0.4, d_visual_raw_m=0.001)
        st.push(row, {})
    g = st.snapshot()["map_gate"]
    assert g["map_updates"] == 1
    # the gate as a number on the page: the local step at a map update is the hand's own millimetre, NOT the
    # 400 mm the map moved. A regression puts the map step in this field.
    assert g["max_local_step_mm"] < 5.0, g
    assert g["last"]["d_map_mm"] == pytest.approx(401.0, abs=0.5)
    assert g["last"]["d_local_mm"] == pytest.approx(1.0, abs=0.5)
    assert g["last"]["absorbed_mm"] == pytest.approx(400.0, abs=0.5)
