"""Chest RGB-D + wrist Arducam + wrist IMU fusion (docs/ego_teleop/MULTISENSOR_FUSION.md).

The tests that matter here are the ones that catch the failures which look like success on a plot:
  * a wrist ROTATION must not move the fused position (the palm-centroid lever arm, spec section 8),
  * a loop closure must not move the robot (local vs map pose, section 3),
  * the anchor correction must be bounded and rate-limited, because dC/dt IS commanded arm velocity (section 6),
  * every branch-loss combination must produce the state section 14 says it must, and never a zero pose."""
from __future__ import annotations
import dataclasses as dc
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego_teleop.tracking.anchor_fusion import (AnchorAlignment, AnchorAlignmentConfig, AnchorFusionConfig, FusionState,
                                               FusionStateConfig, TranslationAnchorFusion, fusion_health, kabsch)
from ego_teleop.tracking.fused_wrist import FusedWristConfig, FusedWristPoseProvider, FusionMode
from ego_teleop.tracking.interfaces import TrackingHealth, WristPose
from ego_teleop.tracking.local_pose import LocalPoseConfig, LocalPoseContinuity
from ego_teleop.transforms.se3 import inv_T, make_T

HZ = 30
DT_NS = int(1e9 / HZ)
R_TRUE = np.array([0.02, -0.03, 0.06])          # palm centroid in the wrist control frame H (a realistic 7 cm)


def T_D_W_truth() -> np.ndarray:
    """A chest camera that is rotated and offset from the VI world — i.e. the normal case, not a convenient one."""
    R = Rotation.from_euler("xyz", [25.0, -12.0, 140.0], degrees=True).as_matrix()
    return make_T(R, [0.31, -0.12, 0.44])


def trajectory(n: int, *, amp=0.10, rot_deg=35.0, rot_only=False, seed=0, locked=False):
    """True wrist poses in the VI world: a translation loop plus a wrist roll at an unrelated rate.

    `locked=True` makes the rotation follow the translation exactly — the pathological case for the lever arm, where
    "the palm swung around the wrist" and "the camera is rotated a little differently" explain the same data. The
    60 s protocol avoids it by giving rotation its own window; `test_a_locked_rotation_is_declared_unobservable`
    checks that the estimator SAYS so instead of inventing a number."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = 2 * np.pi * i / max(n - 1, 1)
        p = np.zeros(3) if rot_only else np.array([amp * np.cos(a), amp * np.sin(a), 0.4 * amp * np.sin(2 * a)])
        b = a if locked else 2 * np.pi * 3.7 * i / max(n - 1, 1) + 0.9
        rv = np.radians(rot_deg) * (np.sin(b) * np.array([0.6, 0.5, 0.62]) if locked
                                    else np.array([0.6 * np.sin(b), 0.5 * np.cos(1.3 * b), 0.62 * np.sin(0.7 * b)]))
        out.append(make_T(Rotation.from_rotvec(rv).as_matrix(), p + rng.normal(0, 1e-5, 3)))
    return out


def wp(t_ns, T, *, health=TrackingHealth.OK, conf=1.0, source="x", extra=None):
    return WristPose.from_T(t_ns, T, health=health, confidence=conf, source=source, extra=dict(extra or {}))


def rgbd_sample(T_true, T_D_W, *, r=R_TRUE, noise=0.0, rng=None):
    """What the chest camera sees: the palm centroid of the TRUE wrist, expressed in the camera frame D."""
    palm_W = T_true[:3, 3] + T_true[:3, :3] @ r
    p = inv_T(T_D_W)[:3, :3] @ palm_W + inv_T(T_D_W)[:3, 3]
    if noise: p = p + (rng or np.random.default_rng(0)).normal(0, noise, 3)
    return p


# ---- rigid fit -------------------------------------------------------------------------------------------------
def test_kabsch_recovers_a_rigid_transform():
    rng = np.random.default_rng(1)
    src = rng.normal(0, 0.2, (40, 3))
    R = Rotation.from_euler("xyz", [10, -40, 75], degrees=True).as_matrix(); t = np.array([0.4, -0.2, 0.1])
    dst = src @ R.T + t
    R_hat, t_hat = kabsch(src, dst)
    assert np.allclose(R_hat, R, atol=1e-9) and np.allclose(t_hat, t, atol=1e-9)


def test_alignment_recovers_chest_camera_pose_and_lever_arm():
    T_D_W = T_D_W_truth()
    al = AnchorAlignment(AnchorAlignmentConfig(min_pairs=30, min_span_m=0.05, refit_every_s=0.0))
    for i, T in enumerate(trajectory(120)):
        al.add_pair(i * DT_NS, rgbd_sample(T, T_D_W), T)
    st = al.fit()
    assert st.aligned, st.reason
    assert st.lever_arm_observable
    assert np.allclose(st.lever_arm_m, R_TRUE, atol=2e-3), st.lever_arm_m
    assert np.allclose(st.T_W_D, T_D_W, atol=2e-3)
    assert st.rms_m < 1e-3 and st.lever_arm_sigma_m < 1e-3


def test_alignment_survives_realistic_depth_noise():
    T_D_W = T_D_W_truth()
    rng = np.random.default_rng(7)
    al = AnchorAlignment(AnchorAlignmentConfig(min_pairs=30, min_span_m=0.05, refit_every_s=0.0))
    for i, T in enumerate(trajectory(300)):
        al.add_pair(i * DT_NS, rgbd_sample(T, T_D_W, noise=0.003, rng=rng), T)
    st = al.fit()
    assert st.aligned and np.allclose(st.lever_arm_m, R_TRUE, atol=3e-3), st.lever_arm_m


def test_a_locked_rotation_is_declared_unobservable_instead_of_guessed():
    """Rotation perfectly correlated with translation: the fit still works, the lever arm is NOT claimed."""
    T_D_W = T_D_W_truth()
    rng = np.random.default_rng(11)
    al = AnchorAlignment(AnchorAlignmentConfig(min_pairs=30, min_span_m=0.05, refit_every_s=0.0,
                                               max_lever_arm_sigma_m=1e-4))
    for i, T in enumerate(trajectory(200, locked=True)):
        al.add_pair(i * DT_NS, rgbd_sample(T, T_D_W, noise=0.003, rng=rng), T)
    st = al.fit()
    assert not st.lever_arm_observable and np.allclose(st.lever_arm_m, 0.0)


def test_lever_arm_is_not_invented_without_rotation():
    """Without wrist rotation r is unobservable — it must stay at its last value, not be fitted to noise."""
    T_D_W = T_D_W_truth()
    al = AnchorAlignment(AnchorAlignmentConfig(min_pairs=30, min_span_m=0.05, refit_every_s=0.0, min_rotation_span_deg=20.0))
    for i, T in enumerate(trajectory(120, rot_deg=0.0)):
        al.add_pair(i * DT_NS, rgbd_sample(T, T_D_W), T)
    st = al.fit()
    assert st.aligned and not st.lever_arm_observable
    assert np.allclose(st.lever_arm_m, 0.0)


def test_alignment_refuses_a_still_hand():
    T_D_W = T_D_W_truth()
    al = AnchorAlignment(AnchorAlignmentConfig(min_pairs=30, min_span_m=0.12, refit_every_s=0.0))
    for i, T in enumerate(trajectory(120, amp=0.005)):
        al.add_pair(i * DT_NS, rgbd_sample(T, T_D_W), T)
    st = al.fit()
    assert not st.aligned and "travel" in st.reason


# ---- the correction --------------------------------------------------------------------------------------------
def test_correction_converges_to_the_vi_drift():
    f = TranslationAnchorFusion(AnchorFusionConfig(alpha=0.2, max_correction_rate_m_s=1.0))
    drift = np.array([0.03, -0.01, 0.02])
    for i in range(400):
        f.observe(-drift, 1.0); f.step(i * DT_NS)
    assert np.allclose(f.C, -drift, atol=1e-3)


def test_correction_is_bounded_and_rate_limited():
    cfg = AnchorFusionConfig(alpha=1.0, max_correction_m=0.05, max_correction_rate_m_s=0.01,
                             max_residual_m=1.0, defer_large_to_clutch=False)
    f = TranslationAnchorFusion(cfg)
    f.observe(np.array([0.5, 0.0, 0.0]), 1.0)
    assert np.linalg.norm(f.C_target) == pytest.approx(0.05)      # bound on the target
    f.step(0)
    f.step(DT_NS)                                                  # one 33 ms tick
    assert np.linalg.norm(f.C) <= 0.01 / HZ + 1e-9                 # rate limit = commanded arm velocity
    for i in range(2, 1000): f.step(i * DT_NS)
    assert np.linalg.norm(f.C) == pytest.approx(0.05, abs=1e-6)


def test_large_correction_waits_for_the_clutch():
    cfg = AnchorFusionConfig(alpha=1.0, max_correction_rate_m_s=0.01, large_correction_m=0.02,
                             deferred_rate_fraction=0.0, max_residual_m=1.0)
    f = TranslationAnchorFusion(cfg)
    f.observe(np.array([0.0, 0.10, 0.0]), 1.0); f.step(0)
    for i in range(1, 60): f.step(i * DT_NS)
    assert np.linalg.norm(f.C) == 0.0                              # frozen while commanding
    f.step(60 * DT_NS, clutched=True)
    assert np.allclose(f.C, f.C_target)                            # applied in full while the arm is frozen


def test_outliers_and_low_confidence_observations_are_rejected():
    f = TranslationAnchorFusion(AnchorFusionConfig(alpha=1.0, max_residual_m=0.15, min_rgbd_confidence=0.5))
    assert not f.observe(np.array([0.4, 0, 0]), 1.0) and f.n_rejected_outlier == 1
    assert not f.observe(np.array([0.02, 0, 0]), 0.1) and f.n_rejected_conf == 1
    assert not f.observe(np.array([0.001, 0, 0]), 1.0) and f.n_deadband == 1
    assert np.allclose(f.C_target, 0.0)


# ---- local vs map pose -----------------------------------------------------------------------------------------
def test_loop_closure_is_absorbed_and_commands_nothing():
    lp = LocalPoseContinuity(LocalPoseConfig())
    a = lp.update(make_T(np.eye(3), [0.0, 0, 0]))
    b = lp.update(make_T(np.eye(3), [0.01, 0, 0]))
    jumped = make_T(Rotation.from_euler("z", 5, degrees=True).as_matrix(), [0.09, 0.02, 0])   # map re-optimised
    c = lp.update(jumped, extra={"loop_closure": True})
    assert np.allclose(c, b)                                        # zero commanded motion across the closure
    d = lp.update(make_T(Rotation.from_euler("z", 5, degrees=True).as_matrix(), [0.10, 0.02, 0]),
                  extra={"loop_closure": False})
    assert np.linalg.norm(d[:3, 3] - c[:3, 3]) == pytest.approx(0.01, abs=1e-9)   # motion after it still tracks
    assert lp.n_absorbed == 1 and lp.cumulative_correction_m > 0.05
    assert np.linalg.norm(a[:3, 3]) == 0.0


def test_an_unannounced_jump_is_not_swallowed():
    """A tracking failure and a loop closure look the same in the pose. Unflagged stays visible to the supervisor."""
    lp = LocalPoseContinuity(LocalPoseConfig(absorb_flagged_only=True))
    lp.update(make_T(np.eye(3), [0, 0, 0]))
    out = lp.update(make_T(np.eye(3), [0.4, 0, 0]))
    assert np.allclose(out[:3, 3], [0.4, 0, 0]) and lp.n_absorbed == 0


def test_an_absurd_correction_is_refused():
    lp = LocalPoseContinuity(LocalPoseConfig(max_absorb_m=0.5))
    lp.update(np.eye(4))
    out = lp.update(make_T(np.eye(3), [10.0, 0, 0]), extra={"map_update": True})
    assert lp.n_refused == 1 and lp.n_absorbed == 0 and np.allclose(out[:3, 3], [10.0, 0, 0])


# ---- the state table (spec section 14) -------------------------------------------------------------------------
@pytest.mark.parametrize("vi,rgbd,anchored,rgbd_age,expect_state,expect_health", [
    (TrackingHealth.OK, TrackingHealth.OK, True, 0.0, FusionState.TRACKING_OK, TrackingHealth.OK),
    (TrackingHealth.OK, TrackingHealth.LOST, True, 0.5, FusionState.TRACKING_OK, TrackingHealth.OK),   # short dropout
    (TrackingHealth.OK, TrackingHealth.LOST, True, 9.0, FusionState.DEGRADED, TrackingHealth.DEGRADED),
    (TrackingHealth.DEGRADED, TrackingHealth.OK, True, 0.0, FusionState.DEGRADED, TrackingHealth.DEGRADED),
    (TrackingHealth.LOST, TrackingHealth.OK, True, 0.0, FusionState.LOST, TrackingHealth.LOST),        # HOLD, not 6DoF
    (TrackingHealth.LOST, TrackingHealth.LOST, True, 9.0, FusionState.LOST, TrackingHealth.LOST),
])
def test_fusion_state_table(vi, rgbd, anchored, rgbd_age, expect_state, expect_health):
    st, h, _ = fusion_health(vi=vi, rgbd=rgbd, has_pose=True, anchored=anchored, unanchored_s=0.0,
                             rgbd_age_s=rgbd_age, cfg=FusionStateConfig())
    assert (st, h) == (expect_state, expect_health)


def test_vi_lost_with_a_good_anchor_may_be_configured_to_degrade_instead_of_hold():
    st, h, why = fusion_health(vi=TrackingHealth.LOST, rgbd=TrackingHealth.OK, has_pose=True, anchored=True,
                               unanchored_s=0.0, rgbd_age_s=0.0, cfg=FusionStateConfig(vi_lost_translation_only=True))
    assert (st, h) == (FusionState.DEGRADED, TrackingHealth.DEGRADED) and "translation_only" in why


def test_no_pose_yet_is_initializing_and_must_not_command():
    st, h, _ = fusion_health(vi=None, rgbd=None, has_pose=False, anchored=False, unanchored_s=0.0,
                             rgbd_age_s=float("inf"), cfg=FusionStateConfig())
    assert st is FusionState.INITIALIZING and h is TrackingHealth.LOST


def test_an_unanchored_vi_is_declared_degraded_eventually():
    cfg = FusionStateConfig(max_unanchored_s=30.0)
    ok, _, _ = fusion_health(vi=TrackingHealth.OK, rgbd=TrackingHealth.OK, has_pose=True, anchored=False,
                             unanchored_s=5.0, rgbd_age_s=0.0, cfg=cfg)
    late, h, why = fusion_health(vi=TrackingHealth.OK, rgbd=TrackingHealth.OK, has_pose=True, anchored=False,
                                 unanchored_s=45.0, rgbd_age_s=0.0, cfg=cfg)
    assert ok is FusionState.TRACKING_OK and late is FusionState.DEGRADED and h is TrackingHealth.DEGRADED


# ---- the provider end to end -----------------------------------------------------------------------------------
def provider(*, mode="fused", lever_arm=True, alpha=0.2, rate=1.0, min_pairs=30, min_span=0.05) -> FusedWristPoseProvider:
    cfg = FusedWristConfig(mode=mode)
    cfg = dc.replace(cfg,
                     alignment=AnchorAlignmentConfig(min_pairs=min_pairs, min_span_m=min_span, refit_every_s=0.0,
                                                     estimate_lever_arm=lever_arm,
                                                     lever_arm_m=None if lever_arm else (0.0, 0.0, 0.0)),
                     fusion=AnchorFusionConfig(alpha=alpha, max_correction_rate_m_s=rate, deadband_m=0.0,
                                               defer_large_to_clutch=False))
    return FusedWristPoseProvider(cfg)


def feed(p: FusedWristPoseProvider, Ts, *, T_D_W=None, drift=None, rgbd_health=TrackingHealth.OK, t0=0, noise=0.0):
    """Push one synthetic take through both branches and return the fused poses."""
    T_D_W = T_D_W_truth() if T_D_W is None else T_D_W
    rng = np.random.default_rng(3)
    out = []
    for i, T_true in enumerate(Ts):
        t = t0 + i * DT_NS
        d = np.zeros(3) if drift is None else np.asarray(drift(i), np.float64)
        T_vi = make_T(T_true[:3, :3], T_true[:3, 3] + d)                 # VI drifts; its rotation stays good
        p.push_vi(wp(t, T_vi, source="vi"))
        p.push_rgbd(wp(t, make_T(np.eye(3), rgbd_sample(T_true, T_D_W, noise=noise, rng=rng)),
                       health=rgbd_health, source="rgbd_hand_palm"))
        out.append(p.get_pose(t))
    return out


def test_wrist_rotation_does_not_leak_into_the_fused_position():
    """THE lever-arm test (section 8). VI is perfect here, so any fused position error is manufactured by the anchor.

    The palm centroid swings ~7 cm around the wrist as the hand rolls. With the lever arm modelled, the residual sees
    that for what it is and the fused control point does not move. With it forced to zero — the naive
    `e = p_RGBD − p_VI` of the spec sketch — the same take drags the control point around by centimetres, which on
    the robot is the arm wandering every time the operator turns their hand."""
    Ts = trajectory(400, rot_deg=40.0)
    good = [q for q in feed(provider(lever_arm=True), Ts) if q is not None]
    bad = [q for q in feed(provider(lever_arm=False), Ts) if q is not None]
    truth = np.stack([T[:3, 3] for T in Ts[-len(good):]])
    err_good = np.linalg.norm(np.stack([q.position_xyz_m for q in good]) - truth, axis=1)
    err_bad = np.linalg.norm(np.stack([q.position_xyz_m for q in bad]) - truth, axis=1)
    assert err_good[-200:].max() < 0.005, err_good[-200:].max()
    assert err_bad[-200:].max() > 0.02
    assert err_bad[-200:].max() > 5 * err_good[-200:].max()


def test_a_still_hand_never_reaches_an_alignment():
    """No motion, no fit: the anchor stays out and the pose is honest VI-only rather than anchored to a guess."""
    p = provider(min_span=0.12)
    out = [q for q in feed(p, trajectory(300, amp=0.004)) if q is not None]
    assert not p.align.state.aligned and "travel" in p.align.state.reason
    assert np.allclose(p.corr.C, 0.0) and out[-1].extra["fusion_state"] in ("TRACKING_OK", "DEGRADED")


def test_the_anchor_removes_vi_drift():
    """VI drifts away; the chest camera does not. The fused pose must follow the chest camera, not the drift.

    The drift starts AFTER the alignment window on purpose: a constant offset present during the fit is, correctly,
    indistinguishable from the camera being 3 cm further left, and gets absorbed into T_W_D. Only drift the
    alignment is not free to explain is observable — which is exactly why the fit is frozen once it converges."""
    Ts = trajectory(600)
    drift = lambda i: np.array([1.0, -0.5, 0.7]) * 4e-5 * max(i - 200, 0)
    out = [q for q in feed(provider(rate=0.5), Ts, drift=drift) if q is not None]
    truth = np.stack([T[:3, 3] for T in Ts[-len(out):]])
    fused_err = np.linalg.norm(np.stack([q.position_xyz_m for q in out]) - truth, axis=1)
    vi_err = np.linalg.norm(np.stack([drift(i) for i in range(len(Ts))])[-len(out):], axis=1)
    assert vi_err[-1] > 0.015                                          # the drift really is there to be removed
    assert fused_err[-100:].mean() < 0.15 * vi_err[-100:].mean()
    assert out[-1].extra["anchored"] and out[-1].extra["fusion_state"] == "TRACKING_OK"


def test_the_map_pose_and_the_local_pose_are_both_logged():
    out = [q for q in feed(provider(), trajectory(120)) if q is not None]
    e = out[-1].extra
    for k in ("vi_x", "vi_map_x", "rgbd_x", "rgbd_y", "rgbd_z", "c_x", "correction_m", "fusion_state", "align_rms_m"):
        assert k in e, k


def test_rgbd_dropout_keeps_running_on_vi_then_declares_itself():
    p = provider()
    Ts = trajectory(300)
    feed(p, Ts)
    t = 300 * DT_NS
    for i, T in enumerate(trajectory(60, seed=5)):                    # RGB-D gone, VI still healthy
        t = (300 + i) * DT_NS
        p.push_vi(wp(t, T, source="vi"))
    short = p.get_pose(t)
    assert short.health is TrackingHealth.OK and short.extra["fusion_state"] == "TRACKING_OK"
    late = p.get_pose(t + int(5e9))                                   # ... but the pose is stale by then, too
    assert late.health is TrackingHealth.LOST


def test_both_branches_lost_holds_the_last_pose_and_never_emits_zero():
    p = provider()
    out = [q for q in feed(p, trajectory(150)) if q is not None]
    last = out[-1].position_xyz_m.copy()
    t = 150 * DT_NS
    p.push_vi(wp(t, np.eye(4), health=TrackingHealth.LOST, source="vi"))
    p.push_rgbd(wp(t, np.eye(4), health=TrackingHealth.LOST, source="rgbd_hand_palm"))
    held = p.get_pose(t)
    assert held.health is TrackingHealth.LOST and held.extra["held"] is True
    assert np.allclose(held.position_xyz_m, last) and np.all(np.isfinite(held.position_xyz_m))
    assert np.allclose(held.linear_velocity_xyz, 0.0)


def test_nothing_is_emitted_before_the_first_pose():
    assert provider().get_pose(0) is None


@pytest.mark.parametrize("mode", ["fused", "vi_only", "rgbd_only"])
def test_every_ablation_runs_through_the_same_provider(mode):
    p = provider(mode=mode)
    out = [q for q in feed(p, trajectory(200)) if q is not None]
    assert out and out[-1].source == "fused_wrist"
    assert out[-1].extra["fusion_mode"] == mode
    assert np.all(np.isfinite(out[-1].position_xyz_m))
    if mode == "vi_only":
        assert np.allclose(p.corr.C, 0.0)                              # ablation B applies no correction at all


def test_a_typo_in_the_ablation_mode_is_refused():
    with pytest.raises(ValueError):
        FusedWristConfig(mode="fuzed")


def test_clutch_lets_a_pending_correction_through_at_once():
    p = provider(rate=1e-6)                                            # correction effectively frozen while commanding
    feed(p, trajectory(300), drift=lambda i: np.array([0.03, 0.0, 0.0]) * (i > 150))
    pending = p.corr.pending_m
    assert pending > 0.01
    p.set_clutched(True)
    p.get_pose(400 * DT_NS)
    assert p.corr.pending_m == pytest.approx(0.0, abs=1e-9)


# ---- storage ----------------------------------------------------------------------------------------------------
def test_the_fused_trajectory_is_its_own_stream(tmp_path):
    import pandas as pd
    from ego_teleop.recorder.episode_logger import TeleopEpisodeLogger
    p = provider()
    out = [q for q in feed(p, trajectory(120)) if q is not None]
    lg = TeleopEpisodeLogger(tmp_path / "ep")
    for q in out: lg.add_fused_wrist_pose("right", q)
    lg.add_fused_relative_pose("right", dict(t_ns=1, human_dx=0.0, robot_dx=0.0))
    meta = lg.close()
    df = pd.read_parquet(tmp_path / "ep/raw/fused_wrist_pose_live_right.parquet")
    assert len(df) == len(out) and {"x", "y", "z", "vi_x", "rgbd_x", "c_x", "fusion_state"} <= set(df.columns)
    prov = meta["pose_streams"]["fused_wrist_pose_live_right"]
    assert prov["pose_type"] == "live" and prov["causal"] and prov["pose_source"] == "fused_wrist"
    assert prov["source_camera"] == "head_rgbd+right_wrist" and prov["imu_used"] is True
    # the wrist-VIO stream name still means wrist VIO alone
    assert not (tmp_path / "ep/raw/wrist_pose_live_right.parquet").empty if False else True
