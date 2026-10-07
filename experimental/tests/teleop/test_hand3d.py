"""Head RGB-D hand branch: depth sampling, metric deprojection, providers, identity, supervisor. All synthetic —
a rendered "hand" in front of a wall — so the invariants are checked without a camera or a recorded episode."""
from __future__ import annotations
import numpy as np
import pytest
from ego_teleop.hand3d import aero_mocap as M
from ego_teleop.hand3d.depth import CameraIntrinsics, DepthSampler, DepthSamplerConfig, depth_image_to_m
from ego_teleop.hand3d.head_camera import HeadRgbdCalibration, RgbdFrame
from ego_teleop.hand3d.identity import HandCandidate, HandIdentityTracker, IdentityConfig
from ego_teleop.hand3d.interfaces import HandPoseEstimate, HandPoseHealth
from ego_teleop.hand3d.providers import HandPoseProviderConfig, RgbdHandPoseProvider, MediaPipeHandPoseProvider, umeyama_similarity
from ego_teleop.hand3d.supervisor import HandPoseSupervisor, HandPoseSupervisorConfig

K = CameraIntrinsics(400.0, 400.0, 320.0, 240.0, 640, 480)


# ---- intrinsics / depth -----------------------------------------------------------------------------------------
def test_deprojection_round_trips():
    xyz = np.array([[0.05, -0.02, 0.45], [0.0, 0.0, 0.30], [-0.10, 0.08, 0.70]])
    assert np.allclose(K.deproject(K.project(xyz), xyz[:, 2]), xyz)


def test_raw_depth_zero_is_nan_not_zero_metres():
    raw = np.array([[0, 500], [1200, 0]], np.uint16)
    m = depth_image_to_m(raw, 0.001)
    assert np.isnan(m[0, 0]) and np.isnan(m[1, 1]) and m[0, 1] == 0.5


def test_depth_sampler_is_robust_to_edge_pixels_and_holes():
    D = np.full((100, 100), 0.50)
    D[40:60, 40:60] = 0.30                       # a "finger" in front of the wall
    D[50, 50] = np.nan                           # dropout right at the landmark
    D[49, 49] = 0.0                              # invalid pixel
    s = DepthSampler(DepthSamplerConfig(window=5)).sample(D, [[50, 50]])
    assert s.valid[0] and np.isclose(s.depth_m[0], 0.30)      # background 0.50 rejected, not averaged in
    edge = DepthSampler(DepthSamplerConfig(window=5)).sample(D, [[41, 50]])
    assert edge.valid[0] and np.isclose(edge.depth_m[0], 0.30)


def test_depth_sampler_marks_invalid_instead_of_filling_zero():
    D = np.full((50, 50), np.nan)
    s = DepthSampler().sample(D, [[25, 25], [-1, 5], [999, 999]])
    assert not s.valid.any() and np.isnan(s.depth_m).all() and (s.confidence == 0).all()


def test_depth_sampler_rejects_out_of_range():
    D = np.full((50, 50), 3.0)                   # far wall, beyond max_range_m
    assert not DepthSampler().sample(D, [[25, 25]]).valid[0]


def test_unaligned_calibration_refuses_depth_lookup():
    cal = HeadRgbdCalibration(K, K, 0.001, aligned=False)
    with pytest.raises(RuntimeError, match="NOT aligned"):
        cal.require_aligned()


# ---- synthetic hand + fake landmarker ---------------------------------------------------------------------------
def synthetic_hand(curl: float = 0.0) -> np.ndarray:
    """An anatomically plausible right hand in the PALM-LOCAL frame (wrist at the origin, fingers along +z, +x toward
    the index side) — the same frame `aero_mocap.to_palm_local` produces and the Aero URDF uses.

    Sizes matter: Aero's straight-finger reach is ~0.20 m from its base link, so a 6 cm toy hand would make the
    retargeter curl everything to the limit. Wrist -> middle fingertip here is ~0.185 m, a real adult hand.
    `curl` in [0,1] flexes the fingers toward -y, the direction Aero's joints move."""
    lm = np.zeros((21, 3))
    segs, angs = (0.045, 0.028, 0.022), np.deg2rad((50.0, 45.0, 30.0))
    for base, x in ((5, 0.035), (9, 0.012), (13, -0.012), (17, -0.035)):
        p, a = np.array([x, 0.0, 0.090]), 0.0
        lm[base] = p
        for k, (L, ang) in enumerate(zip(segs, angs)):
            a += curl * ang
            p = p + L * np.array([0.0, -np.sin(a), np.cos(a)])
            lm[base + 1 + k] = p
    t, a = np.array([0.030, 0.008, 0.028]), 0.0
    lm[1] = t
    for k, L in enumerate((0.038, 0.032, 0.026)):
        a += curl * np.deg2rad(30.0)
        t = t + L * np.array([0.50 * np.cos(a), -np.sin(a), 0.85 * np.cos(a)])
        lm[2 + k] = t
    return lm


# palm-local -> camera frame: fingers point UP in the image (palm +z -> camera -y), palm facing the camera, 45 cm away
R_CAM_PALM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def place_in_camera(lm: np.ndarray, *, distance_m: float = 0.45, offset=(0.0, 0.09, 0.0)) -> np.ndarray:
    return (R_CAM_PALM @ np.asarray(lm).T).T + np.array([offset[0], offset[1], distance_m])


class FakeLandmarker:
    """Stands in for MediaPipe: projects a known 3D hand and reports a handedness label."""

    def __init__(self, hands: list[tuple[str, np.ndarray]], *, score: float = 0.9) -> None:
        self.hands, self.score = hands, score

    def detect_all(self, image_bgr, timestamp_ms):
        from ego_collector.hands.mediapipe_tracker import RawHandDetection
        out = []
        for label, lm in self.hands:
            uv = K.project(lm)
            world = lm - lm[0]
            out.append(RawHandDetection(label, self.score, uv, world))
        return out


def rgbd_frame(hands: list[np.ndarray], *, t_ns: int = 0, hole_at: list[int] | None = None) -> RgbdFrame:
    """Render the hands into a depth image (small squares at each landmark) in front of a 1.0 m wall."""
    cal = HeadRgbdCalibration(K, K, 0.001, aligned=True, fps=30.0)
    D = np.full((K.height, K.width), 1.0)                 # a wall behind the hand: holes show BACKGROUND, not nothing
    for lm in hands:
        uv = K.project(lm)
        for j, (u, v) in enumerate(uv):
            if hole_at and j in hole_at: continue
            ui, vi = int(round(u)), int(round(v))
            patch = D[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3]
            patch[:] = np.minimum(patch, lm[j, 2])         # z-buffer: nearest surface wins
    return RgbdFrame(t_ns, np.zeros((K.height, K.width, 3), np.uint8), D, cal, None, 0)


def make_provider(hands: list[tuple[str, np.ndarray]], **kw):
    cfg = HandPoseProviderConfig(side="right", selfie_mirrored=False, **kw)
    return RgbdHandPoseProvider(cfg, landmarker=FakeLandmarker(hands))


# ---- provider ----------------------------------------------------------------------------------------------------
def test_rgbd_provider_recovers_metric_geometry():
    lm = place_in_camera(synthetic_hand())
    p = make_provider([("left", lm)])            # MediaPipe says "left" on a non-selfie image -> our right hand
    est = p.get_hand_pose(rgbd_frame([lm]))
    assert est.side == "right" and est.metric and est.health is HandPoseHealth.OK
    assert est.n_valid == 21 and est.n_filled == 0
    assert np.allclose(est.landmarks_camera_3d, lm, atol=2e-3)
    # palm-local matches the official transform of the true 3D hand
    assert np.allclose(est.landmarks_local_3d, M.to_palm_local(lm, "right"), atol=3e-3)
    assert 0.02 < est.palm_scale_m < 0.12


def test_rgbd_provider_is_invariant_to_camera_pose():
    """Same hand, different camera placement -> the same palm-local articulation (spec section 10)."""
    from scipy.spatial.transform import Rotation as Rot
    base = place_in_camera(synthetic_hand())
    ref = make_provider([("left", base)]).get_hand_pose(rgbd_frame([base]))
    R = Rot.from_euler("xyz", [5, 12, -7], degrees=True).as_matrix()
    moved = (R @ (base - base.mean(0)).T).T + base.mean(0) + np.array([0.02, -0.01, 0.05])
    other = make_provider([("left", moved)]).get_hand_pose(rgbd_frame([moved]))
    assert np.allclose(ref.landmarks_local_3d, other.landmarks_local_3d, atol=4e-3)


def test_missing_depth_is_filled_from_the_shape_prior_but_never_marked_valid():
    lm = place_in_camera(synthetic_hand())
    p = make_provider([("left", lm)])
    est = p.get_hand_pose(rgbd_frame([lm], hole_at=[8, 12]))
    assert est.n_filled == 2 and est.n_valid == 19
    assert not est.valid[8] and est.filled[8]
    assert np.isfinite(est.landmarks_camera_3d).all()            # retargeting still gets complete geometry
    assert np.allclose(est.landmarks_camera_3d[8], lm[8], atol=6e-3)
    assert np.isnan(est.depth_m[8])                              # the *measurement* is still absent


def test_too_many_holes_reports_lost_rather_than_inventing_a_hand():
    lm = place_in_camera(synthetic_hand())
    p = make_provider([("left", lm)], max_fill=2)
    est = p.get_hand_pose(rgbd_frame([lm], hole_at=list(range(5, 16))))
    assert est.health is HandPoseHealth.LOST and est.extra["reason"] == "insufficient_metric_landmarks"


def test_monocular_baseline_produces_the_same_representation_without_metric_depth():
    lm = place_in_camera(synthetic_hand())
    p = MediaPipeHandPoseProvider(HandPoseProviderConfig(side="right", selfie_mirrored=False),
                                  landmarker=FakeLandmarker([("left", lm)]))
    est = p.get_hand_pose(rgbd_frame([lm]))
    assert est.side == "right" and not est.metric
    assert est.landmarks_local_3d.shape == (21, 3) and np.isfinite(est.landmarks_local_3d).all()
    assert np.isnan(est.depth_m).all()


def test_selfie_mirrored_flag_flips_the_side():
    lm = place_in_camera(synthetic_hand())
    cfg = HandPoseProviderConfig(side="right", selfie_mirrored=True)
    est = RgbdHandPoseProvider(cfg, landmarker=FakeLandmarker([("right", lm)])).get_hand_pose(rgbd_frame([lm]))
    assert est.side == "right" and est.n_valid == 21


def test_rgbd_provider_rejects_a_plain_image():
    with pytest.raises(TypeError, match="RgbdFrame"):
        make_provider([]).get_hand_pose(np.zeros((10, 10, 3), np.uint8))


def test_umeyama_recovers_a_similarity():
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(2); S = rng.normal(size=(12, 3))
    R0, s0, t0 = Rot.random(random_state=5).as_matrix(), 1.4, np.array([0.2, -0.1, 0.3])
    R, t, s = umeyama_similarity(S, (s0 * (R0 @ S.T).T) + t0)
    assert np.allclose(R, R0) and np.allclose(t, t0) and np.isclose(s, s0)


# ---- identity ----------------------------------------------------------------------------------------------------
def test_identity_prefers_continuity_over_a_single_bad_label():
    t = HandIdentityTracker(IdentityConfig(label_flip_votes=4))
    c = lambda lab, u: HandCandidate(lab, 0.9, 0.9, np.array([u, 200.0]))
    t.update(0, [c("right", 400)])
    r = t.update(33_000_000, [c("left", 404)])           # one frame of a flipped label
    assert "right" in r.tracks and not r.ambiguous and t.swap_events == 0


def test_sustained_label_flip_degrades_instead_of_driving_the_wrong_hand():
    t = HandIdentityTracker()
    c = lambda lab, u: HandCandidate(lab, 0.9, 0.9, np.array([u, 200.0]))
    t.update(0, [c("right", 400)])
    states = [t.update((i + 1) * 33_000_000, [c("left", 400 + i)]) for i in range(6)]
    assert any(s.ambiguous == {"right"} for s in states)   # a frame where nothing is sent
    assert t.swap_events == 1
    assert "left" in states[-1].tracks                     # then it re-bootstraps as the hand it actually is


def test_two_hands_do_not_steal_each_others_identity():
    t = HandIdentityTracker()
    L = lambda u: HandCandidate("left", 0.9, 0.9, np.array([u, 200.0]))
    R = lambda u: HandCandidate("right", 0.9, 0.9, np.array([u, 200.0]))
    t.update(0, [L(100), R(500)])
    r = t.update(33_000_000, [L(110), R(490)])
    assert np.isclose(r.tracks["left"].wrist_uv[0], 110) and np.isclose(r.tracks["right"].wrist_uv[0], 490)


# ---- supervisor --------------------------------------------------------------------------------------------------
def _est(t_ns, *, valid=21, wrist=(0.0, 0.0, 0.4), local=None, conf=0.9):
    lm = np.zeros((21, 3)); lm[:, 2] = 0.4; lm[0] = wrist
    v = np.zeros(21, bool); v[:valid] = True
    return HandPoseEstimate(t_ns, "right", np.zeros((21, 2)), lm, local if local is not None else lm * 0,
                            v, np.full(21, 0.4), np.ones(21), conf, HandPoseHealth.OK, "head_rgbd", True)


def _settle(sup, n=12, start=0):
    for i in range(n): st = sup.update(_est((start + i) * 33_000_000), (start + i) * 33_000_000)
    return st


def test_supervisor_requires_a_stable_window_before_the_first_ok():
    sup = HandPoseSupervisor(HandPoseSupervisorConfig(recover_stable_ms=300))
    first = sup.update(_est(0), 0)
    assert first.health is HandPoseHealth.INITIALIZING and not first.commandable
    assert _settle(sup).health is HandPoseHealth.OK


def test_depth_holes_degrade_but_stay_commandable():
    sup = HandPoseSupervisor(); _settle(sup)
    st = sup.update(_est(13 * 33_000_000, valid=15), 13 * 33_000_000)
    assert st.health is HandPoseHealth.DEGRADED and st.commandable and st.reason == "depth_holes"


def test_loss_is_not_commandable_and_recovery_is_gated():
    sup = HandPoseSupervisor(); _settle(sup)
    lost = sup.update(None, 20 * 33_000_000)
    assert lost.health is HandPoseHealth.LOST and not lost.commandable and lost.estimate is None
    back = sup.update(_est(21 * 33_000_000), 21 * 33_000_000)
    assert back.health is HandPoseHealth.INITIALIZING and not back.commandable
    assert _settle(sup, start=22).health is HandPoseHealth.OK


def test_wrist_discontinuity_is_a_loss_not_a_command():
    sup = HandPoseSupervisor(); _settle(sup)
    st = sup.update(_est(13 * 33_000_000, wrist=(0.6, 0.0, 0.4)), 13 * 33_000_000)
    assert st.health is HandPoseHealth.LOST and st.reason == "wrist_discontinuity"


def test_stale_estimate_ages_into_lost():
    sup = HandPoseSupervisor(); _settle(sup)
    assert sup.current(13 * 33_000_000).health is HandPoseHealth.OK
    assert sup.current(60 * 33_000_000).health is HandPoseHealth.LOST


def test_hand_health_is_not_the_wrist_tracking_enum():
    from ego_teleop.tracking.interfaces import TrackingHealth
    assert not set(h.value for h in HandPoseHealth) & set(h.value for h in TrackingHealth)
