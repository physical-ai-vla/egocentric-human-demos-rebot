"""Spec 37: coordinate adapter identity, xyz mapping, +-90 rotations, relative retargeting, motion scaling, left/right identity."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from visual_teleop.retargeting.frame_adapter import FrameAdapter
from visual_teleop.retargeting.motion_scaling import MotionScaling
from visual_teleop.retargeting.relative_se3 import RelativeRetargeter
from ego_teleop.robot.bimanual import build_bimanual_action
from ego_teleop.robot.rebot_client import SIDE_SLICE, GRIPPER_INDEX, FLIP_IDX
from .conftest import T_of


def rv(R): return np.degrees(Rotation.from_matrix(R).as_rotvec())


def test_identity_adapter_is_identity():
    A = FrameAdapter(); D = T_of((0.1, -0.2, 0.3), (10, 20, -30))
    np.testing.assert_allclose(A.map_delta(D), D, atol=1e-12)


@pytest.mark.parametrize("axis,vec", [("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))])
def test_identity_xyz_translation(axis, vec):
    D = T_of(np.array(vec) * 0.05)
    np.testing.assert_allclose(FrameAdapter().map_delta(D)[:3, 3], np.array(vec) * 0.05, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("deg", [90, -90])
def test_identity_pm90_rotations(axis, deg):
    r = np.zeros(3); r[axis] = deg
    np.testing.assert_allclose(rv(FrameAdapter().map_delta(T_of(rotvec_deg=r))[:3, :3]), r, atol=1e-9)


def test_permuted_adapter_maps_axes_and_rotations_consistently():
    # robot x <- human z, robot y <- -human y, robot z <- human x  (det = +1)
    A = FrameAdapter(("z", "y", "x"), (1, -1, 1)); assert not A.is_mirror
    np.testing.assert_allclose(A.map_delta(T_of((0.05, 0, 0)))[:3, 3], [0, 0, 0.05], atol=1e-12)     # human +X -> robot +Z
    np.testing.assert_allclose(A.map_delta(T_of((0, 0.05, 0)))[:3, 3], [0, -0.05, 0], atol=1e-12)    # human +Y -> robot -Y
    np.testing.assert_allclose(A.map_delta(T_of((0, 0, 0.05)))[:3, 3], [0.05, 0, 0], atol=1e-12)     # human +Z -> robot +X
    np.testing.assert_allclose(rv(A.map_delta(T_of(rotvec_deg=(90, 0, 0)))[:3, :3]), [0, 0, 90], atol=1e-9)    # roll -> about robot z
    np.testing.assert_allclose(rv(A.map_delta(T_of(rotvec_deg=(0, 90, 0)))[:3, :3]), [0, -90, 0], atol=1e-9)
    np.testing.assert_allclose(rv(A.map_delta(T_of(rotvec_deg=(0, 0, -90)))[:3, :3]), [-90, 0, 0], atol=1e-9)
    D = T_of((0.01, 0.02, 0.03), (5, -7, 11)); np.testing.assert_allclose(A.unmap_delta(A.map_delta(D)), D, atol=1e-12)


def test_adapter_preserves_composition_and_rotation_validity():
    A = FrameAdapter(("y", "z", "x"), (1, 1, -1)); D1, D2 = T_of((0.1, 0, 0), (0, 30, 0)), T_of((0, 0.1, 0), (20, 0, 0))
    np.testing.assert_allclose(A.map_delta(D1 @ D2), A.map_delta(D1) @ A.map_delta(D2), atol=1e-12)
    R = A.map_delta(D1)[:3, :3]; assert abs(np.linalg.det(R) - 1) < 1e-12


def test_adapter_rejects_nan():
    D = T_of((0.1, 0, 0)); D[0, 3] = np.nan
    with pytest.raises(ValueError): FrameAdapter().map_delta(D)


def test_relative_retarget_no_absolute_leak_and_body_frame():
    rt = RelativeRetargeter({"left": FrameAdapter(), "right": FrameAdapter()}, MotionScaling())
    T_H0 = T_of((1.0, 2.0, 0.5), (0, 0, 90)); T_R0 = T_of((0.3, 0.2, 0.15))
    rt.anchor("left", T_H0, T_R0)
    tgt, _ = rt.target("left", T_H0)                                 # human still at anchor -> robot at its anchor
    np.testing.assert_allclose(tgt, T_R0, atol=1e-12)
    # human moves +X in WORLD while rotated 90 deg about Z: that is -Y in the human body frame -> robot body -Y
    tgt, D = rt.target("left", T_of((1.05, 2.0, 0.5), (0, 0, 90)))
    np.testing.assert_allclose(D[:3, 3], [0, -0.05, 0], atol=1e-12); np.testing.assert_allclose(tgt[:3, 3], [0.3, 0.15, 0.15], atol=1e-12)


def test_motion_scaling_per_axis_and_rotation():
    S = MotionScaling((0.5, 1.0, 2.0), 0.5); D = S.apply(T_of((0.1, 0.1, 0.1), (0, 0, 60)))
    np.testing.assert_allclose(D[:3, 3], [0.05, 0.1, 0.2]); np.testing.assert_allclose(rv(D[:3, :3]), [0, 0, 30], atol=1e-9)
    with pytest.raises(ValueError): MotionScaling((-1, 1, 1))


def test_left_right_identity_in_the_bimanual_vector():
    qL = np.array([0.1, -0.2, -0.3, 0.4, 0.5, 0.6]); qR = -qL
    a = build_bimanual_action(qL, qR, 0.0, 1.0)
    lead = lambda q, sl: [(-v if (sl.start + i) in FLIP_IDX else v) for i, v in enumerate(q)]
    np.testing.assert_allclose(a[SIDE_SLICE["left"]], lead(qL, SIDE_SLICE["left"])); np.testing.assert_allclose(a[SIDE_SLICE["right"]], lead(qR, SIDE_SLICE["right"]))
    assert a[GRIPPER_INDEX["left"]] == 0.0 and a[GRIPPER_INDEX["right"]] == 45.0          # g 0 = closed -> cmd 0; g 1 = open -> 45


def test_orientation_modes_lock_yaw_full():
    T_H0 = T_of((0.1, 0.2, 0.3), (20, -10, 35)); T_R0 = T_of((0.3, 0.2, 0.15))
    # human: +5 cm along world X, then rotated 30 deg about WORLD z and 25 deg about its own x
    R_new = Rotation.from_rotvec([0, 0, np.radians(30)]).as_matrix() @ T_H0[:3, :3] @ Rotation.from_rotvec([np.radians(25), 0, 0]).as_matrix()
    T_H = T_of(T_H0[:3, 3] + [0.05, 0, 0]); T_H[:3, :3] = R_new
    for mode in ("lock", "yaw", "full"):
        rt = RelativeRetargeter({"left": FrameAdapter(), "right": FrameAdapter()}, MotionScaling(), orientation_mode=mode)
        rt.anchor("left", T_H0, T_R0); tgt, D = rt.target("left", T_H)
        np.testing.assert_allclose(tgt[:3, 3] - T_R0[:3, 3], T_H0[:3, :3].T @ [0.05, 0, 0], atol=1e-9)   # translation identical in every mode
        ang = np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude())
        if mode == "lock": assert ang < 1e-9
        if mode == "full": np.testing.assert_allclose(D[:3, :3], T_H0[:3, :3].T @ R_new, atol=1e-9)
        if mode == "yaw":
            Rw = T_H0[:3, :3] @ D[:3, :3] @ T_H0[:3, :3].T                       # back to a world-frame rotation: pure z
            rv = Rotation.from_matrix(Rw).as_rotvec(); assert abs(rv[0]) < 1e-9 and abs(rv[1]) < 1e-9 and abs(np.degrees(rv[2])) > 20


def test_heading_translation_frame_world_aligned():
    # UMI anchor: level, jaws (H z) pointing along world +Y; robot gripper in an arbitrary (folded) orientation
    R_W_H0 = np.column_stack([[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0]])             # x_H down, y_H = -X, z_H (jaws) = world +Y
    T_H0 = T_of((0.5, 0.2, 0.3)); T_H0[:3, :3] = R_W_H0; T_R0 = T_of((0.3, -0.25, 0.15), (40, -70, 10))
    rt = RelativeRetargeter({"left": FrameAdapter(), "right": FrameAdapter()}, MotionScaling(), orientation_mode="lock", translation_frame="heading")
    rt.anchor("right", T_H0, T_R0)
    for d_W, d_R in (([0, 0.1, 0], [0.1, 0, 0]),        # forward (push heading) -> robot +x
                     ([0, 0, 0.1], [0, 0, 0.1]),        # up -> robot +z
                     ([-0.1, 0, 0], [0, 0.1, 0])):      # left of the heading (heading +Y, left = -X) -> robot +y
        T = T_H0.copy(); T[:3, 3] += d_W; tgt, _ = rt.target("right", T)
        np.testing.assert_allclose(tgt[:3, 3] - T_R0[:3, 3], d_R, atol=1e-9)
        np.testing.assert_allclose(tgt[:3, :3], T_R0[:3, :3], atol=1e-9)                # lock: orientation held


def test_heading_fixed_at_init_survives_vertical_reanchor():
    """10-15 live bug: a clutch/loss re-anchor with the jaws pointing straight down raised 'too vertically' -> HOLD."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from visual_teleop.retargeting.relative_se3 import RelativeRetargeter
    from visual_teleop.retargeting.frame_adapter import FrameAdapter
    from visual_teleop.retargeting.motion_scaling import MotionScaling
    ad = FrameAdapter.from_cfg({"axis_map": ["x", "y", "z"], "sign": [1, 1, 1]})
    rt = RelativeRetargeter({"left": ad, "right": ad}, MotionScaling(translation=(1.0, 1.0, 1.0), rotation=1.0), translation_frame="heading")
    T0 = np.eye(4); T0[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()          # jaws (z) forward along W +x
    rt.anchor("right", T0, np.eye(4)); T = T0.copy(); T[0, 3] += 0.05; out, _ = rt.target("right", T)
    np.testing.assert_allclose(out[:3, 3], [0.05, 0, 0], atol=1e-9)
    Tv = np.eye(4); Tv[:3, :3] = R.from_euler("y", 180, degrees=True).as_matrix()           # jaws straight down
    rt.anchor("right", Tv, np.eye(4)); T = Tv.copy(); T[0, 3] += 0.05; out, _ = rt.target("right", T)   # no raise
    np.testing.assert_allclose(out[:3, 3], [0.05, 0, 0], atol=1e-9)                         # heading from INIT still used
    rt.reset_heading()
    import pytest
    with pytest.raises(ValueError): rt.target("right", T)                                    # a NEW init must not be vertical


def test_world_rotation_frame_yaw_and_tilt_follow_in_space():
    """10-15: a hand yaw about W vertical yaws the gripper about RB vertical, whatever the gripper's own orientation."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from visual_teleop.retargeting.relative_se3 import RelativeRetargeter
    from visual_teleop.retargeting.frame_adapter import FrameAdapter
    from visual_teleop.retargeting.motion_scaling import MotionScaling
    ad = FrameAdapter.from_cfg({"axis_map": ["x", "y", "z"], "sign": [1, 1, 1]})
    rt = RelativeRetargeter({"left": ad, "right": ad}, MotionScaling(translation=(1.0, 1.0, 1.0), rotation=1.0), translation_frame="heading", rotation_frame="world")
    H0 = np.eye(4); H0[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()            # UMI jaws forward (W +x) -> heading = W x
    R0 = np.eye(4); R0[:3, :3] = R.from_euler("y", 180, degrees=True).as_matrix()           # gripper pointing DOWN
    rt.anchor("right", H0, R0)
    for axis in ("z", "y"):                                                                  # yaw, then a tilt about W y
        H = H0.copy(); H[:3, :3] = R.from_euler(axis, 20, degrees=True).as_matrix() @ H0[:3, :3]
        out, _ = rt.target("right", H)
        np.testing.assert_allclose(out[:3, :3], R.from_euler(axis, 20, degrees=True).as_matrix() @ R0[:3, :3], atol=1e-9)
