"""Synthetic-render tests for the head-camera AprilTag backend (pinhole + fisheye)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.calibration.spatial import CameraIntrinsics
from handumi.robots.utils import IDENTITY_POSE7, mat_to_pose7, pose7_to_mat
from handumi.tracking.apriltag import (
    AprilTagDetector,
    BundleConfig,
    BundleSpec,
    CameraGeometry,
    TagSpec,
    WorldMap,
    apriltag_features,
    build_pair_sample,
    calibrate_bundle_geometry,
    calibrate_world_map,
    process_frame,
    solve_bundle,
    tag_corners_local,
)

W, H = 1920, 1080
K_PIN = np.array([[1100.0, 0, W / 2], [0, 1100.0, H / 2], [0, 0, 1]])
K_FISH = np.array([[560.0, 0, W / 2], [0, 560.0, H / 2], [0, 0, 1]])
D_FISH = np.array([0.06, -0.02, 0.005, 0.0]).reshape(4, 1)


def _intr(model, K, D):
    return CameraIntrinsics(camera="head", width=W, height=H, matrix=K, distortion=D, rms_px=0.0, mean_error_px=0.0, views=0, model=model)


INTR_PIN = _intr("pinhole", K_PIN, np.zeros((5, 1)))
INTR_FISH = _intr("fisheye", K_FISH, D_FISH)


def _pose(xyz, rotvec_deg):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    T[:3, 3] = xyz
    return T


def _project(pts_cam: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    if intr.model == "fisheye":
        proj, _ = cv2.fisheye.projectPoints(pts_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intr.matrix, intr.distortion)
        return proj.reshape(-1, 2)
    proj = (intr.matrix @ pts_cam.T).T
    return proj[:, :2] / proj[:, 2:3]


def _render(faces: list[tuple[int, float, np.ndarray]], det: AprilTagDetector, intr: CameraIntrinsics) -> np.ndarray:
    """faces = [(tag_id, size_m, T_cam_tag)] -> gray uint8 image with the tags warped in (far first)."""
    canvas = np.full((H, W), 140, dtype=np.uint8)
    M, P = 160, 24
    visible = []
    for tag_id, size, T_ct in faces:
        normal_cam = T_ct[:3, :3] @ np.array([0, 0, 1.0])
        if np.dot(normal_cam, T_ct[:3, 3]) > 0:
            continue
        visible.append((float(T_ct[2, 3]), tag_id, size, T_ct))
    for _, tag_id, size, T_ct in sorted(visible, key=lambda f: -f[0]):
        scale = (M + 2 * P) / M
        obj = tag_corners_local(size * scale)
        pts_cam = (T_ct[:3, :3] @ obj.T).T + T_ct[:3, 3]
        proj = _project(pts_cam, intr)
        marker = det.marker_image(tag_id, M)
        padded = np.full((M + 2 * P, M + 2 * P), 255, dtype=np.uint8)
        padded[P : P + M, P : P + M] = marker
        src = np.array([[0, 0], [M + 2 * P, 0], [M + 2 * P, M + 2 * P], [0, M + 2 * P]], dtype=np.float32)
        Hm = cv2.getPerspectiveTransform(src, proj.astype(np.float32))
        warped = cv2.warpPerspective(padded, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(padded, 255), Hm, (W, H), borderValue=0) > 127
        canvas[mask] = warped[mask]
    return canvas


SECOND_FACE = {"position": [0, -0.07, -0.02], "quaternion": [-0.38268343, 0, 0, 0.92387953]}


def _config(size_m: float = 0.05):
    return BundleConfig.from_dict(
        {
            "bundles": {
                "left": {"anchor_tag": 10, "tags": [{"id": 10, "size_m": size_m}, {"id": 11, "size_m": size_m, "anchor_from_tag": SECOND_FACE}]},
                "right": {"anchor_tag": 20, "tags": [{"id": 20, "size_m": size_m}, {"id": 21, "size_m": size_m, "anchor_from_tag": SECOND_FACE}]},
            }
        }
    )


def _world_map():
    # four 8 cm tags flat on the table (tag +Z up = world +Z), around the workspace
    tags = []
    for tid, (x, y) in zip((100, 101, 102, 103), ((-0.35, 0.45), (0.35, 0.45), (-0.35, 0.10), (0.35, 0.10))):
        tags.append(TagSpec(id=tid, size_m=0.08, anchor_from_tag=mat_to_pose7(_pose([x, y, 0.0], [0, 0, 0])).astype(np.float64)))
    return WorldMap(bundle=BundleSpec(side="world", anchor_tag=100, tags=tuple(tags)), frame="table")


def _scene(cfg, world, T_cam_world, hands_world: dict[str, np.ndarray]):
    """Compose faces for render: world tags + hand bundles given world-frame poses."""
    faces = []
    for spec in world.bundle.tags:
        faces.append((spec.id, spec.size_m, T_cam_world @ pose7_to_mat(spec.anchor_from_tag)))
    for side, T_wa in hands_world.items():
        for spec in cfg.bundles[side].tags:
            faces.append((spec.id, spec.size_m, T_cam_world @ T_wa @ pose7_to_mat(spec.anchor_from_tag)))
    return faces


def _angle_deg(T_a, T_b):
    return np.degrees((Rotation.from_matrix(T_a[:3, :3]).inv() * Rotation.from_matrix(T_b[:3, :3])).magnitude())


# Head camera 0.55 m above the table edge (y=-0.35), looking along +Y and tilted 40 deg down.
# Camera frame: +Z forward, +X right, +Y down.  World: +X right, +Y away, +Z up.
_S, _C = np.sin(np.radians(40)), np.cos(np.radians(40))
HEAD_WORLD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -_S, _C, -0.35],  # cam y -> world (0,-s,-c), cam z -> world (0,c,-s)
        [0.0, -_C, -_S, 0.55],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# hand anchors above the table with tag +Z pointing up-and-back toward the head (normal ~ (0,-0.7,0.7))
HANDS_WORLD = {
    "left": _pose([-0.15, 0.20, 0.12], [45, 0, 0]),
    "right": _pose([0.15, 0.20, 0.12], [45, 0, 0]),
}


def test_bundle_config_and_world_map_roundtrip(tmp_path):
    cfg = _config()
    again = BundleConfig.from_dict(cfg.to_dict())
    assert again.bundles["left"].tag_ids == (10, 11) and cfg.side_of(21) == "right" and cfg.side_of(99) is None
    world = _world_map()
    path = tmp_path / "world_map.yaml"
    world.write_yaml(path)
    back = WorldMap.from_yaml(path)
    assert back.tag_ids == (100, 101, 102, 103) and back.frame == "table"
    np.testing.assert_allclose(back.bundle.tag(101).anchor_from_tag[:3], [0.35, 0.45, 0.0], atol=1e-6)
    feats = apriltag_features()
    assert feats["observation.apriltag.world_corners_px"]["shape"] == (64,)
    assert feats["observation.apriltag.left_corners_px"]["shape"] == (32,)


def _run_scene(intr, hand_tag_m: float = 0.05):
    cfg, world = _config(hand_tag_m), _world_map()
    det = AprilTagDetector(cfg.dictionary)
    geom = CameraGeometry.for_intrinsics(intr)
    T_cam_world = np.linalg.inv(HEAD_WORLD)
    img = _render(_scene(cfg, world, T_cam_world, HANDS_WORLD), det, intr)
    result = process_frame(img, detector=det, config=cfg, geometry=geom, world_map=world)
    return cfg, world, geom, T_cam_world, result


def test_head_localisation_and_world_frame_eef_pinhole():
    _check_scene(INTR_PIN, pos_tol_mm=8.0, ang_tol_deg=1.5)


def test_head_localisation_and_world_frame_eef_fisheye():
    # 160 deg fisheye at 1080p has f~560 px: a 50 mm tag at 0.7 m is only ~40 px (4 px/module) and
    # starts dropping out, so the egocentric rig needs >= 70 mm hand tags (or closer hands).
    _check_scene(INTR_FISH, pos_tol_mm=25.0, ang_tol_deg=4.0, hand_tag_m=0.07)  # ~2 cm / ~3 deg: fisheye f~560 px limit


def test_fisheye_small_hand_tags_drop_out():
    cfg, world, geom, T_cam_world, result = _run_scene(INTR_FISH, hand_tag_m=0.04)
    tracked = sum(int(r.solve is not None) for r in result.hands.values())
    assert tracked < 2  # documents the size limit rather than hiding it


def _check_scene(intr, *, pos_tol_mm, ang_tol_deg, hand_tag_m=0.05):
    cfg, world, geom, T_cam_world, result = _run_scene(intr, hand_tag_m)
    ids = sorted(d.id for d in result.detections)
    assert {100, 101, 102, 103} & set(ids), ids
    assert result.world is not None and result.world.solve is not None, ids
    assert len(result.world.solve.tag_ids) >= 2
    T_est_cw = pose7_to_mat(result.world.solve.camera_from_anchor)
    assert np.linalg.norm(T_est_cw[:3, 3] - T_cam_world[:3, 3]) * 1000 < pos_tol_mm
    assert _angle_deg(T_est_cw, T_cam_world) < ang_tol_deg
    cal = ControllerTcpCalibration(left=IDENTITY_POSE7.copy(), right=IDENTITY_POSE7.copy())
    s = build_pair_sample(result, calibration=cal, capture_time_ns=5, sequence=1)
    assert s.hmd_tracked and s.left_tracked and s.right_tracked, ids
    for side in ("left", "right"):
        got = pose7_to_mat(getattr(s, f"{side}_controller_pose"))
        gt = HANDS_WORLD[side]
        assert np.linalg.norm(got[:3, 3] - gt[:3, 3]) * 1000 < pos_tol_mm, (side, got[:3, 3], gt[:3, 3])
        assert _angle_deg(got, gt) < ang_tol_deg
    np.testing.assert_allclose(pose7_to_mat(s.hmd_pose)[:3, 3], HEAD_WORLD[:3, 3], atol=pos_tol_mm / 1000)
    frame = s.tracking_frame()
    assert frame["observation.apriltag.world_num_tags"][0] >= 2
    assert frame["observation.apriltag.corner_space"][0] == (1 if intr.model == "fisheye" else 0)


def test_world_not_visible_marks_frame_invalid():
    cfg, world = _config(), _world_map()
    det = AprilTagDetector(cfg.dictionary)
    geom = CameraGeometry.for_intrinsics(INTR_PIN)
    T_cam_world = np.linalg.inv(HEAD_WORLD)
    img = _render(_scene(cfg, world, T_cam_world, HANDS_WORLD), det, INTR_PIN)
    # a world map expecting tags nobody printed -> world anchor invisible
    ghost = WorldMap(bundle=BundleSpec(side="world", anchor_tag=200, tags=(TagSpec(id=200, size_m=0.08),)))
    result = process_frame(img, detector=det, config=cfg, geometry=geom, world_map=ghost)
    cal = ControllerTcpCalibration(left=IDENTITY_POSE7.copy(), right=IDENTITY_POSE7.copy())
    s = build_pair_sample(result, calibration=cal, capture_time_ns=1, sequence=1, require_world=True)
    assert not s.hmd_tracked and not s.left_tracked and not s.right_tracked
    assert s.left_device_tracked  # the hand *was* seen; only the world anchor is missing
    # static-camera fallback keeps the frame usable
    s2 = build_pair_sample(result, calibration=cal, capture_time_ns=1, sequence=1, fixed_world_from_camera=mat_to_pose7(HEAD_WORLD))
    assert s2.left_tracked and not s2.hmd_tracked
    assert np.linalg.norm(pose7_to_mat(s2.left_controller_pose)[:3, 3] - HANDS_WORLD["left"][:3, 3]) < 0.01


def test_single_visible_tag_still_solves():
    cfg = _config()
    det = AprilTagDetector(cfg.dictionary)
    geom = CameraGeometry.for_intrinsics(INTR_PIN)
    T_ct = _pose([0.0, 0.0, 0.5], [180, 0, 0])
    img = _render([(10, 0.05, T_ct)], det, INTR_PIN)
    result = process_frame(img, detector=det, config=cfg, geometry=geom)
    r = result.hands["left"]
    assert r.solve is not None and r.solve.tag_ids == (10,)
    assert np.linalg.norm(pose7_to_mat(r.solve.camera_from_anchor)[:3, 3] - T_ct[:3, 3]) < 0.005
    assert solve_bundle([], cfg.bundles["left"], geom) is None


def test_bundle_geometry_calibration_recovers_second_face():
    cfg = _config()
    det = AprilTagDetector(cfg.dictionary)
    geom = CameraGeometry.for_intrinsics(INTR_PIN)
    true_bundle = cfg.bundles["left"]
    wrong = BundleSpec.from_dict("left", {"anchor_tag": 10, "tags": [{"id": 10, "size_m": 0.05}, {"id": 11, "size_m": 0.05, "measured": False}]})
    frames = []
    for k in range(6):
        T_ca = _pose([-0.1 + 0.03 * k, 0.02 * k, 0.5 + 0.02 * k], [180 - 25 + 6 * k, -10 + 4 * k, 5 * k])
        faces = [(s.id, s.size_m, T_ca @ pose7_to_mat(s.anchor_from_tag)) for s in true_bundle.tags]
        frames.append(det.detect(_render(faces, det, INTR_PIN)))
    fixed, metrics = calibrate_bundle_geometry(frames, wrong, geom)
    est = pose7_to_mat(fixed.tag(11).anchor_from_tag)
    true = pose7_to_mat(true_bundle.tag(11).anchor_from_tag)
    assert np.linalg.norm(est[:3, 3] - true[:3, 3]) * 1000 < 4.0
    assert _angle_deg(est, true) < 2.0 and fixed.tag(11).measured and metrics["11"]["views"] >= 3


def test_world_map_calibration_from_reference_poses():
    cfg, world = _config(), _world_map()
    det = AprilTagDetector(cfg.dictionary)
    geom = CameraGeometry.for_intrinsics(INTR_PIN)
    frames = []
    for k in range(8):
        head = HEAD_WORLD @ _pose([0.05 * (k % 3) - 0.05, 0.03 * k, 0.02 * k], [3 * k, -2 * k, 4 * k])
        T_cam_world = np.linalg.inv(head)
        img = _render(_scene(cfg, world, T_cam_world, {}), det, INTR_PIN)
        frames.append((det.detect(img), mat_to_pose7(T_cam_world)))  # reference = ChArUco board in practice
    est, metrics = calibrate_world_map(frames, {i: 0.08 for i in (100, 101, 102, 103)}, geom, frame_name="table", anchor_tag=100)
    assert set(est.tag_ids) == {100, 101, 102, 103}
    for tid in est.tag_ids:
        e = pose7_to_mat(est.bundle.tag(tid).anchor_from_tag)
        t = pose7_to_mat(world.bundle.tag(tid).anchor_from_tag)
        assert np.linalg.norm(e[:3, 3] - t[:3, 3]) * 1000 < 10.0, (tid, e[:3, 3], t[:3, 3])
        assert _angle_deg(e, t) < 2.0
        assert metrics[str(tid)]["views"] >= 3


def test_fisheye_geometry_pixel_roundtrip():
    geom = CameraGeometry.for_intrinsics(INTR_FISH)
    assert geom.rectify and geom.corner_space == 1
    pts_cam = np.array([[0.1, -0.05, 0.6], [-0.3, 0.2, 0.7], [0.0, 0.0, 1.0]])
    raw = _project(pts_cam, INTR_FISH)
    rect = (geom.K @ (pts_cam / pts_cam[:, 2:3]).T).T[:, :2]
    back = geom.to_raw_pixels(rect)
    np.testing.assert_allclose(back, raw, atol=0.05)
