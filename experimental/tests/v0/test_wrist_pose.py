from __future__ import annotations

import numpy as np

from ego_collector.tracking.detector import OpenCVTagDetector
from ego_collector.tracking.transforms import T_from_xyz_rpy, rotation_angle_deg, translation_distance
from ego_collector.tracking.world_pose import solve_camera_pose
from ego_collector.tracking.wrist_pose import WristExtrinsics, solve_wrists
from synth import INTR, head_pose, render, scene_faces, world_map, wrist_pose


def test_world_frame_wrist_poses_from_head_camera():
    wm = world_map()
    wr = WristExtrinsics.default("tag36h11")
    T_wc = head_pose(x=-0.03, y=-0.28, z=0.62, tilt_deg=50, yaw_deg=-3)
    gt = {"left": wrist_pose(-0.15, 0.30, 0.10), "right": wrist_pose(0.18, 0.33, 0.14)}
    wrists = {10: (0.05, gt["left"]), 20: (0.05, gt["right"])}
    img = render(scene_faces(wm, T_wc, wrists), INTR)
    dets = OpenCVTagDetector("tag36h11").detect(img)
    cam = solve_camera_pose(dets, wm, INTR)
    assert cam is not None
    solved = solve_wrists(dets, wr, INTR, cam.T_world_camera)
    assert set(solved) == {"left", "right"}
    for side in ("left", "right"):
        T = solved[side].T_world_wrist
        assert T is not None
        assert translation_distance(T, gt[side]) * 1000 < 10.0, (side, T[:3, 3], gt[side][:3, 3])
        assert rotation_angle_deg(T, gt[side]) < 2.0
    # without camera localisation the camera-frame pose is still available
    solved_cam = solve_wrists(dets, wr, INTR, None)
    assert solved_cam["left"].T_world_wrist is None and solved_cam["left"].T_camera_wrist is not None


def test_wrist_extrinsics_yaml_and_offset(tmp_path):
    y = tmp_path / "w.yaml"
    y.write_text(
        "tag_family: tag36h11\nleft: {tag_id: 10, tag_size_m: 0.05, T_tag_wrist: {xyz: [0, 0, -0.03], rpy: [0, 0, 0]}}\n"
        "right: {tag_id: 20, tag_size_m: 0.05}\n"
    )
    wr = WristExtrinsics.from_yaml(y)
    assert wr.side_of(20) == "right" and wr.side_of(99) is None
    np.testing.assert_allclose(wr.sides["left"].T_tag_wrist[:3, 3], [0, 0, -0.03])
    np.testing.assert_allclose(wr.sides["right"].T_tag_wrist, np.eye(4))
    # the offset is applied along the tag frame axes
    wm = world_map()
    T_wc = head_pose()
    T_tag = wrist_pose(0.0, 0.30, 0.12)
    img = render(scene_faces(wm, T_wc, {10: (0.05, T_tag)}), INTR)
    dets = OpenCVTagDetector("tag36h11").detect(img)
    cam = solve_camera_pose(dets, wm, INTR)
    w = solve_wrists(dets, wr, INTR, cam.T_world_camera)["left"]
    expected = T_tag @ T_from_xyz_rpy([0, 0, -0.03], [0, 0, 0])
    assert translation_distance(w.T_world_wrist, expected) * 1000 < 10.0
