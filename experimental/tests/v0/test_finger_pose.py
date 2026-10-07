"""Finger (thumb/index) tag tracking -> fingertip points, aperture, pinch centre."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ego_collector.hands.grasp import ApertureCalibration
from ego_collector.tracking.detector import OpenCVTagDetector
from ego_collector.tracking.transforms import T_from_xyz_rpy
from ego_collector.tracking.world_pose import solve_camera_pose
from ego_collector.tracking.wrist_pose import WristExtrinsics, solve_fingers
from synth import INTR, head_pose, render, scene_faces, world_map, wrist_pose

FINGER_SIZE = 0.016


def finger_tag_pose(x, y, z, face_deg=50.0):
    return T_from_xyz_rpy([x, y, z], [np.radians(face_deg), 0, 0])


def _scene(th, ix):
    wm = world_map()
    faces = {20: (0.035, wrist_pose(0.10, 0.32, 0.16)), 21: (FINGER_SIZE, th), 22: (FINGER_SIZE, ix)}
    T_wc = head_pose(z=0.55, tilt_deg=52)
    return render(scene_faces(wm, T_wc, faces), INTR), wm, T_wc


def test_fingertips_and_aperture_world_frame():
    th = finger_tag_pose(0.05, 0.30, 0.12)
    ix = finger_tag_pose(0.11, 0.31, 0.13)
    img, wm, _ = _scene(th, ix)
    dets = OpenCVTagDetector("tag36h11").detect(img)
    cam = solve_camera_pose(dets, wm, INTR)
    assert cam is not None
    wr = WristExtrinsics.default("tag36h11", with_fingers=True, finger_size_m=FINGER_SIZE)
    solved = solve_fingers(dets, wr, INTR, cam.T_world_camera)
    assert set(solved) == {("right", "thumb"), ("right", "index")}
    gt_ap = np.linalg.norm(th[:3, 3] - ix[:3, 3])
    p_t, p_i = solved[("right", "thumb")].p_world, solved[("right", "index")].p_world
    assert np.linalg.norm(p_t - th[:3, 3]) * 1000 < 8.0
    assert abs(np.linalg.norm(p_t - p_i) - gt_ap) * 1000 < 5.0  # grip-width error budget from the spec
    # without camera localisation the camera-frame point is still there and aperture is frame-invariant
    solved_cam = solve_fingers(dets, wr, INTR, None)
    ap_cam = np.linalg.norm(solved_cam[("right", "thumb")].p_camera - solved_cam[("right", "index")].p_camera)
    assert abs(ap_cam - gt_ap) * 1000 < 5.0
    assert solved_cam[("right", "thumb")].p_world is None


def test_fingertip_offset_applied_in_tag_frame():
    th = finger_tag_pose(0.05, 0.30, 0.12)
    ix = finger_tag_pose(0.11, 0.31, 0.13)
    img, wm, _ = _scene(th, ix)
    dets = OpenCVTagDetector("tag36h11").detect(img)
    cam = solve_camera_pose(dets, wm, INTR)
    wr = WristExtrinsics.default("tag36h11", with_fingers=True, finger_size_m=FINGER_SIZE)
    offset = T_from_xyz_rpy([0.0, -0.015, -0.008], [0, 0, 0])
    spec = wr.fingers["right"]["thumb"]
    wr.fingers["right"]["thumb"] = type(spec)(spec.side, spec.finger, spec.tag_id, spec.tag_size_m, offset)
    solved = solve_fingers(dets, wr, INTR, cam.T_world_camera)
    expected = (th @ offset)[:3, 3]
    assert np.linalg.norm(solved[("right", "thumb")].p_world - expected) * 1000 < 8.0


def test_extrinsics_yaml_fingers(tmp_path):
    y = tmp_path / "w.yaml"
    y.write_text(
        "tag_family: tag36h11\n"
        "left: {tag_id: 10, tag_size_m: 0.028}\n"
        "right:\n  tag_id: 20\n  tag_size_m: 0.028\n"
        "  thumb: {tag_id: 21, tag_size_m: 0.016, T_tag_tip: {xyz: [0, -0.015, 0]}}\n"
        "  index: {tag_id: 22, tag_size_m: 0.016}\n"
    )
    wr = WristExtrinsics.from_yaml(y)
    assert wr.has_fingers and wr.finger_of(21) == ("right", "thumb") and wr.finger_of(99) is None
    assert not wr.fingers.get("left")
    np.testing.assert_allclose(wr.fingers["right"]["thumb"].T_tag_tip[:3, 3], [0, -0.015, 0])
    # a config without finger blocks has none
    y2 = tmp_path / "w2.yaml"
    y2.write_text("tag_family: tag36h11\nleft: {tag_id: 10, tag_size_m: 0.028}\nright: {tag_id: 20, tag_size_m: 0.028}\n")
    assert not WristExtrinsics.from_yaml(y2).has_fingers


def test_aperture_calibration_mapping():
    cal = ApertureCalibration(right_open_m=0.09, right_closed_m=0.01)
    a = np.array([0.01, 0.05, 0.09, 0.12, np.nan])
    g = cal.grasp("right", a)
    np.testing.assert_allclose(g[:4], [0.0, 0.5, 1.0, 1.0])
    assert np.isnan(g[4])
    cal2 = ApertureCalibration.from_samples(
        {"right": np.full(20, 0.088) + np.linspace(0, 0.004, 20)}, {"right": np.full(20, 0.012) - np.linspace(0, 0.004, 20)}
    )
    assert 0.085 < cal2.right_open_m < 0.095 and 0.005 < cal2.right_closed_m < 0.013
    assert cal2.left_open_m == ApertureCalibration().left_open_m  # left untouched without samples
