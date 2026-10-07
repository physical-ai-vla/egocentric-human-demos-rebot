from __future__ import annotations

import numpy as np
import pytest

from ego_collector.tracking.detector import OpenCVTagDetector, make_detector
from ego_collector.tracking.transforms import rotation_angle_deg, translation_distance
from ego_collector.tracking.world_pose import WorldTagMap, solve_camera_pose
from synth import INTR, head_pose, render, scene_faces, world_map


def test_world_map_yaml_roundtrip(tmp_path):
    wm = world_map()
    wm.save(tmp_path / "wm.yaml")
    back = WorldTagMap.from_yaml(tmp_path / "wm.yaml")
    assert back.ids == (100, 101, 102, 103) and back.tag_size_m == 0.07 and back.tag_family == "tag36h11"
    np.testing.assert_allclose(back.tags[103], wm.tags[103], atol=1e-9)
    c = back.corners_world(100)
    assert c.shape == (4, 3) and np.allclose(c.mean(axis=0), [-0.35, 0.10, 0.0])


@pytest.mark.parametrize("backend", ["opencv", "pupil"])
def test_camera_localisation_from_world_tags(backend):
    wm = world_map()
    T_wc = head_pose(x=0.05, y=-0.30, z=0.60, tilt_deg=48, yaw_deg=4)
    img = render(scene_faces(wm, T_wc, {}), INTR)
    try:
        det = make_detector("tag36h11", backend=backend)
    except ImportError:
        pytest.skip("pupil-apriltags not installed")
    dets = det.detect(img)
    ids = sorted(d.tag_id for d in dets)
    assert len(set(ids) & {100, 101, 102, 103}) >= 3, ids  # one corner tag may fall outside the view
    solve = solve_camera_pose(dets, wm, INTR)
    assert solve is not None and solve.tag_count >= 3 and solve.corner_count == 4 * solve.tag_count
    assert translation_distance(solve.T_world_camera, T_wc) * 1000 < 6.0, (backend, solve.T_world_camera[:3, 3], T_wc[:3, 3])
    assert rotation_angle_deg(solve.T_world_camera, T_wc) < 0.6
    assert solve.reprojection_error_px < 1.0


def test_pupil_and_opencv_corner_order_agree():
    try:
        pupil = make_detector("tag36h11", backend="pupil")
    except ImportError:
        pytest.skip("pupil-apriltags not installed")
    wm = world_map()
    img = render(scene_faces(wm, head_pose(), {}), INTR)
    a = {d.tag_id: d.corners for d in OpenCVTagDetector("tag36h11").detect(img)}
    b = {d.tag_id: d.corners for d in pupil.detect(img)}
    common = sorted(set(a) & set(b))
    assert len(common) >= 3
    for tid in common:
        assert np.abs(a[tid] - b[tid]).max() < 1.5, (tid, a[tid], b[tid])


def test_single_world_tag_and_none():
    wm = world_map()
    T_wc = head_pose()
    only_100 = WorldTagMap(tag_family="tag36h11", tag_size_m=0.07, tags={100: wm.tags[100]})
    img = render(scene_faces(wm, T_wc, {}), INTR)
    dets = OpenCVTagDetector("tag36h11").detect(img)
    solve = solve_camera_pose(dets, only_100, INTR)
    assert solve is not None and solve.tag_count == 1
    assert translation_distance(solve.T_world_camera, T_wc) * 1000 < 25.0  # one 7 cm tag: cm-level, as expected
    assert solve_camera_pose([d for d in dets if d.tag_id > 200], wm, INTR) is None
