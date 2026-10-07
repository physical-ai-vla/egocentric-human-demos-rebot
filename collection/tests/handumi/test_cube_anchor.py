"""Cube-face anchor: does a known pose come back out, and does detection refuse what it should?

These are synthetic on purpose. The real-data question -- is a cube a usable anchor for wrist translation -- is a
measurement, and it is worthless if the solver itself is wrong. An earlier version of this solve returned confident
poses with several hundred pixels of reprojection because the object-point order did not match what
SOLVEPNP_IPPE_SQUARE documents, and nothing failed; it just answered wrongly."""
import cv2
import numpy as np
import pytest

from handumi_collector.pose.cube_anchor import CUBE_M, Face, detect_faces, pair_distances, solve_face

K = np.array([[780.0, 0, 960.0], [0, 780.0, 540.0], [0, 0, 1.0]])
D0 = np.zeros((4, 1))


def _project(t_xyz, rvec=(0.0, 0.0, 0.0), size=CUBE_M, K=K):
    obj = np.array([[-size / 2,  size / 2, 0], [size / 2,  size / 2, 0],
                    [ size / 2, -size / 2, 0], [-size / 2, -size / 2, 0]], np.float64)
    pts, _ = cv2.projectPoints(obj, np.asarray(rvec, np.float64), np.asarray(t_xyz, np.float64), K, None)
    return pts.reshape(4, 2)


@pytest.mark.parametrize("t_true", [(0.0, 0.0, 0.30), (0.05, -0.03, 0.45), (-0.08, 0.06, 0.22)])
def test_a_known_face_pose_comes_back(t_true):
    f = Face("red", _project(t_true), area_px=5000.0, solidity=0.99, clipped=False)
    got = solve_face(f, K, D0, fisheye=False)
    assert got is not None
    assert np.allclose(got.t, t_true, atol=1e-3), f"{got.t} vs {t_true}"
    assert got.rms_px < 0.5, got.rms_px


def test_the_solve_is_indifferent_to_which_corner_came_first():
    """A square is symmetric under 90 degree turns and a blob has no canonical first corner, so every rotation and
    both windings have to give the same centre. Guessing one and hoping is what produced 300 px reprojections."""
    t_true = (0.02, 0.01, 0.35)
    base = _project(t_true)
    seen = []
    for k in range(4):
        for pts in (np.roll(base, k, axis=0), np.roll(base, k, axis=0)[::-1]):
            got = solve_face(Face("blue", pts, 4000.0, 0.99, False), K, D0, fisheye=False)
            assert got is not None and got.rms_px < 0.5
            seen.append(got.t)
    assert np.allclose(np.ptp(np.array(seen), axis=0), 0, atol=1e-6), np.array(seen)


def test_a_tilted_face_still_recovers_its_centre():
    t_true = (0.0, 0.0, 0.33)
    f = Face("purple", _project(t_true, rvec=(0.5, -0.35, 0.2)), 3000.0, 0.99, False)
    got = solve_face(f, K, D0, fisheye=False)
    assert got is not None and np.allclose(got.t, t_true, atol=2e-3), got.t


def test_range_error_scales_the_way_the_geometry_says():
    """dZ ~= Z^2 dpx / (f s). This is the floor a perfect detector would still live with, and the reason one 50 mm
    square is a weak depth constraint: at 0.3 m it is ~2.3 mm per pixel of corner error."""
    t_true = np.array([0.0, 0.0, 0.30])
    pts = _project(t_true)
    edge_px = float(np.linalg.norm(pts[1] - pts[0]))
    # Range follows APPARENT SIZE, so the perturbation has to be a size one: shifting a single corner sideways is
    # absorbed as translation and moves the depth almost not at all (0.24 mm, measured), which is why this test
    # first claimed the geometry was 10x better than it is.
    pts = pts.mean(axis=0) + (pts - pts.mean(axis=0)) * ((edge_px - 1.0) / edge_px)
    got = solve_face(Face("red", pts, 5000.0, 0.99, False), K, D0, fisheye=False)
    assert got is not None
    predicted_mm = 1000 * t_true[2] ** 2 / (K[0, 0] * CUBE_M)      # ~2.3 mm per pixel at 0.3 m
    err_mm = abs(got.t[2] - t_true[2]) * 1000
    assert 0.5 * predicted_mm < err_mm < 2.0 * predicted_mm, f"{err_mm:.2f} mm vs predicted {predicted_mm:.2f}"


def _canvas(colour_bgr, quad, size=(1920, 1080)):
    img = np.full((size[1], size[0], 3), 250, np.uint8)
    cv2.fillPoly(img, [np.asarray(quad, np.int32)], colour_bgr)
    return img


def test_detection_finds_a_cube_face_and_reports_its_geometry():
    quad = [(900, 500), (1020, 500), (1020, 620), (900, 620)]
    faces = detect_faces(_canvas((40, 40, 200), quad))       # BGR red
    assert "red" in faces, list(faces)
    f = faces["red"]
    assert f.area_px > 10_000 and f.solidity > 0.9 and not f.clipped


def test_strict_detection_refuses_a_face_running_off_the_edge():
    """A clipped face's corners are not the cube's corners, so its pose would be confidently wrong."""
    quad = [(-30, 500), (90, 500), (90, 620), (-30, 620)]
    img = _canvas((200, 60, 40), quad)                       # BGR blue
    assert "blue" not in detect_faces(img, strict=True)
    loose = detect_faces(img, strict=False)
    assert "blue" in loose and loose["blue"].clipped


def test_pair_distances_measure_the_scene_not_the_camera():
    from handumi_collector.pose.cube_anchor import FacePose
    poses = {"red": FacePose("red", np.array([0.0, 0, 0.3]), np.eye(3), 0.1),
             "blue": FacePose("blue", np.array([0.2, 0, 0.3]), np.eye(3), 0.1)}
    assert pytest.approx(pair_distances(poses)["blue-red"], abs=1e-9) == 0.2
