"""Synthetic ChArUco calibration: render the board at known poses with a known K, recover K."""

from __future__ import annotations

import cv2
import numpy as np

from ego_collector.camera.calibration import CharucoSpec, calibrate_pinhole, detect_charuco
from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.tracking.transforms import T_from_xyz_rpy

W, H = 1280, 720
K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]])
DIST = np.array([0.08, -0.12, 0.001, -0.001, 0.0])


def _render_board(spec: CharucoSpec, T_cam_board: np.ndarray) -> np.ndarray:
    px_per_m = 4000.0
    board_img = spec.image(px_per_m, margin_px=0)
    bw, bh = spec.squares_x * spec.square_length_m, spec.squares_y * spec.square_length_m
    # board image corners (pixels) <-> board-frame metres (x right, y down in the image == +y in board coords)
    obj = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], dtype=np.float64)
    pts_cam = (T_cam_board[:3, :3] @ obj.T).T + T_cam_board[:3, 3]
    proj, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)
    src = np.array([[0, 0], [board_img.shape[1], 0], [board_img.shape[1], board_img.shape[0]], [0, board_img.shape[0]]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src, proj.reshape(-1, 2).astype(np.float32))
    canvas = np.full((H, W), 128, dtype=np.uint8)
    warped = cv2.warpPerspective(board_img, Hm, (W, H), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(board_img, 255), Hm, (W, H), borderValue=0) > 127
    canvas[mask] = warped[mask]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_intrinsics_yaml_roundtrip(tmp_path):
    intr = CameraIntrinsics(image_width=W, image_height=H, camera_matrix=K, distortion_coefficients=DIST, rms_reprojection_error=0.3, num_views=12)
    path = intr.save(tmp_path / "cam.yaml")
    back = CameraIntrinsics.load(path)
    np.testing.assert_allclose(back.camera_matrix, K)
    np.testing.assert_allclose(back.distortion_coefficients, DIST)
    assert back.image_width == W and back.model == "pinhole"
    pts = np.array([[100.0, 80.0], [1200.0, 700.0], [640.0, 360.0]])
    und = back.undistort_points(pts)
    assert np.linalg.norm(und[2] - pts[2]) < 1e-3  # principal point unchanged
    assert np.linalg.norm(und[0] - pts[0]) > 1.0  # corners move


def test_synthetic_charuco_calibration_recovers_intrinsics():
    spec = CharucoSpec()
    detections = []
    rng = np.random.default_rng(0)
    for k in range(16):
        T = T_from_xyz_rpy(
            [-0.12 + 0.02 * (k % 5) + rng.normal(0, 0.01), -0.14 + 0.03 * (k % 4), 0.45 + 0.04 * (k % 3)],
            [np.radians(180 - 25 + 6 * (k % 6)), np.radians(-20 + 5 * (k % 7)), np.radians(10 * (k % 3) - 10)],
        )
        # board +Z points into the paper; the camera looks at the printed side, so flip about X
        T = T @ T_from_xyz_rpy([0, 0, 0], [np.pi, 0, 0])
        img = _render_board(spec, T)
        det = detect_charuco(img, spec)
        if det is not None:
            detections.append(det)
    assert len(detections) >= 10, len(detections)
    intr = calibrate_pinhole(detections, (W, H))
    assert abs(intr.fx - 1000) < 15 and abs(intr.fy - 1000) < 15, (intr.fx, intr.fy)
    assert abs(intr.cx - 640) < 15 and abs(intr.cy - 360) < 15
    # the homography-based render only distorts the 4 board corners exactly, so k1 is not recoverable here
    assert abs(intr.distortion_coefficients[0]) < 0.2
    assert intr.mean_reprojection_error < 1.0
