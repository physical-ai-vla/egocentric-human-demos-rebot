"""Synthetic tag scenes for tests: render tag36h11 markers at known poses with a known camera."""

from __future__ import annotations

import cv2
import numpy as np

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.tracking.detector import OpenCVTagDetector
from ego_collector.tracking.pnp import tag_corners_local
from ego_collector.tracking.transforms import T_from_xyz_rpy, invert_T
from ego_collector.tracking.world_pose import WorldTagMap

W, H = 1920, 1080
K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1]])  # C922-like at 1080p
DIST = np.array([0.06, -0.10, 0.0005, -0.0005, 0.0])
INTR = CameraIntrinsics(image_width=W, image_height=H, camera_matrix=K, distortion_coefficients=DIST, rms_reprojection_error=0.0, mean_reprojection_error=0.0)
INTR_IDEAL = CameraIntrinsics.ideal(W, H, 1400.0)

_DET = OpenCVTagDetector("tag36h11")


def world_map() -> WorldTagMap:
    tags = {
        100: T_from_xyz_rpy([-0.35, 0.10, 0.0], [0, 0, 0]),
        101: T_from_xyz_rpy([0.35, 0.10, 0.0], [0, 0, 0]),
        102: T_from_xyz_rpy([-0.35, 0.70, 0.0], [0, 0, 0]),
        103: T_from_xyz_rpy([0.35, 0.70, 0.0], [0, 0, 0]),
    }
    return WorldTagMap(tag_family="tag36h11", tag_size_m=0.07, tags=tags, measured=True)


def head_pose(x=0.0, y=-0.25, z=0.65, tilt_deg=45.0, yaw_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """T_world_camera: camera above the near table edge looking along +Y and tilted down.

    Camera frame: +Z forward, +X right, +Y down. World: +X right, +Y away, +Z up.
    """
    s, c = np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))
    R = np.array([[1, 0, 0], [0, -s, c], [0, -c, -s]])  # cam y -> (0,-s,-c), cam z -> (0,c,-s)
    T = np.eye(4)
    T[:3, :3] = R @ T_from_xyz_rpy([0, 0, 0], [np.radians(roll_deg), np.radians(yaw_deg), 0])[:3, :3]
    T[:3, 3] = [x, y, z]
    return T


def wrist_pose(x, y, z=0.10, face_deg=45.0) -> np.ndarray:
    """T_world_tag for a wrist tag whose normal points up-and-back toward the head (rotvec about x)."""
    return T_from_xyz_rpy([x, y, z], [np.radians(face_deg), 0, 0])


def render(faces: list[tuple[int, float, np.ndarray]], intr: CameraIntrinsics = INTR, *, bg: int = 120) -> np.ndarray:
    """faces = [(tag_id, black_size_m, T_cam_tag)] -> BGR image (far faces first, back-facing culled)."""
    canvas = np.full((H, W), bg, dtype=np.uint8)
    M, P = 160, 20  # marker px in the source image, white quiet-zone px
    vis = []
    for tid, size, T_ct in faces:
        n = T_ct[:3, :3] @ np.array([0, 0, 1.0])
        if np.dot(n, T_ct[:3, 3]) > 0 or T_ct[2, 3] <= 0.05:
            continue
        vis.append((float(T_ct[2, 3]), tid, size, T_ct))
    for _, tid, size, T_ct in sorted(vis, key=lambda f: -f[0]):
        scale = (M + 2 * P) / M
        obj = tag_corners_local(size * scale)
        pts_cam = (T_ct[:3, :3] @ obj.T).T + T_ct[:3, 3]
        proj = intr.project(pts_cam)
        marker = _DET.marker_image(tid, M)
        padded = np.full((M + 2 * P, M + 2 * P), 255, dtype=np.uint8)
        padded[P : P + M, P : P + M] = marker
        src = np.array([[0, 0], [M + 2 * P, 0], [M + 2 * P, M + 2 * P], [0, M + 2 * P]], dtype=np.float32)
        Hm = cv2.getPerspectiveTransform(src, proj.astype(np.float32))
        warped = cv2.warpPerspective(padded, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(padded, 255), Hm, (W, H), borderValue=0) > 127
        canvas[mask] = warped[mask]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def scene_faces(wm: WorldTagMap, T_world_camera: np.ndarray, wrists: dict[int, tuple[float, np.ndarray]]) -> list[tuple[int, float, np.ndarray]]:
    """World tags + wrist tags (id -> (size, T_world_tag)) as camera-frame faces."""
    T_cw = invert_T(T_world_camera)
    faces = [(tid, wm.size_of(tid), T_cw @ T) for tid, T in wm.tags.items()]
    faces += [(tid, size, T_cw @ T) for tid, (size, T) in wrists.items()]
    return faces
