"""Head RGB-D <-> wrist RGB metric anchors — the part of the IMU-less V0 that supplies SCALE.

    head RGB + metric depth        wrist RGB (fisheye -> pinhole)
              |                              |
              +---------- matched features --+
              |                              |
      3D point in head frame  <---->  2D pixel in wrist camera
              |                              |
              +-------- RANSAC PnP ----------+
                             |
                        T_head_wrist   (METRIC, from the depth camera's own scale)

Why this exists: a monocular wrist VO trajectory has no metric scale, so its translations cannot be an action label.
The head Orbbec already measures metric 3D, so every frame where the two cameras see the same surface yields a metric
pose anchor for the wrist — no IMU required. Anchors are SPARSE and are expected to fail (FOV overlap, occlusion, blur,
texture); the wrist VO carries the trajectory between them (see metric_align.py).

Reuses: rgbd_io (head reader, backproject/project), backends.opencv_vo.FisheyeRectifier (Kannala-Brandt -> pinhole),
se3. Nothing here estimates a trajectory: one pair in, one anchor out."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from .rgbd_io import CameraIntrinsics
from .se3 import inv_T, make_T

# Provisional gates. They are NOT measured yet: freeze them against the first real Vpilot distribution, the same way
# the depth-tracker gates are handled. Everything is reported raw so re-freezing never needs a re-run.
DEFAULT_GATES = dict(min_matches=40, min_depth_valid=25, min_inliers=15, min_inlier_ratio=0.30,
                     max_reprojection_px=4.0, ratio_test=0.75, max_depth_m=2.5, min_depth_m=0.15)


@dataclass
class CrossViewAnchor:
    """One metric pose anchor, or the recorded reason there is none. A failed anchor is data, not an error."""
    t_ns: int
    side: str
    head_frame_index: int = -1
    wrist_frame_index: int = -1
    T_head_wrist: np.ndarray | None = None       # 4x4, metric, wrist camera expressed in the head camera frame
    n_keypoints_head: int = 0
    n_keypoints_wrist: int = 0
    n_matches: int = 0
    n_depth_valid: int = 0                        # matches whose head pixel has usable depth
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error_px: float | None = None
    depth_median_m: float | None = None
    valid: bool = False
    reason: str = ""

    def to_row(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "T_head_wrist"}
        if self.T_head_wrist is not None:
            from .se3 import T_to_pose7
            p = T_to_pose7(self.T_head_wrist)
            d.update(x=p[0], y=p[1], z=p[2], qx=p[3], qy=p[4], qz=p[5], qw=p[6])
        else:
            d.update(x=np.nan, y=np.nan, z=np.nan, qx=np.nan, qy=np.nan, qz=np.nan, qw=np.nan)
        return d


def _detector(kind: str = "sift", n: int = 3000):
    import cv2
    if kind == "sift":
        return cv2.SIFT_create(nfeatures=n), cv2.NORM_L2
    if kind == "orb":
        return cv2.ORB_create(nfeatures=n), cv2.NORM_HAMMING
    raise ValueError(f"unknown detector {kind!r}")


def match_features(img_a, img_b, *, detector: str = "sift", ratio: float = 0.75, n_features: int = 3000,
                   mask_a=None, mask_b=None):
    """Lowe-ratio matched keypoints between two images. Returns (pts_a (N,2), pts_b (N,2), n_kp_a, n_kp_b).

    A wide-baseline, different-lens, different-exposure pair is the hard case SIFT exists for; ORB is offered for speed
    but is expected to do worse here. The wrist image must already be rectified to a pinhole model."""
    import cv2
    det, norm = _detector(detector, n_features)
    ga = img_a if img_a.ndim == 2 else cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = img_b if img_b.ndim == 2 else cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    # Masks keep detection out of a rectified image's extrapolated periphery (FisheyeRectifier.valid_mask). Passing the
    # mask here, rather than filtering matches later, is deliberate: by the time a match exists its bearing is already
    # wrong, and the descriptor may have matched on the mask's own black seam.
    ka, da = det.detectAndCompute(ga, mask_a)
    kb, db = det.detectAndCompute(gb, mask_b)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), len(ka or []), len(kb or [])
    pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    if not good:
        return np.zeros((0, 2)), np.zeros((0, 2)), len(ka), len(kb)
    pa = np.array([ka[m.queryIdx].pt for m in good], np.float64)
    pb = np.array([kb[m.trainIdx].pt for m in good], np.float64)
    return pa, pb, len(ka), len(kb)


def solve_pnp_metric(P_A: np.ndarray, pts_B: np.ndarray, K_B, *, max_reprojection_px: float = 4.0,
                     min_inliers: int = 15, min_inlier_ratio: float = 0.30):
    """3D points in frame A + their 2D observations in camera B  ->  (T_A_B, stats).

    The one piece of metric geometry the IMU-less pipeline needs, shared by the head<->wrist anchor and by head
    temporal odometry (the same math: 3D from one frame's depth, 2D in another view). solvePnP returns the pose of the
    A-frame points in camera B, i.e. T_B_A, so the returned pose is its inverse — where camera B sits in frame A.
    Returns (None, stats) when the gates are not met; `stats['reason']` always says why."""
    import cv2
    P_A = np.ascontiguousarray(P_A, np.float64).reshape(-1, 3)
    pts_B = np.ascontiguousarray(pts_B, np.float64).reshape(-1, 2)
    K = np.asarray(K_B, np.float64)
    st = dict(n_points=len(P_A), n_inliers=0, inlier_ratio=0.0, reprojection_error_px=None, reason="")
    if len(P_A) < max(min_inliers, 6):
        st["reason"] = f"only {len(P_A)} 3D-2D correspondences"
        return None, st
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(P_A, pts_B, K, None, reprojectionError=float(max_reprojection_px),
                                                 iterationsCount=500, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error as exc:
        st["reason"] = f"solvePnPRansac failed: {exc}"
        return None, st
    if not ok or inl is None or len(inl) < min_inliers:
        st["n_inliers"] = 0 if inl is None else int(len(inl))
        st["inlier_ratio"] = st["n_inliers"] / max(len(P_A), 1)
        st["reason"] = f"PnP found {st['n_inliers']} inliers (< {min_inliers})"
        return None, st
    idx = inl.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(P_A[idx], pts_B[idx], K, None, rvec, tvec)
    st["n_inliers"] = int(len(idx))
    st["inlier_ratio"] = st["n_inliers"] / max(len(P_A), 1)
    proj, _ = cv2.projectPoints(P_A[idx], rvec, tvec, K, None)
    st["reprojection_error_px"] = float(np.linalg.norm(proj.reshape(-1, 2) - pts_B[idx], axis=1).mean())
    if st["inlier_ratio"] < min_inlier_ratio:
        st["reason"] = f"inlier ratio {st['inlier_ratio']:.2f} < {min_inlier_ratio}"
        return None, st
    if st["reprojection_error_px"] > max_reprojection_px:
        st["reason"] = f"reprojection {st['reprojection_error_px']:.2f} px > {max_reprojection_px}"
        return None, st
    return inv_T(make_T(cv2.Rodrigues(rvec)[0], np.asarray(tvec, np.float64).reshape(3))), st


def lift_to_3d(pts_uv: np.ndarray, depth_m: np.ndarray, K, *, min_depth_m: float, max_depth_m: float):
    """Pixels + the depth map they were found in -> (3D points in that camera's frame, mask of which pixels survived)."""
    K = np.asarray(K, np.float64)
    h, w = depth_m.shape
    u = np.round(pts_uv[:, 0]).astype(int); v = np.round(pts_uv[:, 1]).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = np.zeros(len(pts_uv)); z[inb] = depth_m[v[inb], u[inb]]
    ok = inb & (z > min_depth_m) & (z < max_depth_m)
    P = np.stack([(pts_uv[ok, 0] - K[0, 2]) * z[ok] / K[0, 0],
                  (pts_uv[ok, 1] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], axis=1)
    return P, ok, z


def anchor_from_pair(head_rgb, head_depth_m, head_intr: CameraIntrinsics, wrist_img, wrist_K, *, t_ns: int, side: str,
                     gates: dict | None = None, detector: str = "sift", wrist_valid_mask=None,
                     head_frame_index: int = -1, wrist_frame_index: int = -1) -> CrossViewAnchor:
    """One synchronized (head RGB-D, wrist RGB) pair -> one metric anchor.

    `wrist_img` must already be pinhole-rectified and `wrist_K` its intrinsics (use FisheyeRectifier). Depth is metres.
    Pass that rectifier's `valid_mask` as `wrist_valid_mask` so no keypoint comes from the extrapolated periphery.
    Every rejection path records WHY, because the anchor availability ratio is itself a headline V0 metric."""
    import cv2
    g = dict(DEFAULT_GATES); g.update(gates or {})
    a = CrossViewAnchor(t_ns=int(t_ns), side=side, head_frame_index=head_frame_index, wrist_frame_index=wrist_frame_index)

    pa, pb, a.n_keypoints_head, a.n_keypoints_wrist = match_features(head_rgb, wrist_img, detector=detector,
                                                                     ratio=float(g["ratio_test"]),
                                                                     mask_b=wrist_valid_mask)
    a.n_matches = len(pa)
    if a.n_matches < g["min_matches"]:
        a.reason = f"only {a.n_matches} matches (< {g['min_matches']})"
        return a

    h, w = head_depth_m.shape
    u = np.round(pa[:, 0]).astype(int); v = np.round(pa[:, 1]).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = np.zeros(len(pa)); z[inb] = head_depth_m[v[inb], u[inb]]
    ok = inb & (z > g["min_depth_m"]) & (z < g["max_depth_m"])
    a.n_depth_valid = int(ok.sum())
    if a.n_depth_valid < g["min_depth_valid"]:
        a.reason = f"only {a.n_depth_valid} matches have usable depth (< {g['min_depth_valid']})"
        return a
    a.depth_median_m = float(np.median(z[ok]))

    K = head_intr.K
    P = np.stack([(pa[ok, 0] - K[0, 2]) * z[ok] / K[0, 0],
                  (pa[ok, 1] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], axis=1)   # 3D in the HEAD camera frame
    try:
        okp, rvec, tvec, inl = cv2.solvePnPRansac(P.astype(np.float64), pb[ok].astype(np.float64),
                                                  np.asarray(wrist_K, np.float64), None,
                                                  reprojectionError=float(g["max_reprojection_px"]),
                                                  iterationsCount=500, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error as exc:
        a.reason = f"solvePnPRansac failed: {exc}"
        return a
    if not okp or inl is None or len(inl) < g["min_inliers"]:
        a.n_inliers = 0 if inl is None else int(len(inl))
        a.inlier_ratio = a.n_inliers / max(a.n_depth_valid, 1)
        a.reason = f"PnP found {a.n_inliers} inliers (< {g['min_inliers']})"
        return a
    idx = inl.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(P[idx], pb[ok][idx], np.asarray(wrist_K, np.float64), None, rvec, tvec)

    a.n_inliers = int(len(idx))
    a.inlier_ratio = a.n_inliers / max(a.n_depth_valid, 1)
    proj, _ = cv2.projectPoints(P[idx], rvec, tvec, np.asarray(wrist_K, np.float64), None)
    a.reprojection_error_px = float(np.linalg.norm(proj.reshape(-1, 2) - pb[ok][idx], axis=1).mean())
    if a.inlier_ratio < g["min_inlier_ratio"]:
        a.reason = f"inlier ratio {a.inlier_ratio:.2f} < {g['min_inlier_ratio']}"
        return a
    if a.reprojection_error_px > g["max_reprojection_px"]:
        a.reason = f"reprojection {a.reprojection_error_px:.2f} px > {g['max_reprojection_px']}"
        return a

    # solvePnP returns the pose of the HEAD-frame points in the WRIST camera, i.e. T_wrist_head. The anchor is its
    # inverse: where the wrist camera sits in the head camera's metric frame.
    T_wrist_head = make_T(cv2.Rodrigues(rvec)[0], np.asarray(tvec, np.float64).reshape(3))
    a.T_head_wrist = inv_T(T_wrist_head)
    a.valid = True
    return a
