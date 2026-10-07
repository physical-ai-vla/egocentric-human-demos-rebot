"""Metric head trajectory from the head RGB-D alone — temporal 3D-2D PnP.

    head RGB(t) features, lifted to 3D by depth(t)      head RGB(t+1), same features tracked
                        |                                              |
                        +------------ RANSAC PnP + LM -----------------+
                                          |
                               T_head(t)_head(t+1)   (METRIC)
                                          |
                                     chained -> T_world_head(t)

Why this module exists: the head is worn, so an anchor expressed in the head frame is head-relative, and the canonical
action dT_TCP needs a workspace frame. No RGB-D odometry exists in this environment to borrow — the orbslam3 adapter is
mono-inertial only (and needs the IMU we do not have), dpvo is not installed, cv2.rgbd ships as an empty namespace here,
and open3d segfaults under numpy 2. So this is built from the geometry that is already in the repo rather than from a
new SLAM stack: the SAME metric PnP as the head<->wrist anchor (cross_view.solve_pnp_metric), with the two views being
two TIMES of one camera instead of two cameras at one time. Depth makes it metric; there is no scale to estimate.

What it deliberately does not do: no loop closure, no bundle adjustment, no map. It is a frame-to-frame chain, so drift
accumulates — and measuring that drift (HOME return, static segments) is the point of V0-B, not something to hide. A
frame whose relative pose cannot be solved yields NO pose, and the resulting break in the chain is counted and
reported, because a silently bridged gap would corrupt exactly the drift number we are trying to measure."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from .cross_view import lift_to_3d, solve_pnp_metric
from .rgbd_io import CameraIntrinsics
from .se3 import T_to_pose7, inv_T

DEFAULT_OPTS = dict(max_features=1200, feature_quality=0.01, feature_min_distance=12, redetect_below=400,
                    klt_win=21, klt_levels=3, fb_error_px=1.0, min_depth_m=0.15, max_depth_m=3.0,
                    min_inliers=30, min_inlier_ratio=0.35, max_reprojection_px=2.0)


@dataclass
class HeadPoseEstimate:
    t_ns: int
    frame_index: int
    T_world_head: np.ndarray | None = None     # 4x4; None when this frame's relative pose could not be solved
    valid: bool = False
    keyframe: bool = False                      # features were (re)detected on this frame
    n_tracked: int = 0
    n_depth_valid: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error_px: float | None = None
    step_translation_mm: float | None = None
    step_rotation_deg: float | None = None
    chain_broken: bool = False                  # the previous frame had no pose: motion across the gap is UNMEASURED
    reason: str = ""

    def to_row(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "T_world_head"}
        p = T_to_pose7(self.T_world_head) if self.T_world_head is not None else [np.nan] * 7
        d.update(x=p[0], y=p[1], z=p[2], qx=p[3], qy=p[4], qz=p[5], qw=p[6])
        return d


class HeadRgbdOdometry:
    """Streaming, one head camera. Push frames in timestamp order; each push returns that frame's world pose.

    The first frame defines the world frame (identity), exactly like the episode-local canonicalisation the VIO path
    already uses — this produces a trajectory, not a georeferenced map."""

    def __init__(self, intrinsics: CameraIntrinsics, **opts) -> None:
        self.intr = intrinsics
        self.opt = dict(DEFAULT_OPTS)
        self.opt.update({k: v for k, v in opts.items() if k in DEFAULT_OPTS})
        self.reset()

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_depth = None
        self._prev_pts = None                   # (N,2) float32 tracked pixels in the previous frame
        self._T_world = np.eye(4)
        self._have_pose = False
        self.n_frames = 0
        self.n_valid = 0
        self.broken_links = 0

    # ------------------------------------------------------------------ internals
    def _detect(self, gray, depth_m):
        import cv2
        mask = ((depth_m > self.opt["min_depth_m"]) & (depth_m < self.opt["max_depth_m"])).astype(np.uint8)
        p = cv2.goodFeaturesToTrack(gray, int(self.opt["max_features"]), float(self.opt["feature_quality"]),
                                    float(self.opt["feature_min_distance"]), mask=mask, blockSize=5)
        return np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2).astype(np.float32)

    def _track(self, g0, g1, p0):
        """KLT forward-backward: a point survives only if it tracks back to where it came from. Optical flow is the
        right matcher between consecutive frames of one camera — a descriptor match would be slower and no better."""
        import cv2
        if len(p0) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        win = (int(self.opt["klt_win"]), int(self.opt["klt_win"]))
        lv = int(self.opt["klt_levels"])
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, winSize=win, maxLevel=lv)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=win, maxLevel=lv)
        p1 = p1.reshape(-1, 2); p0b = p0b.reshape(-1, 2)
        ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
        ok &= np.linalg.norm(p0b - p0, axis=1) < float(self.opt["fb_error_px"])
        return p0[ok], p1[ok]

    # ------------------------------------------------------------------ interface
    def push(self, rgb: np.ndarray, depth_m: np.ndarray, t_ns: int, frame_index: int = -1) -> HeadPoseEstimate:
        import cv2
        gray = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        depth_m = np.asarray(depth_m, np.float32)
        e = HeadPoseEstimate(t_ns=int(t_ns), frame_index=int(frame_index))
        self.n_frames += 1

        if self._prev_gray is None:                              # first frame defines the world frame
            self._prev_gray, self._prev_depth = gray, depth_m
            self._prev_pts = self._detect(gray, depth_m)
            self._T_world = np.eye(4)
            self._have_pose = True
            e.T_world_head, e.valid, e.keyframe = self._T_world.copy(), True, True
            e.n_tracked = len(self._prev_pts)
            self.n_valid += 1
            return e

        p0, p1 = self._track(self._prev_gray, gray, self._prev_pts)
        e.n_tracked = len(p0)
        P_prev, ok, _z = lift_to_3d(p0, self._prev_depth, self.intr.K,
                                    min_depth_m=self.opt["min_depth_m"], max_depth_m=self.opt["max_depth_m"])
        e.n_depth_valid = int(ok.sum())
        T_rel, st = solve_pnp_metric(P_prev, p1[ok], self.intr.K, max_reprojection_px=self.opt["max_reprojection_px"],
                                     min_inliers=self.opt["min_inliers"], min_inlier_ratio=self.opt["min_inlier_ratio"])
        e.n_inliers, e.inlier_ratio = st["n_inliers"], st["inlier_ratio"]
        e.reprojection_error_px, e.reason = st["reprojection_error_px"], st["reason"]

        if T_rel is None:
            # No pose for this frame. The chain stops here; the next solved frame is flagged chain_broken so the
            # unmeasured motion across the gap can never be mistaken for a measured one.
            self._have_pose = False
            self._prev_gray, self._prev_depth = gray, depth_m
            self._prev_pts = self._detect(gray, depth_m)
            e.keyframe = True
            return e

        e.chain_broken = not self._have_pose
        if e.chain_broken:
            self.broken_links += 1
        self._T_world = self._T_world @ T_rel
        self._have_pose = True
        e.T_world_head, e.valid = self._T_world.copy(), True
        e.step_translation_mm = float(np.linalg.norm(T_rel[:3, 3]) * 1e3)
        from scipy.spatial.transform import Rotation
        e.step_rotation_deg = float(np.degrees(Rotation.from_matrix(T_rel[:3, :3]).magnitude()))
        self.n_valid += 1

        self._prev_gray, self._prev_depth = gray, depth_m
        if len(p1) < self.opt["redetect_below"]:
            self._prev_pts = self._detect(gray, depth_m)
            e.keyframe = True
        else:
            self._prev_pts = p1.astype(np.float32)
        return e

    def summary(self) -> dict:
        return dict(frames=self.n_frames, valid=self.n_valid,
                    valid_ratio=(self.n_valid / self.n_frames if self.n_frames else 0.0),
                    broken_links=self.broken_links,
                    note=("broken_links > 0: the trajectory contains UNMEASURED motion across those gaps, so drift "
                          "metrics over the whole episode are not meaningful — evaluate per unbroken segment"
                          if self.broken_links else ""))
