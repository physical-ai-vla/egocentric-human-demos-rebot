"""`icp` — CPU depth-only rigid tracker (point-to-plane ICP, scene-to-model).

    scene cloud (ROI-cropped depth, camera frame D)  --ICP-->  model cloud (body frame B)   =>   T_B_D, inverted to T_D_B

This is the LOCAL BASELINE, not a replacement for a learned 6D tracker: it needs a good initial registration, it uses
geometry only (colour is ignored), and it can only recover near the last good pose. Its value is that it runs today on
the Mac with no CUDA, so the whole Stage-A pipeline (episode I/O, mesh conventions, QA, overlay) is exercised and
measured before FoundationPose is wired up on a GPU host — and it is an honest fallback number to compare against.

Every threshold is a constructor option (fed from depth_pose.yaml `backend_options.icp`); nothing is hard-coded here."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from ..depth_hand_tracker import DepthBackendInfo, DepthHandPoseTracker, RigidPoseEstimate, TrackerUnavailable, TrackingState
from ..icp_core import ModelCloud, icp_point_to_plane, voxel_down_sample
from ..rgbd_io import CameraIntrinsics, backproject, project
from ..se3 import inv_T, make_T
from ..tracking_mesh import TrackingMesh, load_tracking_mesh


class IcpDepthTracker(DepthHandPoseTracker):
    INFO = DepthBackendInfo(
        name="icp", needs_mesh=True, needs_init_roi=True, metric_scale=True, can_reacquire=True, runs_on="cpu",
        provides=("confidence", "depth_residual_mm", "n_scene_points"),
        notes="numpy/scipy point-to-plane ICP. Geometry only (no RGB). Reacquires only near the last good pose; "
              "there is no global re-detection, so a body that leaves and re-enters elsewhere stays LOST.")

    def __init__(self, *, mesh: TrackingMesh | str | None = None, model_points: int = 8000, model_points_coarse: int = 2000,
                 voxel_m: float = 0.004, voxel_coarse_m: float = 0.010, max_corr_init_m: float = 0.030,
                 max_corr_track_m: float = 0.015, max_corr_recover_m: float = 0.040, icp_iters: int = 50,
                 icp_iters_coarse: int = 20, init_rotations: int = 240, init_seed: int = 0, min_scene_points: int = 250,
                 fitness_min: float = 0.35, fitness_degraded: float = 0.50, rmse_max_mm: float = 8.0,
                 roi_pad_px: int = 28, depth_band_scale: float = 1.6, z_min_m: float = 0.10, z_max_m: float = 2.5,
                 recover_after_lost_frames: int = 3, constant_velocity_seed: bool = False,
                 coverage_drop_degraded: float = 0.85, coverage_drop_lost: float = 0.70, **_ignored) -> None:
        super().__init__()
        if mesh is None:
            raise TrackerUnavailable("`icp` backend needs the HandUMI rigid tracking mesh (mesh=<path|TrackingMesh>)")
        self.mesh = mesh if isinstance(mesh, TrackingMesh) else load_tracking_mesh(mesh)
        self.opt = dict(model_points=int(model_points), model_points_coarse=int(model_points_coarse), voxel_m=float(voxel_m),
                        voxel_coarse_m=float(voxel_coarse_m), max_corr_init_m=float(max_corr_init_m),
                        max_corr_track_m=float(max_corr_track_m), max_corr_recover_m=float(max_corr_recover_m),
                        icp_iters=int(icp_iters), icp_iters_coarse=int(icp_iters_coarse), init_rotations=int(init_rotations),
                        init_seed=int(init_seed), min_scene_points=int(min_scene_points), fitness_min=float(fitness_min),
                        fitness_degraded=float(fitness_degraded), rmse_max_mm=float(rmse_max_mm), roi_pad_px=int(roi_pad_px),
                        depth_band_scale=float(depth_band_scale), z_min_m=float(z_min_m), z_max_m=float(z_max_m),
                        recover_after_lost_frames=int(recover_after_lost_frames),
                        constant_velocity_seed=bool(constant_velocity_seed),
                        coverage_drop_degraded=float(coverage_drop_degraded), coverage_drop_lost=float(coverage_drop_lost))
        self.options = dict(self.opt)
        pts, nrm = self.mesh.sample_points(self.opt["model_points"], seed=self.opt["init_seed"])
        self._model = ModelCloud(pts, nrm)
        self._model_coarse = self._model.subsample(self.opt["model_points_coarse"], seed=self.opt["init_seed"])
        self._bbox_corners = self.mesh.bbox_corners()
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self._last = None
        self._last_good_T = None
        self._prev_good_T = None
        self._ref_coverage = None
        self._lost_run = 0
        self._n_frames = 0
        self._n_valid = 0

    def get_status(self) -> dict:
        d = super().get_status()
        d.update(frames=self._n_frames, valid=self._n_valid, lost_run=self._lost_run,
                 model_points=len(self._model), model_radius_m=round(self._model.radius, 4))
        return d

    # ------------------------------------------------------------------ helpers
    def _scene_points(self, depth_m: np.ndarray, mask: np.ndarray, voxel: float) -> tuple[np.ndarray, int]:
        pts, _px = backproject(depth_m, self.intrinsics.K, mask=mask, z_min=self.opt["z_min_m"], z_max=self.opt["z_max_m"])
        n_raw = len(pts)
        if n_raw == 0:
            return pts, 0
        return voxel_down_sample(pts, voxel), n_raw

    def _depth_band(self, depth_m: np.ndarray, mask: np.ndarray, *, center_m: float | None = None) -> np.ndarray:
        """Drop the background (table, wall) inside the ROI by keeping only a depth slab as thick as the body.

        While tracking, the slab is centred on the depth the model is PREDICTED at — the ROI's own median is useless
        there, because a padded box around a 14 cm body is mostly background and its median is the wall. On the first
        frame there is no prediction, so the slab is anchored on the nearest surface in the operator's ROI (the body is
        in front of whatever is behind it) and then re-centred on that cluster's median."""
        valid = mask & (depth_m > self.opt["z_min_m"]) & (depth_m < self.opt["z_max_m"])
        d = depth_m[valid]
        if len(d) < self.opt["min_scene_points"]:
            return mask
        half = self.opt["depth_band_scale"] * self._model.radius
        if center_m is None:
            near = float(np.percentile(d, 2))
            front = d[(d >= near) & (d <= near + 2 * half)]
            center_m = float(np.median(front)) if len(front) else float(np.median(d))
        return valid & (depth_m > center_m - half) & (depth_m < center_m + half)

    def _roi_mask(self, shape, T_D_B: np.ndarray, pad: int) -> np.ndarray:
        """ROI for the next frame = the projected model bbox at the last good pose, padded."""
        h, w = shape
        C = (T_D_B[:3, :3] @ self._bbox_corners.T + T_D_B[:3, 3:4]).T
        uv = project(C, self.intrinsics.K)
        uv = uv[np.isfinite(uv).all(axis=1)]
        m = np.zeros((h, w), bool)
        if len(uv) < 2:
            return ~m
        x0 = max(int(np.floor(uv[:, 0].min())) - pad, 0); x1 = min(int(np.ceil(uv[:, 0].max())) + pad, w)
        y0 = max(int(np.floor(uv[:, 1].min())) - pad, 0); y1 = min(int(np.ceil(uv[:, 1].max())) + pad, h)
        if x1 <= x0 or y1 <= y0:
            return ~m
        m[y0:y1, x0:x1] = True
        return m

    def _icp(self, scene: np.ndarray, T_D_B_init: np.ndarray, max_corr: float, iters: int, model: ModelCloud | None = None):
        r = icp_point_to_plane(scene, model or self._model, inv_T(T_D_B_init), max_corr_m=float(max_corr), max_iter=int(iters))
        return inv_T(r.T), r.fitness, r.inlier_rmse_m, r.model_coverage

    def _predict(self) -> np.ndarray:
        """ICP seed for this frame. Default: the last good pose.

        A constant-velocity extrapolation (`constant_velocity_seed: true`) predicts the next pose far better on paper
        — against ground truth on the synthetic pilot its p95 error is 1.2 mm versus 8.1 mm for the last pose — but it
        is UNSTABLE here and is off by default: the increment inv(prev)@last carries the tracking error of both frames
        and re-applying it adds that error again, so a 1 mm slip amplifies (measured: 1 -> 2 -> 5 -> 14 -> 37 -> 107 mm
        over six frames, 40/240 frames valid, against 240/240 and 0.97 mm worst error with the plain last-pose seed).
        Leave it off unless the seed is damped or the velocity low-pass filtered."""
        if self._prev_good_T is None or self._lost_run > 0 or not self.opt["constant_velocity_seed"]:
            return self._last_good_T
        d = inv_T(self._prev_good_T) @ self._last_good_T
        return self._last_good_T @ d

    def _verdict(self, fitness: float, rmse_m: float, n_scene: int, coverage: float | None = None) -> TrackingState:
        """Three independent ways to be wrong, three gates.

        `fitness` (scene inliers / scene points) is blind to the dangerous case: after a large inter-frame motion ICP can
        settle with the visible part of the object on the WRONG part of the model — every scene point still has a close
        model point, so fitness stays ~1 and the residual stays small while the pose is tens of millimetres off. The
        model-side coverage, measured against the coverage seen at a good frame, is what catches that."""
        if n_scene < self.opt["min_scene_points"] or fitness < self.opt["fitness_min"] or rmse_m * 1e3 > self.opt["rmse_max_mm"]:
            return TrackingState.LOST
        if coverage is not None and self._ref_coverage is not None:
            drop = coverage / max(self._ref_coverage, 1e-6)
            if drop < self.opt["coverage_drop_lost"]:
                return TrackingState.LOST
            if drop < self.opt["coverage_drop_degraded"]:
                return TrackingState.DEGRADED
        return TrackingState.TRACKING if fitness >= self.opt["fitness_degraded"] else TrackingState.DEGRADED

    def _emit(self, T, state, *, t_ns, frame_index, fitness=None, rmse_m=None, n_scene=None, coverage=None,
              extra=None) -> RigidPoseEstimate:
        self._n_frames += 1
        extra = dict(extra or {})
        if coverage is not None:
            extra["model_coverage"] = round(float(coverage), 4)
            if self._ref_coverage is not None:
                extra["coverage_ratio"] = round(float(coverage) / max(self._ref_coverage, 1e-6), 4)
        if state.valid and T is not None:
            est = RigidPoseEstimate.from_T(T, timestamp_ns=t_ns, side=self.side or "", frame_index=frame_index, state=state,
                                           confidence=None if fitness is None else float(fitness),
                                           depth_residual_mm=None if rmse_m is None else float(rmse_m * 1e3),
                                           n_scene_points=n_scene, extra=extra)
            self._prev_good_T = self._last_good_T
            self._last_good_T = T
            self._lost_run = 0
            self._n_valid += 1
        else:
            est = RigidPoseEstimate(timestamp_ns=int(t_ns), side=self.side or "", frame_index=int(frame_index),
                                    tracking_valid=False, tracking_state=TrackingState.LOST,
                                    confidence=None if fitness is None else float(fitness),
                                    depth_residual_mm=None if rmse_m is None else float(rmse_m * 1e3),
                                    n_scene_points=n_scene, extra=extra)
            self._lost_run += 1
        self._last = est
        return est

    # ------------------------------------------------------------------ interface
    def initialize(self, rgb, depth, camera_intrinsics: CameraIntrinsics, side: str, initial_mask=None, initial_bbox=None,
                   *, timestamp_ns: int = 0, frame_index: int = 0) -> RigidPoseEstimate:
        self.reset()
        self.side = side
        self.intrinsics = camera_intrinsics
        depth_m = np.asarray(depth, np.float32)
        mask = self._depth_band(depth_m, _as_mask(depth_m.shape, initial_mask, initial_bbox))
        scene_fine, n_raw = self._scene_points(depth_m, mask, self.opt["voxel_m"])
        if n_raw < self.opt["min_scene_points"]:
            return self._emit(None, TrackingState.LOST, t_ns=timestamp_ns, frame_index=frame_index, n_scene=n_raw,
                              extra=dict(reason="initial ROI has too few valid depth points"))
        scene_coarse, _ = self._scene_points(depth_m, mask, self.opt["voxel_coarse_m"])
        centroid = scene_fine.mean(axis=0)
        # global search: N random orientations, translation from the centroid match; coarse ICP ranks them
        rots = Rotation.random(self.opt["init_rotations"], random_state=self.opt["init_seed"]).as_matrix()
        corr = max(self.opt["max_corr_init_m"], 1e-9)
        best_T, best_fit, best_rmse, best_score = None, 0.0, float("inf"), -np.inf
        for R in rots:
            T0 = make_T(R, centroid - R @ self._model.centroid)
            T, fit, rmse, _cov = self._icp(scene_coarse, T0, corr, self.opt["icp_iters_coarse"], self._model_coarse)
            score = fit - rmse / corr                  # more inliers is better, tighter residual breaks ties
            if score > best_score:
                best_T, best_fit, best_rmse, best_score = T, fit, rmse, score
        if best_T is None:
            return self._emit(None, TrackingState.LOST, t_ns=timestamp_ns, frame_index=frame_index, n_scene=n_raw,
                              extra=dict(reason="no ICP hypothesis converged"))
        T, fit, rmse, cov = self._icp(scene_fine, best_T, self.opt["max_corr_track_m"], self.opt["icp_iters"])
        state = self._verdict(fit, rmse, n_raw)                 # no reference coverage yet: this frame defines it
        if state.valid:
            self._ref_coverage = cov
        return self._emit(T, state, t_ns=timestamp_ns, frame_index=frame_index, fitness=fit, rmse_m=rmse, n_scene=n_raw,
                          coverage=cov, extra=dict(stage="initialize", init_rotations=self.opt["init_rotations"],
                                                   coarse_fitness=round(float(best_fit), 4)))

    def track(self, rgb, depth, timestamp_ns: int, *, frame_index: int = -1) -> RigidPoseEstimate:
        if self.intrinsics is None:
            raise RuntimeError("track() before initialize()")
        depth_m = np.asarray(depth, np.float32)
        if self._last_good_T is None:
            return self._emit(None, TrackingState.LOST, t_ns=timestamp_ns, frame_index=frame_index,
                              extra=dict(reason="no previous good pose to track from"))
        recovering = self._lost_run >= self.opt["recover_after_lost_frames"]
        seed = self._predict()
        mask = self._roi_mask(depth_m.shape, seed, self.opt["roi_pad_px"] * (3 if recovering else 1))
        mask = self._depth_band(depth_m, mask,
                                center_m=float(seed[2, 3] + (seed[:3, :3] @ self._model.centroid)[2]))
        scene, n_raw = self._scene_points(depth_m, mask, self.opt["voxel_m"])
        if n_raw < self.opt["min_scene_points"]:
            return self._emit(None, TrackingState.LOST, t_ns=timestamp_ns, frame_index=frame_index, n_scene=n_raw,
                              extra=dict(reason="ROI has too few valid depth points (occlusion / out of frame)"))
        max_corr = self.opt["max_corr_recover_m"] if recovering else self.opt["max_corr_track_m"]
        T, fit, rmse, cov = self._icp(scene, seed, max_corr, self.opt["icp_iters"])
        state = self._verdict(fit, rmse, n_raw, cov)
        if state.valid and self._lost_run > 0:
            state = TrackingState.REACQUIRED
        return self._emit(T if state.valid else None, state, t_ns=timestamp_ns, frame_index=frame_index,
                          fitness=fit, rmse_m=rmse, n_scene=n_raw, coverage=cov,
                          extra=dict(recovering=bool(recovering)))


def _as_mask(shape, mask, bbox) -> np.ndarray:
    h, w = shape
    if mask is not None:
        m = np.asarray(mask)
        if m.shape[:2] != (h, w):
            raise ValueError(f"initial_mask shape {m.shape[:2]} != depth shape {(h, w)}")
        return m.astype(bool)
    if bbox is not None:
        x, y, bw, bh = (int(v) for v in bbox)
        out = np.zeros((h, w), bool)
        out[max(y, 0):min(y + bh, h), max(x, 0):min(x + bw, w)] = True
        if not out.any():
            raise ValueError(f"initial_bbox {bbox} does not overlap the {w}x{h} image")
        return out
    raise ValueError("initialize() needs initial_mask or initial_bbox (this backend has no detector)")
