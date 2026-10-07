"""Visual-only monocular VO baseline (Backend B) that runs in this repo with no extra software: fisheye → pinhole
rectification, KLT feature tracking, two-view initialisation (essential matrix), map triangulation, PnP-RANSAC tracking,
keyframe re-triangulation. Monocular ⇒ scale is arbitrary (metric_scale=False): rotations and trajectory *shape* are
usable for the backend benchmark, HOME-return rotation drift and IMU-consistency, but metric translation is not.
IMU is not used inside (IMU-independent QA remains fully independent for this backend)."""
from __future__ import annotations
import numpy as np
import cv2
from ..estimator import BackendInfo, BackendUnavailable, PoseEstimate, PoseEstimator, TrackingState
from ..se3 import inv_T, make_T
from ..virtual_view import valid_radius_maps


class FisheyeRectifier:
    """Kannala-Brandt (cv2.fisheye) → pinhole of a chosen FOV; `pinhole` intrinsics pass through untouched."""

    def __init__(self, intrinsics: dict, *, out_size: tuple[int, int] | None = None, fov_deg: float = 90.0, downscale: int = 1) -> None:
        K = np.asarray(intrinsics["K"], np.float64).copy(); D = np.asarray(intrinsics.get("D", np.zeros(4)), np.float64).reshape(-1)
        w, h = [int(v) for v in intrinsics["image_size"]]
        self.model = intrinsics.get("model", "kannala_brandt")
        if downscale > 1: K[:2] /= downscale; w //= downscale; h //= downscale
        # Radius beyond which D is extrapolated, not measured (see estimate_valid_radius). None = never measured.
        self.valid_radius_px = intrinsics.get("valid_radius_px")
        if downscale > 1 and self.valid_radius_px is not None:
            self.valid_radius_px = float(self.valid_radius_px) / downscale
        if self.model == "pinhole":
            self.K_out = K; self.size = (w, h); self._maps = None
            self.valid_mask = None; self.valid_fraction = 1.0
        else:
            ow, oh = out_size or (w, h)
            f = 0.5 * ow / np.tan(np.radians(fov_deg) / 2)
            self.K_out = np.array([[f, 0, ow / 2 - 0.5], [0, f, oh / 2 - 0.5], [0, 0, 1]]); self.size = (ow, oh)
            # CV_32FC2 (not the packed CV_16SC2 fixed-point form) so the source radius of each target pixel is readable
            # and the extrapolated periphery can be cut before anything is detected in it.
            mx, my = cv2.fisheye.initUndistortRectifyMap(K, D[:4], np.eye(3), self.K_out, (ow, oh), cv2.CV_32FC1)
            mx, my, mask, frac = valid_radius_maps(mx, my, K, self.valid_radius_px)
            self._maps = (mx, my)
            self.valid_mask = None if self.valid_radius_px is None else mask
            self.valid_fraction = frac

    def rectify(self, img: np.ndarray) -> np.ndarray:
        if self._maps is None: return img
        return cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    def pinhole(self) -> tuple[float, float, float, float]:
        K = self.K_out; return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


class OpenCvVoBackend(PoseEstimator):
    INFO = BackendInfo("opencv_vo", uses_imu=False, metric_scale=False, online_capable=True,
                       provides=("confidence", "num_features", "reprojection_error"), notes="in-repo KLT+PnP monocular VO baseline; scale arbitrary")
    info = INFO

    def __init__(self, *, max_features: int = 1200, min_tracked: int = 60, init_min_parallax_px: float = 12.0, kf_min_tracked: int = 150,
                 ransac_reproj_px: float = 1.5, rect_fov_deg: float = 90.0, rect_size: tuple[int, int] | None = None, min_init_inliers: int = 40,
                 new_point_min_parallax_px: float = 8.0, downscale: int = 1, **_ignored) -> None:
        self.max_features, self.min_tracked, self.init_min_parallax, self.kf_min_tracked = max_features, min_tracked, init_min_parallax_px, kf_min_tracked
        self.ransac_px, self.rect_fov, self.rect_size, self.min_init_inliers, self.downscale = ransac_reproj_px, rect_fov_deg, rect_size, min_init_inliers, downscale
        self.new_pt_parallax = new_point_min_parallax_px
        self._rect: FisheyeRectifier | None = None
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, *, intrinsics=None, T_camera_imu=None, imu_noise=None, image_size=None) -> None:
        if intrinsics is None: raise BackendUnavailable("opencv_vo: fisheye/pinhole intrinsics required (configs/calibration/fisheye_<side>_vNNN.yaml)")
        self._rect = FisheyeRectifier(intrinsics, out_size=self.rect_size, fov_deg=self.rect_fov, downscale=self.downscale)
        fx, fy, cx, cy = self._rect.pinhole(); self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def reset(self) -> None:
        self.state = TrackingState.UNINITIALIZED
        self._prev_gray = None; self._prev_pts = None          # (N,2) tracked 2D points in the previous frame
        self._pts3d = None                                     # (N,3) world points for tracked 2D points (None for untriangulated)
        self._init_gray = None; self._init_pts = None; self._init_T = None
        self._T_wc = np.eye(4); self._kf_T = None; self._kf_pts = None; self._kf_ids = None
        self._segment = 0; self._last = None; self._n = 0; self._stats = dict(inits=0, lost=0, keyframes=0)

    # ------------------------------------------------------------------ helpers
    def _detect(self, gray: np.ndarray, existing: np.ndarray | None) -> np.ndarray:
        vm = None if self._rect is None else self._rect.valid_mask
        mask = np.full(gray.shape, 255, np.uint8) if vm is None else vm.copy()
        if existing is not None and len(existing):
            for x, y in existing.astype(int): cv2.circle(mask, (int(x), int(y)), 12, 0, -1)
        need = self.max_features - (0 if existing is None else len(existing))
        if need <= 0: return np.zeros((0, 2), np.float32)
        p = cv2.goodFeaturesToTrack(gray, need, 0.01, 10, mask=mask, blockSize=5)
        return np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2).astype(np.float32)

    def _track(self, g0, g1, p0):
        if p0 is None or len(p0) == 0: return np.zeros((0, 2), np.float32), np.zeros(0, bool)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=3)
        ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (np.linalg.norm(p0b.reshape(-1, 2) - p0, axis=1) < 1.0)
        p1 = p1.reshape(-1, 2)
        h, w = g1.shape; ok &= (p1[:, 0] >= 0) & (p1[:, 1] >= 0) & (p1[:, 0] < w) & (p1[:, 1] < h)
        # A track may start inside the measured region and DRIFT into the extrapolated periphery; drop it when it does,
        # rather than let its bearing quietly go wrong mid-track.
        vm = None if self._rect is None else self._rect.valid_mask
        if vm is not None and ok.any():
            u = np.clip(p1[:, 0].astype(int), 0, w - 1); v = np.clip(p1[:, 1].astype(int), 0, h - 1)
            ok &= vm[v, u] > 0
        return p1, ok

    def _triangulate(self, T_wc_a, pa, T_wc_b, pb) -> tuple[np.ndarray, np.ndarray]:
        """World points from two views with known poses; returns (X (N,3), good mask) — positive depth in both, finite."""
        Pa = self.K @ inv_T(T_wc_a)[:3]; Pb = self.K @ inv_T(T_wc_b)[:3]
        X = cv2.triangulatePoints(Pa, Pb, pa.T.astype(np.float64), pb.T.astype(np.float64)); X = (X[:3] / X[3]).T
        za = (inv_T(T_wc_a) @ np.c_[X, np.ones(len(X))].T)[2]; zb = (inv_T(T_wc_b) @ np.c_[X, np.ones(len(X))].T)[2]
        good = np.isfinite(X).all(axis=1) & (za > 1e-3) & (zb > 1e-3)
        # reprojection sanity
        for T, p in ((T_wc_a, pa), (T_wc_b, pb)):
            proj = (self.K @ inv_T(T)[:3] @ np.c_[X, np.ones(len(X))].T); proj = (proj[:2] / np.where(np.abs(proj[2]) < 1e-9, 1e-9, proj[2])).T
            good &= np.linalg.norm(proj - p, axis=1) < 3 * self.ransac_px
        return X, good

    # ------------------------------------------------------------------ main
    def push_image(self, t_ns: int, frame_index: int, image_bgr) -> PoseEstimate:
        if self._rect is None: raise RuntimeError("opencv_vo: initialize() first")
        self._n += 1
        img = self._rect.rectify(image_bgr)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        est = None
        if self.state in (TrackingState.UNINITIALIZED, TrackingState.LOST, TrackingState.INITIALIZING):
            est = self._step_init(gray, t_ns, frame_index)
        else:
            est = self._step_track(gray, t_ns, frame_index)
        self._prev_gray = gray; self._last = est
        return est

    def _step_init(self, gray, t_ns, frame_index) -> PoseEstimate:
        if self._init_gray is None:                                       # first keyframe of a segment
            self._init_gray = gray; self._init_pts = self._detect(gray, None); self._init_T = self._T_wc.copy()
            self._prev_pts = self._init_pts.copy(); self.state = TrackingState.INITIALIZING
            return PoseEstimate(t_ns, frame_index, None, self.state, num_features=len(self._init_pts), extra=dict(segment=self._segment))
        p1, ok = self._track(self._prev_gray, gray, self._prev_pts)
        self._init_pts, self._prev_pts = self._init_pts[ok], p1[ok]
        if len(self._prev_pts) < self.min_init_inliers:                   # tracking collapsed: restart the init pair
            self._init_gray = None; return self._step_init(gray, t_ns, frame_index)
        parallax = np.median(np.linalg.norm(self._prev_pts - self._init_pts, axis=1))
        if parallax < self.init_min_parallax:
            return PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING, num_features=len(self._prev_pts), extra=dict(parallax_px=float(parallax), segment=self._segment))
        E, inl = cv2.findEssentialMat(self._init_pts, self._prev_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=self.ransac_px)
        if E is None or E.shape != (3, 3) or inl is None or inl.sum() < self.min_init_inliers:
            return PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING, num_features=int(0 if inl is None else inl.sum()), extra=dict(segment=self._segment))
        m = inl.reshape(-1) == 1
        n, R, t, mask_pose = cv2.recoverPose(E, self._init_pts[m], self._prev_pts[m], self.K)
        if n < self.min_init_inliers: return PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING, num_features=int(n), extra=dict(segment=self._segment))
        T_c2_c1 = make_T(R, t.reshape(3))                                 # x2 = R x1 + t ; |t| = 1 (arbitrary scale unit)
        T_wc = self._init_T @ inv_T(T_c2_c1)
        X, good = self._triangulate(self._init_T, self._init_pts[m], T_wc, self._prev_pts[m])
        good &= mask_pose.reshape(-1) > 0
        if good.sum() < self.min_init_inliers:
            return PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING, num_features=int(good.sum()), extra=dict(segment=self._segment))
        self._prev_pts = self._prev_pts[m][good]; self._pts3d = X[good]; self._T_wc = T_wc
        self._kf_T, self._kf_pts, self._kf_gray = T_wc.copy(), self._prev_pts.copy(), gray
        self._kf_new = self._detect(gray, self._prev_pts); self._kf_new_prev = self._kf_new.copy()
        self.state = TrackingState.TRACKING; self._stats["inits"] += 1; self._init_gray = None
        return PoseEstimate(t_ns, frame_index, T_wc.copy(), self.state, confidence=float(good.mean()), num_features=int(good.sum()),
                            reprojection_error=None, extra=dict(segment=self._segment, event="initialized"))

    def _step_track(self, gray, t_ns, frame_index) -> PoseEstimate:
        p1, ok = self._track(self._prev_gray, gray, self._prev_pts)
        pts2d, X = p1[ok], self._pts3d[ok]
        # untriangulated candidates from the last keyframe are tracked alongside (for later triangulation)
        if self._kf_new_prev is not None and len(self._kf_new_prev):
            c1, cok = self._track(self._prev_gray, gray, self._kf_new_prev); self._kf_new, self._kf_new_prev = self._kf_new[cok], c1[cok]
        if len(pts2d) < 6: return self._lose(t_ns, frame_index, "too few tracked points", len(pts2d))
        okp, rvec, tvec, inliers = cv2.solvePnPRansac(X.astype(np.float64), pts2d.astype(np.float64), self.K, None, reprojectionError=self.ransac_px,
                                                     confidence=0.999, iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
        if not okp or inliers is None or len(inliers) < 6: return self._lose(t_ns, frame_index, "pnp failed", len(pts2d))
        inl = np.zeros(len(pts2d), bool); inl[inliers.reshape(-1)] = True
        rvec, tvec = cv2.solvePnPRefineLM(X[inl].astype(np.float64), pts2d[inl].astype(np.float64), self.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec); T_cw = make_T(R, tvec.reshape(3)); T_wc = inv_T(T_cw)
        proj, _ = cv2.projectPoints(X[inl], rvec, tvec, self.K, None); reproj = float(np.linalg.norm(proj.reshape(-1, 2) - pts2d[inl], axis=1).mean())
        jump = float(np.linalg.norm(T_wc[:3, 3] - self._T_wc[:3, 3]))
        self._T_wc = T_wc; self._prev_pts, self._pts3d = pts2d[inl], X[inl]
        n_inl = int(inl.sum()); conf = n_inl / max(len(pts2d), 1)
        state = TrackingState.TRACKING if n_inl >= self.min_tracked else TrackingState.DEGRADED
        if n_inl < self.kf_min_tracked: self._new_keyframe(gray)
        if n_inl < self.min_tracked // 2: return self._lose(t_ns, frame_index, "inliers collapsed", n_inl)
        self.state = state
        return PoseEstimate(t_ns, frame_index, T_wc.copy(), state, confidence=float(conf), num_features=n_inl, reprojection_error=reproj,
                            pose_jump_score=jump, extra=dict(segment=self._segment, map_points=int(len(self._pts3d))))

    def _new_keyframe(self, gray) -> None:
        """Triangulate candidates tracked since the last keyframe (both poses known → scale-consistent), then seed new candidates."""
        if self._kf_new is not None and len(self._kf_new) >= 8:
            X, good = self._triangulate(self._kf_T, self._kf_new, self._T_wc, self._kf_new_prev)
            par = np.linalg.norm(self._kf_new - self._kf_new_prev, axis=1) > self.new_pt_parallax
            good &= par
            if good.sum():
                self._prev_pts = np.vstack([self._prev_pts, self._kf_new_prev[good]]); self._pts3d = np.vstack([self._pts3d, X[good]])
            if good.sum() < 0.3 * len(self._kf_new):          # not enough parallax yet: keep the old keyframe, keep tracking candidates
                self._kf_new, self._kf_new_prev = self._kf_new[~good], self._kf_new_prev[~good]
                return
        self._kf_T, self._kf_gray = self._T_wc.copy(), gray
        self._kf_new = self._detect(gray, self._prev_pts); self._kf_new_prev = self._kf_new.copy(); self._stats["keyframes"] += 1

    def _lose(self, t_ns, frame_index, why: str, n: int) -> PoseEstimate:
        self.state = TrackingState.LOST; self._stats["lost"] += 1; self._segment += 1
        self._init_gray = None; self._pts3d = None; self._kf_new = self._kf_new_prev = None
        # the next segment starts at the last known pose: continuity of world frame is an ASSUMPTION flagged by `segment`
        return PoseEstimate(t_ns, frame_index, None, TrackingState.LOST, num_features=int(n), extra=dict(segment=self._segment, event=f"lost: {why}"))

    def get_quality(self) -> dict:
        return dict(state=self.state.value, frames=self._n, map_points=0 if self._pts3d is None else int(len(self._pts3d)), **self._stats)
