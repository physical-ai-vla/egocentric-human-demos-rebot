"""[2026-10-06 user] live start-pose repeatability WITHOUT fiducials, for ANY HandUMI start pose (REF = the pose at SET START REF = 0,0,0).
v3: reference features that lie on the TABLE get a metric 3D position at REF time (ray from the reference camera cut with the table
plane: plane normal = IMU gravity, camera height above the table = h_mm); a live frame is located against them with PnP-RANSAC
(metric, works whether or not the camera looks down). Rotation also from PnP. Far background features only help matching.
Output: camera displacement vs REF in a gravity-aligned frame (fwd / left / up, mm) + rotation (deg)."""
from __future__ import annotations
import numpy as np, cv2

VW, VH, VF = 960, 720, 420.0          # virtual pinhole: hfov ~98 deg


class StartMatcher:
    def __init__(self, K, D, image_size, ref_height_mm: float = 200.0, min_inliers: int = 15):
        self.K = np.asarray(K, np.float64); self.D = np.asarray(D, np.float64).reshape(4, 1); self.size = tuple(int(x) for x in image_size)
        self.Kv = np.array([[VF, 0, VW / 2], [0, VF, VH / 2], [0, 0, 1.0]]); self.h_ref = float(ref_height_mm); self.min_inl = int(min_inliers)
        self.orb = cv2.ORB_create(3000, fastThreshold=8); self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._maps = {}; self.ref = None
        m = np.full((VH, VW), 255, np.uint8)
        cv2.rectangle(m, (int(VW * 0.30), int(VH * 0.55)), (int(VW * 0.72), VH), 0, -1)       # fingers (bottom centre)
        self.mask = m

    def _virt(self, bgr):
        h, w = bgr.shape[:2]
        if (w, h) not in self._maps:
            K = self.K.copy(); K[0] *= w / self.size[0]; K[1] *= h / self.size[1]
            mp = cv2.fisheye.initUndistortRectifyMap(K, self.D, np.eye(3), self.Kv, (VW, VH), cv2.CV_32FC1)
            valid = ((mp[0] >= 0) & (mp[0] < w - 1) & (mp[1] >= 0) & (mp[1] < h - 1)).astype(np.uint8) * 255
            self._maps[(w, h)] = (cv2.convertMaps(mp[0], mp[1], cv2.CV_16SC2), cv2.erode(valid, np.ones((9, 9), np.uint8)))
        (m1, m2), valid = self._maps[(w, h)]
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.remap(g, m1, m2, cv2.INTER_LINEAR)), cv2.bitwise_and(valid, self.mask)

    def set_reference(self, bgr, up_cam) -> dict:
        g, msk = self._virt(bgr); kp, de = self.orb.detectAndCompute(g, msk)
        up = np.asarray(up_cam, float); up /= np.linalg.norm(up)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        rays = np.c_[(pts - self.Kv[:2, 2]) / VF, np.ones(len(pts))]
        cosd = rays @ (-up) / np.linalg.norm(rays, axis=1)                     # > 0: ray goes down toward the table
        on_table = cosd > 0.25                                                  # < ~75 deg from straight down: hits the table in front
        X = np.where(on_table[:, None], rays * (self.h_ref / np.maximum(rays @ (-up), 1e-6))[:, None], np.nan)
        self.ref = dict(kp=kp, de=de, up=up, X=X); return dict(features=len(kp), table_features=int(on_table.sum()))

    def measure(self, bgr, up_cam=None) -> dict:
        if self.ref is None or self.ref["de"] is None: return dict(ok=False, why="no reference")
        g, msk = self._virt(bgr); kp, de = self.orb.detectAndCompute(g, msk)
        if de is None or len(kp) < 30: return dict(ok=False, why=f"few features ({0 if de is None else len(kp)})")
        mt = self.bf.knnMatch(self.ref["de"], de, k=2); good = [a for a, b in (p for p in mt if len(p) == 2) if a.distance < 0.75 * b.distance]
        tab = [m for m in good if np.isfinite(self.ref["X"][m.queryIdx, 0])]
        if len(tab) < self.min_inl: return dict(ok=False, why=f"table matches {len(tab)} < {self.min_inl} (책상 쪽 특징점 부족)", matches=len(good))
        obj = np.float32([self.ref["X"][m.queryIdx] for m in tab]); img = np.float32([kp[m.trainIdx].pt for m in tab])
        rv0, tv0 = np.zeros((3, 1)), np.zeros((3, 1))                           # the start pose is near REF: identity is the right seed (points are coplanar)
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, self.Kv, None, rv0, tv0, useExtrinsicGuess=True, reprojectionError=3.0, iterationsCount=300, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        n_in = 0 if inl is None else len(inl)
        if not ok or n_in < self.min_inl: return dict(ok=False, why=f"PnP inliers {n_in}", matches=len(good))
        idx = inl.ravel(); rv, tv = cv2.solvePnPRefineLM(obj[idx], img[idx], self.Kv, None, rv, tv)
        R = cv2.Rodrigues(rv)[0]; c = (-R.T @ tv).ravel()                       # current camera centre in the REF frame (mm), REF = (0,0,0)
        z = self.ref["up"]; fwd = np.array([0, 0, 1.0]) - (np.array([0, 0, 1.0]) @ z) * z; fwd /= np.linalg.norm(fwd); left = np.cross(z, fwd)
        return dict(ok=True, fwd_mm=float(c @ fwd), left_mm=float(c @ left), up_mm=float(c @ z), dist_mm=float(np.linalg.norm(c)),
                    rot_deg=float(np.degrees(np.linalg.norm(rv))), inliers=n_in, matches=len(good), table_matches=len(tab), rvec=rv.ravel().tolist(), tvec_mm=tv.ravel().tolist())
