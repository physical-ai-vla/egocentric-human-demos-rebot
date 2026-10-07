"""Point-to-plane ICP on numpy + scipy.cKDTree.

Written out rather than taken from Open3D on purpose: open3d 0.18 (the wheel available for this Mac) segfaults on every
API that converts a numpy array to an Eigen type under numpy 2.x — `translate`, `scale`, `transform` and
`registration_icp` all crash the interpreter. This implementation has no native ABI surface, is deterministic, and
reports the same two numbers Open3D does (fitness = inlier fraction, inlier RMSE = point-to-point).

Convention: `T` maps SCENE (camera frame D) points into the MODEL frame B, i.e. T = T_B_D. The caller inverts it to get
the reported T_D_B. The model is the target because its normals come from CAD and are clean; scene normals from a noisy
depth map are not."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .se3 import make_T


@dataclass
class IcpResult:
    T: np.ndarray                 # T_B_D (scene -> model)
    fitness: float                # inliers / scene points
    inlier_rmse_m: float          # point-to-point RMSE over the inliers (0 when there are none)
    plane_rmse_m: float           # point-to-plane RMSE over the inliers
    model_coverage: float         # fraction of MODEL points that have a scene point within max_corr
    iterations: int
    n_inliers: int


class ModelCloud:
    """Sampled model surface + KD-tree, built once per tracker (the model never moves in its own frame)."""

    def __init__(self, points: np.ndarray, normals: np.ndarray) -> None:
        self.points = np.ascontiguousarray(points, np.float64)
        n = np.ascontiguousarray(normals, np.float64)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.normals = n / np.where(ln < 1e-12, 1.0, ln)
        self.tree = cKDTree(self.points)
        self.centroid = self.points.mean(axis=0)
        self.radius = float(np.linalg.norm(self.points - self.centroid, axis=1).max())

    def __len__(self) -> int:
        return len(self.points)

    def subsample(self, n: int, *, seed: int = 0) -> "ModelCloud":
        if n >= len(self.points):
            return self
        idx = np.random.default_rng(seed).choice(len(self.points), size=int(n), replace=False)
        return ModelCloud(self.points[idx], self.normals[idx])


def voxel_down_sample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """One representative (centroid) point per occupied voxel — the same idea as Open3D's, in numpy."""
    P = np.asarray(points, np.float64)
    if voxel_m <= 0 or len(P) == 0:
        return P
    key = np.floor(P / voxel_m).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    n = inv.max() + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, P)
    cnt = np.bincount(inv, minlength=n).reshape(-1, 1)
    return out / cnt


def icp_point_to_plane(scene: np.ndarray, model: ModelCloud, T_init: np.ndarray, *, max_corr_m: float,
                       max_iter: int = 50, tol_translation_m: float = 1e-5, tol_rotation_rad: float = 1e-5) -> IcpResult:
    S = np.ascontiguousarray(scene, np.float64)
    T = np.asarray(T_init, np.float64).copy()
    it = 0
    for it in range(1, int(max_iter) + 1):
        P = (T[:3, :3] @ S.T + T[:3, 3:4]).T
        d, idx = model.tree.query(P, workers=-1)
        m = d < max_corr_m
        if int(m.sum()) < 6:
            break
        p = P[m]
        q = model.points[idx[m]]
        n = model.normals[idx[m]]
        A = np.hstack([np.cross(p, n), n])
        b = ((q - p) * n).sum(axis=1)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(x)):
            break
        rot = x[:3]
        ang = float(np.linalg.norm(rot))
        if ang > 0.5:                                   # keep the small-angle linearisation honest
            rot = rot / ang * 0.5
        dT = make_T(Rotation.from_rotvec(rot).as_matrix(), x[3:6])
        T = dT @ T
        if float(np.linalg.norm(x[3:6])) < tol_translation_m and ang < tol_rotation_rad:
            break
    P = (T[:3, :3] @ S.T + T[:3, 3:4]).T
    d, idx = model.tree.query(P, workers=-1)
    m = d < max_corr_m
    n_in = int(m.sum())
    if n_in == 0:
        return IcpResult(T, 0.0, 0.0, 0.0, 0.0, it, 0)
    plane = np.abs(((model.points[idx[m]] - P[m]) * model.normals[idx[m]]).sum(axis=1))
    # model-side coverage: fitness alone cannot see a WRONG but locally consistent alignment (ICP settling on a subset of
    # the model after a large motion) — every scene point still has a close model point, so fitness stays ~1. The share of
    # the model that is actually explained by the scene does collapse there, which is what makes it detectable.
    cov = float((cKDTree(P).query(model.points, distance_upper_bound=max_corr_m, workers=-1)[0] < np.inf).mean())
    return IcpResult(T, n_in / len(S), float(np.sqrt((d[m] ** 2).mean())), float(np.sqrt((plane ** 2).mean())), cov, it, n_in)
