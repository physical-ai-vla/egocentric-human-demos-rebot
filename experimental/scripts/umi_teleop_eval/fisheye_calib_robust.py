"""Robust fisheye intrinsics from ChArUco views (2026-10-15, tracking fisheye: cv2.fisheye.calibrate was unstable -- InitExtrinsics
asserts, f wandering 340..650 px with ~1.6-2.4 px residual). Own Levenberg-Marquardt over ALL views jointly, soft-L1 loss, two models:

    KB4  (Kannala-Brandt / OpenCV fisheye):  theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8),  u = fx theta_d x/r + cx
    DS   (Double Sphere, Usenko 2018):       params fx fy cx cy xi alpha  (wide-FOV lenses, used by Basalt/OpenVINS-like VIO)

Per-view poses are initialised by PnP on equidistant-undistorted corners (no OpenCV fisheye init). Held-out views (every 5th) are
scored by re-fitting only their poses with the intrinsics frozen. Outlier corners (residual > 3 px after the robust fit) are counted.
    python scripts/fisheye_calib_robust.py 'datasets/trackcam_calib/<ts>/view_*.png' [--f0 650]"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego_collector.camera.calibration import CharucoSpec, detect_charuco   # noqa: E402


def proj_kb4(P, p):
    fx, fy, cx, cy, k1, k2, k3, k4 = p; x, y, z = P.T; r = np.hypot(x, y); th = np.arctan2(r, z)
    t2 = th * th; thd = th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4); s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
    return np.c_[fx * s * x + cx, fy * s * y + cy]


def proj_ds(P, p):
    fx, fy, cx, cy, xi, al = p; x, y, z = P.T; d1 = np.sqrt(x * x + y * y + z * z); zz = xi * d1 + z; d2 = np.sqrt(x * x + y * y + zz * zz)
    den = al * d2 + (1 - al) * zz
    return np.c_[fx * x / den + cx, fy * y / den + cy]


MODELS = {"kb4": (proj_kb4, lambda f, c: [f, f, c[0], c[1], 0, 0, 0, 0]), "ds": (proj_ds, lambda f, c: [f * 0.6, f * 0.6, c[0], c[1], -0.2, 0.55])}


def init_pose(o, i, f, c):
    """equidistant undistortion (theta = r/f) -> unit rays -> PnP with an identity camera."""
    d = (i - c) / f; r = np.linalg.norm(d, axis=1); th = r; s = np.where(r > 1e-9, np.sin(th) / np.maximum(r, 1e-9), 1.0)
    rays = np.c_[d * s[:, None], np.cos(th)]; uv = rays[:, :2] / rays[:, 2:3]
    ok, rv, tv = cv2.solvePnP(o.astype(np.float64), uv, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    return np.r_[rv.ravel(), tv.ravel()]


def residuals(x, model, obj, img, n_int, fixed_int=None):
    proj = MODELS[model][0]; intr = x[:n_int] if fixed_int is None else fixed_int; off = 0 if fixed_int is not None else n_int; out = []
    for k, (o, i) in enumerate(zip(obj, img)):
        pv = x[off + 6 * k: off + 6 * k + 6]; P = o @ R.from_rotvec(pv[:3]).as_matrix().T + pv[3:]
        out.append((proj(P, intr) - i).ravel())
    return np.concatenate(out)


def fit(model, obj, img, size, f0):
    c = np.array(size, float) / 2; intr0 = np.array(MODELS[model][1](f0, c), float)
    poses = [init_pose(o, i, f0, c) for o, i in zip(obj, img)]; x0 = np.r_[intr0, np.concatenate(poses)]
    sol = least_squares(residuals, x0, args=(model, obj, img, len(intr0)), loss="soft_l1", f_scale=1.0, method="trf", max_nfev=400, x_scale="jac")
    e = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    return sol.x[:len(intr0)], e


def score(model, intr, obj, img, f0, size):
    c = np.array(size, float) / 2; es = []
    for o, i in zip(obj, img):
        x0 = init_pose(o, i, f0, c); s = least_squares(residuals, x0, args=(model, [o], [i], 0, intr), loss="soft_l1", f_scale=1.0, max_nfev=200)
        es.append(np.linalg.norm(s.fun.reshape(-1, 2), axis=1))
    return np.concatenate(es)


def stats(e):
    return dict(rms_px=round(float(np.sqrt(np.mean(e ** 2))), 3), median_px=round(float(np.median(e)), 3), p95_px=round(float(np.percentile(e, 95)), 3),
                rms_inliers_px=round(float(np.sqrt(np.mean(e[e < 3] ** 2))), 3), outliers_gt3px=int((e >= 3).sum()), n=int(len(e)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images"); ap.add_argument("--f0", type=float, default=None); ap.add_argument("--charuco", default="configs/charuco_wrist.yaml"); a = ap.parse_args()
    spec = CharucoSpec.from_yaml(Path(a.charuco)); files = sorted(glob.glob(a.images)); obj, img = [], []; size = None
    for f in files:
        im = cv2.imread(f); size = (im.shape[1], im.shape[0]); d = detect_charuco(im, spec, min_corners=12)
        if d is None: continue
        obj.append(d.object_points.reshape(-1, 3).astype(np.float64)); img.append(d.image_points.reshape(-1, 2).astype(np.float64))
    f0 = a.f0 or 650.0 * size[0] / 1920; val = list(range(0, len(obj), 5)); tr = [k for k in range(len(obj)) if k not in val]
    rep = dict(images=a.images, size=size, views=len(obj), train=len(tr), val=len(val))
    for model in MODELS:
        intr, e_tr = fit(model, [obj[k] for k in tr], [img[k] for k in tr], size, f0)
        e_va = score(model, intr, [obj[k] for k in val], [img[k] for k in val], f0, size)
        rep[model] = dict(intrinsics=np.round(intr, 5).tolist(), train=stats(e_tr), val=stats(e_va))
        print(f"{model}: intrinsics {np.round(intr, 4).tolist()}\n   train {stats(e_tr)}\n   val   {stats(e_va)}", flush=True)
    out = Path(files[0]).parent / "intrinsics_robust.json"; out.write_text(json.dumps(rep, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()
