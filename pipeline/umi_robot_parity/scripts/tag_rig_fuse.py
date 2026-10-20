"""Offline IMU + AprilTag rig fusion v0 (2026-10-20, GPT review step "IMU smoother"): orientation from the gyro, anchored to the tags.
  1. gyro bias = median gyro over the still samples (|w| small) of the take; R_G_I(t) by integrating (gyro - bias) on the host clock
     (+ the calibrated IMU->camera time offset of umi_tag_rig_v001.json `imu.offset_ms`)
  2. rig orientation in the camera: R_C_rig(t) = R_C_G(t) R_G_I(t) R_I_rig, R_I_rig = calibrated (imu.R_I_rig). R_C_G(t) is fitted to the
     tag orientations (multi-corner PnP) with QUALITY weights (2+ tags: 1.0, single tag: --w-single) in a sliding +-window (slow drift)
  3. position: every visible tag's CENTRE from PnP (well measured even for one tag) moved to the rig origin with the FUSED rotation:
     p_rig = p_tag_i - R_C_rig T_rig_tag_i.t, averaged over the visible tags (removes the single-tag rotation -> position leak)
fuse_take(take, hand) -> dict(t_ns, T_C_rig (n,4,4), ntags, ok) on the colour frames where at least one tag of that UMI is visible."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from tag_rig_pose import rig_pose, corners_obj, DepthStore, quad_depth_m      # noqa: E402
SIDE = {"오른손": ("UMI 오른손", "right"), "왼손": ("UMI 왼손", "left")}


def fuse_take(take, hand: str, rig_cfg=None, tag_size: float = 0.048, w_single: float = 0.15, window_s: float = 10.0, use_depth: bool = False, rot_source: str = "imu"):
    from handumi_collector.pose.timing import fit_device_to_host
    E = Path(take); meta = json.loads((E / "meta.json").read_text()); cfg = rig_cfg or json.loads((ROOT / "configs/calibration/umi_tag_rig_v001.json").read_text())
    name, side = SIDE[hand]; Trt = {int(k): np.array(v) for k, v in cfg[name]["T_rig_tag"].items()}
    R_I_rig = np.array(cfg[name]["imu"]["R_I_rig"]); off = float(cfg[name]["imu"]["offset_ms"]) / 1000
    I = meta["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]]); tc = np.load(E / "t_ns.npy") / 1e9
    z = np.load(E / f"imu_{side}.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]) / 1e9 + off; g = z["gyro"].astype(float)
    still = np.linalg.norm(g, axis=1) < np.radians(3); bias = np.median(g[still], 0) if still.sum() > 200 else np.zeros(3)
    dq = R.from_rotvec((g[:-1] - bias) * np.diff(ti)[:, None]); Rg = [R.identity()]
    for d in dq: Rg.append(Rg[-1] * d)                   # R_G_I, body-frame right multiplication
    Rg = R.concatenate(Rg)
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm); obj = corners_obj(tag_size)
    F = []
    DS = DepthStore(E) if use_depth else None           # 10-20: depth.npz or per-frame depth_%05d.png (long takes)
    for k in range(len(tc)):
        cs, ids, _ = det.detectMarkers(cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE))
        r = rig_pose(cs, ids, Trt, K, tag_size)
        if r is None or r["rmse_px"] > 2.0: continue
        cen = {}; dm = DS.mm(k) if DS is not None else None   # per-tag centre in the camera (PnP translation)
        for c, i in zip(cs, ids.ravel()):
            if int(i) in r["tags"]:
                ok, rv, tv = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    c_ = tv.ravel()
                    z_m = quad_depth_m(dm, c, 31)    # 10-20: range along the ray from the aligned depth (median inside the tag quad)
                    if z_m is not None and abs(z_m - c_[2]) < 0.10: c_ = c_ * (z_m / c_[2])
                    cen[int(i)] = c_
        j = int(np.clip(np.searchsorted(ti, tc[k]), 1, len(ti) - 1))
        F.append(dict(k=k, t=tc[k], R_tag=r["T"][:3, :3], R_GI=Rg[j].as_matrix(), w=1.0 if len(r["tags"]) >= 2 else w_single, cen=cen))
    if not F: return None
    # R_C_G per frame = R_C_rig_tag R_I_rig^T R_G_I^T ; weighted sliding mean
    Rcg = R.from_matrix(np.array([f["R_tag"] @ R_I_rig.T @ f["R_GI"].T for f in F])); tt = np.array([f["t"] for f in F]); ww = np.array([f["w"] for f in F])
    T_out, n_out, t_out, resid = [], [], [], []
    for idx, f in enumerate(F):
        m = np.abs(tt - f["t"]) <= window_s
        Rcg_k = Rcg[np.where(m)[0]].mean(weights=ww[m]).as_matrix()
        Rcr = Rcg_k @ f["R_GI"] @ R_I_rig if rot_source == "imu" else f["R_tag"]
        p = np.mean([f["cen"][i] - Rcr @ Trt[i][:3, 3] for i in f["cen"]], 0) if f["cen"] else None
        if p is None: continue
        T = np.eye(4); T[:3, :3] = Rcr; T[:3, 3] = p; T_out.append(T); n_out.append(len(f["cen"])); t_out.append(int(f["t"] * 1e9))
        resid.append(np.degrees(R.from_matrix(Rcr.T @ f["R_tag"]).magnitude()))
    return dict(t_ns=np.array(t_out), T_C_rig=np.array(T_out), ntags=np.array(n_out), bias=bias, tag_vs_fused_deg=np.array(resid))
