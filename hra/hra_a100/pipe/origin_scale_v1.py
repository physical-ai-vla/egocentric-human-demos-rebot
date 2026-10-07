"""[2026-10-07 user "원점으로 스케일 구하는 거"] metric scale of a HandUMI MASt3R-SLAM trajectory from the ORIGIN HOLD.
At the origin the fingertip (= HandUMI TCP, camera_tcp_v2: t_tip_in_cam = (0, 55.3, 122.5) mm) rests on the plane it touches (the
X on the stand-base plate). Metric camera height above that plane = |t_tip_in_cam . n_cam| (n = plane normal in the camera frame).
SLAM height = distance from the camera centre to the plane triangulated in SLAM units: ORB matches between an origin-hold frame and
frames of the "prep" move (SLAM poses known, undistorted pinhole = ss1.calib.yaml), RANSAC plane through the points nearest the
fingertip ray. scale (m per SLAM unit) = h_metric / h_slam; median over pairs. No cube, no IMU, CPU only."""
from __future__ import annotations
import json, pathlib, re, sys
import numpy as np, cv2, yaml
from scipy.spatial.transform import Rotation as Rot

H = pathlib.Path.home()
CT = yaml.safe_load(open(H / "umi_bridge/trackA_mast3r_pose_v1/data/handumi_camera_tcp_v2.yaml"))
TIP = np.array(CT["translation"]["t_camera_tcp_m"], float)          # fingertip in the camera frame (m)


def load_traj(csv):
    import csv as _c
    rows = list(_c.DictReader(open(csv))); T = {}
    for r in rows:
        if r["state"] not in ("TRACKING", "INIT"): continue
        M = np.eye(4); M[:3, :3] = Rot.from_quat([float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")]).as_matrix(); M[:3, 3] = [float(r["x"]), float(r["y"]), float(r["z"])]
        T[int(r["frame_idx"])] = M
    return T


def undistorter(setting, calib):
    t = pathlib.Path(setting).read_text(); g = lambda k: float(re.search(rf"^{k}:\s*([-\d.eE]+)", t, re.M).group(1))
    K = np.array([[g("Camera1.fx"), 0, g("Camera1.cx")], [0, g("Camera1.fy"), g("Camera1.cy")], [0, 0, 1.0]])
    D = np.array([g(f"Camera1.k{i}") for i in (1, 2, 3, 4)]).reshape(4, 1); W, Hh = int(g("Camera.width")), int(g("Camera.height"))
    c = yaml.safe_load(open(calib)); fx, fy, cx, cy = c["calibration"]; P = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (W, Hh), cv2.CV_32FC1)
    return P, (lambda im: cv2.remap(im, mx, my, cv2.INTER_LINEAR)), (W, Hh)


def frames(video, idxs):
    c = cv2.VideoCapture(str(video)); out = {}; i = -1; want = set(idxs)
    while want:
        ok, f = c.read(); i += 1
        if not ok: break
        if i in want: out[i] = f; want.discard(i)
    return out


def episode_scale(export_dir, run_dir, go_frame, *, n_ref=3, step=3, span=45, debug=False):
    T = load_traj(run_dir / "ss1.csv"); P, und, (W, Hh) = undistorter(export_dir / "orbslam_setting.yaml", run_dir / "ss1.calib.yaml")
    refs = [k for k in np.linspace(8, max(go_frame - 6, 9), n_ref).astype(int) if k in T]
    cands = [k for k in range(go_frame, go_frame + span, step) if k in T]
    F = frames(export_dir / "raw_video.mp4", set(refs) | set(cands)); orb = cv2.ORB_create(4000, fastThreshold=8); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    mask = np.full((Hh, W), 255, np.uint8)
    tip_px = P @ TIP; tip_px = tip_px[:2] / tip_px[2]                       # where the fingertip projects (fingers occlude around it)
    cv2.rectangle(mask, (int(W * 0.33), int(tip_px[1] - 40)), (int(W * 0.67), Hh), 0, -1)
    kd = {}
    for k, f in F.items():
        g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(und(f), cv2.COLOR_BGR2GRAY)); kd[k] = orb.detectAndCompute(g, mask)
    est = []
    for r in refs:
        kr, dr = kd[r]
        if dr is None: continue
        Tr = T[r]; Pr = P @ np.linalg.inv(Tr)[:3]
        for j in cands:
            kj, dj = kd[j]
            if dj is None: continue
            mt = [a for a, b in (p for p in bf.knnMatch(dr, dj, k=2) if len(p) == 2) if a.distance < 0.75 * b.distance]
            if len(mt) < 40: continue
            x0 = np.float32([kr[m.queryIdx].pt for m in mt]).T; x1 = np.float32([kj[m.trainIdx].pt for m in mt]).T
            Tj = T[j]; Pj = P @ np.linalg.inv(Tj)[:3]
            Xh = cv2.triangulatePoints(Pr, Pj, x0, x1); X = (Xh[:3] / Xh[3]).T                       # SLAM world
            cr, cj = Tr[:3, 3], Tj[:3, 3]; base = np.linalg.norm(cj - cr)
            if base < 1e-6: continue
            Xc = (np.linalg.inv(Tr) @ np.c_[X, np.ones(len(X))].T)[:3].T                              # ref camera frame
            ang = np.degrees(np.arccos(np.clip(np.sum((X - cr) * (X - cj), 1) / (np.linalg.norm(X - cr, axis=1) * np.linalg.norm(X - cj, axis=1) + 1e-12), -1, 1)))
            rp = lambda Pm, x: np.linalg.norm(((Pm @ np.c_[X, np.ones(len(X))].T)[:2] / (Pm @ np.c_[X, np.ones(len(X))].T)[2]) - x, axis=0)
            ok = (Xc[:, 2] > 0) & (ang > 1.0) & (rp(Pr, x0) < 2.0) & (rp(Pj, x1) < 2.0)
            if ok.sum() < 30: continue
            Y = Xc[ok]
            # RANSAC plane (ref camera frame); threshold relative to the median depth
            thr = 0.01 * np.median(Y[:, 2]); best = None; rng = np.random.default_rng(0)
            for _ in range(300):
                a, b, c = Y[rng.choice(len(Y), 3, replace=False)]; n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
                if nn < 1e-12: continue
                n /= nn; d = -n @ a; inl = np.abs(Y @ n + d) < thr
                if best is None or inl.sum() > best[0].sum(): best = (inl, n, d)
            inl, n, d = best
            if inl.sum() < 25: continue
            c0 = Y[inl].mean(0); u, s_, vt = np.linalg.svd(Y[inl] - c0); n = vt[2]; d = -n @ c0          # refit
            if n @ np.array([0, 1.0, 0]) < 0: n, d = -n, -d                                               # normal pointing "down" in the image
            h_slam = abs(d)                                                                               # camera centre (origin) to plane
            h_m = abs(TIP @ n - 0) if False else abs(n @ TIP + d * 0)                                     # tip lies ON the plane: h_metric = |n . tip| (plane through tip)
            # plane through the tip in METRIC camera coords has offset -n.tip; its distance to the camera centre = |n . tip|
            if h_slam <= 0 or h_m < 0.02: continue
            est.append(dict(ref=r, j=j, s=h_m / h_slam, h_m=h_m, inliers=int(inl.sum()), n_pts=int(ok.sum()), tilt_deg=float(np.degrees(np.arccos(abs(n[1]))))))
    if not est: return dict(valid=False, reason="no plane", n_pairs=0)
    s = np.array([e["s"] for e in est]); med = float(np.median(s)); spread = float((np.percentile(s, 84) - np.percentile(s, 16)) / 2 / med)
    return dict(valid=spread < 0.15 and len(est) >= 5, s=med, rel_spread=spread, n_pairs=len(est), h_m_med=float(np.median([e["h_m"] for e in est])),
                reason="" if spread < 0.15 and len(est) >= 5 else f"spread {spread:.2f} / pairs {len(est)}", pairs=est if debug else None)


if __name__ == "__main__":
    A = H / "c8/hra_a100"; RAWS = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"; out = {}
    tags = sys.argv[1:] or sorted(p.name for p in (A / "runs").iterdir() if (p / "ss1.csv").exists())
    for tag in tags:
        sess = "HRA_A100_" + "_".join(tag.split("_")[:2]); ep = "episode_" + tag.split("_")[2]
        ev = json.load(open(RAWS / sess / ep / "events.json")); g = [e for e in ev if e["kind"] == "go_cue"]
        go_frame = int(round(g[0]["t_rel_s"] * 30)) if g else 60
        try: r = episode_scale(A / "export" / tag, A / "runs" / tag, go_frame)
        except Exception as e: r = dict(valid=False, reason=f"{type(e).__name__}: {e}")
        out[tag] = r; print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "pairs"}, flush=True)
    json.dump(out, open(A / "origin_scale.json", "w"), indent=1, default=float)
