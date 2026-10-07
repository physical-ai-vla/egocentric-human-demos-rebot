"""[2026-10-07] origin metric scale from the MASt3R KEYFRAME-0 pointmap (frame 0 = origin hold, fingertip on the X plate):
plane of the surface the fingertip touches, fitted to the dense pointmap (camera frame, MASt3R units; X_world = s R X + t):
LARGEST of the parallel RANSAC planes (the table; chosen against cube-PnP 2026-10-07) among confident pixels outside the finger region.
scale (m per SLAM unit) = |n . tip_cam| / (s0 * plane distance), tip_cam = camera_tcp_v2 fingertip (m)."""
import json, pathlib, sys
import numpy as np, yaml
H = pathlib.Path.home(); TIP = np.array(yaml.safe_load(open(H / "umi_bridge/trackA_mast3r_pose_v1/data/handumi_camera_tcp_v2.yaml"))["translation"]["t_camera_tcp_m"], float)


def plane_scale(kf_npz):
    z = np.load(kf_npz, allow_pickle=True); X = z["X_cam"][0].astype(np.float64); C = z["C"][0].astype(np.float64); s0 = float(z["T_WC"][0][7])
    if int(z["frame_idx"][0]) != 0: return dict(valid=False, reason=f"kf0 is frame {int(z['frame_idx'][0])}")
    h, w = C.shape; K = np.asarray(z["K"], float); st = int(z["stride"])
    tip = K @ TIP; tu, tv = tip[0] / tip[2] / st, tip[1] / tip[2] / st                     # fingertip pixel in the dump grid
    vv, uu = np.mgrid[0:h, 0:w]
    finger = (np.abs(uu - tu) < 0.20 * w) & (vv > tv - 0.08 * h)                           # fingers come up from the bottom to the tip
    good = (C > np.percentile(C, 30)) & ~finger & (X[..., 2] > 0)
    Y = X[good]
    if len(Y) < 200: return dict(valid=False, reason="few points")
    rng = np.random.default_rng(0)
    def ransac(P):
        thr = 0.004 * np.median(P[:, 2]); best = None
        for _ in range(400):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
            if nn < 1e-12: continue
            n /= nn; d = -n @ a; inl = np.abs(P @ n + d) < thr
            if best is None or inl.sum() > best[0].sum(): best = (inl, n, d)
        inl = best[0]; c0 = P[inl].mean(0); _, _, vt = np.linalg.svd(P[inl] - c0); n = vt[2]; return inl, n, -n @ c0
    planes = []; rest = Y
    for _ in range(3):
        if len(rest) < 200: break
        inl, n, d = ransac(rest)
        if inl.sum() < 150: break
        planes.append(dict(n=n, d=d, n_in=int(inl.sum()))); rest = rest[~inl]
    if not planes: return dict(valid=False, reason="no plane")
    ref = planes[0]["n"]; par = [p for p in planes if abs(p["n"] @ ref) > np.cos(np.radians(8))]
    p = max(par, key=lambda p: p["n_in"])   # LARGEST parallel plane (= the table): vs cube PnP n=9 ratio 1.06, |log| p50 0.08 (nearest: 1.15 / 0.17)
    n = p["n"] if p["n"][1] > 0 else -p["n"]; d_cam = abs(p["d"])
    h_m = abs(n @ TIP); s = h_m / (s0 * d_cam)
    return dict(valid=True, s=float(s), h_m=float(h_m), d_cam=float(d_cam), s0=s0, planes=len(planes), parallel=len(par),
                gap_planes=float(abs(abs(par[0]["d"]) - abs(par[-1]["d"])) / max(d_cam, 1e-9)) if len(par) > 1 else 0.0)


if __name__ == "__main__":
    A = H / "c8/hra_a100"; R = A / "runs_kf"; out = {}
    for d in sorted(p for p in R.iterdir() if (p / "ss1.kf.npz").exists()):
        try: out[d.name] = plane_scale(d / "ss1.kf.npz")
        except Exception as e: out[d.name] = dict(valid=False, reason=f"{type(e).__name__}: {e}")
    json.dump(out, open(A / "origin_scale_kf.json", "w"), indent=1)
    rows = []
    for t, r in out.items():
        q = A / "raw/_scale_qc" / f"HRA_A100_{t[:15]}_{t.split('_')[2]}.json"
        sp = json.load(open(q)) if q.exists() else {}
        rows.append((t, r.get("s"), sp.get("s_pnp"), sp.get("valid")))
    v = [(a, b) for _, a, b, ok in rows if a and b and ok]
    print("episodes", len(out), "valid", sum(r.get("valid", False) for r in out.values()))
    if v:
        a, b = np.array(v).T; r = a / b; print(f"vs cube-PnP (valid, n={len(v)}): ratio s_origin/s_pnp median {np.median(r):.3f}, |log ratio| p50 {np.median(np.abs(np.log(r))):.3f} p90 {np.percentile(np.abs(np.log(r)), 90):.3f}")
