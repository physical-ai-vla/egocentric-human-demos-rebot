"""[2026-10-06 user, EgoMimic-style shared frame step 1] robot GLOBAL camera (bridge 'middle', fixed) <-> robot base calibration.
A red cube is taped to the right gripper tip. The right arm visits N TCP positions (fixed gripper orientation, via the 8056 UI's
/hra_start_pose absolute target -> IK, left arm held); at each, the global frame's red-blob centroid (px) and the FK TCP (base, m) are
recorded. Solve pinhole fx=fy=f, cx, cy, camera pose T_cam_base (R, t) and the cube offset o in the TCP frame by least squares on the
reprojection of T_base_tcp @ o.   usage: collect_and_solve.py collect [--n 20] | solve
"""
import base64, json, pathlib, sys, time
import cv2, numpy as np, requests
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
OUT = pathlib.Path.home() / "c8/hra_red/global_calib"; UI = "http://localhost:8056"; BR = "http://localhost:8021"
sys.path.insert(0, str(pathlib.Path.home() / "holobrain-mac-model"))


def red_centroid(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (float(cen[i][0]), float(cen[i][1]), int(st[i, cv2.CC_STAT_AREA])) if st[i, cv2.CC_STAT_AREA] >= 40 else None


def tcp_now():
    import infer_core_v4 as IC, eef_kin
    k = object.__new__(IC.V4Inferencer); k.kin = eef_kin.Kin({"max_joint_delta": None})
    q = requests.get(BR + "/observe", timeout=10).json()["joints_rad"]; m, _ = k._tcp_mat(q)
    return m[1]


def probe():
    """find where the taped cube enters the global view: step the TCP forward / down from the current pose, report the blob"""
    T0 = tcp_now(); p0 = T0[:3, 3].copy(); hits = []
    for dx in (0.0, 0.05, 0.10, 0.15, 0.20):
        for dz in (0.0, -0.04):
            p = p0 + np.array([dx, 0.05, dz])
            r = requests.post(UI + f"/hra_start_pose?pitch=15&x_mm={p[0]*1000:.1f}&y_mm={p[1]*1000:.1f}&z_mm={max(p[2], 0.10)*1000:.1f}&steps=6", timeout=120).json()
            if r.get("error"):
                print("move", np.round(p * 1000), r["error"]); continue
            time.sleep(0.8)
            img = cv2.imdecode(np.frombuffer(requests.get(BR + "/frame/middle", timeout=10).content, np.uint8), 1)
            c = red_centroid(img); T = tcp_now()
            print("tcp mm", np.round(T[:3, 3] * 1000, 1), "blob", c, flush=True); hits.append((T[:3, 3].tolist(), c))
    json.dump(hits, open(OUT / "probe.json", "w"), indent=1)


def collect(n):
    rng = np.random.default_rng(0)
    # workspace box in front of the right arm (base frame, m); orientation: gripper 15 deg below horizontal
    lo, hi = np.array([0.34, -0.22, 0.10]), np.array([0.52, -0.06, 0.20])   # visible region found by probe (10-06)
    pts = lo + (hi - lo) * rng.random((n, 3)); recs = []
    for i, p in enumerate(pts):
        r = requests.post(UI + f"/hra_start_pose?pitch=15&x_mm={p[0]*1000:.1f}&y_mm={p[1]*1000:.1f}&z_mm={p[2]*1000:.1f}&steps=6", timeout=120).json()
        if r.get("error"):
            print(i, "move refused/failed:", r["error"]); continue
        time.sleep(0.8)
        T = tcp_now()
        img = cv2.imdecode(np.frombuffer(requests.get(BR + "/frame/middle", timeout=10).content, np.uint8), 1)
        c = red_centroid(img)
        cv2.imwrite(str(OUT / f"img_{i:02d}.jpg"), img)
        recs.append(dict(i=i, T=T.tolist(), uv=None if c is None else c[:2], area=None if c is None else c[2], size=list(img.shape[:2])))
        print(i, "tcp mm", np.round(T[:3, 3] * 1000, 1), "blob", None if c is None else (round(c[0], 1), round(c[1], 1), c[2]), flush=True)
    json.dump(recs, open(OUT / "samples.json", "w"), indent=1)


def solve():
    recs = [r for r in json.load(open(OUT / "samples.json")) if r["uv"] is not None]
    H, W = recs[0]["size"]; T = np.array([r["T"] for r in recs]); uv = np.array([r["uv"] for r in recs])
    def proj(x):
        f, cx, cy = x[0], x[1], x[2]; R = Rot.from_rotvec(x[3:6]).as_matrix(); t = x[6:9]; o = x[9:12]
        Pb = np.einsum("nij,j->ni", T[:, :3, :3], o) + T[:, :3, 3]; Pc = Pb @ R.T + t
        return np.stack([f * Pc[:, 0] / Pc[:, 2] + cx, f * Pc[:, 1] / Pc[:, 2] + cy], 1), Pc
    def res(x): return (proj(x)[0] - uv).ravel()
    # init: camera behind/above looking at the workspace; try a few rotations, keep the best
    best = None
    Pb0 = T[:, :3, 3]
    for f0 in (450, 600, 800):
        K = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1.0]])
        ok, rv, tv = cv2.solvePnP(Pb0.astype(np.float64), uv.astype(np.float64), K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok: continue
        x0 = np.r_[f0, W / 2, H / 2, rv.ravel(), tv.ravel(), 0, 0, 0]
        r = least_squares(res, x0, loss="soft_l1", f_scale=3.0)
        if best is None or r.cost < best.cost: best = r
    x = best.x; e = np.linalg.norm(res(x).reshape(-1, 2), axis=1)
    R = Rot.from_rotvec(x[3:6]).as_matrix(); t = x[6:9]
    T_cam_base = np.eye(4); T_cam_base[:3, :3] = R; T_cam_base[:3, 3] = t; T_base_cam = np.linalg.inv(T_cam_base)
    out = dict(n=len(recs), image_size=[W, H], f=float(x[0]), cx=float(x[1]), cy=float(x[2]), cube_offset_tcp_m=x[9:12].tolist(),
               T_cam_base=T_cam_base.tolist(), T_base_cam=T_base_cam.tolist(), cam_pos_base_m=T_base_cam[:3, 3].tolist(),
               reproj_px_p50=float(np.median(e)), reproj_px_p90=float(np.percentile(e, 90)), reproj_px_max=float(e.max()))
    json.dump(out, open(OUT / "global_cam_calib.json", "w"), indent=1)
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v)) for k, v in out.items() if k not in ("T_cam_base", "T_base_cam")}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20)
    elif sys.argv[1] == "probe":
        probe()
    else:
        solve()
