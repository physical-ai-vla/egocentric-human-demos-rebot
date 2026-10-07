"""[2026-10-06 user] HRA human data -> ROBOT BASE frame (the EgoMimic-style shared frame).
head camera (fixed, = the robot's global cam, calibrated: global_cam_intrinsics.json + board_state.json T_base_cam) sees the cube at the
episode start; the wrist SLAM (metric via cube PnP, TCP via camera_tcp_v2) sees the same static cube. Per episode:
  cube in base   : head PnP (undistorted corners, f/cx/cy from the checkerboard) -> T_base_cam @ T_cam_cube
  cube in SLAM   : wrist PnP on the frames the scale QC accepted -> T_slam_cam(i) @ T_wcam_cube, best-reprojection frame
  rotation SLAM->base: R = R_b_cube S R_s_cube^T over the 24 cube symmetries S, kept if it maps the IMU up (SLAM) onto the table up
                   (base) within 20 deg -> ~4 yaw candidates; t = c_base - R c_slam
  branch choice  : iterative consensus -- the candidate whose start TCP is closest to the median start of all episodes
Output: base_start.json (per episode start TCP in base: position mm, approach-axis pitch, height above the table) + a summary."""
import ast, itertools, json, pathlib, re, sys
import cv2, numpy as np, yaml
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20 import cube_pnp as CP
from ego_cart20.geometry.transforms import pose_to_T
RAW = H / "c8/hra_red/raw"; QC = RAW / "_scale_qc"; EXP = H / "c8/robotlike/export"; PR = H / "c8/hra_red/processed"
GC = H / "c8/hra_red/global_calib"; OUT = H / "c8/hra_red/base_frame"
I = json.load(open(GC / "global_cam_intrinsics.json")); K = np.array(I["K"]); Dh = np.array(I["dist"])
BS = json.load(open(GC / "board_state.json")); T_base_cam = np.array(BS["solve"]["T_base_cam"]); T_base_board = np.array(BS["solve"]["T_base_board"])
up_b = -T_base_board[:3, 2]; up_b = up_b if up_b[2] > 0 else -up_b; table_z = T_base_board[2, 3]
CT = yaml.safe_load(open(H / "umi_bridge/trackA_mast3r_pose_v1/data/handumi_camera_tcp_v2.yaml"))
X = np.load(OUT / "X_cam_tcp_full.npy"); Xi = np.linalg.inv(X)   # cp6X @ HX.X, exactly what right_only.py applied
R_IMU_CORR = np.load(OUT / "R_imu_corr.npy")   # 18 deg about cam x: fitted 2026-10-06 (cube axis 17.8->3.4 deg, SLAM up drift 17.2->4.9 deg)
HD = np.load(H / "c8/hra_red/head_link/head_dets.npy", allow_pickle=True).item()
SYM = [Rot.from_matrix(m).as_matrix() for m in (np.array(p) for p in itertools.product(*[[-1, 0, 1]] * 9)) if False]
# the 24 proper rotations of the cube (signed permutation matrices, det +1)
SYM = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((-1, 1), repeat=3):
        M = np.zeros((3, 3))
        for r, (c, s_) in enumerate(zip(perm, sg)): M[r, c] = s_
        if np.linalg.det(M) > 0: SYM.append(M)


def Tm(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


rows = []
eps = [json.loads(l)["episode_id"] for s in ("train", "val") for l in open(PR / f"{s}_manifest.jsonl")]
for ep in eps:
    try:
        if ep not in HD: raise ValueError("no head cube detection")
        hx = np.asarray(HD[ep], float)
        und = cv2.undistortPoints(hx.reshape(-1, 1, 2), K, Dh, P=K).reshape(-1, 2)
        rh = CP.pnp_cube(und, K)
        if rh is None or rh[2] > 4.0: raise ValueError(f"head PnP failed ({None if rh is None else round(rh[2], 1)} px)")
        R_c_cube = cv2.Rodrigues(rh[1])[0]; T_b_cube = T_base_cam @ Tm(R_c_cube, rh[0])
        num = ep.split("_")[-1]; tag = f"red_20261003_152502_{num}_right"; ex = EXP / tag
        z = np.load(RAW / ep / "raw_episode.npz"); v = z["right_valid"].astype(bool); T = pose_to_T(z["right_position"], z["right_quaternion"])
        vf = z["right_video_frame"]; i0 = int(np.flatnonzero(v)[0]); T0 = T[i0]
        qc = json.load(open(QC / f"{ep}.json")); fr = qc["frames"]; fr = ast.literal_eval(fr) if isinstance(fr, str) else fr
        ok_i = sorted({f["i"] for f in fr if f.get("ok")})
        Kw, Dw, size = CP.read_setting(ex / "orbslam_setting.yaml"); Pw, undw = CP.undistorter(Kw, Dw, size)
        cap = cv2.VideoCapture(str(ex / "raw_video.mp4")); best = None; fi = -1; want = set(ok_i)
        while want:
            okf, img = cap.read(); fi += 1
            if not okf: break
            if fi not in want: continue
            want.discard(fi)
            m = CP.cube_blob(undw(img))
            if m is None: continue
            hxw, sol = CP.hexagon(m)
            if hxw is None or sol < CP.MIN_SOLIDITY: continue
            r = CP.pnp_cube(hxw, Pw)
            if r is None or r[2] > CP.MAX_REPROJ_PX: continue
            k = np.flatnonzero((vf == fi) & v)
            if len(k) == 0: continue
            if best is None or r[2] < best[0]:
                best = (r[2], T[k[0]] @ Xi @ Tm(cv2.Rodrigues(r[1])[0], r[0]))
        cap.release()
        if best is None: raise ValueError("no wrist cube pose")
        T_s_cube = best[1]
        # IMU up in the SLAM world (start still window, rotated by the anchor camera)
        t_ = (ex / "orbslam_setting.yaml").read_text()
        Tbc = np.array([float(x) for x in re.search(r"IMU.T_b_c1:.*?data: \[([^\]]+)\]", t_, re.S).group(1).split(",")]).reshape(4, 4)
        acc = json.load(open(ex / "imu_data.json"))["1"]["streams"]["ACCL"]["samples"]
        a0 = np.mean([s["value"] for s in acc if s["cts"] < 1500], axis=0); up_c = R_IMU_CORR @ Tbc[:3, :3].T @ a0; up_c /= np.linalg.norm(up_c)
        up_s = (T0 @ Xi)[:3, :3] @ up_c
        def horiz(Rcube, up):
            # cube axis closest to horizontal, projected onto the plane normal to up
            i = int(np.argmin(np.abs(Rcube.T @ up))); h = Rcube[:, i] - (Rcube[:, i] @ up) * up; return h / np.linalg.norm(h)
        ax_b = float(np.degrees(np.arccos(np.max(np.abs(T_b_cube[:3, :3].T @ up_b)))))
        ax_s = float(np.degrees(np.arccos(np.clip(np.max(np.abs(T_s_cube[:3, :3].T @ up_s)), -1, 1))))
        Rup = Rot.align_vectors([up_b], [up_s])[0].as_matrix()            # tilt: gravity == table normal, exact
        hb = horiz(T_b_cube[:3, :3], up_b); hs = Rup @ horiz(T_s_cube[:3, :3], up_s)
        yaw0 = np.arctan2(np.cross(hs, hb) @ up_b, hs @ hb)
        cands = []
        for kq in range(4):
            Rz = Rot.from_rotvec(up_b * (yaw0 + kq * np.pi / 2)).as_matrix(); R = Rz @ Rup
            t = T_b_cube[:3, 3] - R @ T_s_cube[:3, 3]; Ts = Tm(R, t) @ T0
            cands.append(dict(up_err_deg=max(ax_b, ax_s), T_start=Ts, T_align=Tm(R, t)))
        if not cands: raise ValueError("no symmetry branch aligns gravity with the table")
        rows.append(dict(ep=ep, cands=cands, ax_b=ax_b, ax_s=ax_s, cube_base_mm=(T_b_cube[:3, 3] * 1000).tolist(), head_reproj=rh[2], wrist_reproj=best[0]))
    except Exception as e:
        print(ep, "skip:", repr(e)[:110])
# consensus branch choice
P0 = np.median(np.array([c["T_start"][:3, 3] for r in rows for c in r["cands"]]), 0)
for _ in range(10):
    sel = [min(r["cands"], key=lambda c: np.linalg.norm(c["T_start"][:3, 3] - P0)) for r in rows]
    P1 = np.median(np.array([c["T_start"][:3, 3] for c in sel]), 0)
    if np.linalg.norm(P1 - P0) < 1e-4: break
    P0 = P1
out = []
for r, c in zip(rows, sel):
    Ts = c["T_start"]; a = Ts[:3, 0]
    out.append(dict(T_start=np.round(Ts, 5).tolist(), ep=r["ep"], start_tcp_mm=(Ts[:3, 3] * 1000).round(1).tolist(), above_table_mm=round(float((Ts[2, 3] - table_z) * 1000), 1),
                    approach_pitch_below_horiz_deg=round(float(np.degrees(np.arcsin(np.clip(-a @ up_b, -1, 1)))), 1),
                    approach_heading_deg=round(float(np.degrees(np.arctan2(a[1], a[0]))), 1), up_err_deg=round(c["up_err_deg"], 1),
                    n_cands=len(r["cands"]), cube_base_mm=np.round(r["cube_base_mm"], 1).tolist(), head_reproj_px=round(r["head_reproj"], 2)))
json.dump(out, open(OUT / "base_start.json", "w"), indent=1)
A = np.array([o["start_tcp_mm"] for o in out]); q = lambda x: "p10 %.0f p50 %.0f p90 %.0f" % tuple(np.percentile(x, [10, 50, 90]))
print(f"episodes linked {len(out)} / {len(eps)}")
for k_, n in ((0, "x"), (1, "y"), (2, "z")): print(f"human start TCP {n} (base mm): ", q(A[:, k_]))
print("height above table (mm):        ", q([o["above_table_mm"] for o in out]))
print("approach pitch below horiz (deg)", q([o["approach_pitch_below_horiz_deg"] for o in out]))
print("approach heading in base (deg): ", q([o["approach_heading_deg"] for o in out]))
print("cube-axis vs up (worst of head/wrist) (deg)", q([o["up_err_deg"] for o in out]))
print("cube in base (mm) x", q([o["cube_base_mm"][0] for o in out]), " y", q([o["cube_base_mm"][1] for o in out]), " z", q([o["cube_base_mm"][2] for o in out]))
print(f"table z in base {table_z*1000:.1f} mm")
