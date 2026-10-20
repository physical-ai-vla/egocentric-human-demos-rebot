"""Stereo Phase S2 + S3 from one stereo_imu_record --proto calib take (2026-10-15): KB4 intrinsics of BOTH fisheyes + T_CR_CL (baseline).
  detect   Kalibr AprilGrid (OpenCV aruco DICT_APRILTAG_36h11, markerBorderBits=1 works on the printed grid, 25 mm, spacing 0.3) every 2nd frame
  S2       per camera: up to 80 views spread over the image (slow first 42 s), own soft-L1 LM (scripts/fisheye_calib_robust.fit, KB4),
           held-out every 5th view scored with intrinsics frozen; left also compared with trackcam_fisheye_v001 (/save_scale)
  S3       frame pairs |t_L - t_R| < 20 ms with low gyro rate (< 0.3 rad/s, skew -> < 0.3 deg) and >= 8 tags in both: board pose per camera
           (pose-only LM, intrinsics frozen), T_CR_CL = T_CR_B inv(T_CL_B) per pair, robust mean, then joint refine of T_CR_CL with the
           left board poses held -> right reprojection; reports baseline, rotation, residuals, per-pair spread
Writes <take>/stereo_calib.json and configs/calibration/stereo_trackcam_v001.yaml (intrinsics at FULL 1920x1080 and the saved size).
    python scripts/stereo_calib.py datasets/stereo_imu/<ts>_calib"""
from __future__ import annotations
import argparse, datetime, json, sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from fisheye_calib_robust import fit, proj_kb4, residuals, init_pose, stats   # noqa: E402
from trackcam_board_lever import board_corners                              # noqa: E402
from handumi_collector.pose.timing import fit_device_to_host                 # noqa: E402


def detect(video: Path, n: int, step: int = 2, size=0.025):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), (lambda p: (setattr(p, "cornerRefinementMethod", cv2.aruco.CORNER_REFINE_SUBPIX), p)[-1])(cv2.aruco.DetectorParameters()))
    cap = cv2.VideoCapture(str(video)); out = {}; k = 0
    while k < n:
        ok, f = cap.read()
        if not ok: break
        if k % step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); c, ids, _ = det.detectMarkers(g)
            if ids is not None:
                keep = [(cc.reshape(4, 2), int(i)) for cc, i in zip(c, ids.ravel()) if i < 36]
                if len(keep) >= 6: out[k] = (np.vstack([board_corners(i, size) for _, i in keep]), np.vstack([cc for cc, _ in keep]).astype(np.float64), len(keep))
        k += 1
    return out, (g.shape[1], g.shape[0])


def pose_only(intr, o, i, f0, size):
    c = np.array(size, float) / 2; x0 = init_pose(o, i, f0, c)
    s = least_squares(residuals, x0, args=("kb4", [o], [i], 0, intr), loss="soft_l1", f_scale=1.0, max_nfev=200)
    T = np.eye(4); T[:3, :3] = R.from_rotvec(s.x[:3]).as_matrix(); T[:3, 3] = s.x[3:]; return T, np.linalg.norm(s.fun.reshape(-1, 2), axis=1)


def spread_select(dets: dict, keys, size, nmax=80):
    """greedy: cover a 4x3 grid of board-centre positions and many tag counts, then time-uniform fill."""
    cells = {}
    for k in keys:
        cx, cy = dets[k][1].mean(0); cell = (int(cx * 4 / size[0]), int(cy * 3 / size[1])); cells.setdefault(cell, []).append(k)
    sel = []
    while len(sel) < nmax and any(cells.values()):
        for cell in list(cells):
            if cells[cell] and len(sel) < nmax:
                lst = cells[cell]; sel.append(lst.pop(len(lst) // 2)); cells[cell] = lst[::2] if len(lst) > 6 else lst
    return sorted(sel)


STILL = dict(on=False, gyro=0.3, win_ns=50e6, ti=None, wn=None)


def is_still(t_ns):
    m = np.abs(STILL["ti"] - t_ns) < STILL["win_ns"]; return bool(m.any()) and float(STILL["wn"][m].max()) < STILL["gyro"]


def intrinsics(name, dets, size, f0, t_rel, t_abs=None):
    if STILL["on"]: keys = [k for k in dets if dets[k][2] >= 10 and is_still(t_abs[k])]; sel = spread_select(dets, keys, size, nmax=120)
    else: keys = [k for k in dets if t_rel[k] < 42 and dets[k][2] >= 10]; sel = spread_select(dets, keys, size)
    tr, te = [k for j, k in enumerate(sel) if j % 5], [k for j, k in enumerate(sel) if j % 5 == 0]
    intr, e = fit("kb4", [dets[k][0] for k in tr], [dets[k][1] for k in tr], size, f0)
    eh = np.concatenate([pose_only(intr, dets[k][0], dets[k][1], f0, size)[1] for k in te])
    print(f"[{name}] views {len(tr)}+{len(te)}  fx {intr[0]:.1f} fy {intr[1]:.1f} cx {intr[2]:.1f} cy {intr[3]:.1f} k {np.round(intr[4:], 4).tolist()}  train {stats(e)}  held-out {stats(eh)}", flush=True)
    return intr, dict(views_train=len(tr), views_heldout=len(te), train=stats(e), heldout=stats(eh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--f0-left", type=float, default=None); ap.add_argument("--f0-right", type=float, default=None)
    ap.add_argument("--still", action="store_true", help="still/move take: use only frames with |gyro| < 0.05 rad/s within +-150 ms (intrinsics AND pairs)")
    ap.add_argument("--out-yaml", default="configs/calibration/stereo_trackcam_v001.yaml"); a = ap.parse_args()
    E = Path(a.take); meta = json.loads((E / "meta.json").read_text()); sc = meta["save_scale"]
    tl, tr = np.load(E / "cam_left_t_ns.npy"), np.load(E / "cam_right_t_ns.npy"); t0 = min(tl[0], tr[0])
    dL, size = detect(E / "cam_left.mp4", len(tl)); dR, _ = detect(E / "cam_right.mp4", len(tr))
    z0 = np.load(E / "imu_right.npz"); STILL.update(ti=fit_device_to_host(z0["device_us"], z0["host_ns"]).to_host_ns(z0["device_us"]), wn=np.linalg.norm(z0["gyro"], axis=1))
    if a.still: STILL.update(on=True, gyro=0.05, win_ns=150e6)
    print(f"보드 검출: 왼쪽 {len(dL)} / 오른쪽 {len(dR)} 프레임 (2프레임마다)", flush=True)
    iL, sL = intrinsics("left", dL, size, a.f0_left or 790.0 / sc, (tl - t0) / 1e9, tl)
    iR, sR = intrinsics("right", dR, size, a.f0_right or 790.0 / sc, (tr - t0) / 1e9, tr)
    yi = yaml.safe_load((ROOT / "configs/calibration/trackcam_fisheye_v001.yaml").read_text()); K0 = np.asarray(yi["K"], float).reshape(3, 3) / sc
    sL["vs_trackcam_fisheye_v001"] = dict(fx=round(float(K0[0, 0]), 2), fy=round(float(K0[1, 1]), 2), cx=round(float(K0[0, 2]), 2), cy=round(float(K0[1, 2]), 2))
    # --- S3 pairs
    z = np.load(E / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]); wn = np.linalg.norm(z["gyro"], axis=1)
    pairs = []
    for k in dL:
        j = int(np.argmin(np.abs(tr - tl[k])))
        if j not in dR and j - 1 in dR: j -= 1
        if j not in dR and j + 1 in dR: j += 1
        if j not in dR or abs(tr[j] - tl[k]) > 20e6 or dL[k][2] < 8 or dR[j][2] < 8: continue
        m = np.abs(ti - tl[k]) < 50e6
        if not is_still(tl[k]): continue
        pairs.append((k, j))
    print(f"스테레오 쌍 {len(pairs)}개 (시간차 < 20 ms, 회전 < 0.3 rad/s)", flush=True)
    TL, TRr, keep = [], [], []
    for k, j in pairs:
        A, eA = pose_only(iL, *dL[k][:2], iL[0], size); B, eB = pose_only(iR, *dR[j][:2], iR[0], size)
        if np.median(eA) < 2 and np.median(eB) < 2: TL.append(A); TRr.append(B); keep.append((k, j))
    Trl = [b @ np.linalg.inv(a_) for a_, b in zip(TL, TRr)]; rv = np.array([R.from_matrix(T[:3, :3]).as_rotvec() for T in Trl]); tv = np.array([T[:3, 3] for T in Trl])
    r0 = R.from_rotvec(np.median(rv, 0)); t0v = np.median(tv, 0)
    dev_r = np.degrees([(R.from_rotvec(x) * r0.inv()).magnitude() for x in rv]); dev_t = np.linalg.norm(tv - t0v, axis=1)
    inl = (dev_r < 2.0) & (dev_t < 0.01); print(f"쌍별 외부 파라미터: 회전 편차 중앙 {np.median(dev_r):.2f}°, 이동 편차 중앙 {np.median(dev_t) * 1000:.1f} mm, 인라이어 {inl.sum()}/{len(inl)}", flush=True)
    sub = [i for i in np.flatnonzero(inl)][:150]
    def res(x):
        Tx = np.eye(4); Tx[:3, :3] = R.from_rotvec(x[:3]).as_matrix(); Tx[:3, 3] = x[3:]; out = []
        for i in sub:
            k, j = keep[i]; Pb = dL[k][0]; Pr = (Tx @ TL[i] @ np.c_[Pb, np.ones(len(Pb))].T).T[:, :3]
            # match right detections by object point (same tag ids give same corner coordinates)
            o, im = dR[j][0], dR[j][1]; idx = {tuple(np.round(p, 5)): q for p, q in zip(o, im)}
            for p, q in zip(Pb, Pr):
                if tuple(np.round(p, 5)) in idx: out.append(proj_kb4(q[None], iR)[0] - idx[tuple(np.round(p, 5))])
        return np.concatenate(out)
    x0 = np.r_[r0.as_rotvec(), t0v]; sol = least_squares(res, x0, loss="soft_l1", f_scale=1.0, max_nfev=200)
    e = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1); T_RL = np.eye(4); T_RL[:3, :3] = R.from_rotvec(sol.x[:3]).as_matrix(); T_RL[:3, 3] = sol.x[3:]
    base = float(np.linalg.norm(T_RL[:3, 3])); ang = float(np.degrees(np.linalg.norm(sol.x[:3])))
    print(f"T_CR_CL: 기선 {base * 100:.2f} cm, 이동 {np.round(T_RL[:3, 3] * 100, 2).tolist()} cm, 상대 회전 {ang:.2f}°, 오른쪽 재투영 {stats(e)}", flush=True)
    full = lambda p: dict(fx=float(p[0] * sc), fy=float(p[1] * sc), cx=float(p[2] * sc), cy=float(p[3] * sc), k=[float(v) for v in p[4:]])
    saved = lambda p: dict(fx=float(p[0]), fy=float(p[1]), cx=float(p[2]), cy=float(p[3]), k=[float(v) for v in p[4:]])
    S = dict(take=E.name, save_scale=sc, saved_size=list(size), detections=dict(left=len(dL), right=len(dR)), left=dict(intr_saved=saved(iL), **sL), right=dict(intr_saved=saved(iR), **sR),
             stereo=dict(pairs=len(pairs), pairs_used=len(sub), per_pair_rot_dev_deg_p50=round(float(np.median(dev_r)), 3), per_pair_trans_dev_mm_p50=round(float(np.median(dev_t) * 1000), 2),
                         T_CR_CL=T_RL.round(6).tolist(), baseline_cm=round(base * 100, 2), rel_rotation_deg=round(ang, 2), right_reproj=stats(e)))
    (E / "stereo_calib.json").write_text(json.dumps(S, indent=1))
    doc = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), source=str(E), model="kannala_brandt (KB4), own soft-L1 LM",
               image_size_full=[1920, 1080], left=dict(location=meta["left"]["location"], **full(iL)), right=dict(location=meta["right"]["location"], **full(iR)),
               T_CR_CL=T_RL.round(8).tolist(), T_CR_CL_note="maps a point in the LEFT camera frame into the RIGHT camera frame", baseline_cm=round(base * 100, 2),
               quality=dict(left_heldout=sL["heldout"], right_heldout=sR["heldout"], stereo_right_reproj=stats(e)))
    # --- joint stereo bundle: both KB4 intrinsics + T_CR_CL + one board pose per pair, soft-L1 (the per-camera fits above are only the init)
    J = [i for i in np.flatnonzero(inl)]; J = J[:: max(1, len(J) // 120)]
    def match(k, j):
        o, im = dR[j][0], dR[j][1]; idx = {tuple(np.round(p_, 5)): q for p_, q in zip(o, im)}
        oL, iLm = dL[k][0], dL[k][1]; m = [tuple(np.round(p_, 5)) in idx for p_ in oL]
        return oL, iLm, o, im
    data = [(dL[keep[i][0]][0], dL[keep[i][0]][1], dR[keep[i][1]][0], dR[keep[i][1]][1]) for i in J]
    x0 = np.r_[iL, iR, sol.x, np.concatenate([np.r_[R.from_matrix(TL[i][:3, :3]).as_rotvec(), TL[i][:3, 3]] for i in J])]
    def jres(x):
        pl, pr, ex = x[:8], x[8:16], x[16:22]; Tx = np.eye(4); Tx[:3, :3] = R.from_rotvec(ex[:3]).as_matrix(); Tx[:3, 3] = ex[3:]; out = []
        for n, (oL, imL, oR, imR) in enumerate(data):
            pv = x[22 + 6 * n: 28 + 6 * n]; Rb = R.from_rotvec(pv[:3]).as_matrix()
            PL = oL @ Rb.T + pv[3:]; out.append((proj_kb4(PL, pl) - imL).ravel())
            PR = (oR @ Rb.T + pv[3:]) @ Tx[:3, :3].T + Tx[:3, 3]; out.append((proj_kb4(PR, pr) - imR).ravel())
        return np.concatenate(out)
    js = least_squares(jres, x0, loss="soft_l1", f_scale=1.0, max_nfev=300, x_scale="jac", method="trf")
    iL, iR = js.x[:8], js.x[8:16]; T_RL = np.eye(4); T_RL[:3, :3] = R.from_rotvec(js.x[16:19]).as_matrix(); T_RL[:3, 3] = js.x[19:22]
    rr = js.fun.reshape(-1, 2); nl = sum(len(d[0]) for d in data); eLj = np.linalg.norm(rr[:0], axis=1)
    sizes = []; 
    for d in data: sizes += [len(d[0])] * 0
    allres = []; off = 0
    for d in data:
        allres.append(("L", np.linalg.norm(rr[off:off + len(d[0])], axis=1))); off += len(d[0]); allres.append(("R", np.linalg.norm(rr[off:off + len(d[2])], axis=1))); off += len(d[2])
    eLj = np.concatenate([e_ for s_, e_ in allres if s_ == "L"]); eRj = np.concatenate([e_ for s_, e_ in allres if s_ == "R"])
    base = float(np.linalg.norm(T_RL[:3, 3])); ang = float(np.degrees(np.linalg.norm(js.x[16:19])))
    print(f"[joint] pairs {len(data)}: left fx {iL[0]:.1f} cx {iL[2]:.1f} cy {iL[3]:.1f} | right fx {iR[0]:.1f} cx {iR[2]:.1f} cy {iR[3]:.1f}", flush=True)
    print(f"[joint] 기선 {base * 100:.2f} cm, 이동 {np.round(T_RL[:3, 3] * 100, 2).tolist()} cm, 상대 회전 {ang:.2f}°, 왼쪽 재투영 {stats(eLj)}, 오른쪽 {stats(eRj)}", flush=True)
    S["joint"] = dict(pairs=len(data), left=saved(iL), right=saved(iR), T_CR_CL=T_RL.round(6).tolist(), baseline_cm=round(base * 100, 2), rel_rotation_deg=round(ang, 2), left_reproj=stats(eLj), right_reproj=stats(eRj))
    (E / "stereo_calib.json").write_text(json.dumps(S, indent=1))
    doc.update(left=dict(location=meta["left"]["location"], **full(iL)), right=dict(location=meta["right"]["location"], **full(iR)), T_CR_CL=T_RL.round(8).tolist(),
               baseline_cm=round(base * 100, 2), quality=dict(joint_left_reproj=stats(eLj), joint_right_reproj=stats(eRj), pairs=len(data)))
    p = ROOT / a.out_yaml; p.write_text(yaml.safe_dump(doc, sort_keys=False)); print("저장", p)


if __name__ == "__main__":
    main()
