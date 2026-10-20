"""Stereo calibration from the FULL-RESOLUTION still frames of a stereo_imu_record --proto still take (2026-10-15).
cam_left_full/<t_ns>.png and cam_right_full/<t_ns>.png are saved only while the UMI is held still (>= 0.8 s after a "멈추세요" cue),
so L/R skew does not matter. Steps:
  1. AprilGrid detection at 1920x1080 (OpenCV aruco 36h11, subpix), >= 10 tags.
  2. Per camera and per model (KB4, Double Sphere): own soft-L1 LM on 4/5 of the views, the other 1/5 scored with intrinsics frozen
     (pose-only re-fit) -> median / p95 / edge (r > 0.7 half diagonal) / radial signed pattern. The model with the better held-out p95 wins.
  3. L/R pairs = nearest stamps < 60 ms (still); joint bundle: both intrinsics + T_CR_CL + one board pose per pair.
  4. Baseline vs the ruler value (--ruler-cm).
Writes <take>/stereo_calib_full.json and configs/calibration/stereo_trackcam_v004.yaml (FULL-resolution intrinsics).
    python scripts/stereo_calib_full.py datasets/stereo_imu/<ts>_still [--ruler-cm 3.5]"""
from __future__ import annotations
import argparse, datetime, json, sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from fisheye_calib_robust import MODELS, fit, residuals, init_pose, stats   # noqa: E402
from trackcam_board_lever import board_corners                              # noqa: E402
SIZE = (1920, 1080)


def detect_dir(d: Path):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), (lambda p: (setattr(p, "cornerRefinementMethod", cv2.aruco.CORNER_REFINE_SUBPIX), p)[-1])(cv2.aruco.DetectorParameters()))
    out = {}
    for f in sorted(d.glob("*.png")):
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE); c, ids, _ = det.detectMarkers(g)
        if ids is None: continue
        keep = [(cc.reshape(4, 2), int(i)) for cc, i in zip(c, ids.ravel()) if i < 36]
        if len(keep) >= 10: out[int(f.stem)] = (np.vstack([board_corners(i, 0.025) for _, i in keep]), np.vstack([cc for cc, _ in keep]).astype(np.float64), np.array([i for _, i in keep]))
    return out


def fit_sparse(model, obj, img, f0):
    """same objective as fisheye_calib_robust.fit (soft-L1, all views jointly) but with the Jacobian sparsity given:
    view k's residuals depend on the intrinsics + its own 6 pose parameters only -> finite differences in a few column groups."""
    c = np.array(SIZE, float) / 2; intr0 = np.array(MODELS[model][1](f0, c), float); n = len(intr0)
    x0 = np.r_[intr0, np.concatenate([init_pose(o, i, f0, c) for o, i in zip(obj, img)])]
    rows = sum(2 * len(o) for o in obj); J = lil_matrix((rows, len(x0)), dtype=int); r0 = 0
    for k, o in enumerate(obj):
        m = 2 * len(o); J[r0:r0 + m, :n] = 1; J[r0:r0 + m, n + 6 * k: n + 6 * k + 6] = 1; r0 += m
    sol = least_squares(residuals, x0, args=(model, obj, img, n), jac_sparsity=J, loss="soft_l1", f_scale=1.0, method="trf", max_nfev=300, x_scale="jac")
    return sol.x[:n], np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)


def pose_only(model, intr, o, i, f0):
    c = np.array(SIZE, float) / 2; x0 = init_pose(o, i, f0, c)
    s = least_squares(residuals, x0, args=(model, [o], [i], 0, intr), loss="soft_l1", f_scale=1.0, max_nfev=200)
    return s.x, s.fun.reshape(-1, 2)


def evaluate(model, intr, views, f0):
    E, Rr, S = [], [], []; cc = intr[2:4]; hd = np.hypot(*SIZE) / 2
    for o, i, _ in views:
        _, d = pose_only(model, intr, o, i, f0); v = i - cc; r = np.linalg.norm(v, axis=1)
        E.append(np.linalg.norm(d, axis=1)); Rr.append(r); S.append(np.sum(d * v / np.maximum(r[:, None], 1e-9), axis=1))
    e, r, s = map(np.concatenate, (E, Rr, S)); edge = r > 0.7 * hd; bins = np.linspace(0, r.max(), 7)
    pat = [round(float(s[(r >= lo) & (r < hi)].mean()), 2) if ((r >= lo) & (r < hi)).sum() > 30 else None for lo, hi in zip(bins[:-1], bins[1:])]
    return dict(n=int(len(e)), median_px=round(float(np.median(e)), 3), p95_px=round(float(np.percentile(e, 95)), 3), edge_n=int(edge.sum()),
                edge_median_px=round(float(np.median(e[edge])), 3) if edge.any() else None, radial_pattern=pat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--ruler-cm", type=float, default=None); ap.add_argument("--f0", type=float, default=790.0)
    ap.add_argument("--out-yaml", default="configs/calibration/stereo_trackcam_v004.yaml"); a = ap.parse_args()
    E = Path(a.take); meta = json.loads((E / "meta.json").read_text())
    dL, dR = detect_dir(E / "cam_left_full"), detect_dir(E / "cam_right_full"); print(f"정지 프레임 보드 검출: 왼쪽 {len(dL)} / 오른쪽 {len(dR)}", flush=True)
    best, rep = {}, {}
    for cam, D in (("left", dL), ("right", dR)):
        keys = sorted(D); keys = keys[:: max(1, len(keys) // 150)]; tr = [D[k] for j, k in enumerate(keys) if j % 5]; te = [D[k] for j, k in enumerate(keys) if j % 5 == 0]; rep[cam] = {}
        for model in ("kb4", "ds"):
            intr, e = fit_sparse(model, [v[0] for v in tr], [v[1] for v in tr], a.f0); h = evaluate(model, intr, te, a.f0)
            rep[cam][model] = dict(intr=[float(x) for x in intr], train=stats(e), heldout=h)
            print(f"[{cam} {model}] {np.round(intr, 4).tolist()}  train median {stats(e)['median_px']} p95 {stats(e)['p95_px']} | held-out {h}", flush=True)
        # KB4 unless DS is clearly (>5 %) better: OpenVINS (equidistant) and the trackers only take KB4 (10-16: DS won by 0.1 % and broke that)
        m = "ds" if rep[cam]["ds"]["heldout"]["p95_px"] < 0.95 * rep[cam]["kb4"]["heldout"]["p95_px"] else "kb4"
        best[cam] = (m, np.array(rep[cam][m]["intr"])); print(f"[{cam}] 선택: {m}", flush=True)
    # joint bundle over L/R still pairs
    tR = np.array(sorted(dR)); pairs = []
    for k in sorted(dL):
        j = tR[np.argmin(np.abs(tR - k))]
        if abs(j - k) < 60e6: pairs.append((k, int(j)))
    pairs = pairs[:: max(1, len(pairs) // 120)]
    mL, pL0 = best["left"]; mR, pR0 = best["right"]; nL, nR = len(pL0), len(pR0); projL, projR = MODELS[mL][0], MODELS[mR][0]
    TL, TRr = [], []
    for k, j in pairs:
        x, _ = pose_only(mL, pL0, *dL[k][:2], a.f0); y, _ = pose_only(mR, pR0, *dR[j][:2], a.f0)
        A = np.eye(4); A[:3, :3] = R.from_rotvec(x[:3]).as_matrix(); A[:3, 3] = x[3:]; B = np.eye(4); B[:3, :3] = R.from_rotvec(y[:3]).as_matrix(); B[:3, 3] = y[3:]; TL.append(A); TRr.append(B)
    Trl = [b @ np.linalg.inv(a_) for a_, b in zip(TL, TRr)]; rv = np.median([R.from_matrix(T[:3, :3]).as_rotvec() for T in Trl], 0); tv = np.median([T[:3, 3] for T in Trl], 0)
    dev_t = np.array([np.linalg.norm(T[:3, 3] - tv) for T in Trl]); ok = dev_t < 0.01; P = [p for p, g in zip(pairs, ok) if g]; T0 = [t for t, g in zip(TL, ok) if g]
    print(f"L/R 정지 쌍 {len(pairs)}개, 일관된 쌍 {len(P)}개 (쌍별 이동 편차 중앙 {np.median(dev_t) * 1000:.1f} mm)", flush=True)
    x0 = np.r_[pL0, pR0, rv, tv, np.concatenate([np.r_[R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]] for T in T0])]
    def jres(x):
        pl, pr, ex = x[:nL], x[nL:nL + nR], x[nL + nR:nL + nR + 6]; Rx = R.from_rotvec(ex[:3]).as_matrix(); out = []; o0 = nL + nR + 6
        for n, (k, j) in enumerate(P):
            pv = x[o0 + 6 * n: o0 + 6 * n + 6]; Rb = R.from_rotvec(pv[:3]).as_matrix()
            out.append((projL(dL[k][0] @ Rb.T + pv[3:], pl) - dL[k][1]).ravel())
            out.append((projR((dR[j][0] @ Rb.T + pv[3:]) @ Rx.T + ex[3:], pr) - dR[j][1]).ravel())
        return np.concatenate(out)
    nres = sum(2 * (len(dL[k][0]) + len(dR[j][0])) for k, j in P); Jp = lil_matrix((nres, len(x0)), dtype=int); r0 = 0; o0 = nL + nR + 6
    for n_, (k, j) in enumerate(P):
        mL_, mR_ = 2 * len(dL[k][0]), 2 * len(dR[j][0])
        Jp[r0:r0 + mL_, :nL] = 1; Jp[r0:r0 + mL_, o0 + 6 * n_: o0 + 6 * n_ + 6] = 1; r0 += mL_
        Jp[r0:r0 + mR_, nL:nL + nR + 6] = 1; Jp[r0:r0 + mR_, o0 + 6 * n_: o0 + 6 * n_ + 6] = 1; r0 += mR_
    js = least_squares(jres, x0, jac_sparsity=Jp, loss="soft_l1", f_scale=1.0, max_nfev=200, x_scale="jac", method="trf")
    pl, pr = js.x[:nL], js.x[nL:nL + nR]; ex = js.x[nL + nR:nL + nR + 6]; T = np.eye(4); T[:3, :3] = R.from_rotvec(ex[:3]).as_matrix(); T[:3, 3] = ex[3:]
    rr = np.linalg.norm(js.fun.reshape(-1, 2), axis=1); nLr = sum(len(dL[k][0]) for k, _ in P)
    # residuals are interleaved per pair (L then R) -> split
    eL, eR, off = [], [], 0
    for k, j in P: eL.append(rr[off:off + len(dL[k][0])]); off += len(dL[k][0]); eR.append(rr[off:off + len(dR[j][0])]); off += len(dR[j][0])
    eL, eR = np.concatenate(eL), np.concatenate(eR); base = float(np.linalg.norm(T[:3, 3]) * 100); ang = float(np.degrees(np.linalg.norm(ex[:3])))
    hL, hR = evaluate(mL, pl, [dL[k] for j_, k in enumerate(sorted(dL)) if j_ % 5 == 0], a.f0), evaluate(mR, pr, [dR[k] for j_, k in enumerate(sorted(dR)) if j_ % 5 == 0], a.f0)
    print(f"[joint] 기선 {base:.2f} cm (자 {a.ruler_cm} cm), 이동 {np.round(T[:3, 3] * 100, 2).tolist()} cm, 상대 회전 {ang:.2f}°", flush=True)
    print(f"[joint] 재투영 왼쪽 {stats(eL)} | 오른쪽 {stats(eR)}", flush=True)
    print(f"[joint] 공동 내부값으로 held-out: 왼쪽 {hL} | 오른쪽 {hR}", flush=True)
    S = dict(take=E.name, per_camera=rep, chosen=dict(left=mL, right=mR), joint=dict(pairs=len(P), left=[float(v) for v in pl], right=[float(v) for v in pr], T_CR_CL=T.round(6).tolist(),
             baseline_cm=round(base, 2), ruler_cm=a.ruler_cm, rel_rotation_deg=round(ang, 2), left_reproj=stats(eL), right_reproj=stats(eR), left_heldout=hL, right_heldout=hR))
    (E / "stereo_calib_full.json").write_text(json.dumps(S, indent=1))
    names = {"kb4": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"], "ds": ["fx", "fy", "cx", "cy", "xi", "alpha"]}
    doc = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), source=str(E), image_size=list(SIZE),
               left=dict(location=meta["left"]["location"], model=mL, **{n: float(v) for n, v in zip(names[mL], pl)}),
               right=dict(location=meta["right"]["location"], model=mR, **{n: float(v) for n, v in zip(names[mR], pr)}),
               T_CR_CL=T.round(8).tolist(), T_CR_CL_note="maps a point in the LEFT camera frame into the RIGHT camera frame", baseline_cm=round(base, 2), ruler_cm=a.ruler_cm,
               quality=dict(left_heldout=hL, right_heldout=hR, joint_left=stats(eL), joint_right=stats(eR), pairs=len(P)))
    p = ROOT / a.out_yaml; p.write_text(yaml.safe_dump(doc, sort_keys=False)); print("저장", p)


if __name__ == "__main__":
    main()
