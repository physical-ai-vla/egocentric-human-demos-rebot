"""Robot base <-> global Orbbec solve (2026-10-20, GPT plan step 5). Unknowns T_B_C (camera in the robot base) and T_E_tag (tag 220 on
the gripper, E = <side>_tcp of the URDF) from the robot_cam_poses holds + a tag_rig_record --proto robotcal take (same host monotonic clock).
Chain per hold i: T_C_tag(i) = T_C_B T_B_E(i) T_E_tag.
  1. per hold: tag corners in every Orbbec frame inside the hold window (0.2 s trimmed), median over frames (>= --min-frames)
  2. init: cv2.calibrateRobotWorldHandEye (A = T_tag_C, B = T_E_B -> X = T_C_B, Z = T_tag_E)
  3. refine: reprojection of the 4 tag corners of every hold, Huber 2 px, + the Orbbec DEPTH at the tag (median in the quad, --depth-sigma-mm;
     10-20 synthetic: at ~1 m a 48 mm tag's PnP depth is too noisy to start from) (optional --fit-size: tag size, the FK gives metric scale)
  4. held-out: --folds k-fold (default 5) -> predicted tag centre vs the PnP of the held-out holds (mm, deg) and reprojection px
    .venv/bin/python scripts/robot_cam_solve.py datasets/tag_rig/<take> datasets/robot_cam/<run> [--write]
    .venv/bin/python scripts/robot_cam_solve.py --synthetic                # self-test on simulated data (known answer)
--write -> configs/calibration/robot_cam_v001.json (T_B_C, T_E_tag, stats)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from tag_rig_pose import corners_obj, DepthStore, quad_depth_m      # noqa: E402


def T_of(rv, t):
    T = np.eye(4); T[:3, :3] = R.from_rotvec(rv).as_matrix(); T[:3, 3] = t; return T


def inv(T):
    Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return Ti


def project(T_C_tag, K, size):
    P = (T_C_tag[:3, :3] @ corners_obj(size).T).T + T_C_tag[:3, 3]; uv = (K @ P.T).T; return uv[:, :2] / uv[:, 2:3]


def pnp(px, K, size):
    ok, rv, tv = cv2.solvePnP(corners_obj(size), px.astype(np.float64), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return T_of(rv.ravel(), tv.ravel()) if ok else None


def collect(take: Path, run: Path, tags, min_frames: int, trim_s: float = 0.2):
    meta = json.loads((take / "meta.json").read_text()); I = meta["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]])
    t = np.load(take / "t_ns.npy"); poses = [json.loads(l) for l in (run / "poses.jsonl").read_text().splitlines() if l.strip()]
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm); obs = []; DS = DepthStore(take)
    for p in poses:
        ks = np.where((t >= p["t_hold0_ns"] + trim_s * 1e9) & (t <= p["t_hold1_ns"] - trim_s * 1e9))[0]
        if not len(ks): print(f"  포즈 {p['i']:2d}: 녹화 프레임 없음 (녹화기와 시간이 안 겹침?)"); continue
        per = {g: [] for g in tags}; dep = {g: [] for g in tags}
        for k in ks:
            cs, ids, _ = det.detectMarkers(cv2.imread(str(take / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE)); dm = None
            for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
                if int(i) in per:
                    per[int(i)].append(c.reshape(4, 2)); dm = DS.mm(k) if dm is None else dm; z = quad_depth_m(dm, c)
                    if z is not None: dep[int(i)].append(z)
        for g, L in per.items():
            if len(L) < min_frames: continue
            A = np.array(L); px = np.median(A, 0); jit = float(np.sqrt(((A - px) ** 2).sum(-1).mean()))
            obs.append(dict(pose=p["i"], tag=g, px=px, jitter_px=jit, n=len(L), T_B_E=np.array(p["T_B_E"]), depth_m=float(np.median(dep[g])) if len(dep[g]) >= 3 else None,
                            side_px=float(np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1)))))
        if not any(o["pose"] == p["i"] for o in obs): print(f"  포즈 {p['i']:2d}: 태그 {list(tags)} 안 보임 (프레임 {len(ks)})")
    return obs, K, meta


def pnp2(px, K, size):
    """both IPPE solutions (small tags flip: the 20 px synthetic case gave 300-400 mm hand-eye init errors)"""
    n, rvs, tvs, _ = cv2.solvePnPGeneric(corners_obj(size), px.astype(np.float64), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return [T_of(r.ravel(), t.ravel()) for r, t in zip(rvs, tvs)]


def p3(o, K, size):
    """tag centre in the camera: pixel ray x Orbbec depth when there is depth, else the PnP translation"""
    if o.get("depth_m"): return np.linalg.inv(K) @ np.r_[o["px"].mean(0), 1.0] * o["depth_m"]
    return pnp(o["px"], K, size)[:3, 3]


def kabsch(P, Q):                 # R, t with Q ~ R P + t
    mp, mq = P.mean(0), Q.mean(0); U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T; return Rm, mq - Rm @ mp


def init_positions(obs_m, K, size):
    """flip-free init (10-20): tag CENTRES only (PnP translation is fine even when the rotation flips).
    p_C(i) = R_CB (p_BE(i) + R_BE(i) t_Et) + t_CB : alternate Kabsch (R_CB, t_CB) and linear LS (t_Et, t_CB); then R_Et by inlier voting
    over both IPPE solutions of every hold."""
    pc = np.array([p3(o, K, size) for o in obs_m]); TBE = np.array([o["T_B_E"] for o in obs_m]); t_et = np.zeros(3)
    for _ in range(30):
        Rcb, tcb = kabsch(TBE[:, :3, 3] + np.einsum("nij,j->ni", TBE[:, :3, :3], t_et), pc)
        A = np.concatenate([np.concatenate([Rcb @ T[:3, :3], np.eye(3)], 1) for T in TBE]); b = np.concatenate([pc[i] - Rcb @ TBE[i, :3, 3] for i in range(len(TBE))])
        sol = np.linalg.lstsq(A, b, rcond=None)[0]; t_et = sol[:3]
    Rcb, tcb = kabsch(TBE[:, :3, 3] + np.einsum("nij,j->ni", TBE[:, :3, :3], t_et), pc)
    cand = [TBE[i, :3, :3].T @ Rcb.T @ T[:3, :3] for i, o in enumerate(obs_m) for T in pnp2(o["px"], K, size)]
    best = max(cand, key=lambda Rc: sum(np.degrees(R.from_matrix(Rc.T @ Rq).magnitude()) < 15 for Rq in cand))
    inl = [Rq for Rq in cand if np.degrees(R.from_matrix(best.T @ Rq).magnitude()) < 15]
    T_C_B = np.eye(4); T_C_B[:3, :3] = Rcb; T_C_B[:3, 3] = tcb; T_E_tag = np.eye(4); T_E_tag[:3, :3] = R.from_matrix(np.array(inl)).mean().as_matrix(); T_E_tag[:3, 3] = t_et
    return T_C_B, T_E_tag


def solve(obs, K, size, fit_size=False, depth_sigma_mm=6.0):
    tags = sorted({o["tag"] for o in obs}); main = max(tags, key=lambda g: sum(o["tag"] == g for o in obs))
    obs_m = [o for o in obs if o["tag"] == main]
    if len(obs_m) < 4: raise SystemExit(f"태그 {main} 가 보인 포즈가 {len(obs_m)}개 -- 최소 4개 (권장 25+)")
    inits = [init_positions(obs_m, K, size)]
    try:                                             # classic AX=ZB as a second start (fine with big tags, flips with small ones)
        A_R, A_t, B_R, B_t = [], [], [], []
        for o in obs_m:
            Ttc = inv(pnp(o["px"], K, size)); Teb = inv(o["T_B_E"]); A_R.append(Ttc[:3, :3]); A_t.append(Ttc[:3, 3]); B_R.append(Teb[:3, :3]); B_t.append(Teb[:3, 3])
        Rx, tx, Rz, tz = cv2.calibrateRobotWorldHandEye(A_R, A_t, B_R, B_t, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
        X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx.ravel(); Z = np.eye(4); Z[:3, :3] = Rz; Z[:3, 3] = tz.ravel(); inits.append((X, inv(Z)))
    except cv2.error: pass
    best = None
    for T_C_B0, Tm0 in inits:
        T_E_tag0 = {main: Tm0}
        for g in tags:                               # extra tags: average of T_E_B T_B_C T_C_tag over their holds
            if g == main: continue
            Ts = [inv(o["T_B_E"]) @ inv(T_C_B0) @ pnp(o["px"], K, size) for o in obs if o["tag"] == g]
            T = np.eye(4); T[:3, :3] = R.from_matrix(np.array([x[:3, :3] for x in Ts])).mean().as_matrix(); T[:3, 3] = np.mean([x[:3, 3] for x in Ts], 0); T_E_tag0[g] = T
        r_ = refine(obs, K, size, fit_size, tags, T_C_B0, T_E_tag0, depth_sigma_mm)
        if best is None or r_["cost"] < best["cost"]: best = r_
    return best


def refine(obs, K, size, fit_size, tags, T_C_B0, T_E_tag0, depth_sigma_mm=6.0):
    def unpack(x):
        T_C_B = T_of(x[:3], x[3:6]); Te = {g: T_of(x[6 + 6 * j:9 + 6 * j], x[9 + 6 * j:12 + 6 * j]) for j, g in enumerate(tags)}
        return T_C_B, Te, (size * np.exp(x[-1]) if fit_size else size)

    def res(x):
        T_C_B, Te, sz = unpack(x)
        out = []
        for o in obs:
            T = T_C_B @ o["T_B_E"] @ Te[o["tag"]]; out.append((project(T, K, sz) - o["px"]).ravel())
            if o.get("depth_m") and depth_sigma_mm > 0: out.append([(T[2, 3] - o["depth_m"]) * 1000 / depth_sigma_mm])   # 1 sigma ~ 1 px
        return np.concatenate(out)
    x0 = np.concatenate([R.from_matrix(T_C_B0[:3, :3]).as_rotvec(), T_C_B0[:3, 3]] + [np.r_[R.from_matrix(T_E_tag0[g][:3, :3]).as_rotvec(), T_E_tag0[g][:3, 3]] for g in tags] + ([[0.0]] if fit_size else []))
    sol = least_squares(res, x0, loss="huber", f_scale=2.0, x_scale="jac")
    T_C_B, Te, sz = unpack(sol.x)
    per = np.array([np.sqrt(((project(T_C_B @ o["T_B_E"] @ Te[o["tag"]], K, sz) - o["px"]) ** 2).sum(-1).mean()) for o in obs])
    dz = np.array([(T_C_B @ o["T_B_E"] @ Te[o["tag"]])[2, 3] - o["depth_m"] for o in obs if o.get("depth_m")]) * 1000
    return dict(T_C_B=T_C_B, T_B_C=inv(T_C_B), T_E_tag=Te, size=sz, rms_px=per, depth_res_mm=dz, init_T_C_B=T_C_B0, cost=float(sol.cost))


def heldout(obs, K, size, folds, fit_size, depth_sigma_mm=6.0):
    poses = sorted({o["pose"] for o in obs}); out = []
    for f in range(folds):
        test = set(poses[f::folds]); tr = [o for o in obs if o["pose"] not in test]
        if len({o["pose"] for o in tr}) < 3: continue
        S = solve(tr, K, size, fit_size, depth_sigma_mm)
        for o in obs:
            if o["pose"] not in test or o["tag"] not in S["T_E_tag"]: continue
            Tp = S["T_C_B"] @ o["T_B_E"] @ S["T_E_tag"][o["tag"]]; pm = p3(o, K, S["size"])
            Tm = min(pnp2(o["px"], K, S["size"]), key=lambda T: R.from_matrix(Tp[:3, :3].T @ T[:3, :3]).magnitude())   # the IPPE twin closest
            px = float(np.sqrt(((project(Tp, K, S["size"]) - o["px"]) ** 2).sum(-1).mean()))
            dp = Tp[:3, 3] - pm; ray = pm / np.linalg.norm(pm)
            out.append(dict(pose=o["pose"], px=px, mm=float(np.linalg.norm(dp) * 1e3), lateral_mm=float(np.linalg.norm(dp - ray * (dp @ ray)) * 1e3),
                            deg=float(np.degrees(R.from_matrix(Tp[:3, :3].T @ Tm[:3, :3]).magnitude()))))
    return out


def report(S, obs, H):
    T = S["T_B_C"]; z_cam = T[:3, 2]
    print(f"\n카메라 위치 (로봇 베이스 기준) {np.round(T[:3, 3], 3)} m, 거리 {np.linalg.norm(T[:3, 3]):.2f} m, 시선 방향 {np.round(z_cam, 2)}")
    for g, Te in S["T_E_tag"].items():
        print(f"태그 {g} (그리퍼 TCP 기준) 위치 {np.round(Te[:3, 3] * 100, 1)} cm, 회전 rotvec {np.round(np.degrees(R.from_matrix(Te[:3, :3]).as_rotvec()), 1)}°")
    print(f"태그 크기 {S['size'] * 1000:.1f} mm | 재투영 오차 중앙 {np.median(S['rms_px']):.2f}px, p95 {np.percentile(S['rms_px'], 95):.2f}px (관측 {len(obs)}개)" +
          (f" | depth 잔차 중앙 {np.median(np.abs(S['depth_res_mm'])):.1f} mm (depth 있는 관측 {len(S['depth_res_mm'])})" if len(S["depth_res_mm"]) else " | depth 없음"))
    worst = np.argsort(-S["rms_px"])[:3]
    print("  가장 나쁜 포즈:", ", ".join(f"{obs[w]['pose']}({S['rms_px'][w]:.1f}px, 태그 {obs[w]['side_px']:.0f}px)" for w in worst))
    if H:
        a = lambda k: np.array([h[k] for h in H])
        print(f"검증 (k-fold, 학습에 안 쓴 포즈 {len(H)}개): 재투영 중앙 {np.median(a('px')):.2f}px p95 {np.percentile(a('px'), 95):.2f}px | "
              f"태그 위치 오차 중앙 {np.median(a('mm')):.1f} mm (시선 수직 {np.median(a('lateral_mm')):.1f} mm, p95 {np.percentile(a('lateral_mm'), 95):.1f}) | 회전 중앙 {np.median(a('deg')):.2f}°")
        print("  (위치 오차 = 예측 vs 측정(depth 광선, 없으면 PnP); 시선 방향은 depth 잡음 포함이라 시선 수직 mm 과 px 이 판정 기준)")
        ok = np.median(a("px")) < 3 and np.percentile(a("lateral_mm"), 95) < 10
        print("판정:", "PASS" if ok else "확인 필요", "(기준: 검증 재투영 중앙 < 3px, 시선 수직 p95 < 10 mm)")


def synthetic(fit_size, no_depth=False):
    rng = np.random.default_rng(1); K = np.array([[690.0, 0, 640], [0, 690, 360], [0, 0, 1]]); size = 0.048
    eye, tgt = np.array([0.45, 0.6, 0.75]), np.array([0.45, -0.3, 0.25])                 # camera ~1.3 m away, looking at the workspace
    z = tgt - eye; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    T_B_C = np.eye(4); T_B_C[:3, :3] = np.c_[x, y, z]; T_B_C[:3, 3] = eye
    T_E_tag = T_of(np.radians([5, -80, 3]), [0.0, 0.03, -0.06]); obs = []
    for i in range(32):
        T_B_E = np.eye(4); T_B_E[:3, 3] = tgt + rng.uniform(-1, 1, 3) * [0.08, 0.08, 0.06]
        for _ in range(100):                          # rotation excitation, keep the tag facing the camera
            Rr = R.from_rotvec(rng.normal(size=3) * np.radians(15)).as_matrix(); T_B_E[:3, :3] = Rr @ T_B_C[:3, :3] @ R.from_euler("x", 180, degrees=True).as_matrix() @ inv(T_E_tag)[:3, :3]
            T_C_tag = inv(T_B_C) @ T_B_E @ T_E_tag
            if T_C_tag[2, 2] < -0.5: break
        px = project(T_C_tag, K, size) + rng.normal(0, 0.4, (4, 2))
        T_B_E_meas = T_B_E.copy(); T_B_E_meas[:3, 3] += rng.normal(0, 0.0007, 3)          # FK / joint noise ~1 mm
        obs.append(dict(pose=i, tag=220, px=px, jitter_px=0.4, n=20, T_B_E=T_B_E_meas, depth_m=None if no_depth else T_C_tag[2, 3] + rng.normal(0, 0.004), side_px=float(np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1)))))
    S = solve(obs, K, size * (1.04 if fit_size else 1.0), fit_size); H = heldout(obs, K, size, 5, fit_size); report(S, obs, H)
    e_t = np.linalg.norm(S["T_B_C"][:3, 3] - T_B_C[:3, 3]) * 1e3; e_r = np.degrees(R.from_matrix(S["T_B_C"][:3, :3].T @ T_B_C[:3, :3]).magnitude())
    e_g = np.linalg.norm(S["T_E_tag"][220][:3, 3] - T_E_tag[:3, 3]) * 1e3
    print(f"\n[합성 시험] 정답 대비 T_B_C {e_t:.1f} mm / {e_r:.2f}°, T_E_tag {e_g:.1f} mm, 태그 크기 {S['size'] * 1000:.2f} mm (정답 48.00)")
    ok = e_t < 10 and e_r < 0.5 and e_g < 5; print("합성 시험", "PASS" if ok else "FAIL"); return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?"); ap.add_argument("run", nargs="?"); ap.add_argument("--tags", type=int, nargs="+", default=[220])
    ap.add_argument("--tag-size", type=float, default=0.050); ap.add_argument("--fit-size", action="store_true"); ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--no-depth", action="store_true"); ap.add_argument("--depth-sigma-mm", type=float, default=6.0); ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "configs/calibration/robot_cam_v001.json")); a = ap.parse_args()
    if a.synthetic: sys.exit(0 if synthetic(a.fit_size, a.no_depth) else 1)
    take, run = Path(a.take), Path(a.run); obs, K, meta = collect(take, run, a.tags, a.min_frames)
    print(f"관측: 포즈 {len({o['pose'] for o in obs})}개 (depth 있음 {sum(bool(o['depth_m']) for o in obs)}), 태그별 {[(g, sum(o['tag'] == g for o in obs)) for g in a.tags]}, 태그 크기 중앙 {np.median([o['side_px'] for o in obs]):.0f}px, "
          f"정지 중 코너 흔들림 중앙 {np.median([o['jitter_px'] for o in obs]):.2f}px")
    if a.no_depth:
        for o in obs: o["depth_m"] = None
    S = solve(obs, K, a.tag_size, a.fit_size, a.depth_sigma_mm); H = heldout(obs, K, a.tag_size, a.folds, a.fit_size, a.depth_sigma_mm); report(S, obs, H)
    if a.write:
        run_meta = json.loads((run / "plan.json").read_text())
        Path(a.out).write_text(json.dumps(dict(T_B_C=S["T_B_C"].tolist(), T_C_B=S["T_C_B"].tolist(), T_E_tag={str(g): T.tolist() for g, T in S["T_E_tag"].items()},
                                               ee_frame=f"{run_meta['side']}_tcp", side=run_meta["side"], tag_size_m=S["size"], intrinsics=meta["intrinsics"],
                                               rms_px_median=float(np.median(S["rms_px"])), heldout=H, take=str(take), run=str(run)), indent=1, ensure_ascii=False))
        print("기록", a.out)


if __name__ == "__main__":
    main()
