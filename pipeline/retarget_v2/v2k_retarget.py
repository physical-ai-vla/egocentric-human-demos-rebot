#!/usr/bin/env python3
"""[2026-09-28] C-old v1 vs v2-K retarget on the SAME human segments (only the retargeting differs).

Segments = exactly those of ~/ego_collector/scripts/robotlike_offline_check.py (== frozen c8_phase3 chain: imu_vi scale +
guard, camera_tcp_v2, |dt| < 20 ms, 65-frame hard-break segments). For each segment and each arm the human motion is
reduced to its RELATIVE trajectory  rel(t) = C^-1 F^-1 inv(T_hu(0)) T_hu(t) F C  (Retarget.targets), then attached to a
robot anchor: T_tgt(t) = FK(q0) rel(t)  -- "robot posture + human relative task motion".

  v1   q0 = r150_active_median (the single anchor C-old v1 used)
  v2-K q0 in the frozen kinematic seed bank (v2k/seedbank_q12.npy, sha in seedbank.json):
       1. prescreen every seed: both arms' targets >= 95 % inside the R150 workspace (c8_phase3.ws_in definition)
       2. rank the survivors by mean workspace NN position distance (margin), roll out continuity IK from the top K
       3. hard gates on each rollout: every row IK status in SUCCESS (residual 15 mm / 5 deg, limits, dq), clearance
          d_min >= 0.075 m on every 5th row
       4. score = mean pos_err/15 + mean rot_err/5 + mean|dq|/0.6 + 0.5 mean|ddq|/0.6 (lower is better); best = top-1,
          variants = further survivors >= VARIANT_RAD (mean 12-D joint trajectory distance) from every kept one, max 3
       One anchor per segment; the seed never changes inside a segment.
Nothing here was tuned on v2 results (weights, K, radii fixed before the first run).

usage: v2k_retarget.py --episodes-json <json "episodes": [C8 ids]> --raw-root <dataset dir with <prefix>_<sess>/episode_N>
                       --runs <dir> --export <dir> --out <dir> [--prefix Hpilot]
"""
import argparse, importlib.util, json, pathlib, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree

H = pathlib.Path.home()
spec = importlib.util.spec_from_file_location("rlc", H / "ego_collector/scripts/robotlike_offline_check.py"); RLC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RLC)
P3 = RLC.P3; P = None
K_ROLL, VARIANT_RAD, MAX_VARIANTS, WS_MIN, D_HARD = 8, 0.5, 3, 0.95, 0.075
BANK = np.load(pathlib.Path(__file__).resolve().parent / "seedbank_q12.npy")
MODE, LAMBDA, K_MANI = "v2k", 0.0, 8        # --mode v2kr: v2kr_frozen.json (lambda 0.5, +8 manifold-nearest seeds)
Q_TREE, BANK_D = None, None


def ws_batch(T, a):
    """T (..., 4, 4) -> inside mask (...,) and NN position distance (...,), == c8_phase3.ws_in, vectorised."""
    sh = T.shape[:-2]; Tf = T.reshape(-1, 4, 4)
    dp, ii = P3.TREE[a].query(Tf[:, :3, 3], k=50)
    tr = np.einsum("nkij,nij->nk", P3.WR[a][ii], Tf[:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(1)
    return ((dp[:, 0] < 0.05) & (ang < 20)).reshape(sh), dp[:, 0].reshape(sh)


def rel_traj(T_hu_seg):
    rel = np.linalg.inv(T_hu_seg[0])[None] @ T_hu_seg
    return np.linalg.inv(P.C)[None] @ (np.linalg.inv(P.F)[None] @ rel @ P.F[None]) @ P.C[None]


def fk_T(q12):
    return [P._h(Rot.from_quat(p[3:]).as_matrix(), p[:3]) for p in P3.ik.kin.fk_pose7(q12)]


def rollout(q0, relL, relR, clr):
    T0 = fk_T(q0); tgL, tgR = T0[0][None] @ relL, T0[1][None] @ relR
    res = P3.ik.solve(tgL, tgR, q12_start=q0); q = np.where(np.isfinite(res["q"]), res["q"], res["q_last_good"])
    ik_ok = bool(np.isin(res["status"], P.SUCCESS).all())
    dmin = float(min(clr.dmin(q[i])[0] for i in range(0, len(q), 5)))
    dq = np.abs(np.diff(q, axis=0)); ddq = np.abs(np.diff(q, 2, axis=0))
    pe, re_ = res["pos_err"], res["rot_err"]
    score = float(np.nanmean(pe) / 15 + np.nanmean(re_) / 5 + dq.mean() / 0.6 + 0.5 * ddq.mean() / 0.6)
    d_r150 = float(np.median(Q_TREE.query(q[::3])[0])) if Q_TREE is not None else None
    if MODE == "v2kr": score += LAMBDA * d_r150                                    # (r30 mode keeps the raw score; see do_segment)
    wsL, _ = ws_batch(tgL, 0); wsR, _ = ws_batch(tgR, 1)
    return dict(q=q.astype(np.float32), ik_ok=ik_ok, dmin=dmin, collision_ok=dmin >= D_HARD, score=score, d_r150=d_r150,
                ws=[float(wsL.mean()), float(wsR.mean())], ws_ok=bool(wsL.mean() >= WS_MIN and wsR.mean() >= WS_MIN),
                pos_err_p50=float(np.nanpercentile(pe, 50)), pos_err_p90=float(np.nanpercentile(pe, 90)),
                rot_err_p50=float(np.nanpercentile(re_, 50)), rot_err_p90=float(np.nanpercentile(re_, 90)),
                status={k: int(v) for k, v in zip(*np.unique(res["status"], return_counts=True))})


# ---------------------------------------------------------------- v2-TR-R30 (v2tr_frozen.json)
WIN = dict(q=None, feat=None, tree=None)
TR_T = np.arange(8, 65, 8)                                   # feature sample times inside the 65-frame window
TR_ROT_M = 0.1                                               # m per rad


def _feat(relL, relR):
    f = []
    for rel in (relL, relR):
        f.append(rel[TR_T, :3, 3].ravel()); f.append(Rot.from_matrix(rel[TR_T, :3, :3]).as_rotvec().ravel() * TR_ROT_M)
    return np.concatenate(f)


def build_windows(d150, m):
    """every 65-frame window (stride 5) inside an R30 episode: q_ref and the FK-frame relative TCP motion feature"""
    Qs, F = [], []
    ep = d150["ep"][m]; J = np.radians(np.c_[d150["J"][m, :6], d150["J"][m, 7:13]])
    for e in np.unique(ep):
        Je = J[ep == e]
        T = np.array([np.stack(fk_T(q)) for q in Je])                                  # (n, 2, 4, 4)
        for s0 in range(0, len(Je) - 65 + 1, 5):
            w = T[s0:s0 + 65]; rels = [np.linalg.inv(w[0, a])[None] @ w[:, a] for a in (0, 1)]
            Qs.append(Je[s0:s0 + 65]); F.append(_feat(*rels))
    WIN.update(q=np.array(Qs, np.float64), feat=np.array(F), tree=cKDTree(np.array(F)))
    print(f"TR window library: {len(Qs)} R30 windows", flush=True)


def solve_ref(TL, TR, q_ref):
    """ContinuityIK.solve with ONE change: the next-row seed = last good q + reference joint step q_ref(t+1) - q_ref(t)."""
    ik = P3.ik; c = ik.cfg; n = len(TL)
    q = np.zeros((n, 12)); q_last_good = np.zeros((n, 12))
    pos_err = np.full((n, 2), np.nan); rot_err = np.full((n, 2), np.nan); status = np.zeros((n, 2), "<U14"); status[:] = "OK"
    good = np.array(q_ref[0], float); seed = ik.kin._q(ik.kin._svc(good)); tgt = (TL, TR)
    for t in range(n):
        lp, lq = TL[t, :3, 3], Rot.from_matrix(TL[t, :3, :3]).as_quat(); rp, rq = TR[t, :3, 3], Rot.from_matrix(TR[t, :3, :3]).as_quat()
        qf = seed.copy(); q_last_good[t] = good
        try:
            for it in range(c.iters):
                qn = np.asarray(ik.kin.solver.ik(qf.astype(np.float32), left_pose=(lp.astype(np.float32), lq.astype(np.float32)),
                                                 right_pose=(rp.astype(np.float32), rq.astype(np.float32))), np.float64)
                step = np.abs(qn - qf).max(); qf = qn
                if step < c.conv_dq: break
        except Exception:
            status[t] = "IK_FAIL"; q[t] = np.nan; continue
        q12 = ik._q12(qf); q[t] = q12
        l7, r7 = ik.kin.solver.fk_pose7(qf.astype(np.float32))
        for a, (p7, Tt) in enumerate(((l7, TL[t]), (r7, TR[t]))):
            p7 = np.asarray(p7, float); pos_err[t, a] = np.linalg.norm(p7[:3] - Tt[:3, 3]) * 1e3
            rot_err[t, a] = np.degrees(np.linalg.norm(Rot.from_matrix(Rot.from_quat(p7[3:]).as_matrix().T @ Tt[:3, :3]).as_rotvec()))
        tol = np.radians(P.LIMIT_TOL_DEG)
        for a, sl in ((0, slice(0, 6)), (1, slice(6, 12))):
            d = float(np.abs(q12[sl] - good[sl]).max())                                   # frozen C4 rule, unchanged
            Tp = tgt[a][t - 1] if t else tgt[a][t]
            dp_t = np.linalg.norm(tgt[a][t, :3, 3] - Tp[:3, 3]) * 1e3
            dth_t = np.degrees(np.linalg.norm(Rot.from_matrix(Tp[:3, :3].T @ tgt[a][t, :3, :3]).as_rotvec()))
            if (q12[sl] < ik.lo[sl] - tol).any() or (q12[sl] > ik.hi[sl] + tol).any(): status[t, a] = "LIMIT"
            elif d > c.dq_fail: status[t, a] = "DQ_FAIL"
            elif d > c.branch_dq and dp_t < c.branch_dp_mm and dth_t < c.branch_dth_deg: status[t, a] = "BRANCH_SWITCH"
            elif pos_err[t, a] > c.pos_ok_mm or rot_err[t, a] > c.rot_ok_deg: status[t, a] = "NOT_CONVERGED"
            elif d > c.dq_warn: status[t, a] = "DQ_WARN"
            if status[t, a] in P.SUCCESS: good[sl] = q12[sl]
        nxt = good + (q_ref[min(t + 1, n - 1)] - q_ref[t])
        seed = ik.kin._q(ik.kin._svc(nxt))
    return dict(q=q, q_last_good=q_last_good, pos_err=pos_err, rot_err=rot_err, status=status)


def rollout_tr(q_ref, relL, relR, clr):
    T0 = fk_T(q_ref[0]); tgL, tgR = T0[0][None] @ relL, T0[1][None] @ relR
    res = solve_ref(tgL, tgR, q_ref); q = np.where(np.isfinite(res["q"]), res["q"], res["q_last_good"])
    ik_ok = bool(np.isin(res["status"], P.SUCCESS).all()); dmin = float(min(clr.dmin(q[i])[0] for i in range(0, len(q), 5)))
    dq = np.abs(np.diff(q, axis=0)); ddq = np.abs(np.diff(q, 2, axis=0)); pe, re_ = res["pos_err"], res["rot_err"]
    d_ref = float(np.median(Q_TREE.query(q[::3])[0]))
    score = float(np.nanmean(pe) / 15 + np.nanmean(re_) / 5 + dq.mean() / 0.6 + 0.5 * ddq.mean() / 0.6) + LAMBDA * d_ref
    wsL, _ = ws_batch(tgL, 0); wsR, _ = ws_batch(tgR, 1)
    return dict(q=q.astype(np.float32), ik_ok=ik_ok, dmin=dmin, collision_ok=dmin >= D_HARD, score=score, d_r150=d_ref,
                adherence_rad=float(np.linalg.norm(q - q_ref, axis=1).mean()),
                ws=[float(wsL.mean()), float(wsR.mean())], ws_ok=bool(wsL.mean() >= WS_MIN and wsR.mean() >= WS_MIN),
                pos_err_p50=float(np.nanpercentile(pe, 50)), pos_err_p90=float(np.nanpercentile(pe, 90)),
                rot_err_p50=float(np.nanpercentile(re_, 50)), rot_err_p90=float(np.nanpercentile(re_, 90)),
                status={k: int(v) for k, v in zip(*np.unique(res["status"], return_counts=True))})


def do_segment_tr(TL, TR, clr):
    relL, relR = rel_traj(TL), rel_traj(TR)
    v1 = rollout(P.ANCHORS["r150_active_median"], relL, relR, clr)
    _, nn = WIN["tree"].query(_feat(relL, relR), k=64); cand = []
    for w in np.atleast_1d(nn):
        T0 = fk_T(WIN["q"][w][0]); inL, _ = ws_batch(T0[0][None] @ relL, 0); inR, _ = ws_batch(T0[1][None] @ relR, 1)
        if inL.mean() >= WS_MIN and inR.mean() >= WS_MIN: cand.append(int(w))
        if len(cand) >= K_ROLL: break
    rolls = [(w, rollout_tr(WIN["q"][w], relL, relR, clr)) for w in cand]
    surv = sorted([(r["score"], w, r) for w, r in rolls if r["ik_ok"] and r["collision_ok"] and r["ws_ok"]], key=lambda x: x[0])
    kept = []
    for x in surv:
        if len(kept) >= MAX_VARIANTS: break
        if not kept or all(np.linalg.norm(x[2]["q"] - y[2]["q"], axis=1).mean() >= VARIANT_RAD for y in kept): kept.append(x)
    return v1, dict(n_ws_candidates=len(cand), n_rolled=len(rolls), n_feasible=len(surv), kept=kept)


def segments(ep):
    """(TL_hu, TR_hu) per segment, exactly as robotlike_offline_check.check_episode builds them."""
    tags = {s: RLC.tag_of(ep, s) for s in ("left", "right")}
    if any(not (P3.RUNS / t / "ss1.csv").is_file() for t in tags.values()): return None, "mast3r_missing"
    S = {s: RLC.side_nogrip(tags[s], ep, s) for s in ("left", "right")}; L, R = S["left"], S["right"]
    fm = P3.frame_meta(ep)
    capL = np.array([c for _, c in fm["left_wrist"][L["off"]:L["off"] + L["n"]]], float)
    capR = np.array([c for _, c in fm["right_wrist"][R["off"]:R["off"] + R["n"]]], float); capH = np.array([c for _, c in fm["head"]], float)
    jR, jH = P3.nearest(capR, capL), P3.nearest(capH, capL)
    row_ok = L["ok"] & R["ok"][jR] & (np.abs(capR[jR] - capL) / 1e6 < P3.DT_MS) & (np.abs(capH[jH] - capL) / 1e6 < P3.DT_MS)
    idx = np.flatnonzero(row_ok); runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1) if len(idx) else []
    segs = [r[k:k + P3.SEG] for r in runs for k in range(0, len(r) - P3.SEG + 1, P3.SEG)]
    if not (L["scale_valid"] and R["scale_valid"]): return [(int(s_[0]), None, None) for s_ in segs], "B_metric"
    return [(int(s_[0]), L["T_tcp"][s_], R["T_tcp"][jR[s_]]) for s_ in segs], None


def do_segment(TL, TR, clr):
    relL, relR = rel_traj(TL), rel_traj(TR)
    v1 = rollout(P.ANCHORS["r150_active_median"], relL, relR, clr)
    # v2-K prescreen: all seeds at once
    T0 = BANK_T0                                                                   # (N, 2, 4, 4), computed once
    tL = T0[:, 0][:, None] @ relL[None]; tR = T0[:, 1][:, None] @ relR[None]      # (N, 65, 4, 4)
    inL, dL = ws_batch(tL, 0); inR, dR = ws_batch(tR, 1)
    fracL, fracR = inL.mean(1), inR.mean(1); cand = np.flatnonzero((fracL >= WS_MIN) & (fracR >= WS_MIN))
    order = cand[np.argsort((dL[cand].mean(1) + dR[cand].mean(1)))][:K_ROLL]
    if MODE in ("v2kr", "r30"):                                                    # + seeds whose own posture is nearest the reference
        extra = cand[np.argsort(BANK_D[cand])][:K_MANI]; order = np.array(list(dict.fromkeys(order.tolist() + extra.tolist())))
    rolls = [(int(i), rollout(BANK[i], relL, relR, clr)) for i in order]
    ok_ = lambda r: r["ik_ok"] and r["collision_ok"] and r["ws_ok"]
    if MODE == "r30":                                                               # v2-K-R30 (margin top-8, raw S) and v2-Kr-R30 (union, S + lambda d)
        k_set = set(cand[np.argsort((dL[cand].mean(1) + dR[cand].mean(1)))][:K_ROLL].tolist())      # exactly v2-K's rollout set
        sK = sorted([(r["score"], i, r) for i, r in rolls if ok_(r) and i in k_set], key=lambda x: x[0])
        sR = sorted([(r["score"] + LAMBDA * r["d_r150"], i, r) for i, r in rolls if ok_(r)], key=lambda x: x[0])
        def pick(sv):
            kept = []
            for x in sv:
                if len(kept) >= MAX_VARIANTS: break
                if not kept or all(np.linalg.norm(x[2]["q"] - y[2]["q"], axis=1).mean() >= VARIANT_RAD for y in kept): kept.append(x)
            return kept
        base = dict(n_ws_candidates=int(len(cand)), n_rolled=len(rolls))
        return v1, dict(base, n_feasible=len(sK), kept=pick(sK)), dict(base, n_feasible=len(sR), kept=pick(sR))
    surv = sorted([(r["score"], i, r) for i, r in rolls if ok_(r)], key=lambda x: x[0])
    kept = []
    for sc, i, r in surv:
        if len(kept) >= MAX_VARIANTS: break
        if all(np.linalg.norm(r["q"] - k[2]["q"], axis=1).mean() >= VARIANT_RAD for k in kept) or not kept: kept.append((sc, i, r))
    return v1, dict(n_ws_candidates=int(len(cand)), n_rolled=len(rolls), n_feasible=len(surv), kept=kept,
                    best_any=min(rolls, key=lambda x: x[1]["score"])[1] if rolls else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes-json", required=True); ap.add_argument("--raw-root", required=True)
    ap.add_argument("--prefix", default="Hpilot"); ap.add_argument("--runs", required=True); ap.add_argument("--export", required=True, action="append", help="repeatable; first root holding <tag>/imu_data.json wins")
    ap.add_argument("--out", required=True); ap.add_argument("--part", default="0/1")
    ap.add_argument("--mode", choices=["v2k", "v2kr", "r30", "tr"], default="v2k"); a = ap.parse_args()
    global P, MODE, LAMBDA, Q_TREE, BANK_D, BANK
    P3.RUNS = pathlib.Path(a.runs); roots = [pathlib.Path(x) for x in a.export]
    P3.export_dir = lambda tag: next((r / tag for r in roots if (r / tag / "imu_data.json").exists()), roots[0] / tag)
    global BANK_T0
    t0 = time.time(); P3.init(); P = P3.P; t1 = time.time(); clr = RLC.Clearance(P3.ik.kin)
    BANK_T0 = np.array([np.stack(fk_T(q)) for q in BANK]); print(f"init {t1 - t0:.1f}s, bank FK {time.time() - t1:.1f}s", flush=True)
    from scipy.spatial import cKDTree
    d150 = np.load(H / "umi_bridge/track_c/data/r150_q_tcp.npz"); Q_TREE = cKDTree(np.radians(np.c_[d150["J"][:, :6], d150["J"][:, 7:13]])[::3])
    BANK_D = Q_TREE.query(BANK)[0]
    MODE = a.mode
    if MODE in ("r30", "tr"):
        # 2026-09-28 R30 contract: every real-derived quantity from the frozen R30 calibration episodes ONLY
        v2d = pathlib.Path(__file__).resolve().parent; r30 = json.load(open(v2d / "r30_episodes.json"))["episodes"]
        m = np.isin(d150["ep"], r30); J30 = np.radians(np.c_[d150["J"][m, :6], d150["J"][m, 7:13]])
        BANK = np.load(v2d / "seedbank_R30_q12.npy"); sb = json.load(open(v2d / "seedbank_R30.json"))
        assert sb["sha256_q12"] == __import__("hashlib").sha256(BANK.tobytes()).hexdigest() and sb["reference"] == "r30"
        # workspace gate rebuilt exactly like c8_phase3.init() but on R30 frames (6000 random, rng 0)
        ix = np.random.default_rng(0).choice(len(J30), 6000, replace=False); ref = {0: [], 1: []}
        for i in ix:
            for a_, p7 in enumerate(P3.ik.kin.fk_pose7(J30[i])): ref[a_].append(p7)
        ref = {a_: np.array(v) for a_, v in ref.items()}
        P3.TREE = {a_: cKDTree(ref[a_][:, :3]) for a_ in (0, 1)}; P3.WR = {a_: Rot.from_quat(ref[a_][:, 3:]).as_matrix() for a_ in (0, 1)}
        Q_TREE = cKDTree(J30); BANK_D = Q_TREE.query(BANK)[0]
        BANK_T0 = np.array([np.stack(fk_T(q)) for q in BANK])
        fz = json.load(open(v2d / "v2kr_frozen.json")); LAMBDA = float(fz["lambda"])
        print(f"R30 contract: {len(r30)} episodes / {len(J30)} frames, bank {len(BANK)} (sha {sb['sha256_q12'][:12]}), lambda {LAMBDA}", flush=True)
        if MODE == "tr": build_windows(d150, m)
    if MODE == "v2kr":
        fz = json.load(open(pathlib.Path(__file__).resolve().parent / "v2kr_frozen.json")); LAMBDA = float(fz["lambda"])
        assert fz["seed_bank"]["sha256_q12"] == __import__("hashlib").sha256(BANK.tobytes()).hexdigest(), "seed bank changed"
        print(f"v2kr frozen: lambda {LAMBDA}, K {K_ROLL}+{K_MANI}", flush=True)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    k, n = map(int, a.part.split("/")); eps = json.load(open(a.episodes_json))["episodes"][k::n]
    for e in eps:
        dst = out / f"{e}.json"
        if dst.exists(): continue
        sess, num = e.rsplit("_", 1); ep = pathlib.Path(a.raw_root) / f"{a.prefix}_{sess}" / f"episode_{num}"; t0 = time.time()
        try:
            segs, why = segments(ep)
            rec = dict(episode=e, first_fail=why if segs is None else None, segments=[]); qs = {}
            for start, TL, TR in (segs or []):
                if TL is None: rec["segments"].append(dict(start=start, first_fail="B_metric")); continue
                res_ = do_segment_tr(TL, TR, clr) if MODE == "tr" else do_segment(TL, TR, clr)
                v1, v2 = res_[0], res_[1]; v2r = res_[2] if len(res_) > 2 else None
                strip = lambda r: {k_: v for k_, v in r.items() if k_ != "q"}
                pack = lambda v: dict(n_ws_candidates=v["n_ws_candidates"], n_rolled=v["n_rolled"], n_feasible=v["n_feasible"], pass_=bool(v["kept"]),
                                      kept=[dict(seed=i, sel_score=sc, **strip(r)) for sc, i, r in v["kept"]],
                                      variant_dist_rad=[float(np.linalg.norm(v["kept"][j][2]["q"] - v["kept"][0][2]["q"], axis=1).mean()) for j in range(1, len(v["kept"]))])
                seg_rec = dict(start=start, v1=strip(v1), v1_pass=bool(v1["ik_ok"] and v1["collision_ok"] and v1["ws_ok"]), v2=pack(v2))
                if v2r is not None: seg_rec["v2r"] = pack(v2r)
                rec["segments"].append(seg_rec)
                qs[f"{start}_v1"] = v1["q"]
                for j, (sc, i, r) in enumerate(v2["kept"]): qs[f"{start}_v2_{j}"] = r["q"]
                if v2r is not None:
                    for j, (sc, i, r) in enumerate(v2r["kept"]): qs[f"{start}_v2r_{j}"] = r["q"]
            if qs: np.savez_compressed(out / f"{e}.npz", **qs)
            rec["seconds"] = round(time.time() - t0, 1)
        except Exception as exc:
            rec = dict(episode=e, error=f"{type(exc).__name__}: {exc}")
        dst.write_text(json.dumps(rec, indent=1, default=float))
        ss = [s for s in rec.get("segments", []) if "v1" in s]
        print(f"{e}: segs {len(ss)}  v1 pass {sum(s['v1_pass'] for s in ss)}  v2 pass {sum(s['v2']['pass_'] for s in ss)}  "
              + (f"v2r pass {sum(s['v2r']['pass_'] for s in ss)}  " if ss and "v2r" in ss[0] else "") + "  "
              f"feasible seeds/seg {np.mean([s['v2']['n_feasible'] for s in ss]) if ss else 0:.1f}  {rec.get('seconds', rec.get('error'))}s", flush=True)


if __name__ == "__main__":
    main()
