#!/usr/bin/env python3
"""[2026-09-29] Phase A0/A offline IK comparison on the SAME held-out tuples (declared before any LIK0 number is read).

Backends (all seeded at the measured q_t, one solve per horizon step k, exactly as solve_waypoint does at exec time):
  N0   deployed numerical IK: HandUMI solver with max_joint_delta 0.35 (pos 125 / ori 0.6 / rest 20)
  N1   same solver, internal clip off (max_joint_delta None) -- OFFLINE DIAGNOSTIC
  LIK  LearnedIK-v0 checkpoint(s), all 16 steps in one forward pass
Common post-chain (the deploy's): dq clip +-DQ_MAX 0.6, then LO/HI. Metrics are reported on the RAW backend output and on the
post-chain command. FK for every metric = the deploy's own infer_core_v4 _tcp_mat (eef_kin FK + V4_FRAME_FIX -> dataset frame).
Target T_k = T_t(dataset frame, from q_t) @ A_k, A_k = GT REL16 (test split) -- mode "gt".
Mode "pred" (3-way decomposition) takes a v2_ckpt_eval npz: policy error (T_t Â vs T_t A_gt), IK-only (A_gt -> backend),
total (Â -> backend vs T_t A_gt), per draw.
Sample (fixed now): test-split tuples with running index % 20 == 0 (P12 test, ~4.2k), all 16 k.
Thresholds (fixed now): spike = |second difference over k| > 0.1 rad on any joint; moving = |GT REL k16 pos| > 20 mm.
usage: eval_ik_backends.py gt <out.json> [--lik runs/LIK0-P12/best.pt ...]
       eval_ik_backends.py pred <v2eval npz> <out.json> [--lik ...]
"""
import argparse, json, os, sys, time
import numpy as np, torch
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lik0_common as C
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
import infer_core_v4 as IC, eef_kin

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["gt", "pred"]); ap.add_argument("args", nargs="+")
ap.add_argument("--lik", nargs="*", default=[]); ap.add_argument("--every", type=int, default=20)
a = ap.parse_args()
SPIKE, MOVING_MM = 0.1, 20.0


class FKDeploy:
    """dataset-frame TCP of a q12, through the deploy's own _tcp_mat."""
    def __init__(self):
        self.kin = eef_kin.Kin()
    def mats(self, q12):
        q14 = np.zeros(14); q14[IC.ARM_IDX] = q12
        m, _ = IC.V4Inferencer._tcp_mat(self, q14)
        return np.stack([m[0], m[1]])


FKD = FKDeploy()
KIN = {"N0": eef_kin.Kin({"max_joint_delta": 0.35}), "N1": eef_kin.Kin({"max_joint_delta": None})}


def numerical(name, q12, Tk):
    """Tk (16, 2, 4, 4) dataset-frame targets -> raw q (16, 12), ok (16,), ms per solve."""
    kin = KIN[name]; out = np.zeros((len(Tk), 12)); ok = np.ones(len(Tk), bool); t0 = time.perf_counter()
    for k in range(len(Tk)):
        pos = Tk[k, :, :3, 3]
        quat = np.stack([Rot.from_matrix(Tk[k, r, :3, :3] @ IC.R_DS_TO_FK).as_quat() for r in (0, 1)]) if IC.USE_FRAME_FIX \
            else np.stack([Rot.from_matrix(Tk[k, r, :3, :3]).as_quat() for r in (0, 1)])
        q, _, o = kin.solve_chunk(q12, pos[None], quat[None])
        out[k] = q[0]; ok[k] = bool(o.all())
    return out, ok, (time.perf_counter() - t0) * 1000 / len(Tk)


def load_lik(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = C.LIK0(ck["stats"]); m.load_state_dict(ck["model"]); m.eval()
    return m


def rel_from_T(Tt, Tk):
    """(2,4,4), (16,2,4,4) -> rel (16, 18) in the dataset convention (pos + first two rows of R)."""
    A = np.einsum("aij,kajl->kail", np.linalg.inv(Tt), Tk)
    r = np.concatenate([A[..., :3, 3], A[..., :2, :3].reshape(16, 2, 6)], -1)
    return r.reshape(16, 18)


def post_chain(q12, q_raw):
    return np.clip(q12[None] + np.clip(q_raw - q12[None], -IC.DQ_MAX, IC.DQ_MAX), IC.LO, IC.HI)


def errs(q_seq, Tk):
    M = np.stack([FKD.mats(q) for q in q_seq])                                     # (16, 2, 4, 4)
    pe = np.linalg.norm(M[..., :3, 3] - Tk[..., :3, 3], axis=-1) * 1000             # (16, 2) mm
    Rrel = np.einsum("kaji,kajl->kail", M[..., :3, :3], Tk[..., :3, :3])
    re = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    return pe, re


def summarize(rec, dq_gt, big):
    """rec: dict of arrays per sample. Returns a JSON-able summary, L/R separate."""
    out = {}
    for tag in ("raw", "cmd"):
        pe, re, dq = rec[f"pe_{tag}"], rec[f"re_{tag}"], rec[f"dq_{tag}"]         # (N,16,2), (N,16,2), (N,16,12)
        s = {}
        for ai, arm in ((0, "L"), (1, "R")):
            p16 = pe[:, 15, ai]; r16 = re[:, 15, ai]; pall = pe[..., ai].reshape(-1)
            js = slice(0, 6) if ai == 0 else slice(6, 12)
            d2 = np.abs(np.diff(dq[..., js], 2, axis=1)); d1 = np.abs(np.diff(dq[..., js], 1, axis=1))
            s[arm] = dict(
                pos_mm_k16=dict(p50=float(np.median(p16)), p90=float(np.percentile(p16, 90)), p95=float(np.percentile(p16, 95)), max=float(p16.max())),
                pos_mm_allk=dict(p50=float(np.median(pall)), p95=float(np.percentile(pall, 95)), max=float(pall.max())),
                rot_deg_k16=dict(p50=float(np.median(r16)), p95=float(np.percentile(r16, 95))),
                pos_mm_k16_big=None if big.sum() == 0 else dict(n=int(big.sum()), p50=float(np.median(p16[big])), p90=float(np.percentile(p16[big], 90))),
                pos_mm_k16_notbig=dict(p50=float(np.median(p16[~big])), p90=float(np.percentile(p16[~big], 90))),
                dq_mae_k16_per_joint=[float(x) for x in np.abs(dq[:, 15, js] - dq_gt[:, 15, js]).mean(0)],
                dq_mae_allk=float(np.abs(dq[..., js] - dq_gt[..., js]).mean()),
                first_diff_max_p95=float(np.percentile(d1.max((1, 2)), 95)),
                second_diff_max_p95=float(np.percentile(d2.max((1, 2)), 95)),
                spike_rate=float((d2.max((1, 2)) > SPIKE).mean()),
                largest_jump=float(d1.max()))
        out[tag] = s
    out["limit_violation_raw"] = float(rec["lim_raw"].mean()); out["failure_rate"] = float(rec["fail"].mean())
    out["nan"] = int(rec["nan"].sum()); out["ms_per_sample"] = float(np.mean(rec["ms"]))
    out["dqmax_hit_rate_k16"] = float(rec["dqmax_hit"].mean())
    return out


def run_backend(name, samples, lik=None):
    rec = {k: [] for k in ("pe_raw", "re_raw", "dq_raw", "pe_cmd", "re_cmd", "dq_cmd", "lim_raw", "fail", "nan", "ms", "dqmax_hit")}
    for q12, Tk in samples:
        if lik is None:
            q_raw, ok, ms = numerical(name, q12, Tk); ms *= 16
        else:
            Tt = FKD.mats(q12); rel = rel_from_T(Tt, Tk); t0 = time.perf_counter()
            with torch.no_grad():
                dq = lik(torch.tensor(q12, dtype=torch.float32)[None], torch.tensor(rel, dtype=torch.float32)[None])[0].double().numpy()
            ms = (time.perf_counter() - t0) * 1000; q_raw = q12[None] + dq; ok = np.isfinite(q_raw).all(1)
        nan = int(~np.isfinite(q_raw).all()); q_raw = np.nan_to_num(q_raw)
        q_cmd = post_chain(q12, q_raw)
        for tag, qs in (("raw", q_raw), ("cmd", q_cmd)):
            pe, re = errs(qs, Tk); rec[f"pe_{tag}"].append(pe); rec[f"re_{tag}"].append(re); rec[f"dq_{tag}"].append(qs - q12[None])
        rec["lim_raw"].append(bool(((q_raw < IC.LO) | (q_raw > IC.HI)).any())); rec["fail"].append(not ok.all()); rec["nan"].append(nan)
        rec["ms"].append(ms); rec["dqmax_hit"].append(bool((np.abs(q_raw[15] - q12) > IC.DQ_MAX).any()))
    return {k: np.array(v) for k, v in rec.items()}


liks = {f"LIK:{p}": load_lik(p) for p in a.lik}
if a.mode == "gt":
    out_path = a.args[0]
    D = C.load_data(); te = np.flatnonzero(D["split"] == 2)[::a.every]
    print(f"[gt] {len(te)} test tuples (every {a.every}th of the P12 test split), 16 k each", flush=True)
    samples, dq_gt = [], D["dq"][te].astype(np.float64)
    for i in te:
        q12 = D["q"][i].astype(np.float64); Tt = FKD.mats(q12)
        pos, R = C.rel_split(torch.tensor(D["rel"][i][None], dtype=torch.float64))
        A = np.tile(np.eye(4), (16, 2, 1, 1)); A[..., :3, :3] = R[0].numpy(); A[..., :3, 3] = pos[0].numpy()
        samples.append((q12, np.einsum("aij,kajl->kail", Tt, A)))
    big = (np.abs(dq_gt[:, 15]) > 0.35).any(1)
    res = dict(n=len(te), share_big_dq_k16=float(big.mean()), gt_fk_mm_max=float(D["gt_fk_mm"][te].max()),
               gt_self=summarize(dict(pe_raw=np.zeros((len(te), 16, 2)), re_raw=np.zeros((len(te), 16, 2)), dq_raw=dq_gt,
                                      pe_cmd=np.zeros((len(te), 16, 2)), re_cmd=np.zeros((len(te), 16, 2)), dq_cmd=dq_gt,
                                      lim_raw=np.zeros(1), fail=np.zeros(1), nan=np.zeros(1), ms=np.zeros(1),
                                      dqmax_hit=(np.abs(dq_gt[:, 15]) > IC.DQ_MAX).any(1)), dq_gt, big))
    for name in ["N0", "N1"] + list(liks):
        t0 = time.time(); r = run_backend(name, samples, liks.get(name))
        res[name] = summarize(r, dq_gt, big); np.savez(out_path.replace(".json", f"_{name.replace('/', '_').replace(':', '_')}.npz"), **r)
        s = res[name]["cmd"]
        print(f"[{name}] {time.time() - t0:.0f}s | cmd k16 pos p50/p90/p95 L {s['L']['pos_mm_k16']['p50']:.1f}/{s['L']['pos_mm_k16']['p90']:.1f}/"
              f"{s['L']['pos_mm_k16']['p95']:.1f} R {s['R']['pos_mm_k16']['p50']:.1f}/{s['R']['pos_mm_k16']['p90']:.1f}/{s['R']['pos_mm_k16']['p95']:.1f} mm "
              f"| big-dq subset p50 L {s['L']['pos_mm_k16_big']['p50']:.1f} R {s['R']['pos_mm_k16_big']['p50']:.1f} | dq MAE k16 L "
              f"{np.mean(s['L']['dq_mae_k16_per_joint']):.3f} R {np.mean(s['R']['dq_mae_k16_per_joint']):.3f} rad | spike L {s['L']['spike_rate']:.3f} "
              f"R {s['R']['spike_rate']:.3f} | fail {res[name]['failure_rate']:.3f}", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
else:
    npz, out_path = a.args[0], a.args[1]
    d = np.load(npz, allow_pickle=True); n = int(d["done"]); pick = d["pick"][:n]; gt = d["gt"][:n]; pr = d["pred_real"][:n]
    D = C.load_data(); idx = {int(i): j for j, i in enumerate(np.arange(len(D["q"])))}
    # v2eval "pick" = dataset global index; lik0_data rows are in the same (parquet) order
    import glob, pyarrow as pa, pyarrow.parquet as pq
    root = os.path.expanduser("~/holobrain-data/lerobot/r180_umi76_rel16_v3d")
    col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
    IX = np.concatenate([col(pq.read_table(f, columns=["index"]), "index").to_numpy() for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))])
    row = {int(v): j for j, v in enumerate(IX)}

    def to_T(Tt, a20):
        A = np.tile(np.eye(4), (16, 2, 1, 1))
        for r in (0, 1):
            m = IC.pose10d_to_mat(a20[:, r * 10:r * 10 + 9]); A[:, r] = m
        return np.einsum("aij,kajl->kail", Tt, A)
    res = dict(npz=npz, n=n, draws=int(pr.shape[1]))
    rows = [row[int(i)] for i in pick]
    dq_gt = D["dq"][rows].astype(np.float64)
    sam_gt, sam_pr, pol = [], [], []
    for j, i in enumerate(rows):
        q12 = D["q"][i].astype(np.float64); Tt = FKD.mats(q12); Tg = to_T(Tt, gt[j])
        for dr in range(pr.shape[1]):
            Tp = to_T(Tt, pr[j, dr]); sam_pr.append((q12, Tp, Tg))
            pol.append((np.linalg.norm(Tp[15, :, :3, 3] - Tg[15, :, :3, 3], axis=-1) * 1000,
                        np.degrees(np.arccos(np.clip((np.trace(np.einsum("aji,ajl->ail", Tp[15, :, :3, :3], Tg[15, :, :3, :3]), axis1=-2, axis2=-1) - 1) / 2, -1, 1)))))
        sam_gt.append((q12, Tg))
    pol_p = np.stack([p for p, _ in pol]); pol_r = np.stack([r for _, r in pol])
    mv = np.repeat(np.linalg.norm(gt[:, 15][:, [0, 1, 2]], axis=1) * 1000 > MOVING_MM, pr.shape[1])
    res["policy_error_k16"] = {arm: dict(pos_mm_p50=float(np.median(pol_p[:, ai])), pos_mm_p90=float(np.percentile(pol_p[:, ai], 90)),
                                         pos_mm_p50_moving=float(np.median(pol_p[mv, ai])) if mv.any() else None,
                                         rot_deg_p50=float(np.median(pol_r[:, ai]))) for ai, arm in ((0, "L"), (1, "R"))}
    print(f"[policy] T_t Â16 vs T_t A16_gt: L p50 {res['policy_error_k16']['L']['pos_mm_p50']:.1f} p90 {res['policy_error_k16']['L']['pos_mm_p90']:.1f} mm, "
          f"R p50 {res['policy_error_k16']['R']['pos_mm_p50']:.1f} p90 {res['policy_error_k16']['R']['pos_mm_p90']:.1f} mm", flush=True)
    for name in ["N0", "N1"] + list(liks):
        rg = run_backend(name, sam_gt, liks.get(name))                                  # IK-only
        # total: predicted target through the backend, error against the GT target
        rec = {k: [] for k in ("pe", "re")}
        for q12, Tp, Tg in sam_pr:
            if name in KIN:
                q_raw, _, _ = numerical(name, q12, Tp)
            else:
                with torch.no_grad():
                    q_raw = q12[None] + liks[name](torch.tensor(q12, dtype=torch.float32)[None],
                                                   torch.tensor(rel_from_T(FKD.mats(q12), Tp), dtype=torch.float32)[None])[0].double().numpy()
            pe, re = errs(post_chain(q12, q_raw), Tg); rec["pe"].append(pe[15]); rec["re"].append(re[15])
        pe = np.stack(rec["pe"])
        ik = rg["pe_cmd"][:, 15]
        res[name] = dict(ik_only_k16={arm: dict(p50=float(np.median(ik[:, ai])), p90=float(np.percentile(ik[:, ai], 90))) for ai, arm in ((0, "L"), (1, "R"))},
                         total_k16={arm: dict(p50=float(np.median(pe[:, ai])), p90=float(np.percentile(pe[:, ai], 90)),
                                              p50_moving=float(np.median(pe[mv, ai])) if mv.any() else None) for ai, arm in ((0, "L"), (1, "R"))})
        print(f"[{name}] IK-only k16 L p50 {res[name]['ik_only_k16']['L']['p50']:.1f} p90 {res[name]['ik_only_k16']['L']['p90']:.1f} | "
              f"R p50 {res[name]['ik_only_k16']['R']['p50']:.1f} p90 {res[name]['ik_only_k16']['R']['p90']:.1f} mm || total k16 L p50 "
              f"{res[name]['total_k16']['L']['p50']:.1f} p90 {res[name]['total_k16']['L']['p90']:.1f} | R p50 {res[name]['total_k16']['R']['p50']:.1f} "
              f"p90 {res[name]['total_k16']['R']['p90']:.1f} mm", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
