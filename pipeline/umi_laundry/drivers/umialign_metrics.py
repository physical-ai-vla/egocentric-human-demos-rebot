#!/usr/bin/env python3
"""[2026-10-14, generalised 10-15] UMIalign per-session metric tables from run_smoother outputs (+ MASt3R restart_v1 segment files).

Table A (per episode x side): MASt3R coverage / segments / longest visual gap, full-6DoF coverage, Orbbec residual p50/p90,
max frame jump (pos / rot), camera<->IMU delay + corr, Orbbec rejected fraction.
Table B (short-init-hold episodes): first-3 s pose stability, first-segment Orbbec residual, initial L/R TCP distance.
usage: umialign_metrics.py --session <dir> [--out <prefix>] [--short 3,5,..] (default: hold_s < 2 s from <session>/raw_qa.json)   -> <prefix>.md, <prefix>_A.csv, <prefix>_B.csv
"""
import argparse, csv, json, pathlib

import numpy as np
from scipy.spatial.transform import Rotation as Rot

H = pathlib.Path.home()
SESSION = H / "ego_collector/datasets/human_handumi_raw/UMIalign/UMIalign_20261014_112920"
RR = H / "c8/robotlike/runs_restart"
SHORT_INIT = [3, 5, 6, 11, 23, 25, 27, 29, 33]
# flags: thresholds for "look at this" marks, not acceptance gates
FLAG = dict(mast3r_frac=0.85, gap_s=2.0, full6=0.95, res_p90=30.0, jump_mm=100.0, cam_corr=0.5)


def segs_of(tag):
    p = RR / tag / "ss1.segments.json"
    return len(json.load(open(p))["segments"]) if p.exists() else None


def row_a(ep, side, q):
    c = q["coverage"]; r = dict(ep=ep, side=side[0].upper())
    r.update(m3_frac=c["mast3r_valid"]["frac"], m3_gap_s=c["mast3r_valid"]["longest_gap_s"], segs=segs_of(q["tag"]),
             full6=c["full_6dof_valid"]["frac"], res_p50=q["e_p_mm"]["p50"], res_p90=q["e_p_mm"]["p90"],
             jump_mm=q["step_mm"]["max"], jump_deg=q["rot_step_deg"]["max"], cam_ms=q["camera_delay"]["estimated_ms"],
             cam_corr=q["camera_delay"]["corr"], orb_rej=q["orbbec_rejected"]["frac_of_orbbec"])
    f = []
    if r["m3_frac"] < FLAG["mast3r_frac"]: f.append("m3cov")
    if r["m3_gap_s"] > FLAG["gap_s"]: f.append("gap")
    if r["full6"] < FLAG["full6"]: f.append("6dof")
    if r["res_p90"] > FLAG["res_p90"]: f.append("res")
    if r["jump_mm"] > FLAG["jump_mm"]: f.append("jump")
    if r["cam_corr"] < FLAG["cam_corr"]: f.append("camcorr")
    r["flags"] = ",".join(f)
    return r


def row_b(ep, epdir):
    out = dict(ep=ep)
    z = {s: np.load(epdir / f"derived/umi_fusion/smoothed_{s}.npz", allow_pickle=True) for s in ("left", "right")}
    for s, k in (("left", "L"), ("right", "R")):
        t = z[s]["t"] - z[s]["t"][0]; T = z[s]["T_W_tcp"]; m = (t < 3.0) & z[s]["full_6dof"]
        p = T[m, :3, 3]; R = Rot.from_matrix(T[m, :3, :3])
        out[f"{k}_pos_std_mm"] = float(np.linalg.norm(p.std(0)) * 1e3) if m.sum() > 2 else np.nan
        out[f"{k}_rot_spread_deg"] = float(np.degrees(np.max((R.mean().inv() * R).magnitude()))) if m.sum() > 2 else np.nan
        seg = z[s]["segment"]; first = seg[seg >= 0].min() if (seg >= 0).any() else -1
        res = z[s]["orb_res_mm"][(seg == first) & np.isfinite(z[s]["orb_res_mm"])]
        out[f"{k}_seg0_res_p50_mm"] = float(np.median(res)) if res.size else np.nan
    tl, tr = z["left"]["t"], z["right"]["t"]; t0 = max(tl[0], tr[0])
    ml = (tl >= t0) & (tl < t0 + 1.0); pl = z["left"]["T_W_tcp"][ml, :3, 3]
    pr = np.stack([np.interp(tl[ml], tr, z["right"]["T_W_tcp"][:, i, 3]) for i in range(3)], 1)
    out["LR_tcp_dist_first1s_m"] = float(np.median(np.linalg.norm(pl - pr, axis=1))) if ml.sum() else np.nan
    return out


def md_table(rows, cols):
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 1 else f"{v:.1f}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", default=str(SESSION)); ap.add_argument("--out"); ap.add_argument("--short")
    a = ap.parse_args(); S = pathlib.Path(a.session); A, B, missing = [], [], []; name = S.name
    a.out = a.out or str(H / f"c8/robotlike/metrics_{name}")
    if a.short: short = [int(x) for x in a.short.split(",")]
    else:
        rq = json.load(open(S / "raw_qa.json"))["episodes"]
        short = [int(e["episode"].split("_")[-1]) for e in rq if min((e.get("hold_s") or {"x": 9}).values()) < 2.0]
    for epdir in sorted(S.glob("episode_*")):
        ep = int(epdir.name.split("_")[1]); qa = epdir / "derived/umi_fusion/smooth_qa.json"
        if not qa.exists(): missing.append(ep); continue
        q = json.load(open(qa))
        for side in ("left", "right"):
            if side in q["sides"]: A.append(row_a(ep, side, q["sides"][side]))
        A[-1]["LR_dist_p50_m"] = q["inter_gripper"]["distance_m_p05_p50_p95"][1]
        if ep in short: B.append(row_b(ep, epdir))
    ca = ["ep", "side", "m3_frac", "segs", "m3_gap_s", "full6", "res_p50", "res_p90", "jump_mm", "jump_deg", "cam_ms", "cam_corr", "orb_rej", "flags"]
    cb = ["ep", "L_pos_std_mm", "R_pos_std_mm", "L_rot_spread_deg", "R_rot_spread_deg", "L_seg0_res_p50_mm", "R_seg0_res_p50_mm", "LR_tcp_dist_first1s_m"]
    for rows, cols, suf in ((A, ca, "_A.csv"), (B, cb, "_B.csv")):
        with open(a.out + suf, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    agg = lambda k: np.nanpercentile([np.nan if r[k] is None else r[k] for r in A], [10, 50, 90]) if A else []
    lines = [f"# {name} metrics ({len(A) // 2} episodes smoothed; missing: {missing or 'none'})", "",
             "Flags (look-at marks, not gates): " + ", ".join(f"{k}={v}" for k, v in FLAG.items()), "",
             "| metric | p10 | p50 | p90 |", "|---|---|---|---|"]
    for k in ("m3_frac", "m3_gap_s", "segs", "full6", "res_p50", "res_p90", "jump_mm", "jump_deg", "cam_corr"):
        lines.append(f"| {k} | " + " | ".join(f"{v:.3g}" for v in agg(k)) + " |")
    lines += ["", "## Table A", md_table(A, ca), "", f"## Table B (short init hold < 2 s: {short})", md_table(B, cb)]
    pathlib.Path(a.out + ".md").write_text("\n".join(lines) + "\n"); print("\n".join(lines[:16])); print(f"flagged sides: {sum(1 for r in A if r['flags'])} / {len(A)}")


if __name__ == "__main__":
    main()
