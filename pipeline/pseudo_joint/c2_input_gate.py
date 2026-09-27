#!/usr/bin/env python3
"""[2026-09-23] Track C C2 input-quality gate + census (OPEN-5). RAW data only: no smoothing, no IK, no pseudo q.

Per HandUMI segment (v2 TCP from cam_pos/cam_quat, 15 Hz) and per R150 reference window (zarr TCP, 15 Hz, 22-row
non-overlapping windows = the HandUMI median segment length):

    step_p50/p95/max_mm      per-step translation
    rot_step_p95/max_deg     per-step rotation (body frame)
    rev_rate                 among consecutive step pairs with both > 5 mm: fraction turning > 120 deg (cos < -0.5)
    rot_rev_rate             same on body-frame rotation increments with both > 2 deg
    zigzag_bursts            runs of >= 2 consecutive translation reversals (A-B-A-B with steps > 5 mm)
    lost_adjacent_frac       fraction of rows within +-3 rows of a lost / pose-less frame
    boundary_spike           any step within +-3 rows of a lost frame larger than the R150 p99 step_max

Classification, fixed BEFORE the census ran (R150 ref = both arms, all 150 episodes):
    FAIL_POSE    hard gate (step > 70 mm or rot step > 40 deg), net drift from start > 1.0 m, NaN, or boundary_spike
    FAIL_JITTER  rev_rate, rot_rev_rate or zigzag_bursts above the R150 p99 of that metric
    FAIL_OTHER   fewer than 16 rows (none expected: the segmenter already enforces it)
    PASS_RAW     everything else
Step size alone is NOT a fail (a human may move faster than R150 teleop); it is kept as an outlier score.
Outlier score per metric = R150 empirical CDF of the value (>= 0.99 means beyond the R150 p99).
"""
import collections, csv, glob, json, pathlib, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pseudo_joint_pipeline as P
from umi.common.pose_util import pose_to_mat

import argparse
_ap = argparse.ArgumentParser(description="C2 input gate / OPEN-5 acceptance benchmark (FROZEN 2026-09-23)")
_ap.add_argument("--produce", default=str(HERE / "data/produce"),
                 help="dir of <episode>_<side>.npz with frame_idx, cam_pos, cam_quat (xyzw), seg_id, lost")
_ap.add_argument("--tag", default="raw", help="output subdir under reports/c2 (raw | repaired_<name>)")
_ap.add_argument("--grip", default=str(pathlib.Path.home() / ".claude/jobs/e649aef3/tmp/handumi_grip_corrected"))
ARGS = _ap.parse_args()
OUT = HERE / "reports" / "c2" / ARGS.tag; OUT.mkdir(parents=True, exist_ok=True)
GRIP = pathlib.Path(ARGS.grip)
WIN, MIN_ROWS, NEAR = 22, 16, 3
POS_PAIR_MM, ROT_PAIR_DEG, REV_COS = 5.0, 2.0, -0.5
HARD_DP, HARD_DTH, MAX_DRIFT_M = 70.0, 40.0, 1.0
JITTER = ("rev_rate", "rot_rev_rate", "zigzag_bursts")


def metrics(T, near=None):
    d = np.linalg.inv(T[:-1]) @ T[1:]
    dp_world = np.diff(T[:, :3, 3], axis=0) * 1e3                 # reversal is judged on the path itself
    n = np.linalg.norm(dp_world, axis=1)
    rv = Rot.from_matrix(d[:, :3, :3]).as_rotvec(); th = np.degrees(np.linalg.norm(rv, axis=1))
    ok = (n[:-1] > POS_PAIR_MM) & (n[1:] > POS_PAIR_MM)
    cos = np.einsum("ij,ij->i", dp_world[:-1], dp_world[1:]) / (n[:-1] * n[1:] + 1e-12)
    rev = ok & (cos < REV_COS)
    okr = (th[:-1] > ROT_PAIR_DEG) & (th[1:] > ROT_PAIR_DEG)
    cosr = np.einsum("ij,ij->i", rv[:-1], rv[1:]) / (np.linalg.norm(rv[:-1], axis=1) * np.linalg.norm(rv[1:], axis=1) + 1e-12)
    rrev = okr & (cosr < REV_COS)
    bursts, run = 0, 0
    for r in rev:
        run = run + 1 if r else 0
        if run == 2: bursts += 1
    m = dict(rows=len(T), step_p50_mm=float(np.median(n)), step_p95_mm=float(np.percentile(n, 95)), step_max_mm=float(n.max()),
             rot_step_p95_deg=float(np.percentile(th, 95)), rot_step_max_deg=float(th.max()),
             rev_rate=float(rev.sum() / ok.sum()) if ok.any() else 0.0, rev_pairs=int(ok.sum()),
             rot_rev_rate=float(rrev.sum() / okr.sum()) if okr.any() else 0.0, rot_rev_pairs=int(okr.sum()),
             zigzag_bursts=int(bursts), path_mm=float(n.sum()), drift_m=float(np.linalg.norm(T[:, :3, 3] - T[0, :3, 3], axis=1).max()),
             finite=bool(np.isfinite(T).all()))
    if near is not None:
        m["lost_adjacent_frac"] = float(near.mean())
        m["_near_step"] = n[near[:-1] | near[1:]]
    return m


# ------------------------------------------------------------------ R150 reference
d = np.load(HERE / "data/r150_q_tcp.npz"); ends = d["ends"]; starts = np.r_[0, ends[:-1]]
ref = []
for e in range(len(ends)):
    idx = np.arange(starts[e], ends[e], 2)
    for r in range(2):
        T = pose_to_mat(np.concatenate([d[f"pos{r}"][idx], d[f"rv{r}"][idx]], -1)).astype(np.float64)
        for s in range(0, len(T) - WIN + 1, WIN):
            ref.append(metrics(T[s:s + WIN]))
KEYS = ["step_p95_mm", "step_max_mm", "rot_step_p95_deg", "rot_step_max_deg", "rev_rate", "rot_rev_rate", "zigzag_bursts"]
refv = {k: np.array([m[k] for m in ref]) for k in KEYS}
refstat = {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)))
           for k, v in refv.items()}
THR = {k: refstat[k]["p99"] for k in JITTER}
BOUNDARY_STEP = refstat["step_max_mm"]["p99"]
print(f"R150 reference: {len(ref)} windows of {WIN} rows")
for k, v in refstat.items():
    print(f"   {k:<18} median {v['median']:8.3f}  p95 {v['p95']:8.3f}  p99 {v['p99']:8.3f}")


def cdf(k, x):
    return float((refv[k] <= x).mean())


# ------------------------------------------------------------------ HandUMI census
rows, cam_rows, near_steps, far_steps = [], [], [], []
moving_min_pairs = 3                                   # "moving segment" = >= 3 evaluable >5 mm step pairs
for f in sorted(glob.glob(str(pathlib.Path(ARGS.produce) / "*.npz"))):
    stem = pathlib.Path(f).stem; ep, side = stem.rsplit("_", 1)
    h = P.load_hu(f)
    bad = (h["lost"] > 0) | ~h["pose_valid"]
    near_all = np.zeros(len(bad), bool)
    for k in np.flatnonzero(bad):
        near_all[max(0, k - NEAR):k + NEAR + 1] = True
    z = np.load(f)
    for si, (s, e) in enumerate(P.segments(h["seg_id"], 2)):
        T = h["T_hu"][s:e]; near = near_all[s:e]
        m = metrics(T, near)
        ns = m.pop("_near_step"); near_steps.append(ns)
        spike = bool((ns > BOUNDARY_STEP).any())
        Tc = np.tile(np.eye(4), (e - s, 1, 1)); Tc[:, :3, 3] = z["cam_pos"][s:e]
        Tc[:, :3, :3] = Rot.from_quat(z["cam_quat"][s:e]).as_matrix()
        mc = metrics(Tc); mc.pop("_near_step", None)
        if m["rows"] < MIN_ROWS: cls, why = "FAIL_OTHER", "too short"
        elif (not m["finite"]) or m["step_max_mm"] > HARD_DP or m["rot_step_max_deg"] > HARD_DTH or m["drift_m"] > MAX_DRIFT_M or spike:
            cls = "FAIL_POSE"
            why = ";".join(x for x, c in (("nan", not m["finite"]), ("hard_dp", m["step_max_mm"] > HARD_DP),
                                          ("hard_dth", m["rot_step_max_deg"] > HARD_DTH), ("drift", m["drift_m"] > MAX_DRIFT_M),
                                          ("boundary_spike", spike)) if c)
        elif any(m[k] > THR[k] for k in JITTER):
            cls, why = "FAIL_JITTER", ";".join(k for k in JITTER if m[k] > THR[k])
        else:
            cls, why = "PASS_RAW", ""
        score = {k: cdf(k, m[k]) for k in KEYS}
        rows.append(dict(episode=ep, session="_".join(ep.split("_")[:2]), side=side, seg=si, start=int(s), end=int(e),
                         frame_start=int(h["frame_idx"][s]), frame_end=int(h["frame_idx"][e - 1]),
                         cls=cls, why=why, width_available=(GRIP / f"{ep}.npz").is_file(),
                         outlier_score_max_jitter=max(score[k] for k in JITTER), outlier_score_step=score["step_max_mm"],
                         cam_rev_rate=mc["rev_rate"], cam_step_p95_mm=mc["step_p95_mm"],
                         **{k: v for k, v in m.items() if k != "finite"}))

# ------------------------------------------------------------------ aggregates
n = len(rows); C = collections.Counter(r["cls"] for r in rows)
rw = collections.Counter(); [rw.update({r["cls"]: r["rows"]}) for r in rows]
print(f"\nHandUMI segments {n}  rows {sum(r['rows'] for r in rows)}")
for k in ("PASS_RAW", "FAIL_JITTER", "FAIL_POSE", "FAIL_OTHER"):
    print(f"   {k:<12} {C[k]:>5} ({C[k]/n*100:5.1f} % of segments, {rw[k]/sum(rw.values())*100:5.1f} % of rows)")
why = collections.Counter(w for r in rows if r["why"] for w in r["why"].split(";"))
print("   reasons (a segment can have several):", dict(why))

by_ep = collections.defaultdict(lambda: {"left": [], "right": []})
for r in rows: by_ep[r["episode"]][r["side"]].append(r)
ep_any = sum(1 for v in by_ep.values() if any(x["cls"] == "PASS_RAW" for s in v.values() for x in s))
ep_L = sum(1 for v in by_ep.values() if any(x["cls"] == "PASS_RAW" for x in v["left"]))
ep_R = sum(1 for v in by_ep.values() if any(x["cls"] == "PASS_RAW" for x in v["right"]))
bi_eps, bi_rows, bi_eps_w = 0, 0, 0
for ep, v in by_ep.items():
    fl = np.zeros(0, int); fr = np.zeros(0, int)
    L = [np.arange(x["frame_start"], x["frame_end"] + 1, 2) for x in v["left"] if x["cls"] == "PASS_RAW"]
    R = [np.arange(x["frame_start"], x["frame_end"] + 1, 2) for x in v["right"] if x["cls"] == "PASS_RAW"]
    if not (L and R): continue
    both = np.intersect1d(np.concatenate(L), np.concatenate(R))
    runs = np.split(both, np.flatnonzero(np.diff(both) != 2) + 1) if len(both) else []
    good = sum(len(x) for x in runs if len(x) >= MIN_ROWS)
    if good:
        bi_eps += 1; bi_rows += good
        bi_eps_w += int((GRIP / f"{ep}.npz").is_file())
print(f"\nepisodes {len(by_ep)}: >=1 PASS_RAW segment {ep_any} | left {ep_L} | right {ep_R} | "
      f"bimanual (both arms PASS_RAW overlapping >= {MIN_ROWS} rows) {bi_eps} episodes / {bi_rows} rows, "
      f"of which with width file {bi_eps_w}")

for side in ("left", "right"):
    v = [r for r in rows if r["side"] == side]; c = collections.Counter(r["cls"] for r in v)
    print(f"   {side:<5} segments {len(v):>5}  PASS_RAW {c['PASS_RAW']/len(v)*100:5.1f} %  JITTER {c['FAIL_JITTER']/len(v)*100:5.1f} %  POSE {c['FAIL_POSE']/len(v)*100:5.1f} %")
sess = collections.defaultdict(list)
for r in rows: sess[r["session"]].append(r["cls"])
print("\nsession pass rate (segments):")
for s in sorted(sess): print(f"   {s}  {len(sess[s]):>4}  PASS_RAW {sess[s].count('PASS_RAW')/len(sess[s])*100:5.1f} %")

# diagnostics: is jitter a camera-rotation lever-arm effect? is it concentrated near lost frames?
rv_t = np.array([r["rev_rate"] for r in rows]); rv_c = np.array([r["cam_rev_rate"] for r in rows])
ns = np.concatenate(near_steps)
print(f"\ndiag: rev_rate median TCP {np.median(rv_t):.3f} vs CAMERA {np.median(rv_c):.3f} "
      f"(corr {np.corrcoef(rv_t, rv_c)[0,1]:.2f}); steps near lost frames: {len(ns)}, p95 {np.percentile(ns,95) if len(ns) else 0:.1f} mm")
sens = {str(t): float(np.mean([r["rev_rate"] <= t for r in rows])) for t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)}
print("sensitivity (NOT the rule): share of segments with rev_rate <= t:", {k: round(v, 3) for k, v in sens.items()})

with open(OUT / "census_segments.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
(OUT / "census.json").write_text(json.dumps(dict(
    schema="track_c_c2_input_gate/v1", date="2026-09-23", tag=ARGS.tag, produce=ARGS.produce, rule=__doc__.split("Classification")[1].strip(),
    r150_reference=dict(windows=len(ref), window_rows=WIN, stats=refstat), thresholds=dict(THR, boundary_step_mm=BOUNDARY_STEP,
    hard_dp_mm=HARD_DP, hard_dth_deg=HARD_DTH, max_drift_m=MAX_DRIFT_M),
    counts=dict(segments=n, by_class=dict(C), rows_by_class=dict(rw), reasons=dict(why),
                moving_segments=sum(r["rev_pairs"] >= moving_min_pairs for r in rows),
                moving_pass_raw=sum(r["rev_pairs"] >= moving_min_pairs and r["cls"] == "PASS_RAW" for r in rows),
                static_pass_raw=sum(r["rev_pairs"] == 0 and r["cls"] == "PASS_RAW" for r in rows)),
    episodes=dict(total=len(by_ep), any_pass=ep_any, left_pass=ep_L, right_pass=ep_R, bimanual_pass=bi_eps,
                  bimanual_rows=bi_rows, bimanual_pass_with_width=bi_eps_w),
    session_pass={s: dict(segments=len(v), pass_raw=v.count("PASS_RAW")) for s, v in sess.items()},
    diag=dict(rev_rate_median_tcp=float(np.median(rv_t)), rev_rate_median_camera=float(np.median(rv_c)),
              corr=float(np.corrcoef(rv_t, rv_c)[0, 1])), sensitivity_rev_rate=sens), indent=1))
np.savez(OUT / "hist_data.npz", hu_rev=rv_t, hu_step_p95=np.array([r["step_p95_mm"] for r in rows]),
         hu_rot_rev=np.array([r["rot_rev_rate"] for r in rows]),
         ref_rev=refv["rev_rate"], ref_step_p95=refv["step_p95_mm"], ref_rot_rev=refv["rot_rev_rate"])
print(f"-> {OUT}")
