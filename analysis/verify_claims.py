#!/usr/bin/env python3
"""Check every quantitative claim of report/report.tex against the files in this repository.

Each row: claim as written in the report, value recomputed from the evidence file, PASS/FAIL. Numbers are compared at
the precision the report prints them. Run from the repo root:  python3 analysis/verify_claims.py [--md out.md]
Exit code 1 if any claim fails.
"""
import csv, json, math, re, statistics as st, sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]
J = lambda p: json.load(open(R / p))
rows = []


def check(section, claim, reported, computed, source, tol=None):
    if isinstance(reported, str):
        ok = str(computed) == reported
    else:
        t = tol if tol is not None else 0.5 * 10 ** -max(0, len(repr(float(reported)).split(".")[1].rstrip("0")) if "." in repr(reported) else 0)
        ok = abs(float(computed) - float(reported)) <= t + 1e-9
    rows.append((section, claim, reported, computed if isinstance(computed, str) else round(float(computed), 4), source, "PASS" if ok else "FAIL"))


# ---------------------------------------------------------------- collection
man = list(csv.DictReader(open(R / "results/episode_manifest.csv")))
src = "results/episode_manifest.csv"
check("data", "recorded episodes", 349, len(man), src)
check("data", "sessions", 15, len({m["session"] for m in man}), src)
check("data", "operators", 1, len({m["operator"] for m in man}), src)
check("data", "first day", "2026-09-15", min(m["t_start"] for m in man)[:10], src)
check("data", "last day", "2026-09-17", max(m["t_start"] for m in man)[:10], src)
check("data", "total hours", 2.20, sum(float(m["duration_s"]) for m in man) / 3600, src, tol=0.005)
check("data", "median episode s", 23.05, st.median(float(m["duration_s"]) for m in man), src, tol=0.005)
oc = Counter(m["order"] for m in man)
for o, n in dict(RBP=64, RPB=62, BRP=61, BPR=56, PRB=55, PBR=51).items():
    check("data", f"order {o}", n, oc[o], src)
early = [m for m in man if m["domain"].startswith("early")]
check("data", "early-domain episodes", 55, len(early), src)
check("data", "early-domain sessions", 10, len({m["session"] for m in early}), src)
check("data", "main-domain episodes", 294, len(man) - len(early), src)

# ---------------------------------------------------------------- funnel
cen = J("results/phase3_census.json")["episodes"]; src = "results/phase3_census.json"
ff = Counter(v["first_fail"] for v in cen.values())
check("funnel", "sync failures", 65, ff["sync_pass"], src)
check("funnel", "sync failure share %", 19, 100 * ff["sync_pass"] / len(cen), src, tol=0.5)
check("funnel", "sync-passing episodes", 284, len(cen) - ff["sync_pass"], src)
segs = [s for v in cen.values() for s in v.get("segments", [])]
sf = Counter(s["first_fail"] for s in segs)
check("funnel", "candidate segments", 1306, len(segs), src)
check("funnel", "workspace removals", 677, sf["C_workspace"], src)
check("funnel", "metric-scale removals", 50, sf["B_metric"], src)
check("funnel", "IK removals", 18, sf["D_IK"], src)
check("funnel", "bimanual segments", 561, sf[None], src)
check("funnel", "kept fraction %", 43.0, 100 * sf[None] / len(segs), src, tol=0.05)
check("funnel", "workspace share of removals %", 91, 100 * sf["C_workspace"] / (len(segs) - sf[None]), src, tol=0.5)
check("funnel", "removed fraction %", 57, 100 * (1 - sf[None] / len(segs)), src, tol=0.5)
ws = [s for s in segs if s["first_fail"] == "C_workspace"]
one = sum(1 for s in ws if sum(a == "C_workspace" for a in s["arm_first_fail"]) == 1)
check("funnel", "workspace fails with one arm only", 531, one, src)
check("funnel", "one-arm share %", 78, 100 * one / len(ws), src, tol=0.5)
eps_E = [k for k, v in cen.items() if any(s["first_fail"] is None for s in v.get("segments", []))]
check("funnel", "episodes with >=1 bimanual segment", 259, len(eps_E), src)
starts = [s["start"] for v in cen.values() for s in v.get("segments", []) if s["first_fail"] is None]
check("funnel", "segments starting in first 5 s %", 47, 100 * sum(x < 5 * 30 for x in starts) / len(starts), src, tol=0.5)
q = J("results/c8_quarantine.json")["sessions"]
check("funnel", "quarantined sessions", 2, len(q), "results/c8_quarantine.json")
check("funnel", "quarantined episodes", 6, sum(v["episodes"] for v in q.values()), "results/c8_quarantine.json")
check("funnel", "segments from quarantined sessions", 0,
      sum(1 for k in eps_E if k.rsplit("_", 1)[0] in q), src)

# ---------------------------------------------------------------- split
sp = J("results/split_index.json"); pv = J("results/provenance.json")["dataset"]["report"]
check("split", "val episodes", 26, len(sp["val_episodes"]), "results/split_index.json")
check("split", "train episodes", 233, sp["episodes"] - len(sp["val_episodes"]), "results/split_index.json")
check("split", "train segments", 504, pv["c8old_train"]["segments"], "results/provenance.json")
check("split", "val segments", 57, pv["c8old_val"]["segments"], "results/provenance.json")
check("split", "train frames", 32760, pv["c8old_train"]["rows"], "results/provenance.json")
check("split", "val frames", 3705, pv["c8old_val"]["rows"], "results/provenance.json")
val = [m for m in man if m["split"] == "val"]
vo = Counter(m["order"] for m in val)
check("split", "val episodes per order (min)", 3, min(vo.values()), src := "results/episode_manifest.csv")
check("split", "val episodes per order (max)", 6, max(vo.values()), src)
check("split", "val main-domain episodes", 22, sum(not m["domain"].startswith("early") for m in val), src)
check("split", "val sessions", 9, len({m["session"] for m in val}), src)
check("split", "non-quarantined sessions", 13, len({m["session"] for m in man}) - len(q), src)

# ---------------------------------------------------------------- calibration
import yaml  # noqa: E402
fl, fr = (yaml.safe_load(open(R / f"calibration/{f}")) for f in ("fisheye_left_v002.yaml", "fisheye_right_v003.yaml"))
check("calib", "left fx px", 789, fl["K"][0][0], "calibration/fisheye_left_v002.yaml", tol=0.5)
check("calib", "right fx px", 791, fr["K"][0][0], "calibration/fisheye_right_v003.yaml", tol=0.5)
cl, cr = (yaml.safe_load(open(R / f"calibration/{f}")) for f in ("camera_imu_left_v001.yaml", "camera_imu_right_v002.yaml"))
check("calib", "left cam-IMU offset ms", 30.2, cl["time_offset_ms"], "calibration/camera_imu_left_v001.yaml")
check("calib", "right cam-IMU offset ms", 32.8, cr["time_offset_ms"], "calibration/camera_imu_right_v002.yaml")
rep = lambda f: float(re.search(r"Reprojection error \(cam0\) \[px\]:\s+mean ([0-9.]+)", open(R / f).read()).group(1))
check("calib", "left reprojection mean px", 6.0, rep("calibration/kalibr/calib_left-results-imucam.txt"), "calibration/kalibr/calib_left-results-imucam.txt")
check("calib", "right reprojection mean px", 5.6, rep("calibration/kalibr/calib_right-results-imucam.txt"), "calibration/kalibr/calib_right-results-imucam.txt")
ct = yaml.safe_load(open(R / "calibration/camera_tcp/handumi_camera_tcp_v2.yaml"))["translation"]["t_camera_tcp_m"]
check("calib", "cam->TCP ty mm", 55.3, 1e3 * ct[1], "calibration/camera_tcp/handumi_camera_tcp_v2.yaml")
check("calib", "cam->TCP tz mm", 122.5, 1e3 * ct[2], "calibration/camera_tcp/handumi_camera_tcp_v2.yaml")
g = open(R / "pipeline/grip/correct_grip.py").read()
check("calib", "jaw aperture mm", 67.5, 1e3 * float(re.search(r"APERTURE_M = ([0-9.]+)", g).group(1)), "pipeline/grip/correct_grip.py")
check("calib", "jaw aperture +/- mm", 5, 1e3 * float(re.search(r"APERTURE_UNCERTAINTY_M = ([0-9.]+)", g).group(1)), "pipeline/grip/correct_grip.py")

# ---------------------------------------------------------------- pose benchmark (RTX 5080 run = the one that produced the dataset)
log = open(R / "results/pose_benchmark/cp6_ss1_5080.log").read().split("=== FIT")[0]
row = lambda m: re.search(rf"\n\s+{m}\s+all\s+(\d+)\s+([0-9.]+)%/\s*([0-9.]+)%.*?\|\s+[0-9.]+/\s*([0-9.]+) \([^|]*\|\s*[0-9.]+/\s*([0-9.]+) \([^|]*$", log, re.M)
im, gl = row("imu_vi"), row("global")
src = "results/pose_benchmark/cp6_ss1_5080.log"
check("pose", "held-out tracks", 10, int(im.group(1)), src)
check("pose", "IMU-VI scale err p50 %", 4.6, float(im.group(2)), src)
check("pose", "IMU-VI scale err p90 %", 17.0, float(im.group(3)), src)
check("pose", "global-constant scale err p50 %", 18.5, float(gl.group(2)), src)
check("pose", "global-constant scale err p90 %", 23.8, float(gl.group(3)), src)
check("pose", "k16 rel. position p90 mm", 11.3, float(im.group(4)), src)
check("pose", "k32 rel. position p90 mm", 16.1, float(im.group(5)), src)
c7 = J("results/pose_benchmark/cp7_eval.json")["eval_raw"]
check("pose", "rotation err p50 deg", 0.84, c7["rot_p50"], "results/pose_benchmark/cp7_eval.json")
check("pose", "rotation err p90 deg", 2.13, c7["rot_p90"], "results/pose_benchmark/cp7_eval.json")
qa = J("results/pose_benchmark/qa_all.json")
cov = lambda m: st.median(v[m]["coverage"] for v in qa.values())
check("pose", "benchmark tracks", 28, len(qa), "results/pose_benchmark/qa_all.json")
check("pose", "MASt3R coverage (median)", 1.00, cov("M3_ss1_5080"), "results/pose_benchmark/qa_all.json")
check("pose", "ORB-SLAM3 stored-map coverage (median)", 0.03, cov("ORB_prod"), "results/pose_benchmark/qa_all.json", tol=0.005)
rev = lambda m: st.median(v[m]["tcp"]["reversal_rate"] for v in qa.values() if v[m].get("tcp", {}).get("reversal_rate") is not None)
check("pose", "ORB-SLAM3 reversal rate (median)", 0.24, rev("ORB_prod"), "results/pose_benchmark/qa_all.json")
check("pose", "MASt3R reversal rate (median)", 0.07, rev("M3_ss1_5080"), "results/pose_benchmark/qa_all.json")

# ---------------------------------------------------------------- model / training
cfg, tc = J("results/model_config/config.json"), J("results/model_config/train_config.json")
src = "results/model_config/{config,train_config}.json"
for k, v in dict(depth=24, hidden_size=1024, num_heads=16, len_soft_prompts=32, num_denoising_steps=10, chunk_size=30, num_image_views=3).items():
    check("model", k, v, cfg[k], src)
check("model", "image size px", 224, cfg["resize_imgs_with_padding"][0], src)
check("model", "vision encoder trainable", "False", str(cfg["freeze_vision_encoder"]), src)
check("train", "steps", 300000, tc["steps"], src)
check("train", "batch size", 4, tc["batch_size"], src)
check("train", "lr", 1e-4, tc["optimizer"]["lr"], src, tol=0)
check("train", "weight decay", 1e-4, tc["optimizer"]["weight_decay"], src, tol=0)
check("train", "grad clip", 10, tc["optimizer"]["grad_clip_norm"], src)
check("train", "warm-up steps", 1000, tc["scheduler"]["num_warmup_steps"], src)
check("train", "final lr", 2.5e-6, tc["scheduler"]["decay_lr"], src, tol=0)
check("train", "save every", 10000, tc["save_freq"], src)
ch = open(R / "training/c8old_chain.sh").read()
check("train", "aux C weight", 2.0, float(re.search(r"EE_AUX_LAMBDA=([0-9.]+)", ch).group(1)), "training/c8old_chain.sh")
check("train", "FK D weight", 20, float(re.search(r"EE_FK_LAMBDA=([0-9.]+)", ch).group(1)), "training/c8old_chain.sh")

# ---------------------------------------------------------------- offline evaluation
e160, e300 = J("results/eval/160000.json"), J("results/eval/300000.json"); sel = J("results/selection.json")
check("eval", "samples", 399, e300["samples"], "results/eval/300000.json")
check("eval", "arm-samples", 798, e300["all_samples"]["k1"]["n"], "results/eval/300000.json")
check("eval", "moving arm-samples", 187, e300["motion_tcp"]["k30"]["n"], "results/eval/300000.json")
check("eval", "moving fraction %", 23.4, 100 * e300["motion_tcp"]["k30"]["frac"], "results/eval/300000.json")
check("eval", "static fraction %", 77, 100 * (1 - e300["motion_tcp"]["k30"]["frac"]), "results/eval/300000.json", tol=0.5)
check("eval", "pre-registered selection", "160000", sel["best_geo_pre_registered"], "results/selection.json")
check("eval", "geo score 160k mm", 2.78, e160["geo_score"], "results/eval/160000.json")
check("eval", "geo score 300k mm", 2.88, e300["geo_score"], "results/eval/300000.json")
check("eval", "moving geo score 300k mm", 23.0, e300["motion_tcp"]["geo_score"], "results/eval/300000.json")
check("eval", "all-sample geo score 300k (rounded) mm", 2.9, e300["geo_score"], "results/eval/300000.json")
KS = (1, 4, 8, 16, 30)
tab = {  # Table II as printed
    ("all", "fk", 160000): [1.3, 1.7, 2.2, 3.1, 5.7], ("all", "fk", 300000): [1.5, 1.8, 2.2, 3.2, 5.7],
    ("all", "mae", 300000): [0.30, 0.34, 0.42, 0.57, 0.97], ("all", "mae0", 300000): [0.15, 0.19, 0.25, 0.36, 0.51],
    ("moving", "fk", 160000): [11.2, 15.0, 19.9, 29.2, 39.1], ("moving", "fk", 300000): [11.2, 15.0, 19.4, 29.3, 39.9],
    ("moving", "mae", 300000): [2.02, 2.68, 3.29, 4.30, 6.59], ("moving", "mae0", 300000): [2.32, 3.36, 4.38, 6.34, 8.68],
    ("moving", "cos", 300000): [0.72, 0.80, 0.84, 0.86, 0.90]}
key = dict(fk="fk_mm_p50", mae="joint_mae_deg", mae0="zero_action_mae_deg", cos="dir_cos_p50")
for (sub, m, step), vals in tab.items():
    e = e160 if step == 160000 else e300; blk = e["all_samples" if sub == "all" else "motion_tcp"]
    for k, v in zip(KS, vals):
        check("table2", f"{sub} {m} k{k} @{step // 1000}k", v, blk[f"k{k}"][key[m]], f"results/eval/{step:06d}.json")
mv = e300["motion_tcp"]
red = [100 * (1 - mv[f"k{k}"]["joint_mae_deg"] / mv[f"k{k}"]["zero_action_mae_deg"]) for k in KS]
check("eval", "MAE reduction vs zero, k30 %", 24, red[-1], "results/eval/300000.json", tol=0.5)
check("eval", "MAE reduction range min %", 13, min(red), "results/eval/300000.json", tol=0.5)
check("eval", "MAE reduction range max %", 32, max(red), "results/eval/300000.json", tol=0.5)
check("eval", "moving k30 FK p90 mm", 90, mv["k30"]["fk_mm_p90"], "results/eval/300000.json", tol=0.5)
check("eval", "norm ratio k1", 1.41, e300["k1"]["norm_ratio_p50"], "results/eval/300000.json")
check("eval", "collapse ratio k30", 0.77, e300["collapse_std_ratio_k30"], "results/eval/300000.json")
check("eval", "geo flat after (k): 120k within 0.2 mm of best", "True",
      str(all(abs(sel["table"][s]["geo"] - e160["geo_score"]) < 0.2 for s in sel["table"] if int(s) >= 120000)), "results/selection.json")

# ---------------------------------------------------------------- prompt swap
ps = J("analysis/out/prompt_swap_300k_summary.json"); src = "analysis/out/prompt_swap_300k_summary.json"
p30 = ps["k30"]["moving"]
check("prompt", "prompt spread k30 moving mm", 4.8, p30["prompt_spread_mm_p50"], src)
check("prompt", "noise spread k30 moving mm", 0.09, p30["noise_spread_mm_p50"], src)
check("prompt", "prompt/noise ratio (about 50x)", 50, p30["prompt_over_noise_p50"], src, tol=2)
check("prompt", "predicted displacement k30 mm", 62, p30["pred_disp_mm_p50"], src, tol=0.5)
check("prompt", "spread / displacement %", 8, 100 * p30["prompt_spread_mm_p50"] / p30["pred_disp_mm_p50"], src, tol=0.5)
check("prompt", "err true k30 mm", 39.9, p30["err_true_mm_p50"], src)
check("prompt", "err wrong k30 mm", 39.5, p30["err_wrong_mm_p50"], src)
check("prompt", "mean rank true", 3.55, p30["rank_true_mean"], src, tol=0.005)
check("prompt", "rank-1 frequency %", 16, 100 * p30["frac_rank1"], src, tol=0.5)
check("prompt", "rank near chance at every k/subset (|rank-3.5|<0.25)", "True",
      str(all(abs(ps[f"k{k}"][s]["rank_true_mean"] - 3.5) < 0.25 for k in KS for s in ("all", "moving"))), src)
check("prompt", "reproduces reference eval (< 1e-5 mm)", "True", str(ps["repro_max_abs_diff_mm_vs_reference_eval"] < 1e-5), src)


# ---------------------------------------------------------------- follow-up: retargeting v2, HRL80, final dataset, fine-tuning
rv = J("results/retarget_v2/compare_R30_heldout.json"); src = "results/retarget_v2/compare_R30_heldout.json"
for name, blk, usable, nn50, nn90, w1 in (("v1", rv["r30"]["v1"], 36.5, 1.49, 1.55, 0.00197), ("v2-K", rv["r30"]["v2"], 66.6, 1.21, 1.60, 0.00275),
                                          ("v2-Kr", rv["r30"]["v2r"], 67.0, 0.44, 0.74, 0.00310), ("v2-TR", rv["tr"]["v2"], 56.1, 0.27, 0.45, 0.00208)):
    m = blk["manifold"]
    check("retarget", f"{name} usable %", usable, 100 * blk["rate"], src, tol=0.05)
    check("retarget", f"{name} NN p50", nn50, m["nn_dist_rad_p50"], src, tol=0.005)
    check("retarget", f"{name} NN p90", nn90, m["nn_dist_rad_p90"], src, tol=0.005)
    check("retarget", f"{name} W1 dq", w1, m["W1_dq_vs_heldout120"], src, tol=0.000005)
check("retarget", "v1 start clusters", 1, rv["r30"]["v1"]["manifold"]["start_distinct_clusters"], src)
check("retarget", "TR start clusters", 100, rv["tr"]["v2"]["manifold"]["start_distinct_clusters"], src)
mm = J("results/retarget_v2/hybrid_master_MASTER.json"); src = "results/retarget_v2/hybrid_master_MASTER.json"
check("retarget", "candidate segments", 1229, mm["total_segments"], src)
fr = mm["fallback_reasons"]; none_ws = sum(v for k, v in fr.items() if k.startswith("none:no_workspace"))
none_all = sum(v for k, v in fr.items() if k.startswith("none:"))
check("retarget", "unsolved segments", 402, none_all, src)
check("retarget", "unsolved: no workspace window", 399, none_ws, src)
check("retarget", "matched-segment finding mentions 686 / 0.00207 / 0.00226", "True",
      str(all(t in str(mm["matched_segment_finding"]) for t in ("686", "0.00207", "0.00226"))), src)
hc = J("results/hrl80/hrl80_raw_census.json")["summary"]; src = "results/hrl80/hrl80_raw_census.json"
check("hrl80", "episodes", 60, hc["main_session_episodes"], src)
check("hrl80", "per order min", 10, min(hc["main_session_orders"].values()), src)
check("hrl80", "per order max", 10, max(hc["main_session_orders"].values()), src)
check("hrl80", "main session id", "HRL80_20260928_101010", str(hc["main_session"]), src)
hp = J("results/hrl80/hrl80_prereg_report.json")["T1_TR_usable_fraction_U_e"]; src = "results/hrl80/hrl80_prereg_report.json"
check("hrl80", "live rot median old", 0.144, hp["old259"]["live_rot_speed_median"], src, tol=0.0005)
check("hrl80", "live rot median HRL80", 0.067, hp["HRL80"]["live_rot_speed_median"], src, tol=0.0005)
check("hrl80", "U_e mean old", 0.59, hp["old259"]["U_mean"], src, tol=0.005)
check("hrl80", "U_e mean HRL80", 0.59, hp["HRL80"]["U_mean"], src, tol=0.005)
pk = [k for k in hp["pooled"] if "primary" in k][0]
check("hrl80", "pooled rho", -0.11, hp["pooled"][pk]["rho"], src, tol=0.005)
check("hrl80", "pooled p", 0.05, hp["pooled"][pk]["p"], src, tol=0.005)
check("hrl80", "pooled n", 316, hp["pooled"]["n"], src)
check("hrl80", "TR usable %", 58.9, 100 * J("results/hrl80/compare_hrl80_heldout.json")["tr"]["v2"]["rate"], "results/hrl80/compare_hrl80_heldout.json", tol=0.05)
mf = J("results/dataset_v2/final/MANIFEST_final.json"); src = "results/dataset_v2/final/MANIFEST_final.json"
check("final", "TR segments", 996, mf["composition"]["total"]["segments"], src)
check("final", "old segments", 689, mf["composition"]["old259"]["segments"], src)
check("final", "train segments", 896, mf["counts"]["train_segments"], src)
check("final", "val segments", 100, mf["counts"]["val_segments"], src)
check("final", "source episodes", 316, mf["composition"]["total"]["train_source_episodes"] + mf["composition"]["total"]["val_source_episodes"], src)
check("final", "append invariants pass", "True", str(not J("results/dataset_v2/final/APPEND_INVARIANT.json")["failed"]), "results/dataset_v2/final/APPEND_INVARIANT.json")
bs = J("analysis/out/paired_bootstrap_r150_250k.json"); src = "analysis/out/paired_bootstrap_r150_250k.json"
check("finetune", "geo delta %", -10.0, 100 * bs["point"]["geo"], src, tol=0.05)
check("finetune", "geo CI low %", -14.7, 100 * bs["ci95"]["geo"][0], src, tol=0.05)
check("finetune", "geo CI high %", -5.6, 100 * bs["ci95"]["geo"][1], src, tol=0.05)
check("finetune", "motion geo delta %", -7.2, 100 * bs["point"]["motion_geo"], src, tol=0.05)
check("finetune", "motion geo CI low %", -13.4, 100 * bs["ci95"]["motion_geo"][0], src, tol=0.05)
check("finetune", "motion geo CI high %", -1.8, 100 * bs["ci95"]["motion_geo"][1], src, tol=0.05)
check("finetune", "motion MAE delta %", -6.8, 100 * bs["point"]["motion_k30_mae"], src, tol=0.05)
check("finetune", "motion MAE CI low %", -12.7, 100 * bs["ci95"]["motion_k30_mae"][0], src, tol=0.05)
check("finetune", "motion MAE CI high %", 0.5, 100 * bs["ci95"]["motion_k30_mae"][1], src, tol=0.05)
check("finetune", "episodes favouring ego init", 8, bs["episodes_cold_better_geo"], src)
check("finetune", "eval episodes", 10, bs["n_episodes"], src)
b1, co = J("results/finetune/baseline/B1old_R150_250000.json"), J("results/finetune/r150ft600k/eval/250000.json")
check("finetune", "bootstrap point == eval JSONs (geo)", "True", str(abs((co["geo_score"] - b1["geo_score"]) / b1["geo_score"] - bs["point"]["geo"]) < 1e-6), "results/finetune/*.json")

r90 = open(R / "results/finetune/R90_MATCHED.md").read(); src = "results/finetune/R90_MATCHED.md"
curve = {l.split("|")[1].strip(): [c.strip() for c in l.split("|")[2:-1]] for l in r90.splitlines() if re.match(r"\| \d+k \|", l)}
check("finetune", "R90 matched steps", "10k,20k", ",".join(sorted(curve)), src)
for step, geo, mgeo in (("10k", "-33%", "+2%"), ("20k", "-11%", "+13%")):
    check("finetune", f"R90 {step} all-sample geo delta", geo, curve[step][1].split("/")[-1].strip(), src)
    check("finetune", f"R90 {step} motion geo delta", mgeo, curve[step][0].split("/")[-1].strip(), src)
ho = J("results/finetune/C_OLD_TR_HANDOFF.json"); src = "results/finetune/C_OLD_TR_HANDOFF.json"
check("pretrain_v2", "stopped at 211.7k", "True", str("211.7k" in " ".join(ho["not_used"])), src)
check("pretrain_v2", "primary init 100k", "True", str(ho["primary_init"].startswith("TR100k")), src)
check("pretrain_v2", "secondary 200k", "True", str(ho["secondary_ablation"].startswith("TR200k")), src)
rs = J("results/finetune/RUN_STATUS.json"); src = "results/finetune/RUN_STATUS.json"
check("pretrain_v2", "no fine-tuning result from v2 ckpts", "none", rs["ego_pretrain_v2_final_TR"]["fine_tuning_results_reported"], src)
check("pretrain_v2", "planned steps", 300000, rs["ego_pretrain_v2_final_TR"]["planned_steps"], src)
check("pretrain_v2", "launch schedule steps", 300000, J("results/finetune/FINAL_TR300K_LAUNCH.json")["schedule"]["steps"], "results/finetune/FINAL_TR300K_LAUNCH.json")

# ---------------------------------------------------------------- v3: Cartesian ego dataset
V = "results/v3/"
m2, m2b = J(V + "ego_cart20/metadata_v2.json"), J(V + "ego_cart20/metadata_v2b.json"); src = V + "ego_cart20/metadata_v2.json"
conv = m2["conversion"]
check("v3_data", "episodes converted", 330, len(conv), src)
check("v3_data", "original episodes", 273, sum(c["era"] == "old259_contract" for c in conv), src)
check("v3_data", "HRL80 episodes", 57, sum(c["era"] == "HRL80_v014" for c in conv), src)
check("v3_data", "refused at export", 3, Counter(x["status"] for x in J(V + "ego_cart20/export_log_v2.json"))["refused"], V + "ego_cart20/export_log_v2.json")
check("v3_data", "15 Hz rows", 111533, sum(c["frames_rows"] for c in conv), src)
check("v3_data", "trainable rows", 57283, sum(c["train_rows"] for c in conv), src)
check("v3_data", "trainable share %", 51.4, 100 * sum(c["train_rows"] for c in conv) / sum(c["frames_rows"] for c in conv), src, tol=0.05)
check("v3_data", "UMI_DT ms", 50.05, 1e3 * m2["contract"]["target_dt_s"], src, tol=0.005)
check("v3_data", "v2 train episodes", 298, m2["counts"]["train"]["episodes"], src)
check("v3_data", "v2 train rows", 51612, m2["counts"]["train"]["train_rows"], src)
check("v3_data", "val episodes", 32, m2["counts"]["val"]["episodes"], src)
check("v3_data", "val rows", 5671, m2["counts"]["val"]["train_rows"], src)
src = V + "ego_cart20/metadata_v2b.json"
check("v3_data", "v2b train episodes", 297, m2b["counts"]["train"]["episodes"], src)
check("v3_data", "v2b train rows", 48411, m2b["counts"]["train"]["train_rows"], src)
check("v3_data", "v2b val rows unchanged", 5671, m2b["counts"]["val"]["train_rows"], src)
check("v3_data", "v2b train row drop %", -6.20, 100 * (m2b["counts"]["train"]["train_rows"] / m2["counts"]["train"]["train_rows"] - 1), src, tol=0.005)
man = [json.loads(l) for l in open(R / V / "ego_cart20/manifest_v2b_train.jsonl")]
ev = [(m["episode_id"], e) for m in man for side in ("left", "right") for e in m.get("jump_events", {}).get(side, [])]
check("v3_jump", "jump events (train)", 44, len(ev), V + "ego_cart20/manifest_v2b_train.jsonl")
check("v3_jump", "episodes with jump events", 42, len({e for e, _ in ev}), V + "ego_cart20/manifest_v2b_train.jsonl")
ig = J(V + "ego_cart20/integrity_v2.json"); check("v3_data", "gripper max observed", 0.847, max(ig["gripper_min_max"]) if isinstance(ig["gripper_min_max"], list) else ig["gripper_min_max"]["max"], V + "ego_cart20/integrity_v2.json", tol=0.0005)
co = J(V + "state_jump/v2_vs_v2b_continuity.json"); src = V + "state_jump/v2_vs_v2b_continuity.json"
check("v3_jump", "max adjacent step v2 mm", 422.5, co["v2"]["training_adjacent_rows"]["max_mm"], src, tol=0.05)
check("v3_jump", "max adjacent step v2b mm", 90.5, co["v2b_C"]["training_adjacent_rows"]["max_mm"], src, tol=0.05)
check("v3_jump", "steps >100 mm v2", 18, co["v2"]["training_adjacent_rows"]["gt100mm"], src)
check("v3_jump", "steps >100 mm v2b", 0, co["v2b_C"]["training_adjacent_rows"]["gt100mm"], src)
check("v3_jump", "raw >3 m/s steps left in valid rows", 7, co["v2b_C"]["raw_steps_gt_3mps_left_in_valid"], src)
rj = J(V + "state_jump/raw_jump_events.json"); src = V + "state_jump/raw_jump_events.json"
check("v3_jump", "raw >3 m/s steps (train)", 55, len(rj), src)
check("v3_jump", "episodes with raw >3 m/s steps", 24, len({r[0] for r in rj}), src)
check("v3_jump", "raw jumps that return (spikes)", 5, sum(bool(r[-1]) for r in rj), src)
sd = J(V + "state_jump/raw_jump_stage_dump.json")
check("v3_jump", "raw steps with a lost/missing flag in the previous 15 samples", 0, sum(d["lost_or_missing_in_prev_15"] > 0 for d in sd), V + "state_jump/raw_jump_stage_dump.json")
sj = open(R / V / "state_jump/STATE_JUMP_ROOT_CAUSE.md").read()
check("v3_jump", "apparent jumps that are time gaps", "1,482 out of 2,598 (57%)", re.search(r"(1,482 out of 2,598 \(57%\))", sj).group(1), V + "state_jump/STATE_JUMP_ROOT_CAUSE.md")

# ---------------------------------------------------------------- v3: ego vs robot
ik = J(V + "ego_vs_robot/kinematic_summary.json")["ik_state_feasibility"]; src = V + "ego_vs_robot/kinematic_summary.json"
for key, val in (("ego_pos_L", 99.7), ("ego_pos_R", 99.5), ("ego_full_nolock_L", 72.3), ("ego_full_nolock_R", 74.6), ("robot_full_L", 45.2), ("robot_full_R", 52.2)):
    check("v3_robot", f"IK success {key} %", val, 100 * ik[key]["success"], src, tol=0.05)
it = open(R / V / "ego_vs_robot/INTERPRETATION.md").read(); src = V + "ego_vs_robot/INTERPRETATION.md"
row = lambda lab: [c.strip() for c in re.search(r"\| " + re.escape(lab) + r"[^|]*\|([^|]*)\|([^|]*)\|", it).groups()]
check("v3_robot", "state workspace p95 L ego/robot", "0.221/0.395", "/".join(row("L state workspace p95")), src)
check("v3_robot", "state workspace p95 R ego/robot", "0.214/0.369", "/".join(row("R state workspace p95")), src)
check("v3_robot", "k8 translation p95 ego / robot", "93.1 / 99.2|91.0 / 99.9", "|".join(row("k8 translation p95")), src)
check("v3_robot", "both arms stationary k8 ego/robot", "19.5%/2.4%", "/".join(row("Both arms stationary at k8")), src)
sh = J(V + "robotized/wrist_sharpness_val.json")["p5_p50_p95"]; src = V + "robotized/wrist_sharpness_val.json"
for key, val in (("ego_raw_val/left_wrist", 1854), ("ego_raw_val/right_wrist", 1487), ("robot_R312c_sample/left_wrist", 59), ("robot_R312c_sample/right_wrist", 64),
                 ("robot100_val/left_wrist", 181), ("robot100_val/right_wrist", 132)):
    check("v3_robot", f"wrist sharpness p50 {key}", val, sh[key][1], src, tol=0.5)
for d, p in (("robot100", 1.0), ("mix70", 0.7)):
    for s_ in ("train", "val"):
        inf = J(V + f"robotized/info_{d}_{s_}.json"); check("v3_robot", f"{d} {s_} frames", 48411 if s_ == "train" else 5671, inf["total_frames"], V + f"robotized/info_{d}_{s_}.json")

# ---------------------------------------------------------------- v3: soft-prompt slots
fp = J(V + "domain_slots/lineage_comparison.json")["fingerprint"]["orig"]; src = V + "domain_slots/lineage_comparison.json"
trained = sorted(d["slot"] for d in fp["enc.bias"] if not d["all_zero"])
check("v3_slots", "trained slots (non-zero enc.bias)", "10-17", f"{trained[0]}-{trained[-1]}" if trained == list(range(trained[0], trained[-1] + 1)) else str(trained), src)
check("v3_slots", "number of slots", 30, len(fp["enc.bias"]), src)
pr = {int(a): float(b) for a, b in re.findall(r"slot\s+(\d+): mean loss ([0-9.]+)", open(R / V / "domain_slots/domain_probe_log_excerpt.txt").read())}
for sl, val in ((0, 1.11), (6, 1.12), (15, 1.61), (10, 2.35), (16, 2.65), (17, 4.07), (11, 22.80)):
    check("v3_slots", f"init loss slot {sl}", val, pr[sl], V + "domain_slots/domain_probe_log_excerpt.txt", tol=0.005)

# ---------------------------------------------------------------- v3: transfer diagnostics
dc = open(R / V / "diagnostics/direction_cosine_ego40k_vs_robot40k.txt").read().split("## ")
k8 = {blk.split("_")[0][:3]: [float(x) for x in re.findall(r"k8 cos med ([+-][0-9.]+)", blk)] for blk in dc[1:]}
src = V + "diagnostics/direction_cosine_ego40k_vs_robot40k.txt"
check("v3_transfer", "robot 40k k8 cos L/R", "+0.81/+0.77", "/".join(f"{x:+.2f}" for x in k8["R31"]), src)
check("v3_transfer", "ego-only 40k k8 cos L/R", "-0.05/-0.03", "/".join(f"{x:+.2f}" for x in k8["EGO"]), src)
fl = [float(x) for x in re.findall(r"cos L ([+-][0-9.]+)\s+R ([+-][0-9.]+)", open(R / V / "diagnostics/direction_cosine_flip_swap.txt").read()) for x in x]
check("v3_transfer", "flip/swap variants", 24, len(fl), V + "diagnostics/direction_cosine_flip_swap.txt")
check("v3_transfer", "flip/swap best cosine", 0.16, max(fl), V + "diagnostics/direction_cosine_flip_swap.txt")
ab = {re.match(r"(\S+ \w+ img \+ \w+ state)", l).group(1): [float(x) for x in re.findall(r"k\d+ ([+-][0-9.]+)\(", l)] for l in open(R / V / "diagnostics/image_state_ablation_ego40k.txt") if "img +" in l}
src = V + "diagnostics/image_state_ablation_ego40k.txt"
egoimg = [x for k, v in ab.items() if "ego img" in k for x in v]; robimg = [x for k, v in ab.items() if "robot img" in k for x in v]
egonon = [x for k, v in ab.items() if "ego img" in k and "ego state" not in k for x in v]
check("v3_transfer", "ego img + ego state range", "0.84-0.94", f"{min(ab['1 ego img + ego state']):.2f}-{max(ab['1 ego img + ego state']):.2f}", src)
check("v3_transfer", "ego img + other state range", "0.66-0.90", f"{min(egonon):.2f}-{max(egonon):.2f}", src)
check("v3_transfer", "ego img + robot state range", "0.69-0.90", f"{min(ab['3 ego img + robot state']):.2f}-{max(ab['3 ego img + robot state']):.2f}", src)
check("v3_transfer", "robot img + any state range", "-0.25-0.49", f"{min(robimg):.2f}-{max(robimg):.2f}", src)
lc = lambda n: {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(R / V / f"loss_curves/{n}.csv"))}
A, B = lc("A60_scratch_R312c_150kdecay"), lc("B60_egoinit_R312c_150kdecay"); src = V + "loss_curves/{A60,B60}*.csv"
check("v3_transfer", "step-200 loss scratch", 0.945, A[200], src); check("v3_transfer", "step-200 loss ego init", 0.244, B[200], src)
check("v3_transfer", "step-200 ratio", 3.9, A[200] / B[200], src, tol=0.05)
win = lambda lo, hi: [k for k in sorted(set(A) & set(B)) if lo < k <= hi]
for (lo, hi), val in (((0, 5000), 0.49), ((5000, 20000), 0.875), ((20000, 40000), 0.95), ((55000, 60000), 0.984)):
    s_ = win(lo, hi); check("v3_transfer", f"loss ratio B/A {lo // 1000}k-{hi // 1000}k", val, sum(B[k] for k in s_) / sum(A[k] for k in s_), src, tol=0.0015)
s_ = win(55000, 60000)
check("v3_transfer", "55-60k means A/B", "0.0745/0.0733", f"{sum(A[k] for k in s_) / len(s_):.4f}/{sum(B[k] for k in s_) / len(s_):.4f}", src)
check("v3_transfer", "55-60k points B lower", "13 of 24", f"{sum(B[k] < A[k] for k in s_)} of {len(s_)}", src)
check("v3_transfer", "B60 stop step", 59800, max(B), src)
last = lambda n: max(lc(n))
check("v3_runs", "co-train reached step", 88600, last("COTRAIN_R312c_ROBOT100_4p4"), V + "loss_curves/COTRAIN_R312c_ROBOT100_4p4.csv")
check("v3_runs", "MIX70 reached step", 5000, last("MIX70_pretrain_stopped"), V + "loss_curves/MIX70_pretrain_stopped.csv")
check("v3_runs", "ROBOT100 pretrain step (snapshot)", 74000, last("ROBOT100_pretrain_300k_running"), V + "loss_curves/ROBOT100_pretrain_300k_running.csv")
check("v3_runs", "HRA step (snapshot)", 100000, last("HRA_rightonly_300k_running"), V + "loss_curves/HRA_rightonly_300k_running.csv")
check("v3_runs", "HRA loss at snapshot", 0.019, lc("HRA_rightonly_300k_running")[100000], V + "loss_curves/HRA_rightonly_300k_running.csv")
ds = open(R / V / "loss_curves/dataset_sizes_from_logs.txt").read()
check("v3_runs", "R312c episodes/frames", "312/174012", "/".join(re.search(r"r312c-relonly\S*: dataset.num_episodes=(\d+) dataset.num_frames=(\d+)", ds).groups()), V + "loss_curves/dataset_sizes_from_logs.txt")
hw = list(csv.DictReader(open(R / V / "hardware_cycles_by_ckpt.csv"))); src = V + "hardware_cycles_by_ckpt.csv"
check("v3_robot_runs", "checkpoints executed", 71, len(hw), src)
check("v3_robot_runs", "control cycles", 49431, sum(int(h["cycles"]) for h in hw), src)
check("v3_robot_runs", "run starts", 1681, sum(int(h["starts"]) for h in hw), src)
hh = [h for h in hw if h["checkpoint"].startswith("HRA")]
check("v3_robot_runs", "HRA cycles", 8397, sum(int(h["cycles"]) for h in hh), src)
check("v3_robot_runs", "HRA starts", 658, sum(int(h["starts"]) for h in hh), src)

# ---------------------------------------------------------------- v3: HRA_red
qc = [json.load(open(p)) for p in sorted((R / V / "hra_red/scale_qc_per_episode").glob("*.json"))]; src = V + "hra_red/scale_qc_per_episode/"
check("v3_hra", "recorded takes", 200, len(qc), src)
check("v3_hra", "IMU scale median", 0.012, st.median(q["s_imu"] for q in qc), src, tol=0.0005)
check("v3_hra", "IMU scale <= 0", 95, sum(q["s_imu"] <= 0 for q in qc), src)
check("v3_hra", "cube edge m", 0.038, qc[0]["cube_edge_m"], src)
check("v3_hra", "scale-valid", 181, sum(bool(q["valid"]) for q in qc), src)
rej = J(V + "hra_red/scale_qc_summary_v0_181.json")["rejected"]; src = V + "hra_red/scale_qc_summary_v0_181.json"
check("v3_hra", "too few PnP frames", 14, sum("valid PnP frames" in v for v in rej.values()), src)
check("v3_hra", "spread too large", 5, sum(v.startswith("bootstrap spread") for v in rej.values()), src)
sr = [json.loads(l)["reject_reason"] for l in open(R / V / "hra_red/sanity_rejected.jsonl")]; src = V + "hra_red/sanity_rejected.jsonl"
check("v3_hra", "sanity rejects", 15, len(sr), src)
check("v3_hra", "travel-ratio-only rejects", 10, sum(r.startswith("travel_ratio") and ";" not in r for r in sr), src)
check("v3_hra", "s>1 and travel>0.8 rejects", 3, sum("outside" in r and "> 0.8 m" in r for r in sr), src)
sm = J(V + "hra_red/scale_qc_summary.json"); src = V + "hra_red/scale_qc_summary.json"
check("v3_hra", "accepted", 166, sm["accepted"], src)
check("v3_hra", "train episodes/rows", "151/20319", f"{sm['train_episodes']}/{sm['train_rows']}", src)
check("v3_hra", "val episodes/rows", "15/2011", f"{sm['val_episodes']}/{sm['val_rows']}", src)
check("v3_hra", "s_pnp p5/p50/p95", "0.188/0.354/0.5", "/".join(str(x) for x in sm["s_pnp_p5_p50_p95"]), src)
check("v3_hra", "centre residual p50 cm", 0.17, sm["centre_resid_cm_p50"], src)
hv = J(V + "hra_red/hrl_val_1face.json"); ok = [h for h in hv if h["valid"]]; rr = sorted(h["s_pnp"] / h["s_imu"] for h in ok); src = V + "hra_red/hrl_val_1face.json"
pct = lambda a, q: float(__import__("numpy").percentile(a, q))
check("v3_hra", "HRL80 PnP-valid of episodes", "37/57", f"{len(ok)}/{len(hv)}", src)
check("v3_hra", "HRL80 PnP/IMU median", 1.062, st.median(rr), src, tol=0.0005)
check("v3_hra", "HRL80 PnP/IMU p16", 0.959, pct(rr, 16), src, tol=0.0005)
check("v3_hra", "HRL80 PnP/IMU p84", 1.274, pct(rr, 84), src, tol=0.0005)
vl = J(V + "hra_red/val_loss_results.json")["015000"]; src = V + "hra_red/val_loss_results.json"
check("v3_hra", "val loss 15k", 0.150, vl["val_mean"], src, tol=0.0005)
check("v3_hra", "train-subset loss 15k", 0.066, vl["train_seed0"]["mean_batch_loss"], src, tol=0.0005)
check("v3_hra", "partial val loss 30k seed 0", 0.198, float(re.search(r"\[030000\] val seed 0: \{'mean_batch_loss': ([0-9.]+)", open(R / V / "hra_red/val_loss_run_log_excerpt.txt").read()).group(1)), V + "hra_red/val_loss_run_log_excerpt.txt", tol=0.0005)
rs3 = J(V + "RUN_STATUS_v3.json")
check("v3_runs", "run status entries", 11, len(rs3["runs"]), V + "RUN_STATUS_v3.json")

# ---------------------------------------------------------------- retired values must not reappear in the report
tex = open(R / "report/report.tex").read()
for bad in ("16.5\\%", "20.8/23.8", "versus 0.09", "five early", "6.1 and 5.6", "128\\textdegree"):
    check("text", f"retired value absent: {bad}", "True", str(bad not in tex), "report/report.tex")
check("text", "public repo URL present", "True", str("github.com/physical-ai-vla/egocentric-human-demos-rebot" in tex), "report/report.tex")
for need in ("211.7k", "$-33\\%$", "$+13\\%$", "36.5\\%", "56.1\\%", "58.9\\%", "14.7", "996 segments", "4.6/17.0", "18.5/23.8", "versus 0.03", "ten early", "6.0 and 5.6", "0.24 versus 0.07", "39.9", "3.55", "233 train", "561"):
    check("text", f"value present: {need}", "True", str(need in tex), "report/report.tex")

for need in ("48{,}411", "$-6.20$\\%", "57{,}283 (51.4\\%)", "1{,}482 of the 2{,}598", "422.5 to 90.5", "99.7\\% (left) and 99.5\\%", "72.3\\%/74.6\\%",
             "45.2\\%/52.2\\%", "1{,}854/1{,}487", "59/64", "slots 10--17", "22.80", "$+0.81$/$+0.77$", "$-0.05$/$-0.03$", "$+0.16$",
             "0.69--0.90", "$-0.25$ and $+0.49$", "0.945 for A and 0.244", "3.9$\\times$", "0.49 over the first 5k", "(0.0733 versus 0.0745",
             "181/132", "88.6k", "71\ncheckpoints, 49{,}431", "median of 0.012", "95 of them", "37 of 57", "1.062 (p16--p84 0.959--1.274",
             "151 train episodes (20{,}319 rows)", "0.354 (p5 0.188, p95 0.500)", "0.150", "0.198", "8{,}397", "version 3"):
    check("text", f"v3 value present: {need[:40]}", "True", str(need in tex), "report/report.tex")
# ---------------------------------------------------------------- README: the same verified values, and none of the stale ones
rd = open(R / "README.md").read()
for need in ("349 recorded", "284 pass sync", "259 source", "561 bimanual", "233 train", "26 held-out", "1.5 / 5.7 mm", "11.2 / 39.9 mm",
             "4.8 mm", "3.55 of 6", "36.5 %", "56.1 %", "1.494 / 1.551", "0.272 / 0.454", "58.9 %", "0.067 vs 0.144", "996 TR segments",
             "896 / val 100", "27,776", "−10.0 %", "−14.7 to −5.6 %", "8 of 10", "211.7k", "−33 %", "+13 %", "4.6 / 17.0 %", "11.3 mm"):
    check("readme", f"value present: {need}", "True", str(need in rd), "README.md")
for need in ("48,411 rows (−6.20 %)", "1,482 / 2,598", "99.7 / 99.5 %", "45.2 / 52.2 %", "1,854 / 1,487", "slots 10–17", "−0.05 / −0.03",
             "+0.81 / +0.77", "0.244 vs 0.945", "0.984 (55–60k, tie)", "1.062 (p16–p84 0.959–1.274)", "151 episodes / 20,319 rows", "49,431"):
    check("readme", f"v3 value present: {need}", "True", str(need in rd), "README.md")
for bad in ("is running; no results yet", "234 checks", "This repository reports no closed-loop robot results"):
    check("readme", f"stale text absent: {bad}", "True", str(bad not in rd), "README.md")

# ---------------------------------------------------------------- output
nf = sum(r[-1] == "FAIL" for r in rows)
md = ["| section | claim | report | recomputed | evidence | status |", "|---|---|---|---|---|---|"]
md += [f"| {a} | {b} | {c} | {d} | `{e}` | {f} |" for a, b, c, d, e, f in rows]
md.append(f"\n{len(rows)} claims checked, {nf} failed.")
out = "\n".join(md)
if "--md" in sys.argv:
    open(sys.argv[sys.argv.index("--md") + 1], "w").write(out + "\n")
for r in rows:
    if r[-1] == "FAIL": print("FAIL:", r)
print(f"{len(rows)} claims checked, {nf} failed.")
sys.exit(1 if nf else 0)
