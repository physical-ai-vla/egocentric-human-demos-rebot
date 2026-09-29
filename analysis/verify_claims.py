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

# ---------------------------------------------------------------- retired values must not reappear in the report
tex = open(R / "report/report.tex").read()
for bad in ("16.5\\%", "20.8/23.8", "versus 0.09", "five early", "6.1 and 5.6", "128\\textdegree"):
    check("text", f"retired value absent: {bad}", "True", str(bad not in tex), "report/report.tex")
for need in ("36.5\\%", "56.1\\%", "58.9\\%", "14.7", "996 segments", "4.6/17.0", "18.5/23.8", "versus 0.03", "ten early", "6.0 and 5.6", "0.24 versus 0.07", "39.9", "3.55", "233 train", "561"):
    check("text", f"value present: {need}", "True", str(need in tex), "report/report.tex")

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
