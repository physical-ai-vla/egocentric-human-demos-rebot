#!/usr/bin/env python3
"""[2026-09-23] C8 Phase-1 regression: reference = 104 episodes / 208 sides, MASt3R-SLAM ss1 on the RTX 5080.
Not numeric equality with the legacy ORB poses -- trajectory-quality criteria, fixed BEFORE looking at the results:
  R1 completeness   208/208 ss1.csv, rc 0, no MEM_STOP, csv rows == raw_video frames, frame_idx 0..n-1, and
                    n_video == n_CORI + offset (offset = c8_grip frame mapping: CORI starts `offset` stream frames late)
  index mapping     video_to_cori_index_v1: cori_idx = video_frame - offset. MASt3R csv frame_idx is the VIDEO frame; the
                    IMU (CORI/GYRO/ACCL) and the gripper are CORI-indexed. Benchmark valB/valC had offset 0, C8 has 1..4.
                    cp6_scale.imu_vi reads frame_idx as the CORI index, so it gets a staged re-indexed copy
                    (stage/runs/<tag>/ss1.csv, frames < offset dropped, nothing else changed). The gyro check is also
                    run WITHOUT the mapping (diagnostic): its best lag must then equal the offset.
  validity rule     nonfinite_pose_invalid_v1: a row with any non-finite x/y/z/q (MASt3R-SLAM can diverge to NaN while
                    still reporting TRACKING / is_lost=false; qa_compare.load lets NaN through) counts as invalid. Applied
                    to a staged copy (stage/runs_video, then re-indexed to stage/runs); counted per side (nonfinite_pose_rows, nonfinite_pose_side).
  R2 width          208/208 grip width length == frames, and identical to legacy handumi_grip_corrected (c8_grip.py)
  R3 coverage       MASt3R valid-frame fraction median >= 0.95, and >= the legacy ORB coverage on >= 90 % of sides
  R4 continuity     reversal rate (qa_compare.tcp_metrics, 15 Hz, > 5 mm steps) median <= ORB median; zigzag bursts
                    and step p99 reported (IMU-VI metric scale)
  R5 gyro           pose-derived body angular velocity vs the wrist gyro (independent sensor, rotation is scale-free):
                    vector cosine p50 >= 0.9 on >= 95 % of sides, best lag within +-1 frame on >= 95 %, no axis with
                    negative correlation (catches frame / sign convention flips)
  R6 IMU-VI scale   per side, cp6_scale.imu_vi EXACTLY (exec of the frozen source, W 15, no bias); newly registered,
                    no GT here; |g| in [8.81, 10.81] on >= 90 % of sides. Reported (not a PASS criterion, added
                    2026-09-24): sides failing imu_vi_degenerate_guard_v1 (s outside [0.0305, 9.18], benchmark GT x0.1 / x10)
Functions are exec'd from the frozen sources (cp6_scale.py up to the FIT/EVAL section, qa_compare.py up to the run loop)
on a staging dir of symlinks (~/c8/stage), so nothing is re-implemented. Writes ~/c8/regress_ref208.json.
Env: ~/handumi-sw/.venv/bin/python. Inputs synced from the NFS: ~/c8/runs_m3/<tag>/ss1.{csv,json}, <tag>.log, <tag>.vram.
"""
import collections, json, os, pathlib, sys
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); C8 = H / "c8"; TA = H / "umi_bridge/trackA_mast3r_pose_v1"; STAGE = C8 / "stage"
RUNS = pathlib.Path(os.environ.get("C8_RUNS", C8 / "runs_m3")); PRODUCE = H / "umi_bridge/track_c/data/produce"
TAGS = sys.argv[1:] or open(C8 / "ref104ep_208sides.txt").read().split()
GRIP = json.load(open(C8 / "grip/grip_census.json"))["sides"]
MAN = json.load(open(C8 / "c8_domain_manifest.json"))
EXP = lambda t: C8 / ("export" if MAN[t]["domain"] == "main_calibrated" else "export_early") / t   # rest490 includes early sides

NONFIN = {}
# ---- staging dir: the layout cp6_scale / qa_compare expect, as symlinks (no copies, no edits)
STAGE = pathlib.Path(os.environ.get("C8_STAGE", STAGE)); (STAGE / "qa").mkdir(parents=True, exist_ok=True)
for src, dst in ((TA / "data/qa/qa_all.json", STAGE / "qa/qa_all.json"), (TA / "data/handumi_camera_tcp_v2.yaml", STAGE / "handumi_camera_tcp_v2.yaml")):
    if not dst.exists(): dst.symlink_to(src)
for t in TAGS:
    a, b = t.split("_", 1)
    dst = STAGE / "val" / a / b; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): dst.symlink_to(EXP(t))
    src = RUNS / t / "ss1.csv"; dst = STAGE / "runs" / t / "ss1.csv"; dst.parent.mkdir(parents=True, exist_ok=True)
    dv = STAGE / "runs_video" / t / "ss1.csv"; dv.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        off = GRIP[t]["offset"]; L = open(src).read().splitlines(); hdr = L[0].split(","); fi = hdr.index("frame_idx")
        pc = [hdr.index(k) for k in ("x", "y", "z", "q_x", "q_y", "q_z", "q_w")]; il = hdr.index("is_lost"); outv, out = [L[0]], [L[0]]; NONFIN[t] = 0
        for line in L[1:]:
            c = line.split(",")
            if not all(np.isfinite(float(c[k])) for k in pc) and c[il] == "false": c[il] = "true"; NONFIN[t] += 1   # nonfinite_pose_invalid_v1
            outv.append(",".join(c)); j = int(c[fi]) - off                                                     # video_to_cori_index_v1
            if j >= 0: c[fi] = str(j); out.append(",".join(c))
        dv.write_text("\n".join(outv) + "\n"); dst.write_text("\n".join(out) + "\n")
os.environ["M3_BASE"] = str(STAGE); os.environ["M3_VARIANT"] = "ss1"
cp6, qa = {}, {}
_s = open(TA / "cp6_scale.py").read(); exec(_s[:_s.index("tags = [")], cp6)
_s = open(TA / "qa_compare.py").read(); exec(_s[:_s.index("res = {}")], qa)
W_VI, BIAS_VI = 15, False                                                    # frozen C-P6 choice (valB FIT)


def orb(tag, n):
    z = np.load(PRODUCE / f"{tag}.npz"); T = np.tile(np.eye(4), (n, 1, 1)); ok = np.zeros(n, bool)
    valid = np.linalg.norm(z["cam_quat"], axis=1) > 0.5
    for i, f in enumerate(z["frame_idx"]):
        if valid[i] and 0 <= f < n:
            T[f, :3, :3] = Rot.from_quat(z["cam_quat"][i]).as_matrix(); T[f, :3, 3] = z["cam_pos"][i]; ok[f] = True
    return T, ok


def gyro_check(T, ok, tag):
    """T, ok indexed by CORI frame (pass the unmapped video-indexed arrays only for the diagnostic)."""
    d = json.load(open(EXP(tag) / "imu_data.json"))["1"]["streams"]
    tc = np.array([s["cts"] for s in d["CORI"]["samples"]]) / 1e3
    ti = np.array([s["cts"] for s in d["GYRO"]["samples"]]) / 1e3; w = np.array([s["value"] for s in d["GYRO"]["samples"]], float)
    R_bc = cp6["T_b_c"](EXP(tag) / "orbslam_setting.yaml")[:3, :3]
    n = min(len(tc), len(T)); wp = np.full((n - 1, 3), np.nan); wg = np.full((n - 1, 3), np.nan)
    for i in range(n - 1):
        sel = (ti > tc[i]) & (ti <= tc[i + 1])
        if sel.any(): wg[i] = w[sel].mean(0)
        if ok[i] and ok[i + 1]:
            wp[i] = R_bc @ (Rot.from_matrix(T[i, :3, :3].T @ T[i + 1, :3, :3]).as_rotvec() / (tc[i + 1] - tc[i]))
    def corr_at(L):
        a, b = (wp[L:], wg[:len(wg) - L]) if L >= 0 else (wp[:L], wg[-L:])
        m = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        if m.sum() < 30: return None
        return np.array([np.corrcoef(a[m, k], b[m, k])[0, 1] for k in range(3)]), a[m], b[m]
    lags = {L: corr_at(L) for L in range(-5, 6)}
    lags = {L: v for L, v in lags.items() if v is not None}
    if 0 not in lags: return None
    best = max(lags, key=lambda L: np.nanmean(lags[L][0]))
    c0, a, b = lags[0]; mv = np.linalg.norm(b, axis=1) > 0.3
    cos = np.einsum("ij,ij->i", a[mv], b[mv]) / (np.linalg.norm(a[mv], axis=1) * np.linalg.norm(b[mv], axis=1) + 1e-12)
    return dict(axis_corr=c0.round(4).tolist(), best_lag=int(best), vec_cos_p50=float(np.median(cos)) if mv.any() else float("nan"),
                gain_p50=float(np.median(np.linalg.norm(a[mv], axis=1) / np.linalg.norm(b[mv], axis=1))) if mv.any() else float("nan"),
                n_moving=int(mv.sum()))


rec = {}
for t in TAGS:
    r = dict(); rec[t] = r
    n = len(json.load(open(EXP(t) / "imu_data.json"))["1"]["streams"]["CORI"]["samples"]); r["n_cori"] = n
    off = GRIP[t]["offset"]; r["offset"] = off
    nv = int(cv2.VideoCapture(str(EXP(t) / "raw_video.mp4")).get(cv2.CAP_PROP_FRAME_COUNT)); r["n_video"] = nv
    csvp = STAGE / "runs_video" / t / "ss1.csv"; vr = RUNS / f"{t}.vram"; lg = RUNS / f"{t}.log"
    tok = vr.read_text().split() if vr.is_file() else []; r["vram"] = dict(zip(tok[::2], map(int, tok[1::2]))) if tok else None
    r["present"] = (RUNS / t / "ss1.csv").is_file(); r["nonfinite_pose_rows"] = NONFIN.get(t, 0); r["nonfinite_pose_side"] = r["nonfinite_pose_rows"] > 0
    if not r["present"]: continue
    L = lg.read_text() if lg.is_file() else ""
    r["reloc_attempts"] = L.count("RELOCALIZING"); r["reloc_failed"] = L.count("Failed to relocalize")
    T, ok, step = qa["load"](csvp)
    rows = sum(1 for _ in open(csvp)) - 1; r["rows_match"] = rows == nv == len(ok) and step == 1 and nv == n + off
    Tv, okv = T, ok; T, ok = T[off:off + n], ok[off:off + n]                 # video_to_cori_index_v1
    g = np.load(C8 / "grip" / f"{t}.npz"); r["width_len_match"] = len(g["width_m"]) == n and np.array_equal(g["video_frame"], np.arange(n) + off)
    r["width_legacy_identical"] = bool(GRIP[t].get("legacy_width_identical"))
    r["coverage"] = float(ok.mean()); r["kf"] = int(json.load(open(RUNS / t / "ss1.json")).get("n_keyframes", -1)) if (RUNS / t / "ss1.json").is_file() else -1
    vi = cp6["imu_vi"](t, W_VI, BIAS_VI); r["imu_vi"] = vi
    s = vi["s"] if vi else float("nan")
    r["tcp"] = qa["tcp_metrics"](T, ok, s, 1); r["gyro"] = gyro_check(T, ok, t); r["gyro_unmapped"] = gyro_check(Tv[:n], okv[:n], t)
    if (PRODUCE / f"{t}.npz").is_file():
        To, oko = orb(t, n)
        r["orb"] = dict(coverage=float(oko.mean()), tcp=qa["tcp_metrics"](To, oko, 1.0, 1), gyro=gyro_check(To, oko, t))
    print(t, f"cov {r['coverage']:.2f}", f"s {s:.4f} |g| {vi['g_norm']:.2f}" if vi else "no-VI",
          f"rev {r['tcp'].get('reversal_rate', float('nan')):.3f}", f"gyro cos {r['gyro']['vec_cos_p50']:.3f} lag {r['gyro']['best_lag']} (unmapped lag {r['gyro_unmapped']['best_lag']}, off {off})" if r["gyro"] else "no-gyro",
          f"| ORB cov {r['orb']['coverage']:.2f} rev {r['orb']['tcp'].get('reversal_rate', float('nan')):.3f}" if "orb" in r else "| no ORB", flush=True)

P = [r for r in rec.values() if r["present"]]; has_orb = [r for r in P if "orb" in r]
res = collections.OrderedDict()
res["R1"] = dict(present=len(P), rows_match=sum(r["rows_match"] for r in P), rc0=sum(1 for r in P if r["vram"] and r["vram"]["rc"] == 0))
res["R1"]["PASS"] = res["R1"]["present"] == res["R1"]["rows_match"] == res["R1"]["rc0"] == len(TAGS)
res["R2"] = dict(len_match=sum(r["width_len_match"] for r in P), legacy_identical=sum(r["width_legacy_identical"] for r in P))
res["R2"]["PASS"] = res["R2"]["len_match"] == res["R2"]["legacy_identical"] == len(TAGS)
cov = np.array([r["coverage"] for r in P]); ge = [r["coverage"] >= r["orb"]["coverage"] for r in has_orb]
res["R3"] = dict(cov_p10=float(np.percentile(cov, 10)), cov_p50=float(np.median(cov)), n_below_0_9=int((cov < 0.9).sum()),
                 orb_cov_p50=float(np.median([r["orb"]["coverage"] for r in has_orb])) if has_orb else None,
                 ge_orb_frac=float(np.mean(ge)) if ge else None, n_with_orb=len(has_orb))
res["R3"]["PASS"] = res["R3"]["cov_p50"] >= 0.95 and (res["R3"]["ge_orb_frac"] or 0) >= 0.90
rv = [r["tcp"]["reversal_rate"] for r in P if "reversal_rate" in r["tcp"]]; rvo = [r["orb"]["tcp"]["reversal_rate"] for r in has_orb if "reversal_rate" in r["orb"]["tcp"]]
res["R4"] = dict(rev_p50=float(np.median(rv)), orb_rev_p50=float(np.median(rvo)) if rvo else None,
                 zigzag_total=int(sum(r["tcp"].get("zigzag_bursts", 0) for r in P)), orb_zigzag_total=int(sum(r["orb"]["tcp"].get("zigzag_bursts", 0) for r in has_orb)),
                 step_p99_mm_p50=float(np.median([r["tcp"]["step_p99_mm"] for r in P if "step_p99_mm" in r["tcp"]])))
res["R4"]["PASS"] = rvo != [] and res["R4"]["rev_p50"] <= res["R4"]["orb_rev_p50"]
G = [r["gyro"] for r in P if r["gyro"]]
res["R5"] = dict(n=len(G), cos_ge_0_9_frac=float(np.mean([g["vec_cos_p50"] >= 0.9 for g in G])), lag_within1_frac=float(np.mean([abs(g["best_lag"]) <= 1 for g in G])),
                 n_neg_axis=int(sum(min(g["axis_corr"]) < 0 for g in G)), cos_p50=float(np.nanmedian([g["vec_cos_p50"] for g in G])),
                 gain_p50=float(np.nanmedian([g["gain_p50"] for g in G])),
                 orb_cos_p50=float(np.nanmedian([r["orb"]["gyro"]["vec_cos_p50"] for r in has_orb if r["orb"]["gyro"]])) if has_orb else None)
res["R5"]["PASS"] = res["R5"]["cos_ge_0_9_frac"] >= 0.95 and res["R5"]["lag_within1_frac"] >= 0.95 and res["R5"]["n_neg_axis"] == 0
V = [r["imu_vi"] for r in P if r["imu_vi"]]; gn = np.array([v["g_norm"] for v in V])
res["R6"] = dict(n=len(V), g_in_band_frac=float(np.mean(np.abs(gn - 9.81) <= 1.0)), g_p50=float(np.median(gn)),
                 s_p10_p50_p90=np.percentile([v["s"] for v in V], [10, 50, 90]).round(4).tolist(),
                 degenerate_scale_sides={t: r["imu_vi"]["s"] for t, r in rec.items() if r.get("imu_vi") and not (0.0305 <= r["imu_vi"]["s"] <= 9.18)})
res["R6"]["PASS"] = res["R6"]["g_in_band_frac"] >= 0.90 and len(V) == len(TAGS)
vr = [r["vram"].get("tree_peak_MiB", r["vram"].get("proc_peak_MiB")) for r in P if r["vram"]]
res["vram"] = dict(tree_peak_max_MiB=max(vr), tree_peak_p50=float(np.median(vr)), n_ge_14GB=int(sum(v >= 14336 for v in vr)), gpu_peak_max_MiB=max(r["vram"].get("gpu_peak_MiB", 0) for r in P if r["vram"]),
                   sec_p50=float(np.median([r["vram"]["sec"] for r in P if r["vram"]])))
res["index_mapping"] = dict(name="video_to_cori_index_v1", offsets=dict(collections.Counter(r["offset"] for r in P)),
                            unmapped_lag_equals_offset=sum(1 for r in P if r.get("gyro_unmapped") and r["gyro_unmapped"]["best_lag"] == r["offset"]))
res["nonfinite"] = dict(sides=sum(r.get("nonfinite_pose_side", False) for r in P), rows=sum(r.get("nonfinite_pose_rows", 0) for r in P),
                       which={t: r["nonfinite_pose_rows"] for t, r in rec.items() if r.get("nonfinite_pose_rows", 0)})
res["reloc"] = dict(sides_with_reloc=sum(r["reloc_attempts"] > 0 for r in P), sides_reloc_failed=sum(r["reloc_failed"] > 0 for r in P))
res["ALL_PASS"] = all(res[k]["PASS"] for k in ("R1", "R2", "R3", "R4", "R5", "R6"))
json.dump(dict(summary=res, sides=rec), open(C8 / "regress_ref208.json", "w"), indent=1, default=float)
print("\n" + "\n".join(f"{k}: {v}" for k, v in res.items()))
