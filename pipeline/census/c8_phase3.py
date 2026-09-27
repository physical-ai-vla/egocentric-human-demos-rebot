#!/usr/bin/env python3
"""[2026-09-24] C8 Phase 3: CPU postprocess of the 5080 MASt3R-SLAM runs -> segment census + pseudo q, per episode.

Per side (rules frozen in contract §21, applied in this order, all named):
  nonfinite_pose_invalid_v1   valid_pose = (is_lost == false) AND all_finite(x, y, z, q)
  video_to_cori_index_v1      MASt3R frame_idx is the VIDEO frame; cori_idx = video_frame - offset (c8_grip frame mapping)
  imu_vi (C-P6 frozen)        cp6_scale.imu_vi EXACTLY (exec of the frozen source), SG window 15, no bias
  metric TCP (C-P5)           T_world_tcp = (T_WC with t * s) @ X, X = handumi_camera_tcp_v2
  gripper                     ~/c8/grip/<tag>.npz width_m (Track A §16), CORI-indexed
Per episode (LEFT wrist = master clock, V3 pairing): right / head frame = nearest capture_ns, |dt| < 20 ms.
  row_ok[i] = valid_L[i] AND valid_R[jR[i]] AND dtR < 20 ms AND dtH < 20 ms
  Segments = non-overlapping 65 consecutive row_ok frames, tail dropped. SEGMENT LENGTH (explicit, never re-derive):
    65 rows at 30 fps INCLUDING the anchor row  ==  33 rows at 15 Hz (even frames 0, 2, ..., 64)  ==  anchor + 32 future steps.
  Every
  invalid row (lost, NaN, unsynced) is a HARD BREAK -- no segment spans it.
  Each segment re-anchored at r150_active_median, continuity IK (C4 frozen IKCfg) at 30 fps.
Segment funnel (first fail, mutually exclusive; fixed before any Phase-3 result was seen):
  A_candidates     65-frame run as above (sync <= 20 ms, video_to_cori_index_v1, valid_pose, lost/NaN hard break)
  B_metric         both sides: imu_vi returned, s finite > 0, |g| in [8.81, 10.81], AND imu_vi_degenerate_guard_v1:
                   s in [0.1 x min, 10 x max] of the ArUco GT scales of the 28-side benchmark = [0.0305, 9.18] (added
                   2026-09-24 after the dry run found s = 1.4e-4 with |g| 9.71 on a coverage-0.40 side: |g| is solved
                   independently of s, so the |g| band cannot see a degenerate scale; the bound comes from the benchmark
                   GT only, never from C8 values) -> metric TCP
  C_workspace      BOTH arms >= 95 % rows inside the R150 workspace (pos NN < 50 mm AND ori NN-50 < 20 deg)
  D_IK             BOTH arms: every row status in P.SUCCESS (continuity IK)
  E_pseudo_q       state32 (s32) all finite
  Every stage is judged PER ARM (cumulative: an arm credited at a stage passed all earlier ones). PRIMARY (bimanual) =
  both arms pass; SECONDARY = >= 1 arm passes B..E (the V3 per-arm mask rule). A side with a degenerate / invalid scale
  therefore drops only its own arm from the secondary yield. No residual threshold is used (resid_rms is reported only).
  Thresholds FROZEN 2026-09-24 (incl. imu_vi_degenerate_guard_v1 [0.0305, 9.18], a catastrophic-degeneracy guard, NOT a
  scale-quality gate).
Episode first-fail (before segments): sensor/gripper/sync from census_pre.json, then mast3r_missing, then no_segment.
Writes ~/c8/phase3/<episode>.npz (per usable-or-not segment: frames, status, q, state32, gates) and ~/c8/phase3_census.json.
Env: ~/xvla-mac/bin/python (mcap + eef_kin, same env as build_v3_dataset / census_pre).  Args: [episode ...] (default: every episode of the manifest with both sides run).
"""
import collections, json, os, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); C8 = H / "c8"; TA = H / "umi_bridge/trackA_mast3r_pose_v1"; TC = H / "umi_bridge/track_c"
RUNS = pathlib.Path(os.environ.get("C8_RUNS", C8 / "runs_m3")); STAGE = C8 / "stage_p3"; OUT = C8 / "phase3"; OUT.mkdir(exist_ok=True)
RAW = H / "ego_collector/datasets/human_handumi_raw/Hpilot"
MAN = json.load(open(C8 / "c8_domain_manifest.json")); GRIP = json.load(open(C8 / "grip/grip_census.json"))["sides"]
PRE = json.load(open(C8 / "census_pre.json"))
SEG, DT_MS, G_BAND = 65, 20.0, (8.81, 10.81)
S_GUARD = (0.1 * 0.3049, 10 * 0.9178)                                     # imu_vi_degenerate_guard_v1 (benchmark s_gt min / max)
W_VI, BIAS_VI = 15, False


NF_IDX = {}                                                                   # tag -> CORI indices made invalid by the NaN rule


def export_dir(tag): return C8 / ("export" if MAN[tag]["domain"] == "main_calibrated" else "export_early") / tag


def stage_side(tag):
    """nonfinite_pose_invalid_v1 + video_to_cori_index_v1 into STAGE/runs/<tag>/ss1.csv; returns (n_nonfinite, offset)."""
    a, b = tag.split("_", 1); dst = STAGE / "val" / a / b; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): dst.symlink_to(export_dir(tag))
    src = RUNS / tag / "ss1.csv"; out = STAGE / "runs" / tag / "ss1.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    off = GRIP[tag]["offset"]; L = open(src).read().splitlines(); hdr = L[0].split(","); fi = hdr.index("frame_idx"); il = hdr.index("is_lost")
    pc = [hdr.index(k) for k in ("x", "y", "z", "q_x", "q_y", "q_z", "q_w")]; rows = [L[0]]; nonfin = 0; NF_IDX[tag] = []
    for line in L[1:]:
        c = line.split(",")
        if c[il] == "false" and not all(np.isfinite(float(c[k])) for k in pc):
            c[il] = "true"; nonfin += 1; NF_IDX[tag].append(int(c[fi]) - off)
        j = int(c[fi]) - off
        if j >= 0: c[fi] = str(j); rows.append(",".join(c))
    out.write_text("\n".join(rows) + "\n")
    return nonfin, off


def init():
    global cp6, P, ik, rt, TREE, WR, build_state32
    (STAGE / "qa").mkdir(parents=True, exist_ok=True)
    for s_, d_ in ((TA / "data/qa/qa_all.json", STAGE / "qa/qa_all.json"), (TA / "data/handumi_camera_tcp_v2.yaml", STAGE / "handumi_camera_tcp_v2.yaml")):
        if not d_.exists():
            try: d_.symlink_to(s_)
            except FileExistsError: pass
    os.environ["M3_BASE"] = str(STAGE); os.environ["M3_VARIANT"] = "ss1"
    cp6 = {}; _s = open(TA / "cp6_scale.py").read(); exec(_s[:_s.index("tags = [")], cp6)
    sys.path.insert(0, str(TC)); import pseudo_joint_pipeline as P_; from state32 import build_state32 as b32
    P, build_state32 = P_, b32
    cfg = P.IKCfg(); ik = P.ContinuityIK(cfg); rt = P.Retarget(ik.kin, P.ANCHORS[cfg.anchor])
    d150 = np.load(TC / "data/r150_q_tcp.npz"); ix = np.random.default_rng(0).choice(len(d150["J"]), 6000, replace=False); ref = {0: [], 1: []}
    for i in ix:
        for a, p7 in enumerate(ik.kin.fk_pose7(np.radians(np.r_[d150["J"][i, :6], d150["J"][i, 7:13]]))): ref[a].append(p7)
    ref = {a: np.array(v) for a, v in ref.items()}; TREE = {a: cKDTree(ref[a][:, :3]) for a in (0, 1)}; WR = {a: Rot.from_quat(ref[a][:, 3:]).as_matrix() for a in (0, 1)}


def ws_in(T, a):                                                              # == build_v3_dataset.ws_in
    dp, ii = TREE[a].query(T[:, :3, 3], k=50)
    do = np.array([np.degrees(np.linalg.norm(Rot.from_matrix(WR[a][ii[t]].transpose(0, 2, 1) @ T[t, :3, :3]).as_rotvec(), axis=1)).min() for t in range(len(T))])
    return (dp[:, 0] < 0.05) & (do < 20)


def frame_meta(epdir):
    from mcap.reader import make_reader
    fr = {"left_wrist": [], "right_wrist": [], "head": []}
    with open(epdir / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            if ch.topic.endswith("/frame_meta"):
                s = ch.topic.split("/")[1]
                if s in fr: d = json.loads(msg.data); fr[s].append((int(d["video_frame"]), int(d["capture_ns"])))
    return {k: sorted(v) for k, v in fr.items()}


def nearest(cap, ref):
    j = np.clip(np.searchsorted(cap, ref), 1, len(cap) - 1); return np.where(np.abs(cap[j - 1] - ref) < np.abs(cap[j] - ref), j - 1, j)


def side(tag):
    nonfin, off = stage_side(tag)
    n = len(json.load(open(export_dir(tag) / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
    T, ok = cp6["load_traj"](STAGE / "runs" / tag / "ss1.csv", n)             # CORI-indexed, is_lost rows (incl. NaN) invalid
    vi = cp6["imu_vi"](tag, W_VI, BIAS_VI)
    s = vi["s"] if vi else float("nan")
    sv = bool(vi and np.isfinite(s) and s > 0 and G_BAND[0] <= vi["g_norm"] <= G_BAND[1] and S_GUARD[0] <= s <= S_GUARD[1])
    Tm = T.copy(); Tm[:, :3, 3] *= (s if np.isfinite(s) else 1.0); Tt = Tm @ cp6["X"]; Tt[~ok] = np.eye(4)
    g = np.load(C8 / "grip" / f"{tag}.npz"); assert len(g["width_m"]) == n and g["video_frame"][0] == off, tag
    nfm = np.zeros(n, bool); nfm[[j for j in NF_IDX[tag] if 0 <= j < n]] = True
    return dict(T_tcp=Tt, ok=ok, n=n, off=off, s=s, scale_valid=sv, imu_vi=vi, nonfinite_pose_rows=nonfin, nonfinite_mask=nfm, width=g["width_m"])


def episode(e):
    dom = MAN[f"{e}_left"]["domain"]; pre = PRE[e]; rec = dict(episode=e, domain=dom)
    if pre["first_fail"] is not None: rec["first_fail"] = pre["first_fail"]; return rec
    if not all((RUNS / f"{e}_{s}" / "ss1.csv").is_file() for s in ("left", "right")): rec["first_fail"] = "mast3r_missing"; return rec
    S = {s: side(f"{e}_{s}") for s in ("left", "right")}
    for s in ("left", "right"):
        rec[f"{s}"] = dict(scale=S[s]["s"], scale_valid=S[s]["scale_valid"], g_norm=S[s]["imu_vi"]["g_norm"] if S[s]["imu_vi"] else None, resid_rms=S[s]["imu_vi"]["resid_rms"] if S[s]["imu_vi"] else None,
                          coverage=float(S[s]["ok"].mean()), nonfinite_pose_rows=S[s]["nonfinite_pose_rows"], nonfinite_pose_side=S[s]["nonfinite_pose_rows"] > 0, offset=S[s]["off"])
    sess, ep = e.rsplit("_", 1); fm = frame_meta(RAW / f"Hpilot_{sess}" / f"episode_{ep}")
    L, R = S["left"], S["right"]
    capL = np.array([c for _, c in fm["left_wrist"][L["off"]:L["off"] + L["n"]]], float)
    capR = np.array([c for _, c in fm["right_wrist"][R["off"]:R["off"] + R["n"]]], float); capH = np.array([c for _, c in fm["head"]], float)
    jR, jH = nearest(capR, capL), nearest(capH, capL)
    dtR, dtH = np.abs(capR[jR] - capL) / 1e6, np.abs(capH[jH] - capL) / 1e6
    row_ok = L["ok"] & R["ok"][jR] & (dtR < DT_MS) & (dtH < DT_MS)
    idx = np.flatnonzero(row_ok); runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1) if len(idx) else []
    segs = [r[k:k + SEG] for r in runs for k in range(0, len(r) - SEG + 1, SEG)]
    rec.update(rows=int(L["n"]), row_ok=int(row_ok.sum()), runs_ge65=int(sum(len(r) >= SEG for r in runs)),
               rows_lost_to_nonfinite=int((L["nonfinite_mask"] | R["nonfinite_mask"][jR]).sum()),
               rows_lost_to_pose=int((~(L["ok"] & R["ok"][jR]) & ~(L["nonfinite_mask"] | R["nonfinite_mask"][jR])).sum()), rows_lost_to_sync=int(((L["ok"] & R["ok"][jR]) & ~((dtR < DT_MS) & (dtH < DT_MS))).sum()))
    bv = [L["scale_valid"], R["scale_valid"]]; seg_rec = []; store = collections.defaultdict(list)   # B_metric per ARM
    STAGES = ("B_metric", "C_workspace", "D_IK", "E_pseudo_q")
    for s_ in segs:
        r = dict(start=int(s_[0])); TL, TR = L["T_tcp"][s_], R["T_tcp"][jR[s_]]
        if not any(bv):
            r.update(arm_pass={k: [False, False] for k in STAGES}, first_fail="B_metric", arm_first_fail=["B_metric", "B_metric"], one_arm_usable=False)
            seg_rec.append(r); continue
        tgL, tgR = rt.targets(TL, 0), rt.targets(TR, 1)
        ws = [float(ws_in(tgL, 0).mean()), float(ws_in(tgR, 1).mean())]
        res = ik.solve(tgL, tgR); qn = np.where(np.isfinite(res["q"]), res["q"], res["q_last_good"])
        s32, _ = build_state32([TL @ P.F[None], TR @ P.F[None]], [L["width"][s_].astype(np.float32), R["width"][jR[s_]].astype(np.float32)], [qn[:, :6], qn[:, 6:]])
        # per-arm stage flags, cumulative (an arm that failed B is not credited later); E = that arm's 16 state32 columns finite
        c_ = [ws[a] >= 0.95 for a in (0, 1)]; d_ = [bool(np.isin(res["status"][:, a], P.SUCCESS).all()) for a in (0, 1)]
        e_ = [bool(np.isfinite(s32[:, 16 * a:16 * a + 16]).all()) for a in (0, 1)]
        ap = {"B_metric": list(bv)}; ap["C_workspace"] = [ap["B_metric"][a] and c_[a] for a in (0, 1)]
        ap["D_IK"] = [ap["C_workspace"][a] and d_[a] for a in (0, 1)]; ap["E_pseudo_q"] = [ap["D_IK"][a] and e_[a] for a in (0, 1)]
        arm_ff = [next((k for k in STAGES if not ap[k][a]), None) for a in (0, 1)]
        r.update(ws_in=ws, arm_pass=ap, arm_first_fail=arm_ff, s32_finite=bool(np.isfinite(s32).all()),
                 ik_success_rows=[int(np.isin(res["status"][:, a], P.SUCCESS).sum()) for a in (0, 1)],
                 status=[dict(collections.Counter(res["status"][:, a].tolist())) for a in (0, 1)],
                 first_fail=next((k for k in STAGES if not all(ap[k])), None),               # PRIMARY: both arms
                 one_arm_usable=any(ap["E_pseudo_q"]))                                       # SECONDARY: >= 1 arm
        seg_rec.append(r)
        store["start"].append(s_[0]); store["q"].append(qn.astype(np.float32)); store["state32"].append(s32.astype(np.float32))
        store["status"].append(res["status"]); store["arm_usable"].append(ap["E_pseudo_q"]); store["usable"].append(r["first_fail"] is None)
    if store: np.savez_compressed(OUT / f"{e}.npz", **{k: np.array(v) for k, v in store.items()}, seg_len=SEG, jR=jR, jH=jH)
    rec["segments"] = seg_rec; rec["first_fail"] = None if segs else "no_segment"
    return rec


def _init_worker(): init()


if __name__ == "__main__":
    eps = sys.argv[1:] or sorted({v["episode"] for v in MAN.values()})
    init()
    for e in eps:                                                             # staging is serial (symlinks, csv rewrite)
        for s in ("left", "right"):
            if (RUNS / f"{e}_{s}" / "ss1.csv").is_file() and PRE[e]["first_fail"] is None: stage_side(f"{e}_{s}")
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2), initializer=_init_worker) as ex:
        recs = dict(zip(eps, ex.map(episode, eps)))
    out = dict(schema="c8_phase3/v1", date="2026-09-24", rules=["nonfinite_pose_invalid_v1", "video_to_cori_index_v1", "imu_vi W15 no-bias",
               "camera_tcp_v2", "left-master |dt|<20ms", "65-frame hard-break segments", "r150_active_median re-anchor", "C4 IKCfg", "ws>=95% both arms"],
               provenance=P.provenance(), episodes=recs)
    json.dump(out, open(C8 / ("phase3_census.json" if len(sys.argv) == 1 else "phase3_census_partial.json"), "w"), indent=1, default=float)
    STG = ["A_candidates", "B_metric", "C_workspace", "D_IK", "E_pseudo_q"]; FF = {"B_metric": 1, "C_workspace": 2, "D_IK": 3, "E_pseudo_q": 4}
    for dom in ("main_calibrated", "early_precalibration", "all"):
        v = [r for r in recs.values() if dom == "all" or r["domain"] == dom]
        ef = collections.Counter(r["first_fail"] for r in v)
        segs = [s for r in v for s in r.get("segments", [])]; alive = [len(segs)] + [0] * 4
        for k in range(1, 5): alive[k] = sum(1 for s in segs if s["first_fail"] is None or FF[s["first_fail"]] > k)
        one = sum(1 for s in segs if s.get("one_arm_usable"))
        sec = [len(segs)] + [sum(1 for s in segs if any(s.get("arm_pass", {}).get(k, [False, False]))) for k in STG[1:]]
        nf = sum(1 for r in v for s in ("left", "right") if r.get(s, {}).get("nonfinite_pose_side"))
        print(f"{dom:<22} episodes {len(v)}  episode first-fail {dict((k, c) for k, c in ef.items() if k)}  with segments {ef.get(None, 0)}  nonfinite sides {nf}")
        print(" " * 23 + "Bimanual production yield (both arms)  " + "  ".join(f"{n} {a}" for n, a in zip(STG, alive)))
        print(" " * 23 + "Single-arm salvage yield (>=1 arm)     " + "  ".join(f"{n} {a}" for n, a in zip(STG, sec)))
        E = [r for r in v if "rows" in r]; rs = lambda k: sum(r[k] for r in E); tot = rs("rows") if E else 0
        segf = collections.Counter(s["first_fail"] for s in segs if s["first_fail"])
        print(" " * 23 + f"drop reasons | episodes: sync {ef.get('sync_pass', 0)}  sensor {ef.get('sensor_complete', 0)}  gripper {ef.get('gripper_valid', 0)}"
              f"  mast3r_missing {ef.get('mast3r_missing', 0)}  no_segment {ef.get('no_segment', 0)}"
              f" | rows (of {tot}): lost {rs('rows_lost_to_pose') if E else 0}  nonfinite {rs('rows_lost_to_nonfinite') if E else 0}  unsynced {rs('rows_lost_to_sync') if E else 0}"
              f" | segments (both-arms first fail): metric/scale {segf.get('B_metric', 0)}  workspace {segf.get('C_workspace', 0)}  IK {segf.get('D_IK', 0)}  pseudo_q {segf.get('E_pseudo_q', 0)}")
        wsf = [s["arm_first_fail"] for s in segs if s["first_fail"] == "C_workspace"]
        wb = sum(1 for f in wsf if f == ["C_workspace", "C_workspace"]); wl = sum(1 for f in wsf if f[0] == "C_workspace" and f[1] != "C_workspace")
        wr = sum(1 for f in wsf if f[1] == "C_workspace" and f[0] != "C_workspace")
        print(" " * 23 + f"workspace fails {len(wsf)}: one arm only {wl + wr} (left {wl} / right {wr}), both arms {wb}"
              + (f"  -> one-arm share {100 * (wl + wr) / len(wsf):.1f} %" if wsf else ""))
