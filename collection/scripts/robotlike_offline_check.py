#!/usr/bin/env python3
"""[2026-09-28] robot_like_v1 OFFLINE check: would the reBot be able to do what the human just did?

For an episode whose two wrists already have MASt3R-SLAM runs (<runs>/<tag>/ss1.csv) and ORB-SLAM exports
(<export>/<tag>/{raw_video.mp4, imu_data.json, orbslam_setting.yaml}), this reuses the FROZEN C8 Phase-3 chain from
~/c8/c8_phase3.py function by function -- nonfinite_pose_invalid_v1, video_to_cori_index_v1, imu_vi scale + degenerate
guard, camera_tcp_v2, left-master |dt| < 20 ms sync, 65-frame hard-break segments, r150_active_median re-anchor,
R150 workspace NN, continuity IK (C4 IKCfg) -- and adds, per segment:

  joint margin   min distance of every IK joint to its URDF limit (deg); j2/j3 upper = 0 is the normal operating
                 regime of this arm, so that bound is reported but not gated (same rule as ego_ik_retarget.py)
  collision      c8_collision_v1 link-segment clearance on the IK solution, every 5th row; d_min < 0.075 m = hard
                 fail, < 0.194 m = warn (contract §24 frozen D_HARD / D_SAFE)
  dynamics       robot-space target TCP at 15 Hz, probe_v2_motion.dyn definitions, vs R150 p95

The gripper width is NOT needed for any of these, so the grip census is bypassed: the video->CORI offset is
recomputed as N_video_frames - n_CORI (verified equal to grip_census.json on 8/8 sampled C8 sides, 2026-09-28).

Old C8 episodes are only READ; results go to --out (default: <episode>/derived/robot_like/offline_check.json for new
episodes). Env: ~/xvla-mac/bin/python (pyroki / jax via eef_kin).

usage: robotlike_offline_check.py --episode <raw episode dir> [...] --runs <dir> --export <dir> [--out <dir>]
       robotlike_offline_check.py --session <raw session dir> --runs <dir> --export <dir>
"""
import argparse, collections, json, pathlib, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

H = pathlib.Path.home()
sys.path.insert(0, str(H / "c8"))
import c8_phase3 as P3                                                           # noqa: E402  (frozen chain, reused as is)
from c8_collision_v1 import Clearance                                           # noqa: E402

D_HARD, D_SAFE = 0.075, 0.194
LIM = np.array([[-2.8, 2.8], [-3.14, 0.0], [-3.14, 0.0], [-1.87, 1.57], [-1.57, 1.57], [-3.14, 3.14]])  # URDF rad, per arm
GATED_UPPER = np.array([True, False, False, True, True, True])                  # j2/j3 upper 0 = normal regime, not gated
MARGIN_WARN_DEG = 5.0
R150 = json.load(open(H / "umi_bridge/track_c/reports/probe_v2_motion.json"))["tcp_dyn"]["r150"]
FPS15 = 15.0


def tag_of(ep: pathlib.Path, side: str) -> str:
    """C8 tag convention: <session timestamp>_<episode number>_<side> (session dir = <Dataset>_<timestamp>)."""
    return f"{ep.parent.name.split('_', 1)[1]}_{ep.name.split('_', 1)[1]}_{side}"


def dyn(T):
    """probe_v2_motion.dyn for one segment at 15 Hz (T: (n, 4, 4) robot-space TCP)."""
    p = T[:, :3, 3]; d = np.diff(p, axis=0) * FPS15
    v = np.linalg.norm(d, axis=1); a = np.linalg.norm(np.diff(d, axis=0), axis=1) * FPS15
    om = np.array([Rot.from_matrix(T[i, :3, :3].T @ T[i + 1, :3, :3]).as_rotvec() for i in range(len(T) - 1)]) * FPS15
    w = np.linalg.norm(om, axis=1); al = np.linalg.norm(np.diff(om, axis=0), axis=1) * FPS15
    return dict(lin_v=v, lin_a=a, ang_w=w, ang_a=al)


def side_nogrip(tag: str, ep: pathlib.Path, side: str) -> dict:
    """c8_phase3.side() without the gripper width (offset recomputed from the raw stream counts)."""
    exp = P3.export_dir(tag)
    n = len(json.load(open(exp / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
    nv = len(P3.frame_meta(ep)[f"{side}_wrist"])
    P3.GRIP[tag] = dict(offset=nv - n)
    nonfin, off = P3.stage_side(tag)
    T, ok = P3.cp6["load_traj"](P3.STAGE / "runs" / tag / "ss1.csv", n)
    vi = P3.cp6["imu_vi"](tag, P3.W_VI, P3.BIAS_VI)
    s = vi["s"] if vi else float("nan")
    sv = bool(vi and np.isfinite(s) and s > 0 and P3.G_BAND[0] <= vi["g_norm"] <= P3.G_BAND[1] and P3.S_GUARD[0] <= s <= P3.S_GUARD[1])
    Tm = T.copy(); Tm[:, :3, 3] *= (s if np.isfinite(s) else 1.0); Tt = Tm @ P3.cp6["X"]; Tt[~ok] = np.eye(4)
    return dict(T_tcp=Tt, ok=ok, n=n, off=off, s=s, scale_valid=sv, g_norm=vi["g_norm"] if vi else None)


def _pct(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; return None if not len(x) else round(float(np.percentile(x, p)), 3)


def margins(q12):
    """per arm: min margin (deg) to a gated limit over the rows, and share of rows within MARGIN_WARN_DEG of one."""
    out = []
    for a in (0, 1):
        q = q12[:, 6 * a:6 * a + 6]
        lo = np.degrees(q - LIM[:, 0]); hi = np.degrees(LIM[:, 1] - q); hi = np.where(GATED_UPPER, hi, np.inf)
        m = np.minimum(lo, hi); worst = np.unravel_index(np.argmin(m), m.shape)
        out.append(dict(min_margin_deg=round(float(m.min()), 2), worst_joint=f"j{worst[1] + 1}",
                        near_limit_rows_frac=round(float((m.min(axis=1) < MARGIN_WARN_DEG).mean()), 4)))
    return out


def _rate(segs, key):
    return round(float(np.mean([bool(s[key]) for s in segs])), 4) if segs else None


def check_episode(ep: pathlib.Path, clr: Clearance) -> dict:
    tags = {s: tag_of(ep, s) for s in ("left", "right")}
    rec = dict(episode=str(ep), tags=tags, schema="robot_like_offline/v1")
    missing = [t for t in tags.values() if not (P3.RUNS / t / "ss1.csv").is_file() or not (P3.export_dir(t) / "imu_data.json").is_file()]
    if missing: rec.update(verdict="PENDING", first_fail="mast3r_missing", missing=missing); return rec
    S = {s: side_nogrip(tags[s], ep, s) for s in ("left", "right")}
    for s in ("left", "right"): rec[s] = dict(scale=S[s]["s"], scale_valid=S[s]["scale_valid"], g_norm=S[s]["g_norm"], coverage=round(float(S[s]["ok"].mean()), 4))
    fm = P3.frame_meta(ep); L, R = S["left"], S["right"]
    capL = np.array([c for _, c in fm["left_wrist"][L["off"]:L["off"] + L["n"]]], float)
    capR = np.array([c for _, c in fm["right_wrist"][R["off"]:R["off"] + R["n"]]], float); capH = np.array([c for _, c in fm["head"]], float)
    jR, jH = P3.nearest(capR, capL), P3.nearest(capH, capL)
    dtR, dtH = np.abs(capR[jR] - capL) / 1e6, np.abs(capH[jH] - capL) / 1e6
    row_ok = L["ok"] & R["ok"][jR] & (dtR < P3.DT_MS) & (dtH < P3.DT_MS)
    idx = np.flatnonzero(row_ok); runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1) if len(idx) else []
    segs = [r[k:k + P3.SEG] for r in runs for k in range(0, len(r) - P3.SEG + 1, P3.SEG)]
    rec.update(rows=int(L["n"]), row_ok=int(row_ok.sum()), n_segments=len(segs))
    bv = [L["scale_valid"], R["scale_valid"]]; seg_rec = []; D = collections.defaultdict(list)
    for s_ in segs:
        r = dict(start=int(s_[0]))
        if not any(bv): r.update(first_fail="B_metric"); seg_rec.append(r); continue
        TL, TR = L["T_tcp"][s_], R["T_tcp"][jR[s_]]
        tgL, tgR = P3.rt.targets(TL, 0), P3.rt.targets(TR, 1)
        ws = [float(P3.ws_in(tgL, 0).mean()), float(P3.ws_in(tgR, 1).mean())]
        res = P3.ik.solve(tgL, tgR); qn = np.where(np.isfinite(res["q"]), res["q"], res["q_last_good"])
        ik_ok = [bool(np.isin(res["status"][:, a], P3.P.SUCCESS).all()) for a in (0, 1)]
        dmins = [clr.dmin(qn[i])[0] for i in range(0, len(qn), 5)]; dmin = float(min(dmins))
        mg = margins(qn)
        for a, T_ in ((0, tgL[::2]), (1, tgR[::2])):
            for k, v in dyn(T_).items(): D[(a, k)].append(v)
        ff = ("B_metric" if not all(bv) else "C_workspace" if not all(w >= 0.95 for w in ws) else "D_IK" if not all(ik_ok)
              else "F_collision" if dmin < D_HARD else None)
        pe, re_ = res["pos_err"], res["rot_err"]                           # FK(IK q) vs target, mm / deg, per row x arm
        fk = [dict(pos_mm_p50=_pct(pe[:, a], 50), pos_mm_p90=_pct(pe[:, a], 90), rot_deg_p50=_pct(re_[:, a], 50), rot_deg_p90=_pct(re_[:, a], 90)) for a in (0, 1)]
        for a in (0, 1): D[(a, "fk_pos")].append(pe[:, a][np.isfinite(pe[:, a])]); D[(a, "fk_rot")].append(re_[:, a][np.isfinite(re_[:, a])])
        r.update(ws_pass=all(w >= 0.95 for w in ws), ik_pass=all(ik_ok), collision_pass=dmin >= D_HARD, fk_residual=fk,
                 ws_in=[round(w, 4) for w in ws], ik_ok=ik_ok, status=[dict(collections.Counter(res["status"][:, a].tolist())) for a in (0, 1)],
                 d_min_m=round(dmin, 4), collision_warn=dmin < D_SAFE, joint_margin=mg, first_fail=ff)
        seg_rec.append(r)
    ok = [s for s in seg_rec if s["first_fail"] is None]
    rec["segments"] = seg_rec
    rec["funnel_first_fail"] = dict(collections.Counter(s["first_fail"] for s in seg_rec))
    rec["robot_feasible_frac"] = round(len(ok) / len(seg_rec), 4) if seg_rec else 0.0
    dyn_s = {}
    for a, arm in ((0, "left"), (1, "right")):
        d = {}
        for k in ("lin_v", "lin_a", "ang_w", "ang_a"):
            if D[(a, k)]:
                x = np.concatenate(D[(a, k)]); p95 = float(R150[k][1])
                d[k] = dict(p50=round(float(np.percentile(x, 50)), 4), p95=round(float(np.percentile(x, 95)), 4), r150_p95=round(p95, 4),
                            over_r150_p95_frac=round(float((x > p95).mean()), 4))
        dyn_s[arm] = d
    rec["dynamics_vs_r150"] = dyn_s
    judged = [s for s in seg_rec if "ws_pass" in s]
    rec["pass_rates"] = dict(n=len(judged), workspace=_rate(judged, "ws_pass"), ik=_rate(judged, "ik_pass"), collision=_rate(judged, "collision_pass"),
                             full=round(len(ok) / len(seg_rec), 4) if seg_rec else None)
    rec["fk_residual"] = {arm: dict(pos_mm_p50=_pct(np.concatenate(D[(a, "fk_pos")]), 50) if D[(a, "fk_pos")] else None,
                                    pos_mm_p90=_pct(np.concatenate(D[(a, "fk_pos")]), 90) if D[(a, "fk_pos")] else None,
                                    rot_deg_p50=_pct(np.concatenate(D[(a, "fk_rot")]), 50) if D[(a, "fk_rot")] else None,
                                    rot_deg_p90=_pct(np.concatenate(D[(a, "fk_rot")]), 90) if D[(a, "fk_rot")] else None)
                          for a, arm in ((0, "left"), (1, "right"))}
    if seg_rec:
        rec["collision_min_m"] = round(min(s["d_min_m"] for s in seg_rec if "d_min_m" in s), 4) if any("d_min_m" in s for s in seg_rec) else None
        mm = [s["joint_margin"][a]["min_margin_deg"] for s in seg_rec if "joint_margin" in s for a in (0, 1)]
        rec["joint_margin_min_deg"] = min(mm) if mm else None
    rec["verdict"] = "PASS" if rec["robot_feasible_frac"] >= 0.5 else "FAIL" if seg_rec else "FAIL"
    rec["first_fail"] = None if seg_rec else "no_segment"
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episode", action="append", default=[]); ap.add_argument("--session")
    ap.add_argument("--runs", required=True, help="dir with <tag>/ss1.csv (MASt3R-SLAM)"); ap.add_argument("--export", required=True, help="dir with <tag>/imu_data.json")
    ap.add_argument("--out", default=None, help="write <episode tag>.json here instead of <episode>/derived/robot_like/")
    a = ap.parse_args(argv)
    eps = [pathlib.Path(e) for e in a.episode]
    if a.session: eps += sorted(p for p in pathlib.Path(a.session).glob("episode_*") if (p / ".complete").exists())
    P3.RUNS = pathlib.Path(a.runs); exp_root = pathlib.Path(a.export); P3.export_dir = lambda tag: exp_root / tag
    P3.init(); clr = Clearance(P3.ik.kin); summary = []
    for ep in eps:
        try: rec = check_episode(ep, clr)
        except Exception as exc: rec = dict(episode=str(ep), verdict="ERROR", error=f"{type(exc).__name__}: {exc}")
        dst = (pathlib.Path(a.out) / f"{tag_of(ep, 'x')[:-2]}.json") if a.out else (ep / "derived" / "robot_like" / "offline_check.json")
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(json.dumps(rec, indent=1, default=float))
        summary.append(rec)
        print(f"{ep.parent.name}/{ep.name}: {rec['verdict']:7s} feasible {rec.get('robot_feasible_frac', '-')}  segs {rec.get('n_segments', '-')}"
              f"  funnel {rec.get('funnel_first_fail', rec.get('first_fail'))}  d_min {rec.get('collision_min_m', '-')} m"
              f"  margin {rec.get('joint_margin_min_deg', '-')} deg" + (f"  {rec.get('error', '')}" if rec["verdict"] == "ERROR" else ""), flush=True)
    return summary


if __name__ == "__main__":
    main()
