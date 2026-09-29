#!/usr/bin/env python3
"""[2026-09-28] Does a live-robot-like human trajectory also pass the real robot constraints more often?

Four groups = {old C8 259, new HRL80} x {robot-like, not robot-like}, robot-like cutoff (user-confirmed 2026-09-28):
    worst-wrist share of 15 Hz ticks with |w| > R150 p95 (0.996 rad/s)  <= 10 %   AND   grasps >= 2
Angular acceleration is reported as a diagnostic only (both old and new exceed it; hand-held tremor).
The live judgement comes from robotlike_replay_live.py for BOTH old and new (same method; replay == live verdict/grasps
on 5/5 checked HRL80 episodes). Offline metrics come from robotlike_offline_check.py (segment level, 65-frame segments).

usage: robotlike_compare_groups.py --old-live <dir> --old-offline <dir> --new-live <dir> --new-session <raw session> [...]
"""
import argparse, glob, json, pathlib
import numpy as np

W_MAX, G_MIN = 0.10, 2


def robot_like(s):
    w = max(s["sides"][x]["over_frac_ang_w"] for x in ("left", "right")); return (w <= W_MAX and s["grasps_total"] >= G_MIN), w


def med(x):
    x = [v for v in x if v is not None]; return None if not x else round(float(np.median(x)), 3)


def pct(x, p):
    x = [v for v in x if v is not None]; return None if not x else round(float(np.percentile(x, p)), 3)


def group_stats(offl):
    segs = [s for r in offl for s in r.get("segments", []) if "ws_pass" in s]
    rate = lambda k: None if not segs else round(float(np.mean([bool(s[k]) for s in segs])), 3)
    full = None if not segs else round(float(np.mean([s["first_fail"] is None for s in segs])), 3)
    fkp = [s["fk_residual"][a][k] for s in segs for a in (0, 1) for k in ("pos_mm_p50",)]
    fk90 = [s["fk_residual"][a]["pos_mm_p90"] for s in segs for a in (0, 1)]
    fr50 = [s["fk_residual"][a]["rot_deg_p50"] for s in segs for a in (0, 1)]
    fr90 = [s["fk_residual"][a]["rot_deg_p90"] for s in segs for a in (0, 1)]
    marg = [s["joint_margin"][a]["min_margin_deg"] for s in segs for a in (0, 1)]
    near = [s["joint_margin"][a]["near_limit_rows_frac"] for s in segs for a in (0, 1)]
    dmin = [s["d_min_m"] for s in segs]
    pend = sum(1 for r in offl if r.get("verdict") in ("PENDING", "ERROR"))
    return dict(episodes=len(offl), episodes_pending_or_error=pend, segments=len(segs), workspace=rate("ws_pass"), ik=rate("ik_pass"),
                collision=rate("collision_pass"), full=full,
                fk_pos_mm_p50=med(fkp), fk_pos_mm_p90=med(fk90), fk_rot_deg_p50=med(fr50), fk_rot_deg_p90=med(fr90),
                joint_margin_min_deg_median=med(marg), near_limit_rows_frac_median=med(near),
                collision_dmin_m_median=med(dmin), collision_dmin_m_p10=pct(dmin, 10))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--old-live", required=True); ap.add_argument("--old-offline", required=True)
    ap.add_argument("--new-live", required=True); ap.add_argument("--new-session", action="append", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # old: live replay keyed "Hpilot_<sess>__episode_<n>.json", offline keyed "<sess>_<n>.json"
    old = []
    for p in glob.glob(f"{a.old_live}/*.json"):
        s = json.load(open(p)); sess, ep = pathlib.Path(p).stem.split("__"); eid = f"{sess.split('_', 1)[1]}_{ep.split('_', 1)[1]}"
        f = pathlib.Path(a.old_offline) / f"{eid}.json"; old.append((s, json.load(open(f)) if f.exists() else dict(verdict="PENDING")))
    new = []
    for p in glob.glob(f"{a.new_live}/*.json"):
        s = json.load(open(p)); sess, ep = pathlib.Path(p).stem.split("__")
        f = next((pathlib.Path(d) / ep / "derived/robot_like/offline_check.json" for d in a.new_session if pathlib.Path(d).name == sess), None)
        new.append((s, json.load(open(f)) if f is not None and f.exists() else dict(verdict="PENDING")))
    rows = {}
    for name, S in (("old", old), ("new", new)):
        for rl in (True, False):
            g = [(s, o) for s, o in S if robot_like(s)[0] == rl]
            st = group_stats([o for _, o in g]); st["live_rot_over_p95_median"] = med([robot_like(s)[1] for s, _ in g])
            st["live_rot_acc_over_p95_median"] = med([max(s["sides"][x]["over_frac_ang_a"] for x in ("left", "right")) for s, _ in g])
            rows[f"{name} {'robot-like' if rl else 'NOT robot-like'}"] = st
    out = dict(schema="robot_like_group_compare/v1", cutoff=dict(rot_over_p95_max=W_MAX, grasps_min=G_MIN, rot_acc="diagnostic only"), groups=rows)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    cols = ["episodes", "episodes_pending_or_error", "segments", "workspace", "ik", "collision", "full", "fk_pos_mm_p50", "fk_pos_mm_p90",
            "fk_rot_deg_p50", "fk_rot_deg_p90", "joint_margin_min_deg_median", "near_limit_rows_frac_median", "collision_dmin_m_median",
            "collision_dmin_m_p10", "live_rot_over_p95_median", "live_rot_acc_over_p95_median"]
    print(f"{'metric':32s}" + "".join(f"{k:>22s}" for k in rows))
    for c in cols: print(f"{c:32s}" + "".join(f"{str(rows[k][c]):>22s}" for k in rows))


if __name__ == "__main__":
    main()
