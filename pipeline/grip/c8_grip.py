#!/usr/bin/env python3
"""[2026-09-23] C8 gripper width for all 698 sides (349 episodes), Track A contract §16.
main_calibrated sessions: the STORED anchors of gripper_contract.json (never recomputed).
early_precalibration sessions: the same rule applied fresh (closed = p10 / p90 of per-episode closed extremes, open = closed
+ direction * device span 1136.6 L / 1248.2 R), flagged anchor_source = derived_early.
Frame mapping: val frame i = stream frame (N_stream - n_val + i), verified by CORI intervals == capture_ns intervals
(max |diff| < 1 ms), otherwise the side is REJECTED (census reason grip_frame_mapping).
Regression: for sides whose episode is in handumi_grip_corrected (the 104 reference), the new width must equal the legacy
width_m at the same video_frame. Env: ~/xvla-mac. Writes ~/c8/grip/<tag>.npz, grip_census.json.
"""
import collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path.home() / "umi_bridge"))
from extract_grip import read_episode, align, SIDE_STREAM
H = pathlib.Path.home(); C8 = H / "c8"; OUT = C8 / "grip"; OUT.mkdir(exist_ok=True)
RAW = H / "ego_collector/datasets/human_handumi_raw/Hpilot"; LEG = H / ".claude/jobs/e649aef3/tmp/handumi_grip_corrected"
CON = json.load(open(LEG / "gripper_contract.json")); APER = CON["aperture_m"]; SPAN = {"left": 1136.6, "right": 1248.2}
STORED = {(a["session"], a["side"]): (a["closed_raw"], a["open_raw"], a["direction"]) for a in CON["anchors"]}
man = json.load(open(C8 / "c8_domain_manifest.json"))
eps = sorted({v["episode"] for v in man.values()})
raw, census = {}, {}
for e in eps:
    sess, ep = e.rsplit("_", 1)
    r = read_episode(sess, ep)
    for side, stream in SIDE_STREAM.items():
        tag = f"{e}_{side}"; exp = C8 / ("export" if man[tag]["domain"] == "main_calibrated" else "export_early") / tag
        cts = np.array([c["cts"] for c in json.load(open(exp / "imu_data.json"))["1"]["streams"]["CORI"]["samples"]])
        if r is None: census[tag] = dict(ok=False, reason="no_mcap"); continue
        fr = sorted(r[0][stream]); N, n = len(fr), len(cts); off = N - n
        cap = np.array([x[1] for x in fr], float)
        err = float(np.abs(np.diff(cts) - np.diff(cap[off:]) / 1e6).max()) if off >= 0 and len(cap[off:]) == n else np.inf
        if not err < 1.0: census[tag] = dict(ok=False, reason="grip_frame_mapping", err_ms=err); continue
        al = align(fr[off:], r[1][side])
        if al is None: census[tag] = dict(ok=False, reason="no_gripper_stream"); continue
        raw[tag] = dict(sess=sess, side=side, raw=al["raw"].astype(float), dt=al["dt_ms"], vf=np.array([x[0] for x in fr[off:]]), n=n)
        census[tag] = dict(ok=True, offset=off, interval_match_ms=err, align_dt_ms_p95=float(np.percentile(al["dt_ms"], 95)))
anchors = {}
for sess in sorted({v["sess"] for v in raw.values()}):
    for side in ("left", "right"):
        if (sess, side) in STORED:
            cl, op, sg = STORED[(sess, side)]; anchors[(sess, side)] = (cl, op, sg, "stored_contract"); continue
        meta = json.load(open(RAW / f"Hpilot_{sess}" / "session_meta.json"))["devices"][f"gripper_{side}"]
        sg = 1 if meta["ticks_open"] > meta["ticks_closed"] else -1
        ex = [(v["raw"].min() if sg > 0 else v["raw"].max()) for v in raw.values() if v["sess"] == sess and v["side"] == side]
        cl = float(np.percentile(ex, 10 if sg > 0 else 90)); anchors[(sess, side)] = (cl, cl + sg * SPAN[side], sg, "derived_early")
reg = collections.Counter()
for tag, v in raw.items():
    cl, op, sg, src = anchors[(v["sess"], v["side"])]
    f = np.clip((v["raw"] - cl) / (op - cl), 0, 1); w = (f * APER).astype(np.float32)
    np.savez(OUT / f"{tag}.npz", val_frame=np.arange(v["n"]), width_m=w, open_fraction=f.astype(np.float32), raw=v["raw"], dt_ms=v["dt"], video_frame=v["vf"])
    census[tag].update(anchor_source=src, width_p95_mm=float(np.percentile(w, 95) * 1e3))
    lf = LEG / f"{tag.rsplit('_', 1)[0]}.npz"
    if lf.is_file():
        L = np.load(lf); lvf, lw = L[f"{v['side']}_video_frame"], L[f"{v['side']}_width_m"]
        pos = np.searchsorted(lvf, v["vf"]); okm = (pos < len(lvf)) & (lvf[np.clip(pos, 0, len(lvf) - 1)] == v["vf"])
        same = okm.all() and np.array_equal(lw[pos[okm]], w)
        reg["identical" if same else "different"] += 1; census[tag]["legacy_width_identical"] = bool(same)
json.dump(dict(anchors={f"{k[0]}_{k[1]}": dict(closed=a[0], open=a[1], direction=a[2], source=a[3]) for k, a in anchors.items()},
               sides=census, regression=dict(reg)), open(OUT / "grip_census.json", "w"), indent=1, default=float)
bad = collections.Counter(c["reason"] for c in census.values() if not c["ok"])
by_dom = collections.Counter((man[t]["domain"], c["ok"]) for t, c in census.items())
print(f"sides {len(census)}  ok {sum(c['ok'] for c in census.values())}  rejects {dict(bad)}  by domain {dict(by_dom)}")
print(f"regression vs legacy handumi_grip_corrected: {dict(reg)}")
