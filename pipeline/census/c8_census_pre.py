#!/usr/bin/env python3
"""[2026-09-23] C8 census, GPU-free stages, per EPISODE (349), main294 vs early55, mutually exclusive first-fail reasons:
   1 sensor_complete        sensors.mcap + both wrist videos + IMU both sides + head video
   2 calibration_applicable main = session after the 2026-09-16 11:51 calibration files; early = flagged, NOT rejected here
   3 gripper_valid          c8_grip.py produced width for both sides (frame mapping verified)
   4 sync_pass              left-master pairing with right wrist + head, |dt| < 20 ms, AND at least one 65-frame run
                            (the C7/V3 segment length) exists
Writes ~/c8/census_pre.json (per-episode record) and prints the funnel.
"""
import collections, json, pathlib, sys
import numpy as np
from mcap.reader import make_reader
H = pathlib.Path.home(); C8 = H / "c8"; RAW = H / "ego_collector/datasets/human_handumi_raw/Hpilot"
man = json.load(open(C8 / "c8_domain_manifest.json")); grip = json.load(open(C8 / "grip/grip_census.json"))["sides"]
eps = sorted({v["episode"]: v["domain"] for v in man.values()}.items())
SEG = 65; rec = {}


def frame_meta(epdir):
    fr = {"left_wrist": [], "right_wrist": [], "head": []}
    with open(epdir / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            if ch.topic.endswith("/frame_meta"):
                s = ch.topic.split("/")[1]
                if s in fr: d = json.loads(msg.data); fr[s].append((int(d["video_frame"]), int(d["capture_ns"])))
    return {k: sorted(v) for k, v in fr.items()}


for e, dom in eps:
    sess, ep = e.rsplit("_", 1); d = RAW / f"Hpilot_{sess}" / f"episode_{ep}"; r = dict(domain=dom)
    meta = json.load(open(d / "episode_meta.json")) if (d / "episode_meta.json").is_file() else {}
    r["sensor_complete"] = all((d / f).is_file() for f in ("sensors.mcap", "left_wrist.mp4", "right_wrist.mp4", "head.mp4")) and bool(meta.get("imu_present"))
    r["calibration_applicable"] = dom == "main_calibrated"; r["early_calibration_domain"] = dom != "main_calibrated"
    r["gripper_valid"] = all(grip.get(f"{e}_{s}", {}).get("ok") for s in ("left", "right"))
    if r["sensor_complete"]:
        fm = frame_meta(d)
        offL, offR = grip[f"{e}_left"].get("offset", 0), grip[f"{e}_right"].get("offset", 0)
        exp = C8 / ("export" if dom == "main_calibrated" else "export_early")
        nL = len(json.load(open(exp / f"{e}_left" / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
        nR = len(json.load(open(exp / f"{e}_right" / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
        capL = np.array([c for _, c in fm["left_wrist"][offL:offL + nL]], float); capR = np.array([c for _, c in fm["right_wrist"][offR:offR + nR]], float)
        capH = np.array([c for _, c in fm["head"]], float)
        near = lambda cap: (lambda j: np.where(np.abs(cap[j - 1] - capL) < np.abs(cap[j] - capL), j - 1, j))(np.clip(np.searchsorted(cap, capL), 1, len(cap) - 1))
        dtR = np.abs(capR[near(capR)] - capL) / 1e6; dtH = np.abs(capH[near(capH)] - capL) / 1e6
        good = (dtR < 20) & (dtH < 20); idx = np.flatnonzero(good)
        runs = [len(x) for x in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)] if len(idx) else []
        r.update(sync_good_frac=float(good.mean()), sync_longest_run=int(max(runs) if runs else 0),
                 sync_segments_possible=int(sum(L // SEG for L in runs)), dtR_p95=float(np.percentile(dtR, 95)), dtH_p95=float(np.percentile(dtH, 95)))
        r["sync_pass"] = r["sync_segments_possible"] > 0
    else:
        r["sync_pass"] = False
    stages = ["sensor_complete", "gripper_valid", "sync_pass"]
    r["first_fail"] = next((s for s in stages if not r[s]), None)
    rec[e] = r
json.dump(rec, open(C8 / "census_pre.json", "w"), indent=1, default=float)
for dom in ("main_calibrated", "early_precalibration", "all"):
    v = [r for r in rec.values() if dom == "all" or r["domain"] == dom]
    n = len(v); f = collections.Counter(r["first_fail"] for r in v)
    alive = n; line = f"{dom:<22} raw {n}"
    for s in ("sensor_complete", "gripper_valid", "sync_pass"):
        alive -= f.get(s, 0); line += f" -> {s} {alive}"
    segs = sum(r.get("sync_segments_possible", 0) for r in v if r["first_fail"] is None)
    print(line + f"   | 65-frame segments available {segs}   | first-fail reasons {dict((k, c) for k, c in f.items() if k)}")
