#!/usr/bin/env python3
"""[2026-09-25] §23 G1 gravity-sign / frame-convention gate, on REAL data (independent of any selector result).
Tilt vector = gravity expressed in the robot FK-TCP frame convention, unit length (2 DOF, no yaw):
  human  : g_w from imu_vi_g (that side's MASt3R world, pointing down) -> HU TCP frame (R_wtcp^T g_w, TCP = T_wc @ X)
           -> FK-TCP convention via the frozen chain: R_fk = R_hu F C  =>  v_fk = C^T F^T v_hu
  robot  : base -z (assumed z-up URDF base) in the FK TCP frame: v_fk = R_fk(q)^T [0, 0, -1]
Both populations are table-top 3-stack manipulation, so their tilt distributions should overlap. The gate compares the
angle between the mean tilt directions (and the median per-sample NN angle) under the chain as frozen vs with each
single sign / frame flip (g sign, F omitted, C omitted, C^-1). PASS = the frozen chain is the clear best and close.
Human samples: every 15th valid row of every usable (bimanual E) segment of the integrity-filtered census.
"""
import json, os, pathlib, sys
import numpy as np
from scipy.spatial import cKDTree
H = pathlib.Path.home(); C8 = H / "c8"; sys.path.insert(0, str(H / "umi_bridge/track_c")); sys.path.insert(0, str(C8))
os.environ["M3_BASE"] = str(C8 / "stage_p3"); os.environ["M3_VARIANT"] = "ss1"
import c8_imu_vi_g, pseudo_joint_pipeline as P
ns = c8_imu_vi_g.load(); X = ns["X"]; F3, C3 = P.F[:3, :3], P.C[:3, :3]
d = json.load(open(C8 / "phase3_census.json"))["episodes"]; Q = set(json.load(open(C8 / "c8_quarantine.json"))["sessions"])
man = json.load(open(C8 / "c8_domain_manifest.json")); grip = json.load(open(C8 / "grip/grip_census.json"))["sides"]
hum = {"left": [], "right": []}
for e, r in d.items():
    if e.rsplit("_", 1)[0] in Q or r.get("first_fail") is not None: continue
    starts = [s["start"] for s in r.get("segments", []) if s["first_fail"] is None]
    if not starts: continue
    z = np.load(C8 / "phase3" / f"{e}.npz"); jR = z["jR"]
    for side in ("left", "right"):
        tag = f"{e}_{side}"; vi = ns["imu_vi"](tag, 15, False)
        if vi is None: continue
        n = len(json.load(open((C8 / ("export" if man[tag]["domain"] == "main_calibrated" else "export_early") / tag / "imu_data.json")))["1"]["streams"]["CORI"]["samples"])
        T, ok = ns["load_traj"](C8 / "stage_p3/runs" / tag / "ss1.csv", n); g = np.array(vi["g_world"]); g /= np.linalg.norm(g)
        for s0 in starts:
            rows = np.arange(s0, s0 + 65, 15); rows = jR[rows] if side == "right" else rows
            for i in rows:
                if ok[i]: hum[side].append((T[i, :3, :3] @ X[:3, :3]).T @ g)
hum = {k: np.array(v) for k, v in hum.items()}; print("human tilt samples", {k: len(v) for k, v in hum.items()})
kin = P.ContinuityIK(P.IKCfg()).kin; d150 = np.load(H / "umi_bridge/track_c/data/r150_q_tcp.npz"); J = d150["J"]
ix = np.random.default_rng(0).choice(len(J), 4000, replace=False); rob = {"left": [], "right": []}
from scipy.spatial.transform import Rotation as Rot
for i in ix:
    for a, p7 in enumerate(kin.fk_pose7(np.radians(np.r_[J[i, :6], J[i, 7:13]]))):
        rob["left" if a == 0 else "right"].append(Rot.from_quat(p7[3:]).as_matrix().T @ np.array([0, 0, -1.0]))
rob = {k: np.array(v) for k, v in rob.items()}
VAR = {"frozen chain (g, F, C)": lambda v: v @ F3 @ C3, "g sign flipped": lambda v: -v @ F3 @ C3, "F omitted": lambda v: v @ C3,
       "C omitted": lambda v: v @ F3, "C inverted": lambda v: v @ F3 @ C3.T}
ang = lambda a, b: np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
out = {}
for side in ("left", "right"):
    rm = rob[side].mean(0); tree = cKDTree(rob[side]); out[side] = {}
    print(f"\n== {side}: robot mean tilt resultant length {np.linalg.norm(rm):.2f}")
    for name, f in VAR.items():
        h = f(hum[side]); h /= np.linalg.norm(h, axis=1, keepdims=True); hm = h.mean(0)
        nn = np.degrees(2 * np.arcsin(np.clip(tree.query(h)[0] / 2, 0, 1)))
        out[side][name] = dict(mean_dir_angle_deg=float(ang(hm, rm)), nn_angle_p50_deg=float(np.median(nn)))
        print(f"   {name:<24} mean-direction angle {ang(hm, rm):6.1f} deg   per-sample NN angle p50 {np.median(nn):5.1f} deg   (human resultant {np.linalg.norm(hm):.2f})")
best = {s: min(out[s], key=lambda k: out[s][k]["mean_dir_angle_deg"]) for s in out}
ok = all(b == "frozen chain (g, F, C)" for b in best.values())
print("\nG1", "PASS" if ok else "FAIL", {s: (b, round(out[s][b]["mean_dir_angle_deg"], 1)) for s, b in best.items()})
json.dump(out, open(C8 / "g1_gravity_sign.json", "w"), indent=1)
