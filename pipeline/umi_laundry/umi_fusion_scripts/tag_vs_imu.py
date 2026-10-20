#!/usr/bin/env python3
"""[2026-10-13] GPU-free rotation ablation on one episode: TAG-ONLY vs IMU (AHRS) for each gripper's orientation.

  tag   : top AprilTag (L10 / R20, 30 mm) in the Orbbec colour, IPPE square PnP; the 2-fold flip ambiguity is resolved WITHOUT the
          IMU by temporal continuity (pick the solution closest to the previous accepted tag rotation; restart after a gap).
  imu   : Mahony AHRS (canonical tracker's mahony / ahrs_at), gravity world, yaw free.
  model : R_O_tag(t) = A @ R_g_imu(t) @ B   (A = Orbbec cam <- gravity world, B = imu body <- tag; both constant)
          B from hand-eye on relative rotations (tag pairs 10/20/30 frames apart, Wahba on rotation vectors), A by averaging.
Reported per side:
  coverage       tag frames / all frames; share of frames within 0.5 s of a tag frame; longest tag gap
  imu_drift      A fitted on the FIRST 10 s only; angle(A R_imu B, R_tag) per 10 s block -> IMU-alone drift (yaw) over time
  tag_only_gap   tag-only label in gaps = SLERP between the surrounding tag frames; error vs the IMU-propagated rotation from
                 the nearest tag frame (IMU is accurate over short horizons) -> error of a tag-only pipeline by gap length
  tag_jitter     consecutive-frame tag rotation vs IMU rotation over the same step (high-frequency tag noise)
usage: tag_vs_imu.py <episode> [--tag-mm 30]
"""
import argparse, json, pathlib, sys
import cv2, numpy as np, pupil_apriltags as pa
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from handumi_collector.pose.episode_io import RawEpisode                   # noqa: E402
from handumi_canonical_track import CANON_V1, ahrs_at                      # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--tag-mm", type=float, default=30.0); a = ap.parse_args()
EP = pathlib.Path(a.episode); K = json.load(open(EP.parent / "session_meta.json"))["devices"]["head_depth"]["intrinsics"]
Kc = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1.0]]); h = a.tag_mm / 2000.0
OBJ = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
raw = RawEpisode.load(EP)
from mcap.reader import make_reader
fm = {}
with open(EP / "sensors.mcap", "rb") as f:
    for _, ch, msg in make_reader(f).iter_messages(topics=["/head_depth/frame_meta"]):
        d = json.loads(msg.data); fm[d["video_frame"]] = d["capture_ns"]
det = pa.Detector(families="tag36h11", quad_decimate=1.0, refine_edges=True, decode_sharpening=0.5)
cap = cv2.VideoCapture(str(EP / "head_depth.mp4")); i = 0; sols = {"left": {}, "right": {}}
while True:
    ok, img = cap.read()
    if not ok: break
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for d in det.detect(g):
        side = {10: "left", 20: "right"}.get(d.tag_id)
        if side is None or d.hamming: continue
        n, rvs, tvs, errs = cv2.solvePnPGeneric(OBJ, (d.corners / 2).astype(np.float64), Kc, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols[side][i] = [cv2.Rodrigues(r)[0] for r in rvs]
    i += 1
N = i; t_ns = np.array([fm.get(k, -1) for k in range(N)], np.int64); t = (t_ns - t_ns[0]) / 1e9
out = {}
for side in ("left", "right"):
    Rimu, info = ahrs_at(raw, side, t_ns, CANON_V1)
    if Rimu is None: out[side] = "no imu"; continue
    tag = {}; prev = None; prev_k = None                              # tag-only flip resolution by temporal continuity
    for k in sorted(sols[side]):
        cands = sols[side][k]
        pick = cands[0] if (prev is None or k - prev_k > 15) else min(cands, key=lambda R: Rot.from_matrix(prev.T @ R).magnitude())
        tag[k] = pick; prev, prev_k = pick, k
    ks = np.array(sorted(tag))
    if len(ks) < 30: out[side] = dict(tag_coverage=round(len(ks) / N, 3), note="too few tag frames"); continue
    va, vb = [], []                                                   # hand-eye: rt ~ Bt @ ri
    for x in range(len(ks)):
        for dk in (10, 20, 30):
            y = np.searchsorted(ks, ks[x] + dk)
            if y >= len(ks) or ks[y] != ks[x] + dk: continue
            rt = Rot.from_matrix(tag[ks[x]].T @ tag[ks[y]]).as_rotvec(); ri = Rot.from_matrix(Rimu[ks[x]].T @ Rimu[ks[y]]).as_rotvec()
            if np.linalg.norm(ri) > np.radians(8) and abs(np.linalg.norm(rt) - np.linalg.norm(ri)) < np.radians(6): va.append(ri); vb.append(rt)
    va, vb = np.array(va), np.array(vb)
    U, _, Vt = np.linalg.svd(vb.T @ va); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Bt = U @ D @ Vt; B = Bt.T
    early = ks[t[ks] <= 10.0]
    A = Rot.from_matrix([tag[k] @ Bt @ Rimu[k].T for k in early]).mean().as_matrix()
    e_abs = np.array([np.degrees(Rot.from_matrix((A @ Rimu[k] @ B).T @ tag[k]).magnitude()) for k in ks])
    drift = {f"{lo}-{lo + 10}s": round(float(np.median(e_abs[(t[ks] >= lo) & (t[ks] < lo + 10)])), 2) for lo in (0, 10, 20, 30, 40) if ((t[ks] >= lo) & (t[ks] < lo + 10)).any()}
    gaps = {"<=0.5s": [], "0.5-1s": [], "1-2s": [], ">2s": []}
    for x in range(len(ks) - 1):
        k0, k1 = ks[x], ks[x + 1]
        if k1 - k0 < 2: continue
        sl = Slerp([t[k0], t[k1]], Rot.from_matrix([tag[k0], tag[k1]])); L = t[k1] - t[k0]
        for k in range(k0 + 1, k1):
            near = k0 if (k - k0) <= (k1 - k) else k1
            imu_ref = tag[near] @ Bt @ (Rimu[near].T @ Rimu[k]) @ Bt.T
            gaps["<=0.5s" if L <= 0.5 else "0.5-1s" if L <= 1 else "1-2s" if L <= 2 else ">2s"].append(np.degrees(Rot.from_matrix(sl([t[k]]).as_matrix()[0].T @ imu_ref).magnitude()))
    noise = [np.degrees(Rot.from_matrix((tag[ks[x]].T @ tag[ks[x + 1]]).T @ (Bt @ Rimu[ks[x]].T @ Rimu[ks[x + 1]] @ Bt.T)).magnitude())
             for x in range(len(ks) - 1) if ks[x + 1] - ks[x] == 1]
    dist = np.array([np.min(np.abs(ks - k)) / 30.0 for k in range(N)])
    out[side] = dict(tag_coverage=round(len(ks) / N, 3), frames_within_0p5s_of_tag=round(float((dist <= 0.5).mean()), 3), longest_tag_gap_s=round(float(np.max(np.diff(ks)) / 30), 2),
                     handeye_pairs=int(len(va)), imu_drift_deg_median_by_time=drift,
                     tag_only_gap_error_deg={k: (dict(n=len(v), p50=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2)) if v else None) for k, v in gaps.items()},
                     tag_frame_jitter_deg=dict(p50=round(float(np.median(noise)), 2), p90=round(float(np.percentile(noise, 90)), 2)) if noise else None)
print(json.dumps(out, indent=1))
