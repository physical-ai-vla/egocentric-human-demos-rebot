#!/usr/bin/env python3
"""[2026-10-15] Tag gate for the Orbbec canonical anchor: drop canonical frames that are not on the HandUMI unit.

The laundry_v1 canonical tracker picks dark points above the table; a dark SLEEVE (UMIalign 112920: 16-19 % of frames) can take
over, and then the smoothed camera/TCP lands 26-58 cm off the jaws. The top AprilTag (L10 / R20) is rigid on the unit:
  every head frame: detect the tag, back-project its centre with depth into W (R_WC, t_WC);
  bad = |tag_W - p_W| > BAD_MM (the body point sits 6-10 cm from the tag when correct);
  frames within DILATE of a bad frame, and tag-less frames inside a bad run (bad on both sides within GAP_FILL), are bad too.
-> <episode>/derived/body_track/tag_gate.npz {t_ns, bad_left, bad_right, tag_seen_left, tag_seen_right, d_mm_left, d_mm_right}
usage: orbbec_tag_gate.py <episode|session> ... [-j 4]
"""
import argparse, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

import cv2, numpy as np, pupil_apriltags as pa

TAG = {10: "left", 20: "right"}; BAD_MM = 120.0; DILATE = 5; GAP_FILL = 15


def one(ep):
    ep = pathlib.Path(ep); out = ep / "derived/body_track/tag_gate.npz"
    sm = json.loads((ep.parent / "session_meta.json").read_text())["devices"]["head_depth"]["intrinsics"]
    K = np.array([[sm["fx"], 0, sm["cx"]], [0, sm["fy"], sm["cy"]], [0, 0, 1.0]])
    cz = np.load(ep / "derived/body_track/canonical.npz", allow_pickle=True); R_WC, t_WC, tn = cz["R_WC"], cz["t_WC"], cz["t_ns"]
    det = pa.Detector(families="tag36h11", quad_decimate=1.0, refine_edges=True, decode_sharpening=0.5)
    n = len(tn); seen = {s: np.zeros(n, bool) for s in TAG.values()}; dmm = {s: np.full(n, np.nan) for s in TAG.values()}
    cap = cv2.VideoCapture(str(ep / "head_depth.mp4")); i = -1
    while True:
        ok, img = cap.read(); i += 1
        if not ok or i >= n: break
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dets = [d for d in det.detect(g) if d.tag_id in TAG and not d.hamming]
        if not dets: continue
        depth = cv2.imread(str(ep / "head_depth_depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None: continue
        for d in dets:
            s = TAG[d.tag_id]; c = d.center / 2; u, v = int(round(c[0])), int(round(c[1]))
            if not (2 <= u < depth.shape[1] - 2 and 2 <= v < depth.shape[0] - 2): continue
            z = depth[v - 2:v + 3, u - 2:u + 3].astype(float); z = z[z > 0]
            if len(z) < 5: continue
            z = np.median(z) / 1e3; tagW = R_WC @ np.array([(c[0] - K[0, 2]) * z / K[0, 0], (c[1] - K[1, 2]) * z / K[1, 1], z]) + t_WC
            pW = cz[f"p_W_{s}"][i]; seen[s][i] = True
            if np.isfinite(pW).all(): dmm[s][i] = np.linalg.norm(tagW - pW) * 1e3
    res = dict(t_ns=tn)
    for s in TAG.values():
        b = np.nan_to_num(dmm[s], nan=0) > BAD_MM; bad = b.copy()
        for k in np.flatnonzero(b): bad[max(k - DILATE, 0):k + DILATE + 1] = True
        idx = np.flatnonzero(b)
        for a_, c_ in zip(idx[:-1], idx[1:]):                       # tag-less gap inside a bad run
            if c_ - a_ <= GAP_FILL and not np.any(seen[s][a_ + 1:c_] & ~b[a_ + 1:c_]): bad[a_:c_ + 1] = True
        res.update({f"bad_{s}": bad, f"tag_seen_{s}": seen[s], f"d_mm_{s}": dmm[s]})
    np.savez_compressed(out, **res)
    return f"{ep.parent.name}/{ep.name}: bad L {res['bad_left'].mean():.3f} R {res['bad_right'].mean():.3f} | tag seen L {seen['left'].mean():.2f} R {seen['right'].mean():.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("-j", type=int, default=4); a = ap.parse_args(); eps = []
    for p in map(pathlib.Path, a.paths):
        eps += [p] if (p / "episode_meta.json").exists() else [e for e in sorted(p.glob("episode_*")) if (e / ".complete").exists()]
    eps = [e for e in eps if not json.loads((e / "episode_meta.json").read_text()).get("exclude_from_training")]
    with ProcessPoolExecutor(a.j) as ex:
        for r in ex.map(one, map(str, eps)): print(r, flush=True)
