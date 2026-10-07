#!/usr/bin/env python3
"""[2026-09-18] Metric anchors from the 50 mm cubes, using the RGB we already have.

Generic monocular metric depth was rejected on this footage: Depth Anything V2 over-estimated the grasp distance by 5.7-6.4x
and MetricAnything by 2.8x, because a 1920x1080 fisheye frame is far outside what those models expect. The cubes remove the
ambiguity those models cannot resolve: their physical size is known, so a detected face fixes the range directly, with no
learned scale in the loop.

Per frame this finds each visible cube face in the fisheye image, back-projects its corners through the frozen
Kannala-Brandt intrinsics, and solves for the face's metric position by matching the observed angular size to the known
50 mm edge. The output is an anchor per cube per frame, with a reprojection residual so unreliable detections can be dropped
rather than trusted.
    .venv/bin/python scripts/cube_pnp_anchor.py [--episodes 6] [--side left] [--out report.md]"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, cv2, yaml

ROOT = Path(__file__).resolve().parents[1]; M = Path.home() / "ego_data/manifests"
CUBE = 0.050
HSV = {"R": [((0, 90, 70), (10, 255, 255)), ((170, 90, 70), (180, 255, 255))],
       "B": [((100, 90, 60), (130, 255, 255))],
       "P": [((130, 55, 55), (165, 255, 255))]}


def intrinsics(side):
    f = sorted(glob.glob(str(ROOT / f"configs/calibration/wrist_bundle_{side}_v*.yaml")))[-1]
    c = yaml.safe_load(open(f))["camera"]
    return np.array(c["K"], float), np.array(c["D"], float).ravel()[:4], Path(f).name


def bearing(pts, K, D):
    """pixels -> unit bearing vectors through the fisheye model"""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    u = cv2.fisheye.undistortPoints(p, K, D.reshape(4, 1)).reshape(-1, 2)
    v = np.c_[u, np.ones(len(u))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def detect_faces(im):
    """one quadrilateral per visible cube face, with its colour"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(im, (5, 5), 0), cv2.COLOR_BGR2HSV)
    out = []
    for col, rngs in HSV.items():
        m = np.zeros(im.shape[:2], np.uint8)
        for lo, hi in rngs: m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            a = cv2.contourArea(c)
            if a < 1200: continue
            peri = cv2.arcLength(c, True)
            q = None
            for eps in (0.02, 0.03, 0.045, 0.06):          # the fisheye bends the edges, so the tolerance has to sweep
                qq = cv2.approxPolyDP(c, eps * peri, True)
                if len(qq) == 4 and cv2.isContourConvex(qq): q = qq; break
            if q is None:
                r = cv2.minAreaRect(c)                      # fall back to the best-fitting rectangle
                if cv2.contourArea(cv2.boxPoints(r).astype(np.int32)) < a * 0.75: continue
                q = cv2.boxPoints(r).reshape(4, 1, 2)
            x, y, w, h = cv2.boundingRect(q)
            if a / max(w * h, 1) < 0.55 or not (0.45 < w / max(h, 1) < 2.2): continue
            out.append((col, q.reshape(4, 2).astype(np.float64), a))
    return out


def face_pnp(quad, K, D):
    """Proper PnP on the known 50 mm square face. The four edges of a face seen obliquely are at different ranges, so
    matching each edge's subtended angle to the edge length (the earlier approach) is only valid fronto-parallel and gave a
    46 mm spread. Here the corners are undistorted to normalised bearings and a real pose is solved, which handles the tilt
    and yields a reprojection residual in pixels."""
    # IPPE_SQUARE fixes the object-point order: top-left, top-right, bottom-right, bottom-left with +y up.
    obj = np.array([[-CUBE / 2, CUBE / 2, 0], [CUBE / 2, CUBE / 2, 0],
                    [CUBE / 2, -CUBE / 2, 0], [-CUBE / 2, -CUBE / 2, 0]], np.float64)
    und = cv2.fisheye.undistortPoints(quad.reshape(-1, 1, 2), K, D.reshape(4, 1)).reshape(-1, 2)
    # match that order in the image: sort clockwise from the top-left (image y grows downward)
    c = und.mean(0)
    ang = np.arctan2(und[:, 1] - c[1], und[:, 0] - c[0])
    order = np.argsort(ang)                      # counter-clockwise in image coords
    und = und[order]
    start = int(np.argmin(und[:, 0] + und[:, 1]))  # the top-left corner
    und = np.roll(und, -start, axis=0)
    ok, rvec, tvec = cv2.solvePnP(obj, und.reshape(-1, 1, 2), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok: return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, np.eye(3), None)
    resid = float(np.linalg.norm(proj.reshape(-1, 2) - und, axis=1).mean() * K[0, 0])   # back to pixels
    t = tvec.ravel()
    return float(np.linalg.norm(t)), t, resid


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=6); ap.add_argument("--side", default="left")
    ap.add_argument("--step", type=int, default=5); ap.add_argument("--out", default=str(M / "kin_audit/cube_pnp_anchor.md")); a = ap.parse_args()
    K, D, calname = intrinsics(a.side)
    dirs = [l.strip() for l in open(M / "E30_dirs.txt")][: a.episodes]
    rows = []
    for ep in dirs:
        cap = cv2.VideoCapture(f"{ep}/{a.side}_wrist.mp4"); n = int(cap.get(7))
        seen = {}
        for f in range(0, n, a.step):
            cap.set(1, f); ok, im = cap.read()
            if not ok: continue
            for col, quad, area in detect_faces(im):
                r = face_pnp(quad, K, D)
                if r is None: continue
                rng, t, resid = r
                if not (0.03 < rng < 1.5) or resid > 12.0: continue
                seen.setdefault(col, []).append((f, rng, resid, *t))
        cap.release()
        for col, v in seen.items():
            V = np.array(v)
            if len(V) < 8: continue
            rows.append(dict(ep=os.path.basename(ep), cube=col, n=len(V),
                             rng_p50=float(np.median(V[:, 1])), spread_p50=float(np.median(V[:, 2])),
                             jitter=float(np.median(np.abs(np.diff(V[:, 1]))))))
        print(f"  {os.path.basename(ep)}: " + ", ".join(f"{c} n={len(v)}" for c, v in seen.items()), flush=True)
    if not rows: print("no cube anchors found"); return 1
    import statistics as st
    med = lambda k: st.median([r[k] for r in rows])
    L = [f"# Metric anchors from the known 50 mm cubes ({len(rows)} cube-episode tracks, {a.side} wrist) -- 2026-09-18", "",
         f"Fisheye intrinsics `{calname}`; the range comes from the angle a known 50 mm edge subtends, so no learned scale is involved.", "",
         "| quantity | median |", "|---|---|",
         f"| detections per cube per episode | {med('n'):.0f} |",
         f"| range to the cube | {med('rng_p50')*1000:.0f} mm |",
         f"| PnP reprojection residual | {med('spread_p50'):.2f} px |",
         f"| frame-to-frame range jitter | {med('jitter')*1000:.1f} mm |", "",
         "for comparison on the same footage: Depth Anything V2 Large 5.67x scale error with 11.4 mm jitter, "
         "MetricAnything 2.83x with 9.0 mm jitter", ""]
    Path(a.out).expanduser().write_text("\n".join(L) + "\n"); print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
