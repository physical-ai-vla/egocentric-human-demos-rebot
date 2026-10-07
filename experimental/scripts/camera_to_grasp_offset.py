#!/usr/bin/env python3
"""[2026-09-18] Estimate the rigid offset from the wrist CAMERA optical centre to the actual GRASP point of the HandUMI.

Today the human trajectory is the camera pose, while the robot target is a TCP 70 mm in front of the gripper link, so the two
sides track different physical points and a wrist rotation moves one far more than the other. The CAD in handumi-hw is exported
per part with no assembly placements, so the offset is recovered from the recordings instead: when the jaws are closed on a cube,
the grasp point IS the cube centre, so the cube's position in camera coordinates is the offset we need.

Depth comes from the known cube edge (50 mm) and the frozen Kannala-Brandt intrinsics: a face of known size subtends a known
angle, so the apparent width fixes the range. The cube is found in the jaw region at the bottom centre of the fisheye view,
only on frames the gripper labels as closed, and only when the blob is compact enough to be a single face.
    .venv/bin/python scripts/camera_to_grasp_offset.py [--side left] [--episodes N] [--out report.md]"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, cv2, yaml

ROOT = Path(__file__).resolve().parents[1]; M = Path.home() / "ego_data/manifests"
CUBE_MM = 50.0
HSV = {"R": [((0, 90, 70), (10, 255, 255)), ((170, 90, 70), (180, 255, 255))],
       "B": [((100, 90, 60), (130, 255, 255))],
       "P": [((130, 60, 60), (165, 255, 255))]}


def intrinsics(side):
    f = sorted(glob.glob(str(ROOT / f"configs/calibration/wrist_bundle_{side}_v*.yaml")))[-1]
    c = yaml.safe_load(open(f))["camera"]
    return np.array(c["K"], float), np.array(c["D"], float).ravel()[:4], f


def unproject(px, py, K, D):
    """Kannala-Brandt: pixel -> unit bearing vector in camera coordinates."""
    pts = np.array([[[px, py]]], dtype=np.float64)
    und = cv2.fisheye.undistortPoints(pts, K, D.reshape(4, 1))
    x, y = und[0, 0]
    v = np.array([x, y, 1.0]); return v / np.linalg.norm(v)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--side", default="left"); ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--step", type=int, default=4); ap.add_argument("--out", default=str(M / "kin_audit/camera_to_grasp_offset.md")); a = ap.parse_args()
    K, D, calfile = intrinsics(a.side)
    dirs = [l.strip() for l in open(M / "E30_dirs.txt")][: a.episodes]
    gi = 7 if a.side == "left" else 15
    samples = []
    for ep in dirs:
        z = np.load(ep + "/derived/humanik/ego16_com.npz"); g = z["S16"][:, gi]
        cap = cv2.VideoCapture(f"{ep}/{a.side}_wrist.mp4"); n = int(cap.get(7))
        if n < 10: continue
        closed = np.flatnonzero(g < 0.5)
        for fi in closed[:: a.step]:
            f = int(fi * n / len(g))
            cap.set(1, f); ok, im = cap.read()
            if not ok: continue
            H, W = im.shape[:2]
            roi = im[int(H * 0.55):, int(W * 0.25):int(W * 0.75)]          # the jaw region of the fisheye view
            hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (7, 7), 0), cv2.COLOR_BGR2HSV)
            best = None
            for col, rngs in HSV.items():
                m = np.zeros(roi.shape[:2], np.uint8)
                for lo, hi in rngs: m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cs:
                    area = cv2.contourArea(c)
                    if area < 3000: continue
                    x, y, w, h = cv2.boundingRect(c)
                    fill = area / max(w * h, 1); ar = w / max(h, 1)
                    if fill < 0.6 or not (0.6 < ar < 1.7): continue           # a single cube face, not a smear
                    if best is None or area > best[0]: best = (area, x, y, w, h, col)
            if best is None: continue
            _, x, y, w, h, col = best
            cx = int(W * 0.25) + x + w / 2; cy = int(H * 0.55) + y + h / 2
            # angular size of the face -> range; use the smaller side (the face seen most head-on)
            s = min(w, h)
            b1 = unproject(cx - s / 2, cy, K, D); b2 = unproject(cx + s / 2, cy, K, D)
            ang = np.arccos(np.clip(np.dot(b1, b2), -1, 1))
            if ang < 1e-4: continue
            rng = (CUBE_MM / 2) / np.tan(ang / 2)
            v = unproject(cx, cy, K, D)
            samples.append(np.r_[v * rng, fi, ord(col)])
        cap.release()
    # A cube that is actually GRASPED is nearly stationary in the camera frame while the hand moves; a cube lying on the
    # table is not. Keep only detections that sit still in camera coordinates across a run of consecutive closed frames.
    if samples:
        A = np.array(samples); order = np.argsort(A[:, 3]); A = A[order]
        keepm = np.zeros(len(A), bool); i = 0
        while i < len(A):
            j = i
            while j + 1 < len(A) and A[j + 1, 3] - A[j, 3] <= a.step * 2 and A[j + 1, 4] == A[j, 4]: j += 1
            if j - i >= 3:
                seg = A[i:j + 1, :3]
                if np.median(np.abs(seg - np.median(seg, 0)), 0).max() * 1.4826 < 25.0: keepm[i:j + 1] = True
            i = j + 1
        print(f"  temporal-stability filter: {int(keepm.sum())}/{len(A)} detections are consistent with a HELD cube")
        samples = [r for r, k in zip(A, keepm) if k]
    if len(samples) < 10:
        print(f"only {len(samples)} usable grasp frames -- not enough to estimate the offset"); return 1
    S = np.array(samples); P = S[:, :3]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) * 1.4826
    keep = (np.abs(P - med) < 3 * np.maximum(mad, 1.0)).all(1)
    Pk = P[keep]
    L = [f"# Wrist camera -> grasp point offset, {a.side} hand ({len(Pk)} accepted frames of {len(P)}) -- 2026-09-18", "",
         f"Recovered from recordings, not CAD: on frames the gripper labels as closed, the grasped 50 mm cube IS the grasp point,",
         f"so its position in camera coordinates is the offset. Intrinsics: `{Path(calfile).name}` (Kannala-Brandt).", "",
         "| axis | median (mm) | robust sigma (mm) |", "|---|---|---|"]
    for i, nm in enumerate(("x right", "y down", "z forward")):
        L.append(f"| {nm} | {np.median(Pk[:, i]):+.1f} | {np.median(np.abs(Pk[:, i] - np.median(Pk[:, i]))) * 1.4826:.1f} |")
    off = np.median(Pk, 0)
    L += ["", f"**offset magnitude {np.linalg.norm(off):.0f} mm** (camera optical centre -> grasp point, in the camera frame)", "",
          "Induced position error when the wrist rotates, if this offset is ignored (2 d sin(theta/2)):", "",
          "| wrist rotation | induced grasp-point motion |", "|---|---|"]
    for th in (10, 30, 48):
        L.append(f"| {th} deg | {2 * np.linalg.norm(off) * np.sin(np.radians(th) / 2):.0f} mm |")
    out = Path(a.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(side=a.side, offset_mm_camera_frame=off.tolist(), n=len(Pk), n_raw=len(P),
                   cube_mm=CUBE_MM, calibration=Path(calfile).name),
              open(out.with_suffix(".json"), "w"), indent=1)
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
