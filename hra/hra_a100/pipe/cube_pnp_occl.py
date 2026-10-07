"""[2026-10-07] HRA_A100: the HandUMI fingers split the cube in two near the end of the approach (fingers aimed at its centre), so the
largest red component is half a cube and the solidity gate drops the frame. Variant: every red component whose bbox lies within
1.5 cube-widths of the largest one is merged and the CONVEX HULL is filled (the fingers sit inside the cube outline). The border-cut
and min-area rules are unchanged; hexagon()/pnp/reprojection gates downstream are unchanged."""
import numpy as np, cv2
from ego_cart20 import cube_pnp as CP


def cube_blob_occl(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (((h <= 18) | (h >= 170)) & (s >= 150) & (v >= 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, w, hh, a = st[k]
    reach = 1.5 * max(w, hh); keep = [k]
    for j in range(1, n):
        if j != k and st[j, cv2.CC_STAT_AREA] > 0.15 * a and np.linalg.norm(cen[j] - cen[k]) < reach: keep.append(j)
    mm = np.isin(lab, keep).astype(np.uint8)
    xs, ys, ws, hs = cv2.boundingRect(mm); H, W = m.shape
    if mm.sum() < CP.MIN_AREA_PX or xs <= 1 or ys <= 1 or xs + ws >= W - 1 or ys + hs >= H - 1: return None
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack(cnts)); out = np.zeros_like(mm); cv2.fillPoly(out, [hull], 1)
    return out


def install():
    CP.cube_blob = cube_blob_occl
