"""[2026-10-06 user step 1] HRA head C922 (fixed, 640x480): cube PnP at each episode start (first 15 frames, hand not in view) and the
focal length f that minimises the cube-corner reprojection over all episodes (pinhole, cx = 320, cy = 240, no distortion)."""
import json, pathlib, sys
import cv2, numpy as np
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20 import cube_pnp as CP
S = H / "ego_collector/datasets/human_handumi_raw/HRA_red/HRA_red_20261003_152502"; P = H / "c8/hra_red/processed"
OUT = H / "c8/hra_red/head_link"
eps = [json.loads(l)["episode_id"] for s in ("train", "val") for l in open(P / f"{s}_manifest.jsonl")]
dets = {}
for ep in eps:
    num = ep.split("_")[-1]; c = cv2.VideoCapture(str(S / f"episode_{num}" / "head.mp4")); hx_all = []
    for i in range(15):
        ok, f = c.read()
        if not ok: break
        m = CP.cube_blob(f)
        if m is None: continue
        hx, sol = CP.hexagon(m)
        if hx is not None and sol >= CP.MIN_SOLIDITY and len(hx) in (6, 4): hx_all.append(np.asarray(hx, float))
    c.release()
    if hx_all:
        # use the median corner set of the frames with the most common corner count
        n = max(set(len(h) for h in hx_all), key=lambda k: sum(len(h) == k for h in hx_all))
        dets[ep] = np.median(np.stack([h for h in hx_all if len(h) == n]), 0)
print("episodes with a head cube detection:", len(dets), "/", len(eps), " (6 corners:", sum(len(v) == 6 for v in dets.values()), ")")


def total_err(f):
    Pm = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]]); es = []
    for ep, hx in dets.items():
        r = CP.pnp_cube(hx, Pm)
        if r is not None: es.append(r[2])
    return float(np.median(es)) if es else 1e9, len(es)


grid = [(f,) + total_err(f) for f in range(300, 1101, 25)]
for f, e, n in grid: print(f"f {f:5d}  median reproj {e:6.2f} px  n {n}")
fb = min(grid, key=lambda g: g[1])[0]
fine = [(f,) + total_err(f) for f in np.arange(fb - 25, fb + 26, 5)]
fbest = min(fine, key=lambda g: g[1])
print("best f", fbest)
np.save(OUT / "head_dets.npy", dets, allow_pickle=True)
json.dump(dict(f_best=float(fbest[0]), reproj_median=fbest[1], n=fbest[2], grid=[list(map(float, g)) for g in grid]), open(OUT / "head_f.json", "w"), indent=1)
