import pathlib, cv2, numpy as np, json, sys
sys.path.insert(0, str(pathlib.Path.home() / "c8/hra_a100/pipe"))
from ego_cart20 import cube_pnp as CP
import cube_pnp_occl
orig = CP.cube_blob
E = pathlib.Path.home() / "c8/hra_a100/export"; out = {}
for tag in sorted(p.name for p in E.iterdir()):
    K, D, size = CP.read_setting(E / tag / "orbslam_setting.yaml"); P, und = CP.undistorter(K, D, size)
    c = cv2.VideoCapture(str(E / tag / "raw_video.mp4")); n = {"orig": 0, "occl": 0}; i = 0
    while True:
        ok, f = c.read()
        if not ok: break
        i += 1
        if i % 2: continue
        u = und(f)
        for name, fn in (("orig", orig), ("occl", cube_pnp_occl.cube_blob_occl)):
            m = fn(u)
            if m is None: continue
            hx, sol = CP.hexagon(m)
            if hx is None or sol < CP.MIN_SOLIDITY: continue
            r = CP.pnp_cube(hx, P)
            if r is not None and r[2] <= CP.MAX_REPROJ_PX: n[name] += 1
    out[tag] = n
json.dump(out, open(pathlib.Path.home() / "c8/hra_a100/pnp_census.json", "w"), indent=1)
o = np.array([v["orig"] for v in out.values()]); q = np.array([v["occl"] for v in out.values()]); b = np.maximum(o, q)
print("episodes", len(out), "| >=10 valid PnP frames (every 2nd frame): orig", int((o >= 5).sum()), "occl", int((q >= 5).sum()), "best-of", int((b >= 5).sum()))
