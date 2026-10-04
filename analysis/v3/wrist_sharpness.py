#!/usr/bin/env python3
"""Wrist-image sharpness (variance of the Laplacian of the mean-RGB grey image, the same statistic as
cart20/ego_cart20/scripts/compare_ego_r312c.py section 11) for LeRobot v3 datasets, N frames stride-sampled per stream.
usage: wrist_sharpness.py <out.json> <name>=<lerobot_dataset_root> [...] [--n 200]"""
import glob, json, sys
import av, cv2, numpy as np

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
args = [a for i, a in enumerate(sys.argv[2:], 2) if "=" in a]
out = {"n_frames_per_stream": N, "statistic": "var(Laplacian(mean_rgb)), 224x224 stored frames", "p5_p50_p95": {}}
for spec in args:
    name, root = spec.split("=", 1)
    for side in ("left_wrist", "right_wrist"):
        files = sorted(glob.glob(f"{root}/videos/observation.images.{side}/*/*.mp4"))
        tot = 0
        for f in files:
            with av.open(f) as c: tot += c.streams.video[0].frames or 0
        stride, sh, gi = max(1, tot // N), [], 0
        for f in files:
            with av.open(f) as c:
                for fr in c.decode(video=0):
                    if gi % stride == 0 and len(sh) < N:
                        g = fr.to_ndarray(format="rgb24").astype(np.float64).mean(-1)
                        sh.append(float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var()))
                    gi += 1
            if len(sh) >= N: break
        out["p5_p50_p95"][f"{name}/{side}"] = [round(float(np.percentile(sh, q)), 1) for q in (5, 50, 95)]
        print(name, side, out["p5_p50_p95"][f"{name}/{side}"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
