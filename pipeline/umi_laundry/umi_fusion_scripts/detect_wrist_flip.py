#!/usr/bin/env python3
"""[2026-10-13] Find episodes whose wrist camera rotated on its mount mid-take (user: "오른쪽 카메라가 중간에 뒤집어진").
Seen on real frames: the camera then looks at the room / ceiling and the JAWS disappear from the bottom-centre, where they
always are on a correctly mounted camera. Per sampled frame: jaw = dark-pixel fraction (grey < 70) in the bottom-centre
strip; a frame is "jaw missing" when that fraction < MISS_FRAC (black cloth over the jaws reads as present = conservative).
Flags an episode when the jaws are missing for >= MIN_RUN_S seconds in a row. usage: detect_wrist_flip.py <session>... [--side right]"""
import argparse, json, pathlib, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("sessions", nargs="+"); ap.add_argument("--side", default="right"); ap.add_argument("--step", type=int, default=10)
a = ap.parse_args(); MISS_FRAC, MIN_RUN_S = 0.06, 2.0
out = []
for S in a.sessions:
    for ep in sorted(pathlib.Path(S).glob("episode_*")):
        v = ep / f"{a.side}_wrist.mp4"
        if not v.exists(): continue
        cap = cv2.VideoCapture(str(v)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; i = 0; sc = []; ts = []
        while True:
            ok = cap.grab()
            if not ok: break
            if i % a.step == 0:
                _, img = cap.retrieve(); g = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2GRAY).astype(float)
                strip = g[150:180, 96:224]; sc.append(float((strip < 70).mean())); ts.append(i / fps)   # wide band: open jaws keep their fingers inside it
            i += 1
        sc = np.array(sc); ts = np.array(ts); flip = sc < MISS_FRAC
        best = run = 0; start = None; first = None
        for k, f in enumerate(flip):
            run = run + 1 if f else 0
            if run * a.step / fps >= MIN_RUN_S and first is None: first = ts[k - run + 1]
            best = max(best, run)
        r = dict(episode=f"{ep.parent.name}/{ep.name}", flipped_frac=round(float(flip.mean()), 3), longest_flip_s=round(best * a.step / fps, 1),
                 first_flip_s=None if first is None else round(float(first), 1), flagged=first is not None)
        out.append(r); print(("FLIP " if r["flagged"] else "ok   ") + f"{r['episode']:45s} jaw missing {100 * r['flipped_frac']:5.1f}%  longest {r['longest_flip_s']:5.1f}s  first {r['first_flip_s']}")
json.dump(out, open(pathlib.Path(a.sessions[0]) / f"wrist_flip_{a.side}.json", "w"), indent=1)
