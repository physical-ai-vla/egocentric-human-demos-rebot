#!/usr/bin/env python3
"""[2026-09-25] Frame cache for c8old_mac_eval.py: per-sample random access into AV1 LeRobot videos (pyav) is too slow for ~60
checkpoint evaluations, so the frames of the FIXED eval samples are decoded once, sequentially, and stored as uint8.
Frame location per row: LeRobot v3 meta.episodes -> videos/<key>/{chunk_index, file_index, from_timestamp}; frame in file =
round((from_timestamp + frame_index / fps) * fps). VERIFIED against LeRobotDataset.__getitem__ (pyav) on 5 samples: the cached
uint8 / 255 CHW must equal the loader tensor (max |d| <= 1/255 + 1e-6), else the cache is NOT written.
Usage: c8old_eval_cache.py <ego|r150>   -> ~/c8/c8old_runs/eval_cache_<mode>.npz (starts, images[N, 3, 480, 640, 3])"""
import json, pathlib, sys
import numpy as np, av, torch
H = pathlib.Path.home(); C8 = H / "c8"; MODE = sys.argv[1]
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LEAD, K = 5, 30; KEYS = ("observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist")
if MODE == "ego":
    root = C8 / "c8old_data/c8old_val"; ds = LeRobotDataset("local/c8old_val", root=str(root), video_backend="pyav")
else:
    root = C8 / "r150_ds"; val = json.load(open(C8 / "probe_r150_split.json"))["val"]
    ds = LeRobotDataset("rebot/rebot_3stack_R150_headview", root=str(root), episodes=val, video_backend="pyav")
hf = ds.hf_dataset; EP = np.asarray(hf["episode_index"]); FI = np.asarray(hf["frame_index"]); fps = ds.meta.fps
starts = []
for e in np.unique(EP):
    ix = np.flatnonzero(EP == e)
    starts += list(ix[0] + np.arange(0, 31, 5)) if MODE == "ego" else list(range(ix[0], ix[-1] + 1 - (LEAD + K - 1), 20))
starts = np.array(starts); print(f"{MODE}: {len(starts)} samples")
want = {}                                                                        # (key, file path) -> {frame in file: [(sample i, cam j)]}
for i, t in enumerate(starts):
    e = int(EP[t]); em = ds.meta.episodes[e]
    for j, k in enumerate(KEYS):
        ci, fi = int(em[f"videos/{k}/chunk_index"]), int(em[f"videos/{k}/file_index"]); ts0 = float(em[f"videos/{k}/from_timestamp"])
        path = root / ds.meta.video_path.format(video_key=k, chunk_index=ci, file_index=fi)
        fr = int(round((ts0 + FI[t] / fps) * fps)); want.setdefault((k, str(path)), {}).setdefault(fr, []).append((i, j))
IM = np.zeros((len(starts), 3, 480, 640, 3), np.uint8)
for (k, path), frames in want.items():
    last = max(frames); got = 0
    with av.open(path) as c:
        for n, f in enumerate(c.decode(c.streams.video[0])):
            if n in frames:
                a = f.to_ndarray(format="rgb24")
                for i, j in frames[n]: IM[i, j] = a
                got += 1
            if n >= last: break
    print(f"  {k} {pathlib.Path(path).name}: {got}/{len(frames)} frames", flush=True)
    assert got == len(frames), "missing frames"
mx = 0.0
for i in np.linspace(0, len(starts) - 1, 5).astype(int):
    it = ds[int(starts[i])]
    for j, k in enumerate(KEYS): mx = max(mx, float((torch.from_numpy(IM[i, j]).permute(2, 0, 1).float() / 255 - it[k]).abs().max()))
print(f"cache vs LeRobot loader: max |d| {mx:.2e}")
assert mx <= 1 / 255 + 1e-6, "cache does not match the loader -> not written"
np.savez(C8 / f"c8old_runs/eval_cache_{MODE}.npz", starts=starts, images=IM); print("CACHE_OK", MODE)
