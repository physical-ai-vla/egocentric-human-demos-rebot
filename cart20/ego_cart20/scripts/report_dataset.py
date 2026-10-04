#!/usr/bin/env python3
"""usage: report_dataset.py <processed_root> <out_dir> [n_samples=4]  -> trajectory_samples.png + order / split table (stdout)"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.io.processed_writer import load_episode
from ego_cart20.validation.visualize_trajectory import plot_samples
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 4; out.mkdir(parents=True, exist_ok=True)
man = [json.loads(l) for l in open(root / "train_manifest.jsonl")]; rng = np.random.default_rng(0); C, T = [], []
for r in rng.choice(man, n, replace=False):
    ep = load_episode(root / r["path"]); j = int(rng.integers(len(ep["train_rows"])))
    j = int(np.argmax(np.linalg.norm(ep["cart20"][:, 15, 0:3], axis=1))) if j % 2 else j    # half random, half the most moving row
    C.append(np.asarray(ep["cart20"][j], np.float64)); T.append(f"{r['episode_id']} row {int(ep['train_rows'][j])} ({r['stack_order']})")
plot_samples(C, T, out / "trajectory_samples.png"); print("wrote", out / "trajectory_samples.png")
meta = json.load(open(root / "metadata.json")); print(json.dumps(meta["counts"], indent=1))
