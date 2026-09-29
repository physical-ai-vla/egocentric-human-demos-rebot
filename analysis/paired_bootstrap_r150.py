#!/usr/bin/env python3
"""Paired bootstrap of the R150 fine-tune comparison at a matched step: ego-pretrained (C-old) vs scratch (B1-old).

Both checkpoints were scored by c8old_mac_eval.py in r150 mode on the same 578 start frames from the 10 probe episodes
(which are INSIDE both fine-tuning sets). This resamples EPISODES with replacement (cluster bootstrap, 10k draws) and
recomputes the geo score (mean over k of median FK TCP error) and the moving-subset metrics for both arms on the same
draw, giving a CI for the relative difference (C-old - scratch) / scratch.
Usage: paired_bootstrap_r150.py <scratch npz> <c-old npz> <out json>
"""
import json, pathlib, sys
import numpy as np
H = pathlib.Path.home(); C8 = H / "c8"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
A, B, OUT = sys.argv[1:4]
LEAD, K, KS = 5, 30, (1, 4, 8, 16, 30)
val = json.load(open(C8 / "probe_r150_split.json"))["val"]
ds = LeRobotDataset("rebot/rebot_3stack_R150_headview", root=str(C8 / "r150_ds"), episodes=val, video_backend="pyav")
EP = np.asarray(ds.hf_dataset["episode_index"]); starts = []
for e in np.unique(EP):
    ix = np.flatnonzero(EP == e); starts += list(range(ix[0], ix[-1] + 1 - (LEAD + K - 1), 20))
ep_of = np.repeat(EP[np.array(starts)], 2)                      # raw arrays are sample-major, arm-minor
za, zb = np.load(A), np.load(B)
assert za["k1_fk"].shape == zb["k1_fk"].shape == ep_of.shape, (za["k1_fk"].shape, ep_of.shape)
eps = np.unique(ep_of); idx = {e: np.flatnonzero(ep_of == e) for e in eps}
def metrics(z, sel):
    mt = z["k30_mt"][sel].astype(bool)
    return dict(geo=np.mean([np.median(z[f"k{k}_fk"][sel]) for k in KS]),
                motion_geo=np.mean([np.median(z[f"k{k}_fk"][sel][mt]) for k in KS]),
                k30_fk=np.median(z["k30_fk"][sel]), motion_k30_mae=np.median(z["k30_mae"][sel][mt]))
full = np.concatenate([idx[e] for e in eps]); ma, mb = metrics(za, full), metrics(zb, full)
rng = np.random.default_rng(0); draws = {k: [] for k in ma}
for _ in range(10000):
    sel = np.concatenate([idx[e] for e in rng.choice(eps, len(eps), replace=True)])
    a, b = metrics(za, sel), metrics(zb, sel)
    for k in ma: draws[k].append((b[k] - a[k]) / a[k])
per_ep = {int(e): {k: float((metrics(zb, idx[e])[k] - metrics(za, idx[e])[k]) / metrics(za, idx[e])[k]) for k in ("geo",)} for e in eps}
res = dict(scratch=A.rsplit("/", 1)[-1], cold=B.rsplit("/", 1)[-1], n_samples=int(len(starts)), n_episodes=int(len(eps)), draws=10000, seed=0,
           note="eval episodes are inside both fine-tuning sets: this measures fitting of the training distribution, not generalization",
           point={k: float((mb[k] - ma[k]) / ma[k]) for k in ma}, scratch_abs={k: float(v) for k, v in ma.items()}, cold_abs={k: float(v) for k, v in mb.items()},
           ci95={k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()},
           p_cold_worse={k: float(np.mean(np.array(v) >= 0)) for k, v in draws.items()},
           per_episode_geo_delta=per_ep, episodes_cold_better_geo=int(sum(v["geo"] < 0 for v in per_ep.values())))
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({k: res[k] for k in ("n_samples", "n_episodes", "point", "ci95", "p_cold_worse", "episodes_cold_better_geo")}, indent=1))
