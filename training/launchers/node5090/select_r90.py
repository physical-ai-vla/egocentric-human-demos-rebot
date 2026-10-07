"""[2026-10-03 user "R150이나 r180에서 r90개만 뽑아줘"] R90 selection = 90 HEAD180 episodes of r312c_relcart20_rel16_v4 (0..179 = R150 + R30
headview), 15 per stacking order, numpy default_rng(0). Training reads them via --dataset.episodes (same bytes, no copy)."""
import glob, json, numpy as np, pandas as pd
SRC = "/srv/data/johann/relonly/data/r312c_relcart20_rel16_v4"
e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(SRC + "/meta/episodes/*/*.parquet"))]).sort_values("episode_index")
e = e[e.episode_index < 180].copy(); e["task"] = e.tasks.map(lambda t: t[0])
rng = np.random.default_rng(0); keep = []
for task, g in e.groupby("task"):
    keep += rng.choice(g.episode_index.to_numpy(), 15, replace=False).tolist()
keep = sorted(int(k) for k in keep)
k = e[e.episode_index.isin(keep)]
out = dict(source=SRC, rule="episodes 0..179 (R150 0-149 + R30 150-179), 15 per task, numpy default_rng(0), groupby(task) order",
           episodes=keep, n=len(keep), frames=int(k.length.sum()), from_r150=int((k.episode_index < 150).sum()),
           from_r30=int((k.episode_index >= 150).sum()), per_task={t: int(c) for t, c in k.task.value_counts().items()})
json.dump(out, open("/srv/data/johann/relonly/code/R90_SELECTION.json", "w"), indent=1)
print(json.dumps({x: out[x] for x in ("n", "frames", "from_r150", "from_r30")}), json.dumps(keep))
