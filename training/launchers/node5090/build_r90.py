"""[2026-10-03 user "R150이나 r180에서 r90개만 뽑아줘"] R90 = 90 HEAD180 episodes of r312c_relcart20_rel16_v4 (episodes 0..179 =
R150 headview + R30 day4 headview), 15 per stacking order (6 orders), seed 0. Same bytes per episode (lerobot dataset_tools.delete_episodes),
stats recomputed by the tool for the kept episodes. Output r90_relcart20_rel16_v4 + R90_SELECTION.json."""
import glob, json, sys, pathlib, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "/srv/data/johann/relonly/code/lerobot-seeed/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
R = pathlib.Path("/srv/data/johann/relonly/data"); SRC = R / "r312c_relcart20_rel16_v4"; OUT = R / "r90_relcart20_rel16_v4"
e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SRC / "meta/episodes/*/*.parquet")))]).sort_values("episode_index")
e = e[e.episode_index < 180]; e["task"] = e.tasks.map(lambda t: t[0])
rng = np.random.default_rng(0); keep = []
for task, g in e.groupby("task"):
    assert len(g) >= 15, (task, len(g)); keep += sorted(rng.choice(g.episode_index.to_numpy(), 15, replace=False).tolist())
keep = sorted(int(k) for k in keep); assert len(keep) == 90
src = LeRobotDataset("rebot/r312c_relcart20_rel16_v4", root=SRC)
drop = [i for i in range(src.meta.total_episodes) if i not in set(keep)]
ds = delete_episodes(src, episode_indices=drop, output_dir=OUT, repo_id="rebot/r90_relcart20_rel16_v4")
info = json.load(open(OUT / "meta/info.json"))
sel = dict(source=str(SRC), rule="episodes 0..179 (HEAD180 = R150 + R30), 15 per task, numpy default_rng(0)", kept_source_episodes=keep,
           per_task={t: int(c) for t, c in e[e.episode_index.isin(keep)].task.value_counts().items()},
           total_episodes=info["total_episodes"], total_frames=info["total_frames"], builder_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
json.dump(sel, open(OUT / "R90_SELECTION.json", "w"), indent=1)
print("R90 DONE", info["total_episodes"], info["total_frames"])
