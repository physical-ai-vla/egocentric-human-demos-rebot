"""MPS latency and output shift for dtype x denoising steps, REL16 400k, same noise, audit samples."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ["V4_STATE_MODE"] = "umi76"; os.environ["V4_ACTION_MODE"] = "umi"
import infer_core_v4 as IC
from lerobot.datasets.lerobot_dataset import LeRobotDataset
d = np.load(os.path.expanduser("~/umi_bridge/rel16_audit/step2_preds.npz"), allow_pickle=True)
ds = LeRobotDataset("rebot/r180", root=os.path.expanduser("~/holobrain-data/lerobot/r180_umi76_rel16_v1"), video_backend="pyav")
inf = IC.V4Inferencer(os.path.expanduser("~/holobrain-mac-model/ckpt_B180H-UMI76-REL16_400k"))
import pandas as pd
TS = {int(v): k for k, v in pd.read_parquet(os.path.expanduser("~/holobrain-data/lerobot/r180_umi76_rel16_v1/meta/tasks.parquet"))["task_index"].items()}
idx = [int(i) for i in d["pick"][:12]]
obs_l = []
for i in idx:
    x = ds[i]; o = {f"observation.images.{k}": x[f"observation.images.{k}"] for k in ("global", "left_wrist", "right_wrist")}
    o["observation.state"] = x["observation.state"]; o["task"] = TS[int(x["task_index"])]; obs_l.append(o)
dim = inf.policy.model.dim_action


def run(steps, half):
    inf.policy.config.num_denoising_steps = steps
    outs, ts = [], []
    for j, o in enumerate(obs_l):
        noise = torch.randn(1, 16, dim, generator=torch.Generator().manual_seed(j))
        b = inf.pre(dict(o))
        if half:
            b = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()}
        torch.mps.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            ch = inf.policy.predict_action_chunk(b, noise=noise.to(inf.device, dtype=torch.float16 if half else torch.float32))
        torch.mps.synchronize(); ts.append(time.perf_counter() - t)
        outs.append(inf.post(ch.float())[0].cpu().numpy())
    return np.array(outs), np.median(ts[2:])


ref, t_ref = run(10, False)
res = {("fp32", 10): (0.0, t_ref)}
for steps in (5, 3):
    o, t = run(steps, False); res[("fp32", steps)] = (np.linalg.norm(o[:, 15, [0,1,2]] - ref[:, 15, [0,1,2]], axis=-1).mean() * 1000, t)
inf.policy.half()
for steps in (10, 5, 3):
    o, t = run(steps, True); res[("fp16", steps)] = (np.linalg.norm(o[:, 15, [0,1,2]] - ref[:, 15, [0,1,2]], axis=-1).mean() * 1000, t)
for k, (dmm, t) in res.items():
    print(f"{k[0]} steps {k[1]:2d}: infer {t*1000:6.0f} ms | L-arm A16 shift vs fp32/10 (same noise) {dmm:5.2f} mm")
