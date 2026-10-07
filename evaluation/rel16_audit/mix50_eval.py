"""[2026-10-02 user] per-checkpoint transfer probe for the MIX50 ego pretrain (and the raw-ego reference): median cosine of predicted vs
GT action translation (current-TCP body frame), k4/k8/k16 per arm, on (1) 120 R312c HEAD180 robot frames (same frames/noise as
dir_cos.py), (2) ego val RAW, (3) ego val MIX50, (4) ego val ROBOT100 (every wrist frame robot-like), 100 frames each.
bf16 + 5 denoising steps (deployment). usage: mix50_eval.py <ckpt_dir> <tag>  -> appends one row to rel16_audit/mix50_cosine.csv"""
import os, sys, csv, time, numpy as np, torch, pandas as pd
CK, TAG = sys.argv[1], sys.argv[2]
os.environ.update(V4_STATE_MODE="relcart20", V4_ACTION_MODE="umi", V4_RELONLY="1", V4_DTYPE="bf16", V4_DENOISE_STEPS="5", V4_CKPT=CK, N_FRAMES="120")
H = os.path.expanduser("~")
src = open(f"{H}/umi_bridge/rel16_audit/compare_ckpt_offline.py").read(); src = src[:src.index("GT = A[pick]")]
argv = sys.argv; sys.argv = [argv[0], CK]; g = {"__name__": "x"}; exec(compile(src, "cco", "exec"), g); sys.argv = argv
IC = g["IC"]
ROB = [dict(img=g["obs_cache"][n], state=torch.as_tensor(g["ST"][i], dtype=torch.float32), task=g["TS"][int(g["TI"][i])], gt=g["A"][i]) for n, i in enumerate(g["pick"])]
from lerobot.datasets.lerobot_dataset import LeRobotDataset
def load(root, N=100):
    ds = LeRobotDataset("rebot/x", root=root, video_backend="pyav"); t = pd.read_parquet(f"{root}/meta/tasks.parquet"); TS = {int(v): k for k, v in t["task_index"].items()}
    out = []
    for i in np.random.default_rng(0).choice(len(ds), N, replace=False):
        x = ds[int(i)]; out.append(dict(img={f"observation.images.{k}": x[f"observation.images.{k}"] for k in ("global", "left_wrist", "right_wrist")},
                                        state=x["observation.state"].float(), task=TS[int(x["task_index"])], gt=x["action"].numpy()[:, :20]))
    return out
SETS = {"robot": ROB, "ego_raw": load(f"{H}/c8/ego_cart20_v2b_lerobot/ego_cart20_v2b_val"),
        "ego_mix50": load(f"{H}/c8/ego_cart20_v2b_robotized_lerobot/ego_cart20_v2b_robotized_val"),
        "ego_robot100": load(f"{H}/c8/ego_cart20_v2b_robot100_lerobot/ego_cart20_v2b_robot100_val")}
INF = IC.V4Inferencer(CK); dim = INF.policy.model.dim_action
def pred(d, n):
    o = dict(d["img"]); o["observation.state"] = d["state"]; o["task"] = d["task"]
    noise = torch.randn(1, INF.chunk, dim, generator=torch.Generator().manual_seed(1234 + n)).to(INF.device).to(torch.bfloat16)
    with torch.no_grad():
        return INF.post(INF.policy.predict_action_chunk(INF.pre(o), noise=noise))[0].detach().float().cpu().numpy()[:, :20]
row = dict(t=time.strftime("%F %T"), tag=TAG, ckpt=os.path.basename(CK.rstrip("/")))
for name, D in SETS.items():
    P = np.stack([pred(d, n) for n, d in enumerate(D)]); G = np.stack([d["gt"] for d in D])
    for arm, o in (("L", 0), ("R", 10)):
        for k in (4, 8, 16):
            p, t = P[:, k-1, o:o+3], G[:, k-1, o:o+3]; nt = np.linalg.norm(t, axis=1); m = nt > 0.01
            row[f"{name}_{arm}k{k}"] = round(float(np.median(np.sum(p[m]*t[m], 1) / (np.linalg.norm(p[m], axis=1) * nt[m] + 1e-9))), 3)
    print(name, {k: v for k, v in row.items() if k.startswith(name) and "k8" in k}, flush=True)
out = f"{H}/umi_bridge/rel16_audit/mix50_cosine.csv"; new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if new: w.writeheader()
    w.writerow(row)
print("ROW", row)
