"""[2026-10-02] A: ego40k on ego-val (teacher-forced, same dataset images+state+action) per-arm cosine of k4/k8/k16 translation.
D: image/state ablation on the SAME model: robot frames (R312c HEAD180) vs ego-val frames, with own state / zero state /
the other domain's state (paired by index). Cosine always against the GT of the frame whose IMAGE is used."""
import os, sys, glob, numpy as np, torch, pandas as pd
os.environ.update(V4_STATE_MODE="relcart20", V4_ACTION_MODE="umi", V4_RELONLY="1", V4_DTYPE="bf16", V4_DENOISE_STEPS="5")
CK = sys.argv[1]; os.environ["V4_CKPT"] = CK
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model")); import infer_core_v4 as IC
from lerobot.datasets.lerobot_dataset import LeRobotDataset
N = 100
def load(root, name):
    ds = LeRobotDataset(f"rebot/{name}", root=root, video_backend="pyav")
    tasks = pd.read_parquet(f"{root}/meta/tasks.parquet"); TS = {int(v): k for k, v in tasks["task_index"].items()}
    pick = np.random.default_rng(0).choice(len(ds), N, replace=False); out = []
    for i in pick:
        x = ds[int(i)]
        out.append(dict(img={f"observation.images.{k}": x[f"observation.images.{k}"] for k in ("global", "left_wrist", "right_wrist")},
                        state=x["observation.state"].float(), task=TS[int(x["task_index"])], gt=x["action"].numpy()[:, :20]))
    return out
EGO = load(os.path.expanduser("~/c8/ego_cart20_v2b_lerobot/ego_cart20_v2b_val"), "ego_val")
# robot frames: the same HEAD180 rows as dir_cos.py (r180_umi76_rel16_v3d images + r180_relcart20_rel16_v3d state/action)
os.environ["N_FRAMES"] = str(N)
_src = open(os.path.expanduser("~/umi_bridge/rel16_audit/compare_ckpt_offline.py")).read(); _src = _src[:_src.index("GT = A[pick]")]
_argv = sys.argv; sys.argv = [sys.argv[0], CK]; _g = {"__name__": "x"}; exec(compile(_src, "cco", "exec"), _g); sys.argv = _argv
ROB = [dict(img=_g["obs_cache"][n], state=torch.as_tensor(_g["ST"][i], dtype=torch.float32), task=_g["TS"][int(_g["TI"][i])], gt=_g["A"][i])
       for n, i in enumerate(_g["pick"])]
INF = IC.V4Inferencer(CK); dim = INF.policy.model.dim_action
def pred(img, state, task, n):
    o = dict(img); o["observation.state"] = state; o["task"] = task
    noise = torch.randn(1, INF.chunk, dim, generator=torch.Generator().manual_seed(1234 + n)).to(INF.device).to(torch.bfloat16)
    with torch.no_grad():
        return INF.post(INF.policy.predict_action_chunk(INF.pre(o), noise=noise))[0].detach().float().cpu().numpy()[:, :20]
def cos_rep(P, G):
    s = []
    for arm, o in (("L", 0), ("R", 10)):
        for k in (4, 8, 16):
            p = P[:, k-1, o:o+3]; t = G[:, k-1, o:o+3]; n = np.linalg.norm(t, axis=1); m = n > 0.01
            c = np.sum(p[m]*t[m], 1) / (np.linalg.norm(p[m], axis=1) * n[m] + 1e-9); s.append(f"{arm}k{k} {np.median(c):+.2f}(n{m.sum()})")
    return " ".join(s)
zero = torch.tensor([0,0,0, 1,0,0,0,1,0, 0,0,0, 1,0,0,0,1,0, 0,0], dtype=torch.float32)   # identity rel pose, jaws closed
conds = {
  "1 ego img + ego state":   (EGO, lambda i, d: d[i]["state"]),
  "1b ego img + zero state": (EGO, lambda i, d: zero),
  "3 ego img + robot state": (EGO, lambda i, d: ROB[i]["state"]),
  "4 robot img + robot state": (ROB, lambda i, d: d[i]["state"]),
  "2 robot img + zero state": (ROB, lambda i, d: zero),
  "2b robot img + ego state": (ROB, lambda i, d: EGO[i]["state"]),
}
for name, (D, sf) in conds.items():
    P = np.stack([pred(D[i]["img"], sf(i, D), D[i]["task"], i) for i in range(N)]); G = np.stack([d["gt"] for d in D])
    print(f"{name:28s} {cos_rep(P, G)}", flush=True)
