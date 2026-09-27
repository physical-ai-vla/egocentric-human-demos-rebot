#!/usr/bin/env python3
"""Task-order prompt sensitivity of the C-old ego-pretrained X-VLA (offline, ego val, Mac MPS).

Same samples, preprocessing, and inference path as ~/c8/c8old_mac_eval.py (ego mode): 399 chunk starts from the 57
held-out segments / 26 source episodes, frame cache eval_cache_ego.npz, pred[:, arm] = dq for t + 5 + i (i = 0..29).
For every sample the model is queried with
  - each of the 6 order instructions, all with the SAME noise seed n (torch.manual_seed(n)), so any difference between
    these 6 predictions is caused by the instruction alone;
  - the true instruction with two extra seeds (n + 10000, n + 20000): the flow-matching sampling spread, used as the
    noise floor for "does the prompt change anything".
Saves raw per-sample predictions (FK TCP positions per k, joint dq) to out/prompt_swap_<step>.npz. No summary here;
see summarize_prompt_swap.py.
Usage: prompt_swap.py <pretrained_model dir> <tag>
"""
import os, pathlib, sys, time
import numpy as np, torch
H = pathlib.Path.home(); C8 = H / "c8"; OUT = pathlib.Path(__file__).resolve().parent / "out"
os.environ.setdefault("REBOT_URDF", str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee/reBot_B601_DM_dualarm.urdf"))
sys.path.insert(0, str(C8 / "c8old")); sys.path.insert(0, str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee"))
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import rebot_fk_torch
CK, TAG = sys.argv[1], sys.argv[2]
LEAD, K = 5, 30; KS = (1, 4, 8, 16, 30); ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; GRIP = [6, 13]; DEV = "mps"
RENAME = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2",
          "observation.images.right_wrist": "observation.images.image3"}
IMK = ("observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist")
FK = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float32)
ds = LeRobotDataset("local/c8old_val", root=str(C8 / "c8old_data/c8old_val"), video_backend="pyav")
hf = ds.hf_dataset; ST = np.stack(hf["observation.state"]); AC = np.stack(hf["action"]); EP = np.asarray(hf["episode_index"])
TI = np.asarray(hf["task_index"]); TT = np.stack(hf["observation.ee.tcp_tgt"]).reshape(-1, 2, 4, 4)
starts = []
for e in np.unique(EP):
    ix = np.flatnonzero(EP == e); starts += list(ix[0] + np.arange(0, 31, 5))
_c = np.load(C8 / "c8old_runs/eval_cache_ego.npz"); assert np.array_equal(_c["starts"], np.array(starts)); CI = _c["images"]
TASKS = {int(i): t for i, t in zip(ds.meta.tasks["task_index"], ds.meta.tasks.index)}; NT = len(TASKS)
p = XVLAPolicy.from_pretrained(CK).float().to(DEV).eval()
pre, post = make_pre_post_processors(p.config, pretrained_path=CK,
                                     preprocessor_overrides={"device_processor": {"device": DEV}, "rename_observations_processor": {"rename_map": RENAME}},
                                     postprocessor_overrides={"device_processor": {"device": DEV}})
def run(n, t, task, seed):
    q_now = np.deg2rad(ST[t][ARM]); s14 = np.zeros(14, np.float32); s14[ARM] = q_now; s14[GRIP] = (ST[t][GRIP] <= -135)
    obs = {k: torch.from_numpy(CI[n, j]).permute(2, 0, 1).float() / 255 for j, k in enumerate(IMK)}
    obs["task"] = task; obs["observation.state"] = torch.tensor(s14); torch.manual_seed(seed)
    with torch.no_grad(): ch = p.predict_action_chunk(pre(obs))
    pred = torch.stack([post(ch[:, i, :]).squeeze(0) for i in range(ch.shape[1])]).float().cpu().numpy()[:, :14]
    dq = pred[:, ARM]; P = FK.tcp(torch.tensor(q_now[None] + dq, dtype=torch.float32))[..., :3, 3].numpy()   # [30, 2, 3]
    return dq, P
N = len(starts); SEEDS = (10000, 20000)
DQ = np.zeros((N, NT + len(SEEDS), K, 12), np.float32); PP = np.zeros((N, NT + len(SEEDS), K, 2, 3), np.float32)
GT = np.zeros((N, K, 2, 3), np.float32); P0 = np.zeros((N, 2, 3), np.float32); DQG = np.zeros((N, K, 12), np.float32)
TRUE = np.zeros(N, int); EPI = np.zeros(N, int); t0 = time.time()
for n, t in enumerate(starts):
    q_now = np.deg2rad(ST[t][ARM]); fut = np.arange(t + LEAD, t + LEAD + K)
    GT[n] = TT[fut][..., :3, 3]; DQG[n] = np.deg2rad(AC[fut][:, ARM]) - q_now[None]
    P0[n] = FK.tcp(torch.tensor(q_now[None], dtype=torch.float32))[..., :3, 3].numpy()[0]
    TRUE[n] = int(TI[t]); EPI[n] = int(EP[t])
    for j in range(NT): DQ[n, j], PP[n, j] = run(n, t, TASKS[j], n)
    for s, sd in enumerate(SEEDS): DQ[n, NT + s], PP[n, NT + s] = run(n, t, TASKS[TRUE[n]], n + sd)
    if n % 20 == 0: print(f"{n}/{N} {time.time() - t0:.0f}s", flush=True)
np.savez(OUT / f"prompt_swap_{TAG}.npz", DQ=DQ, PP=PP, GT=GT, P0=P0, DQG=DQG, TRUE=TRUE, EPI=EPI, starts=np.array(starts),
         tasks=np.array([TASKS[j] for j in range(NT)]), checkpoint=CK, seeds=np.array(SEEDS))
print("done", N, f"{time.time() - t0:.0f}s")
