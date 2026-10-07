"""[2026-10-01] Boundary-conditioned bias (user): does the policy pull an arm back inside its training support near the workspace
edge, or keep pushing it out? Live v3 245k drove the RIGHT arm to x 58-160 mm (training: x >= 172 mm for 99% of frames) with
offline signed error e_x -5/-10/-14, e_y +6/+12/+15 mm at k4/8/16.

Frames (TRAINING, teacher-forced; no val split for HEAD180) are binned by the arm's CURRENT base-frame TCP coordinate
(FK(aux.q_t) + frame fix, the deploy _tcp_mat): RIGHT arm x bins >250 / 220-250 / 190-220 / 172-190 / <172 mm, N_PER_BIN random
frames each (seed 0). Same inference path as ybias_eval.py (V4Inferencer pre/post, one seeded flow draw per sample).
Per bin and k in {4, 8}: base-frame Δ (mm) of pred and GT, signed error e = pred - GT, and the near-zero subset (|ΔGT| < 5 mm at k8):
mean |Δpred| there = residual motion when the demo stands still.
usage: CKPT=<local ckpt dir> TAG=<name> [N_PER_BIN=60] boundary_bias_eval.py     (pause robot UIs: shares the MPS)
"""
import os, sys, time, glob
import numpy as np, torch, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ["V4_STATE_MODE"] = "relcart20"; os.environ["V4_ACTION_MODE"] = "umi"; os.environ["V4_CKPT"] = os.environ["CKPT"]
import infer_core_v4 as IC
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/holobrain-data/lerobot/r180_umi76_rel16_v3d"))          # images
FEED_ROOT = os.path.expanduser(os.environ.get("FEED_ROOT", "~/holobrain-data/lerobot/r180_relcart20_rel16_v3d"))  # state + action trained on
TAG = os.environ.get("TAG", os.path.basename(os.environ["CKPT"])); NPB = int(os.environ.get("N_PER_BIN", "60"))
OUT = os.path.expanduser(f"~/umi_bridge/rel16_audit/boundary_{TAG}.npz")
col = lambda t, c: (lambda a: a.storage if isinstance(a, pa.ExtensionArray) else a)(t.column(c).combine_chunks())
ft = [pq.read_table(f, columns=["observation.state", "action", "index", "episode_index", "task_index", "aux.q_t"])
      for f in sorted(glob.glob(f"{FEED_ROOT}/data/chunk-000/*.parquet"))]
IX = np.concatenate([col(t, "index").to_numpy() for t in ft]); E = np.concatenate([col(t, "episode_index").to_numpy() for t in ft])
TI = np.concatenate([col(t, "task_index").to_numpy() for t in ft]); Q = np.concatenate([np.asarray(col(t, "aux.q_t").to_pylist()) for t in ft])
FEED = np.concatenate([np.asarray(col(t, "observation.state").to_pylist(), np.float64) for t in ft])
A = np.concatenate([np.asarray(col(t, "action").to_pylist(), np.float64) for t in ft])
tasks = pd.read_parquet(f"{FEED_ROOT}/meta/tasks.parquet"); TASK_STR = {int(v): k for k, v in tasks["task_index"].items()}

class K_: kin = IC.eef_kin.Kin(); _tcp_mat = IC.V4Inferencer._tcp_mat
kk = K_()
def mats(i):
    q14 = np.zeros(14); q14[IC.ARM_IDX] = Q[i]; return kk._tcp_mat(q14)[0]
# current right-arm x for every 3rd frame (FK is the slow part), then bin
cand = np.arange(0, len(IX), 3); RX = np.array([mats(i)[1][0, 3] * 1000 for i in cand])
BINS = [("x>250", 250, 1e9), ("220-250", 220, 250), ("190-220", 190, 220), ("172-190", 172, 190), ("x<172 (OOD)", -1e9, 172)]
rng = np.random.default_rng(0); pick, binlab = [], []
for name, lo, hi in BINS:
    c = cand[(RX >= lo) & (RX < hi)]; s = rng.choice(c, min(NPB, len(c)), replace=False) if len(c) else []
    print(f"  bin {name:12s}: {len(c)} candidate frames, {len(s)} picked", flush=True); pick += list(map(int, s)); binlab += [name] * len(s)
Rt = np.stack([np.stack([m[0][:3, :3], m[1][:3, :3]]) for m in (mats(i) for i in pick)])
X0 = np.array([mats(i)[1][:3, 3] * 1000 for i in pick])

INF = IC.V4Inferencer(os.environ["CKPT"])
ds = LeRobotDataset(f"rebot/{os.path.basename(ROOT)}", root=ROOT, video_backend="pyav")
dim = INF.policy.model.dim_action; pred = np.full((len(pick), 16, 20), np.nan); t0 = time.time()
for n, i in enumerate(pick):
    x = ds[i]; assert int(x["index"]) == int(IX[i])
    obs = {f"observation.images.{k}": x[f"observation.images.{k}"] for k in ("global", "left_wrist", "right_wrist")}
    obs["observation.state"] = torch.as_tensor(FEED[i], dtype=torch.float32); obs["task"] = TASK_STR[int(TI[i])]
    g = torch.Generator().manual_seed(1234 + n); noise = torch.randn(1, INF.chunk, dim, generator=g)
    with torch.no_grad():
        ch = INF.policy.predict_action_chunk(INF.pre(obs), noise=noise.to(INF.device))
    pred[n] = INF.post(ch)[0].detach().float().cpu().numpy()[:, :20]
    if (n + 1) % 50 == 0: print(f"  {n+1}/{len(pick)} {time.time()-t0:.0f}s", flush=True)
B = np.array(binlab); GT = A[pick][:, :, :20]
np.savez(OUT, pick=np.array(pick), bins=B, pred=pred, gt=GT, Rt=Rt, x0=X0, ckpt=os.environ["CKPT"])

print(f"\n######## {TAG}  RIGHT arm, binned by current x (mm)   + = x fwd (away from base) / y LEFT (toward centre for R)")
print("  bin          n  x0med |k | GT dx  dy  | pred dx  dy  | e_x mean/med  e_y mean/med | near-0 GT: n  |pred| mean")
for name, _, _ in BINS:
    s = B == name
    if not s.any(): continue
    for k in (4, 8):
        gd = np.einsum("nij,nj->ni", Rt[s, 1], GT[s, k - 1, 10:13]) * 1000; pd_ = np.einsum("nij,nj->ni", Rt[s, 1], pred[s, k - 1, 10:13]) * 1000
        e = pd_ - gd; g8 = np.einsum("nij,nj->ni", Rt[s, 1], GT[s, 7, 10:13]) * 1000; z = np.linalg.norm(g8, axis=1) < 5
        print(f"  {name:12s} {s.sum():3d} {np.median(X0[s,0]):5.0f} |{k:2d}| {gd[:,0].mean():+5.1f} {gd[:,1].mean():+5.1f} | {pd_[:,0].mean():+6.1f} {pd_[:,1].mean():+5.1f} | "
              f"{e[:,0].mean():+6.1f}/{np.median(e[:,0]):+5.1f}  {e[:,1].mean():+6.1f}/{np.median(e[:,1]):+5.1f} | {z.sum():3d}  {np.linalg.norm(pd_[z],axis=1).mean() if z.any() else float('nan'):5.1f}")
