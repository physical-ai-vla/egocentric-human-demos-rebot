"""[2026-09-30] Left-arm y-magnitude bias check (user): the live :8033 RELCART20-v3 130k left arm overshoots toward -y (goes to
y -7..-54 mm where training's first blue grasp is at y ~ +102 mm). Is the bias already in the teacher-forced prediction?

Same loading path as step2_predict.py (V4Inferencer pre/post, LeRobotDataset frames of r180_umi76_rel16_v3d, RELCART20 state fed
from r180_relcart20_rel16_v3d with the same row order). TRAINING frames (no val split exists for HEAD180), so this is a
teacher-forced fit check, not generalisation.

Samples (fixed, seed 0): per episode of each order prompt, the L-arm APPROACH phase = frames before the L arm's first
OPEN->CLOSE of the continuous leader gripper label (g_k1 crosses 0.5), cut into early / mid / late (one frame each), plus the
first frame of the R arm's approach for a control. One flow draw per sample (seeded).

Per k in {1, 4, 8, 16}: base-frame (R_t from FK(aux.q_t) + V4_FRAME_FIX, the deploy _tcp_mat) Δ = R_t p_REL,k (mm), for pred and GT:
e = Δpred - ΔGT (x, y, z), mean / median / p90|e|, and the magnitude ratio Δpred_y / ΔGT_y over samples with |ΔGT_y| > 10 mm.
usage: CKPT=<local ckpt dir> TAG=<name> ybias_eval.py        (pause robot UIs: shares the MPS)
"""
import os, sys, time, json, glob
import numpy as np, torch, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ["V4_STATE_MODE"] = "relcart20"; os.environ["V4_ACTION_MODE"] = "umi"; os.environ["V4_CKPT"] = os.environ["CKPT"]
import infer_core_v4 as IC
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.expanduser("~/holobrain-data/lerobot/r180_umi76_rel16_v3d")
FEED_ROOT = os.path.expanduser("~/holobrain-data/lerobot/r180_relcart20_rel16_v3d")
TAG = os.environ.get("TAG", os.path.basename(os.environ["CKPT"]))
OUT = os.path.expanduser(f"~/umi_bridge/rel16_audit/ybias_{TAG}.npz")
col = lambda t, c: (lambda a: a.storage if isinstance(a, pa.ExtensionArray) else a)(t.column(c).combine_chunks())
tabs = [pq.read_table(f, columns=["action", "episode_index", "frame_index", "index", "task_index", "aux.q_t"])
        for f in sorted(glob.glob(f"{ROOT}/data/chunk-000/*.parquet"))]
A = np.concatenate([np.asarray(col(t, "action").to_pylist(), np.float64) for t in tabs])            # (N,16,32)
E = np.concatenate([col(t, "episode_index").to_numpy() for t in tabs]); TI = np.concatenate([col(t, "task_index").to_numpy() for t in tabs])
IX = np.concatenate([col(t, "index").to_numpy() for t in tabs]); Q = np.concatenate([np.asarray(col(t, "aux.q_t").to_pylist()) for t in tabs])
ft = [pq.read_table(f, columns=["observation.state", "index"]) for f in sorted(glob.glob(f"{FEED_ROOT}/data/chunk-000/*.parquet"))]
assert np.array_equal(np.concatenate([col(t, "index").to_numpy() for t in ft]), IX), "FEED rows not in ROOT order"
FEED = np.concatenate([np.asarray(col(t, "observation.state").to_pylist(), np.float64) for t in ft])
tasks = pd.read_parquet(f"{ROOT}/meta/tasks.parquet"); TASK_STR = {int(v): k for k, v in tasks["task_index"].items()}
first = {j: s.split("the ")[1].split(" cube")[0] for j, s in TASK_STR.items()}

# approach-phase picks
rng = np.random.default_rng(0); pick, meta = [], []
for e in np.unique(E):
    idx = np.flatnonzero(E == e); fc = first[int(TI[idx[0]])]
    for arm, gd in ((0, 9), (1, 19)):
        g = A[idx, 0, gd]; op = np.flatnonzero(g > 0.5)
        if len(op) == 0: continue
        cl = op[0] + np.flatnonzero(g[op[0]:] < 0.5)
        if len(cl) == 0 or cl[0] < 6: continue
        if arm == 0:
            for ph, (a, b) in (("early", (0.0, 0.33)), ("mid", (0.33, 0.66)), ("late", (0.66, 1.0))):
                lo, hi = int(a * cl[0]), max(int(a * cl[0]) + 1, int(b * cl[0]))
                pick.append(int(idx[rng.integers(lo, hi)])); meta.append((int(e), fc, "L", ph))
        else:
            pick.append(int(idx[rng.integers(0, max(1, cl[0] // 2))])); meta.append((int(e), fc, "R", "early"))
print(f"[{TAG}] {len(pick)} samples", flush=True)


class K_: kin = IC.eef_kin.Kin(); _tcp_mat = IC.V4Inferencer._tcp_mat
kk = K_(); Rt = []
for i in pick:
    q14 = np.zeros(14); q14[IC.ARM_IDX] = Q[i]; m, _ = kk._tcp_mat(q14); Rt.append(np.stack([m[0][:3, :3], m[1][:3, :3]]))
Rt = np.stack(Rt)

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
    if (n + 1) % 25 == 0: print(f"  {n+1}/{len(pick)} {time.time()-t0:.0f}s", flush=True)
np.savez(OUT, pick=np.array(pick), meta=np.array(meta, dtype=object), pred=pred, gt=A[pick][:, :, :20], Rt=Rt, ckpt=os.environ["CKPT"])

M = np.array(meta, dtype=object)
def report(sel, arm, title):
    r = 0 if arm == "L" else 1
    print(f"\n== {title}  (n {sel.sum()})  base frame mm, + = x fwd / y LEFT / z up")
    print("   k  | e_x mean/med | e_y mean/med  p90|e_y| | e_z mean/med | GT_y med  pred_y med  ratio pred_y/GT_y med (|GT_y|>10)")
    for k in (1, 4, 8, 16):
        gd = np.einsum("nij,nj->ni", Rt[sel, r], A[pick][sel, k - 1, r * 10:r * 10 + 3]) * 1000
        pd_ = np.einsum("nij,nj->ni", Rt[sel, r], pred[sel, k - 1, r * 10:r * 10 + 3]) * 1000
        e = pd_ - gd; mv = np.abs(gd[:, 1]) > 10
        ratio = np.median(pd_[mv, 1] / gd[mv, 1]) if mv.any() else np.nan
        print(f"  {k:3d} | {e[:,0].mean():+6.1f}/{np.median(e[:,0]):+6.1f} | {e[:,1].mean():+6.1f}/{np.median(e[:,1]):+6.1f}  {np.percentile(np.abs(e[:,1]),90):6.1f} | "
              f"{e[:,2].mean():+6.1f}/{np.median(e[:,2]):+6.1f} | {np.median(gd[:,1]):+7.1f}  {np.median(pd_[:,1]):+7.1f}   {ratio:5.2f} (n {mv.sum()})")

print(f"\n######## {TAG}  ckpt {os.environ['CKPT']}")
bl = (M[:, 1] == "blue") & (M[:, 2] == "L")
report(bl, "L", "BLUE-first, LEFT arm, approach (all phases)")
for ph in ("early", "mid", "late"):
    report(bl & (M[:, 3] == ph), "L", f"BLUE-first, LEFT arm, approach {ph}")
report((M[:, 1] != "blue") & (M[:, 2] == "L"), "L", "RED/PURPLE-first, LEFT arm, approach (all phases)")
report(M[:, 2] == "R", "R", "all prompts, RIGHT arm, approach early (control)")
