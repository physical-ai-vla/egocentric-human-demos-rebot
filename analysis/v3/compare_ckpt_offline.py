"""[2026-10-02] Teacher-forced offline comparison of REL-only checkpoints on the SAME frames.
Frames: N random HEAD180 rows (r180_umi76_rel16_v3d images, r180_relcart20_rel16_v3d state/action -- the HEAD180 part of R312c,
so in-training for every R312c run), seed 0; one seeded flow draw per frame, identical across checkpoints. V4_RELONLY=1 (zero mask).
Per k in {1,4,8,16}: base-frame position error |pred - GT| (mm, both arms), signed mean per axis, and for frames whose GT moves
< 5 mm by k8 the predicted motion (residual motion where the demo stands still).
usage: compare_ckpt_offline.py <ckpt_dir> [<ckpt_dir> ...]   (N_FRAMES=200)"""
import os, sys, glob, time
import numpy as np, torch, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
os.environ.update(V4_STATE_MODE="relcart20", V4_ACTION_MODE="umi", V4_RELONLY="1", V4_CKPT=sys.argv[1])
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
import infer_core_v4 as IC
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ROOT = os.path.expanduser("~/holobrain-data/lerobot/r180_umi76_rel16_v3d"); FEED = os.path.expanduser("~/holobrain-data/lerobot/r180_relcart20_rel16_v3d")
col = lambda t, c: (lambda a: a.storage if isinstance(a, pa.ExtensionArray) else a)(t.column(c).combine_chunks())
ft = [pq.read_table(f, columns=["observation.state", "action", "index", "task_index", "aux.q_t"]) for f in sorted(glob.glob(f"{FEED}/data/chunk-000/*.parquet"))]
IX = np.concatenate([col(t, "index").to_numpy() for t in ft]); TI = np.concatenate([col(t, "task_index").to_numpy() for t in ft])
Q = np.concatenate([np.asarray(col(t, "aux.q_t").to_pylist()) for t in ft]); ST = np.concatenate([np.asarray(col(t, "observation.state").to_pylist()) for t in ft])
A = np.concatenate([np.asarray(col(t, "action").to_pylist()) for t in ft])[:, :, :20]
TS = {int(v): k for k, v in pd.read_parquet(f"{FEED}/meta/tasks.parquet")["task_index"].items()}
N = int(os.environ.get("N_FRAMES", "200")); pick = np.random.default_rng(0).choice(len(IX), N, replace=False)
class K_: kin = IC.eef_kin.Kin(); _tcp_mat = IC.V4Inferencer._tcp_mat
kk = K_(); Rt = []
for i in pick:
    q14 = np.zeros(14); q14[IC.ARM_IDX] = Q[i]; m = kk._tcp_mat(q14)[0]; Rt.append(np.stack([m[0][:3, :3], m[1][:3, :3]]))
Rt = np.stack(Rt)
ds = LeRobotDataset(f"rebot/{os.path.basename(ROOT)}", root=ROOT, video_backend="pyav")
obs_cache = []
for i in pick:
    x = ds[int(i)]; assert int(x["index"]) == int(IX[i])
    obs_cache.append({f"observation.images.{k}": x[f"observation.images.{k}"] for k in ("global", "left_wrist", "right_wrist")})
GT = A[pick]
def run(ck):
    IC.CKPT = ck; INF = IC.V4Inferencer(ck); dim = INF.policy.model.dim_action; P = np.zeros((N, 16, 20))
    for n, i in enumerate(pick):
        o = dict(obs_cache[n]); o["observation.state"] = torch.as_tensor(ST[i], dtype=torch.float32); o["task"] = TS[int(TI[i])]
        noise = torch.randn(1, INF.chunk, dim, generator=torch.Generator().manual_seed(1234 + n))
        with torch.no_grad():
            P[n] = INF.post(INF.policy.predict_action_chunk(INF.pre(o), noise=noise.to(INF.device)))[0].detach().float().cpu().numpy()[:, :20]
    del INF; return P
print(f"{N} HEAD180 frames, same noise for every checkpoint")
for ck in sys.argv[1:]:
    t0 = time.time(); P = run(ck); name = os.path.basename(ck.rstrip("/"))
    line = []
    for k in (1, 4, 8, 16):
        gd = np.concatenate([np.einsum("nij,nj->ni", Rt[:, r], GT[:, k - 1, r * 10:r * 10 + 3]) for r in (0, 1)]) * 1000
        pd_ = np.concatenate([np.einsum("nij,nj->ni", Rt[:, r], P[:, k - 1, r * 10:r * 10 + 3]) for r in (0, 1)]) * 1000
        e = pd_ - gd; err = np.linalg.norm(e, axis=1)
        line.append(f"k{k:<2d} err med {np.median(err):5.1f} p90 {np.percentile(err, 90):5.1f} | signed xyz {np.round(e.mean(0), 1)}")
    g8 = np.concatenate([np.einsum("nij,nj->ni", Rt[:, r], GT[:, 7, r * 10:r * 10 + 3]) for r in (0, 1)]) * 1000
    p8 = np.concatenate([np.einsum("nij,nj->ni", Rt[:, r], P[:, 7, r * 10:r * 10 + 3]) for r in (0, 1)]) * 1000
    z = np.linalg.norm(g8, axis=1) < 5
    gerr = np.abs(P[:, :, [9, 19]] - GT[:, :, [9, 19]]).mean()
    print(f"\n## {name}  ({time.time() - t0:.0f}s)"); [print("  " + l) for l in line]
    print(f"  near-still frames (GT k8 < 5 mm): n {z.sum()} | predicted k8 motion mean {np.linalg.norm(p8[z], axis=1).mean():.1f} mm | gripper |err| mean {gerr:.3f}")
