"""[2026-09-28] Prompt-swap sensitivity for the joint-space models that DO stack (C-old R150 FT, scratch B1-old),
on the SAME scenes as the REL16 audit (r180 row f of episode e < 150 == R150 headview source frame 2f+2).

Loaded as the e280 serving core does (IDENTITY normalization, RENAME map, 3 cams, robot prompt), state =
[q_L6 rad, gL (raw <= -135 -> 1), q_R6 rad, gR]. Each prompt is run with the SAME seeded flow samples, so the
prompt effect is compared against the draw-to-draw spread of one prompt. Metric: FK TCP of q + dq at chunk step
12 and 24 (0.4 / 0.8 s at 30 Hz, the REL16 k=8 / k=16 horizons), on the arm with the larger GT REL16 motion.
Usage: promptswap_joint.py <ckpt> [<ckpt> ...]   (reads ep/frame/stratum from step2_preds.npz)
"""
import os, sys, time, glob, json
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ.setdefault("E280_CAMS", "middle,left,right"); os.environ.setdefault("E280_PROMPT_STYLE", "robot")
import infer_core_e280 as E
import eef_kin
from lerobot.datasets.lerobot_dataset import LeRobotDataset

AUD = os.path.expanduser("~/umi_bridge/rel16_audit")
SRC = os.path.expanduser("~/holobrain-data/lerobot/src_rebot_3stack_R150_headview")
K = 2; SEED = 1234; STRATA = {"moving": 15, "rest_go": 15}
TASK_STR = {int(v): k for k, v in pd.read_parquet(os.path.expanduser(
    "~/holobrain-data/lerobot/r180_umi76_rel16_v1/meta/tasks.parquet"))["task_index"].items()}

d = np.load(f"{AUD}/step2_preds.npz", allow_pickle=True)
n = int(d["done"])
gt = d["gt"][:n]
act = np.argmax(np.stack([np.linalg.norm(gt[:, 15, r * 10:r * 10 + 3], axis=1) for r in (0, 1)], 1), 1)
sel = []
for s, q in STRATA.items():
    idx = [i for i in range(n) if d["stratum"][i] == s and d["ep"][i] < 150][:q]
    sel += idx
ds = LeRobotDataset("rebot/rebot_3stack_R150_headview", root=SRC, video_backend="pyav")
kin = eef_kin.Kin()


def tcp_pos(q12):
    L7, R7 = kin.fk_pose7(q12)
    return np.stack([np.asarray(L7)[:3], np.asarray(R7)[:3]]) * 1000.0


out = {}
for ck in sys.argv[1:]:
    inf = E.XVLAInferencer(ck)
    dim = inf.policy.model.dim_action; T = inf.policy.config.chunk_size
    res = []
    t0 = time.time()
    for j, i in enumerate(sel):
        e, f = int(d["ep"][i]), int(d["frame"][i]); src_i = 2 * f + 2
        gi = int(ds.meta.episodes[e]["dataset_from_index"]) + src_i
        x = ds[gi]
        assert int(x["episode_index"]) == e and int(x["frame_index"]) == src_i
        q = x["observation.state"].numpy().astype(np.float64).copy()
        q[E.ARM_IDX] = np.radians(q[E.ARM_IDX])
        g = np.where(q[E.GRIP_IDX] > 180, q[E.GRIP_IDX] - 360, q[E.GRIP_IDX])
        st = q.copy(); st[E.GRIP_IDX] = (g <= E.G_STATE_CLOSED).astype(np.float64)
        imgs = {f"observation.images.{E.CAM_MAP[c]}": x[f"observation.images.{E.CAM_MAP[c]}"] for c in E.CAMS}
        P = np.zeros((6, K, 2, 2, 3))                     # prompt, draw, horizon(12,24), arm, xyz
        G = np.zeros((6, K, 2))
        for p in range(6):
            for k in range(K):
                gen = torch.Generator().manual_seed(SEED + 1000 * j + k)
                noise = torch.randn(1, T, dim, generator=gen)
                obs = dict(imgs); obs["observation.state"] = torch.tensor(st, dtype=torch.float32); obs["task"] = TASK_STR[p]
                b = inf.pre(obs)
                with torch.no_grad():
                    ch = inf.policy.predict_action_chunk(b, noise=noise.to(inf.device))
                pred = np.stack([inf.post(ch[:, t, :]).squeeze(0).float().cpu().numpy() for t in range(T)])[:, :14]
                for h, t in enumerate((11, 23)):
                    P[p, k, h] = tcp_pos(q[E.ARM_IDX] + pred[t, E.ARM_IDX])
                G[p, k] = pred[23, E.GRIP_IDX]
        now = tcp_pos(q[E.ARM_IDX])
        res.append(dict(i=int(i), stratum=str(d["stratum"][i]), ep=e, f=f, task=int(d["task"][i]), arm=int(act[i]),
                        P=P.tolist(), now=now.tolist(), G=G.tolist()))
        print(f"[{os.path.basename(ck)} {j+1}/{len(sel)}] {res[-1]['stratum']} ep{e} f{f} {time.time()-t0:.0f}s", flush=True)
    out[os.path.basename(ck)] = res
    del inf; torch.mps.empty_cache()
    json.dump(out, open(f"{AUD}/promptswap_joint.json", "w"))

for name, res in out.items():
    print(f"\n== {name}")
    for s in STRATA:
        rows = [r for r in res if r["stratum"] == s]
        for h, lab in ((0, "0.4s"), (1, "0.8s")):
            pd_, dd_, mv = [], [], []
            for r in rows:
                P = np.array(r["P"])[:, :, h, r["arm"]]           # (6,K,3)
                M = P.mean(1)
                pd_.append(np.mean([np.linalg.norm(M[a] - M[b]) for a in range(6) for b in range(a + 1, 6)]))
                dd_.append(np.mean([np.linalg.norm(P[p, 0] - P[p, 1]) for p in range(6)]))
                mv.append(np.linalg.norm(M[r["task"]] - np.array(r["now"])[r["arm"]]))
            print(f"{s:8s} {lab}: prompt pairwise {np.median(pd_):6.1f} mm | same-prompt draw spread {np.median(dd_):6.1f} mm | "
                  f"own-prompt motion {np.median(mv):6.1f} mm")
