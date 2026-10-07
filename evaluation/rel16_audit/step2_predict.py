"""[2026-09-28] REL16 audit steps 2-4 -- model predictions under controlled counterfactuals (compute only).

Loads the checkpoint exactly as the deploy UI does (V4Inferencer: same pre/post processors, RENAME map, fp32 MPS)
and feeds TRAINING frames of r180_umi76_rel16_v1 (all 180 episodes were trained on; there is no val split).
For every sample and every condition the SAME initial flow samples are used (K draws, seeded by sample+draw),
so a difference between conditions is the condition, not the sampler.

conditions
    real        the recorded state76
    ident       history := current for both self and cross terms and the gripper (what a stop-and-go
                deployment observes: the arm has settled, so T(t-dt) == T(t))
    rev         history reflected through the current pose: self_h' = inv(D), cross_h' = cross_c @ inv(D),
                grip_h' = 2 g_c - g_h, with D = inv(T_t) T_{t-dt}; i.e. the same speed, opposite direction
    p0..p5      real state, order prompt replaced by task_index j (only j != own task is run)
strata
    moving      max-arm GT |A_16| > 40 mm and history speed > 2 mm / 50 ms
    rest_go     history speed < 0.5 mm / 50 ms on both arms but GT |A_16| > 30 mm (starts from rest)
    rest_rest   history < 0.5 mm and GT |A_16| < 5 mm
    ep_start    frame 0 of an episode
Output: step2_preds.npz (gt, state, task, stratum, preds[cond] (N,K,16,20)).
"""
import os, sys, time, json
import numpy as np, torch, pandas as pd, glob
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ.setdefault("V4_STATE_MODE", "umi76"); os.environ["V4_ACTION_MODE"] = "umi"   # umi94 for v2B
os.environ.setdefault("V4_CKPT", os.path.expanduser("~/holobrain-mac-model/ckpt_B180H-UMI76-REL16-D600K_400k"))
import infer_core_v4 as IC
from umi.common.pose_util import pose10d_to_mat, mat_to_pose10d
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/holobrain-data/lerobot/r180_umi76_rel16_v1"))
GRIP_BINARY = os.environ.get("GRIP_BINARY", "0") == "1"      # v2 datasets: action gripper = 1 OPEN / 0 CLOSE
POOL = os.environ.get("POOL", "")                              # HEAD | FRONT | "" (both): Stage-1 n>=30 HEAD re-check
TRUNC_STATE = int(os.environ.get("TRUNC_STATE", "0"))          # evaluator DRY-RUN ONLY: feed state[:N] to an older ckpt
OUT = os.path.expanduser(os.environ.get("OUT", "~/umi_bridge/rel16_audit/step2_preds.npz"))
K = int(os.environ.get("K", "2")); SEED = 1234
QUOTA = json.loads(os.environ["QUOTA"]) if os.environ.get("QUOTA") else {"moving": 25, "rest_go": 20, "rest_rest": 10, "ep_start": 10}
OFF = {0: dict(pos=0, pos_wrt=6, rot=12, rot_wrt=24, grip=36), 1: dict(pos=38, pos_wrt=44, rot=50, rot_wrt=62, grip=74)}

tasks = pd.read_parquet(f"{ROOT}/meta/tasks.parquet")
TASK_STR = {int(v): k for k, v in tasks["task_index"].items()}
import pyarrow.parquet as pq, pyarrow as pa
def _col(t, c):
    a = t.column(c).combine_chunks()
    return a.storage if isinstance(a, pa.ExtensionArray) else a
_tabs = [pq.read_table(f, columns=["observation.state", "action", "episode_index", "frame_index", "index", "task_index"])
         for f in sorted(glob.glob(f"{ROOT}/data/chunk-000/*.parquet"))]
df = pd.DataFrame({c: np.concatenate([np.asarray(_col(t, c).to_numpy(zero_copy_only=False)) for t in _tabs])
                   for c in ("episode_index", "frame_index", "index", "task_index")})
df["observation.state"] = [np.asarray(v) for t in _tabs for v in _col(t, "observation.state").to_pylist()]
df["action"] = [np.asarray(v) for t in _tabs for v in _col(t, "action").to_pylist()]
S = np.stack(df["observation.state"].values).astype(np.float64)
# [2026-09-29] FEED_ROOT (CART20 ablation): picks, strata and the saved `state` stay those of ROOT (v3d, state76), so both runs
# are scored on the SAME rows/images/noise; only the state FED to the model comes from FEED_ROOT's row with the same index.
# state76-only history ablations (ident/rev) do not exist for another state layout: they are not run and are stored as copies
# of "real" (analyze.py's [3] block is then meaningless for that ckpt).
FEED_ROOT = os.path.expanduser(os.environ.get("FEED_ROOT", ""))
FEED = None
if FEED_ROOT:
    _ft = [pq.read_table(f, columns=["observation.state", "index"]) for f in sorted(glob.glob(f"{FEED_ROOT}/data/chunk-000/*.parquet"))]
    _fi = np.concatenate([np.asarray(_col(t, "index").to_numpy()) for t in _ft])
    assert np.array_equal(_fi, df["index"].values), "FEED_ROOT rows are not in ROOT's order"
    FEED = np.stack([np.asarray(v) for t in _ft for v in _col(t, "observation.state").to_pylist()]).astype(np.float64)
    print(f"[feed] model state from {FEED_ROOT} width {FEED.shape[1]} (picks/strata from {ROOT})", flush=True)
A = np.stack([np.stack(a) for a in df["action"].values]).astype(np.float64)          # (n,16,20)
hist_mm = np.stack([np.linalg.norm(S[:, OFF[r]["pos"]:OFF[r]["pos"] + 3], axis=1) * 1000 for r in (0, 1)], 1)
gt16 = np.stack([np.linalg.norm(A[:, 15, r * 10:r * 10 + 3], axis=1) * 1000 for r in (0, 1)], 1)
H, G = hist_mm.max(1), gt16.max(1)
masks = {"moving": (G > 40) & (H > 2), "rest_go": (H < 0.5) & (G > 30), "rest_rest": (H < 0.5) & (G < 5),
         "ep_start": df["frame_index"].values == 0}
# gripper onset: jaw open (>60 mm) and not moving in the history, GT width closes by >20 mm within 16 steps
_gc = np.stack([S[:, OFF[r]["grip"] + 1] for r in (0, 1)], 1); _gh = np.stack([S[:, OFF[r]["grip"]] for r in (0, 1)], 1)
_g16 = np.stack([A[:, 15, r * 10 + 9] for r in (0, 1)], 1)
_go = (_gc > 0.060) & (np.abs(_gc - _gh) < 0.0005) & (_gc - _g16 > 0.020)
masks["grip_go"] = _go.any(1)
if GRIP_BINARY:
    # v2: jaw open (> 60 mm) and static in the state, label OPEN now and CLOSE by k16 (close intent onset)
    _l0 = np.stack([A[:, 0, r * 10 + 9] for r in (0, 1)], 1); _l16 = np.stack([A[:, 15, r * 10 + 9] for r in (0, 1)], 1)
    masks["grip_go"] = ((_gc > 0.060) & (np.abs(_gc - _gh) < 0.0005) & (_l0 >= 0.5) & (_l16 < 0.5)).any(1)   # binary or continuous
masks["grip_go_armrest"] = _go.any(1) & (H < 0.5)
if POOL:
    _pm = (df["episode_index"].values < 180) if POOL == "HEAD" else (df["episode_index"].values >= 180)
    masks = {k: v & _pm for k, v in masks.items()}
rng = np.random.default_rng(0)
pick, strat = [], []
for s, q in QUOTA.items():
    cand = np.flatnonzero(masks[s]); print(f"{s}: {len(cand)} candidates", flush=True)
    sel = rng.choice(cand, size=min(q, len(cand)), replace=False)
    pick += sorted(sel.tolist()); strat += [s] * len(sel)
pick = np.array(pick)


def m10(v):
    return pose10d_to_mat(np.asarray(v, np.float64)[None])[0]


def ablate(st, mode):
    st = st.copy()
    for r in (0, 1):
        f = OFF[r]
        sh = m10(np.r_[st[f["pos"]:f["pos"] + 3], st[f["rot"]:f["rot"] + 6]])
        cc = m10(np.r_[st[f["pos_wrt"] + 3:f["pos_wrt"] + 6], st[f["rot_wrt"] + 6:f["rot_wrt"] + 12]])
        if mode == "ident":
            nsh, nch, gh = np.eye(4), cc, st[f["grip"] + 1]
        else:                                           # rev
            Di = np.linalg.inv(sh)
            nsh, nch, gh = Di, cc @ Di, 2 * st[f["grip"] + 1] - st[f["grip"]]
        a, b = mat_to_pose10d(nsh[None])[0], mat_to_pose10d(nch[None])[0]
        st[f["pos"]:f["pos"] + 3] = a[:3]; st[f["rot"]:f["rot"] + 6] = a[3:9]
        st[f["pos_wrt"]:f["pos_wrt"] + 3] = b[:3]; st[f["rot_wrt"]:f["rot_wrt"] + 6] = b[3:9]
        st[f["grip"]] = gh
    return st


INF = IC.V4Inferencer(os.environ["V4_CKPT"])
ds = LeRobotDataset(f"rebot/{os.path.basename(ROOT)}", root=ROOT, video_backend="pyav")
dim = INF.policy.model.dim_action
conds = ["real", "ident", "rev"] + [f"p{j}" for j in range(6)]
preds = {c: np.full((len(pick), K, 16, 20), np.nan) for c in conds}
t0 = time.time()
for n, idx in enumerate(pick):
    x = ds[int(idx)]
    assert int(x["index"]) == int(df["index"].iloc[idx]) and np.allclose(x["observation.state"].numpy(), S[idx], atol=1e-6)
    imgs = {f"observation.images.{k}": x[f"observation.images.{k}"] for k in ("global", "left_wrist", "right_wrist")}
    own = int(x["task_index"])
    for c in conds:
        if c.startswith("p") and int(c[1:]) == own:
            continue
        if FEED is not None and c in ("ident", "rev"):
            continue
        st = S[idx] if c in ("real",) or c.startswith("p") else ablate(S[idx], c)
        if FEED is not None:
            st = FEED[idx]
        if TRUNC_STATE:
            st = st[:TRUNC_STATE]
        task = TASK_STR[int(c[1:])] if c.startswith("p") else TASK_STR[own]
        for d in range(K):
            g = torch.Generator().manual_seed(SEED + 1000 * n + d)
            noise = torch.randn(1, INF.chunk, dim, generator=g)
            obs = dict(imgs); obs["observation.state"] = torch.as_tensor(st, dtype=torch.float32); obs["task"] = task
            batch = INF.pre(obs)
            with torch.no_grad():
                ch = INF.policy.predict_action_chunk(batch, noise=noise.to(INF.device))
            preds[c][n, d] = INF.post(ch)[0].detach().float().cpu().numpy()[:, :20]
    if FEED is not None:
        preds["ident"][n] = preds["real"][n]; preds["rev"][n] = preds["real"][n]
    el = time.time() - t0
    print(f"[{n+1}/{len(pick)}] {strat[n]} ep{int(x['episode_index'])} f{int(x['frame_index'])} {el:.0f}s", flush=True)
    if (n + 1) % 10 == 0 or n + 1 == len(pick):
        np.savez(OUT, pick=pick, stratum=np.array(strat), gt=A[pick], state=S[pick], task=df["task_index"].values[pick],
                 ep=df["episode_index"].values[pick], frame=df["frame_index"].values[pick], done=n + 1,
                 **{f"pred_{c}": v for c, v in preds.items()}, ckpt=os.environ["V4_CKPT"])
print("done", OUT)
