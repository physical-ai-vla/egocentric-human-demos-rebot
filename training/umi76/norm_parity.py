"""[2026-09-28] Normalization 3-way parity for a REL16-v2 dataset (+ optionally a checkpoint). Read-only.

 L1 independent recompute from the dataset's parquet (action over frames x 16 steps, state over frames)
 L2 meta/stats.json of the dataset
 L3 the checkpoint's normalizer / unnormalizer tensors (policy_*_normalizer*.safetensors), and the actual
    normalize(x) / unnormalize(y) outputs of the checkpoint's pre/post processors on real batches

Correctness = L1 == L2 == L3 within tolerance, in the SAME semantic ordering (action20 = per arm [pos3 rot6d grip],
gripper dims 9/19; state94 = state76 + TCP18 at 76:94). ACTION is MEAN_STD (the only normalized feature), STATE and
VISUAL are IDENTITY: state stats are still compared (they must not be stale) but a checkpoint may not store them.
Stale-file detector only: the whole stats vector must NOT equal a given older dataset's (e.g. R380).
usage: norm_parity.py <dataset dir> [--ckpt <pretrained_model dir>] [--stale <older dataset dir>]
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser(); ap.add_argument("ds"); ap.add_argument("--ckpt"); ap.add_argument("--stale")
a = ap.parse_args(); fails = []


def need(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def col(t, c):
    x = t.column(c).combine_chunks(); return x.storage if isinstance(x, pa.ExtensionArray) else x


tabs = [pq.read_table(f, columns=["action", "observation.state", "episode_index"]) for f in sorted(glob.glob(f"{a.ds}/data/chunk-*/*.parquet"))]
A32 = np.concatenate([np.asarray(col(t, "action").to_pylist(), np.float32) for t in tabs])
S32 = np.concatenate([np.asarray(col(t, "observation.state").to_pylist(), np.float32) for t in tabs])
EPI = np.concatenate([np.asarray(col(t, "episode_index").to_numpy()) for t in tabs])
AD = A32.shape[-1]; A, S = A32.reshape(-1, AD).astype(np.float64), S32.astype(np.float64)
st = json.load(open(f"{a.ds}/meta/stats.json"))
# L1 = LeRobot's OWN algorithm run independently of every stored stats file: per-episode get_feature_stats on the
# float32 rows (action flattened over frames x 16 steps, as the writer does), then aggregate_stats (parallel variance).
# A plain global std differs from this by float32 accumulation (~2e-4 abs), and on near-constant dims (the identity
# "current" block of state76, std ~1e-7) any relative criterion is meaningless -- so the tolerance is absolute + relative.
from lerobot.datasets.compute_stats import get_feature_stats, aggregate_stats
per = []
for e in np.unique(EPI):
    m = EPI == e
    per.append({"action": get_feature_stats(A32[m].reshape(-1, AD), axis=0, keepdims=False),
                "observation.state": get_feature_stats(S32[m], axis=0, keepdims=False)})
agg = aggregate_stats(per)
L1 = {k: (np.asarray(agg[k]["mean"], np.float64), np.asarray(agg[k]["std"], np.float64)) for k in ("action", "observation.state")}
for k, (m, s_) in L1.items():
    m2, s2 = np.asarray(st[k]["mean"]), np.asarray(st[k]["std"])
    need(m2.shape == m.shape, f"L1/L2 {k} dim {m.shape[0]} == stats.json {m2.shape[0]}")
    dm = np.abs(m - m2).max(); dsa = np.abs(s_ - s2).max(); dsr = (np.abs(s_ - s2) / np.maximum(s_, 1e-3)).max()
    need(dm < 1e-5 and dsa < 1e-5 and dsr < 1e-4, f"L1 independent LeRobot recompute == L2 stats.json {k}: max|dmean| {dm:.1e}, max|dstd| {dsa:.1e}, rel {dsr:.1e}")
if os.environ.get("GRIP", "binary") == "continuous":      # [2026-09-29] variant C: continuous leader command in [0, 1]
    need(A[:, [9, 19]].min() >= 0 and A[:, [9, 19]].max() <= 1, f"gripper dims 9/19 continuous in [0,1]; mean L/R {A[:, 9].mean():.3f}/{A[:, 19].mean():.3f}")
else:
    need(set(np.unique(A[:, [9, 19]]).tolist()) <= {0.0, 1.0}, f"gripper dims 9/19 binary; mean L/R {A[:, 9].mean():.3f}/{A[:, 19].mean():.3f}")
if S.shape[1] == 94:
    need(0.05 < S[:, 76].mean() < 0.6 and S[:, 79:85].std() > 0.01, f"TCP18 at 76:94 looks like pos+rot6d (L x mean {S[:, 76].mean():.3f})")
h = hashlib.sha256(json.dumps({k: st[k] for k in ("action", "observation.state")}, sort_keys=True).encode()).hexdigest()
print(f"stats hash (action+state) {h[:16]}")
if a.stale:
    so = json.load(open(f"{a.stale}/meta/stats.json"))
    ho = hashlib.sha256(json.dumps({k: so[k] for k in ("action", "observation.state")}, sort_keys=True).encode()).hexdigest()
    need(h != ho, f"stale detector: stats differ from {os.path.basename(a.stale.rstrip('/'))} ({ho[:16]})")
    for k in ("action", "observation.state"):
        m, mo = np.asarray(st[k]["mean"]), np.asarray(so[k]["mean"])
        if m.shape == mo.shape:
            print(f"      diagnostic {k}: dims whose mean moved > 1e-6 vs stale: {int((np.abs(m - mo) > 1e-6).sum())}/{m.size}")
if a.ckpt:
    import torch
    from safetensors.torch import load_file
    fs = sorted(glob.glob(f"{a.ckpt}/policy_*normalizer*.safetensors"))
    need(len(fs) >= 2, f"checkpoint has pre+post normalizer files ({[os.path.basename(f) for f in fs]})")
    for f in fs:
        t = load_file(f)
        for key in ("action", "observation.state"):
            mk = [k for k in t if k.startswith(key + ".") and k.endswith("mean")]
            sk = [k for k in t if k.startswith(key + ".") and k.endswith("std")]
            if not mk:
                continue
            m3, s3 = t[mk[0]].double().numpy(), t[sk[0]].double().numpy()
            m2, s2 = np.asarray(st[key]["mean"]), np.asarray(st[key]["std"])
            need(m3.shape == m2.shape and np.abs(m3 - m2).max() < 1e-6 and np.abs(s3 - s2).max() < 1e-6,
                 f"L3 {os.path.basename(f)} {key} mean/std == stats.json (max diff {np.abs(m3 - m2).max():.1e}/{np.abs(s3 - s2).max():.1e})")
    # functional parity: the checkpoint's processors on real rows vs (x - mean) / std from stats.json
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(a.ckpt); cfg.device = "cpu"
    pre, post = make_pre_post_processors(cfg, pretrained_path=a.ckpt, preprocessor_overrides={"device_processor": {"device": "cpu"}},
                                         postprocessor_overrides={"device_processor": {"device": "cpu"}})
    norm = [s for s in pre.steps if "Normalizer" in type(s).__name__][0]
    unn = [s for s in post.steps if "Unnormalizer" in type(s).__name__][0]
    x = torch.tensor(A[np.random.default_rng(0).choice(len(A), 64, replace=False)], dtype=torch.float32)
    m2, s2 = torch.tensor(st["action"]["mean"], dtype=torch.float32), torch.tensor(st["action"]["std"], dtype=torch.float32)
    y = norm._normalize_tensor(x, "action") if hasattr(norm, "_normalize_tensor") else None
    if y is None:
        from lerobot.processor import TransitionKey
        tr = {TransitionKey.ACTION: x.clone(), TransitionKey.OBSERVATION: {}}
        y = norm(tr)[TransitionKey.ACTION]
    ref = (x - m2) / (s2 + 1e-8)
    need(torch.allclose(y, ref, atol=1e-4), f"L3 normalize(action) == (x-mean)/std from stats.json (max {float((y - ref).abs().max()):.1e})")
    back = unn(dict({__import__('lerobot.processor', fromlist=['TransitionKey']).TransitionKey.ACTION: y.clone(),
                     __import__('lerobot.processor', fromlist=['TransitionKey']).TransitionKey.OBSERVATION: {}}))
    from lerobot.processor import TransitionKey
    xb = back[TransitionKey.ACTION]
    need(torch.allclose(xb, x, atol=1e-4), f"L3 unnormalize(normalize(x)) == x (max {float((xb - x).abs().max()):.1e})")
print("\nNORM PARITY PASS" if not fails else f"\nNORM PARITY FAILED: {fails}")
sys.exit(1 if fails else 0)
