"""[2026-09-30] CART-ONLY pre-training preflight (CPU, no GPU) for ego_relcart20task_rel16_v4_nopseudoq-style datasets.

Replaces preflight_rel16v2.py for the ego cart-only pretrain (that one requires aux.q_t at action dim 32). Same code path as
lerobot_train (base config + launcher overrides -> resolve_delta_timestamps -> LeRobotDataset -> make_policy /
make_pre_post_processors with the dataset stats) with rel16_aux_masked installed. Asserts:
  P1 no delta expansion on action; sample action (16, 32), state (sdim,), bit-identical to the stored parquet row
  P2 dataset has no aux.q_t / aux.fk_mask feature; has_dq_supervision == has_fk_supervision == 0 on every row (parquet scan)
  P3 action dims 0:20 finite, dims 20:32 NaN on every row
  P4 full policy forward + backward on one batch: total loss finite, rel_loss finite > 0, dq_loss == fk_loss == 0 exactly
  P5 no q_t access: batch carries no aux.q_t, action space _rel16_qt is None, FK (ReBotFKTorch.tcp) called 0 times
  P6 action_decoder grad for output dims 20:32 exactly 0 in every domain, dims 0:20 non-zero
usage: preflight_cartonly.py <dataset root> <base dir> <state dim>     (env REL16_STATS = <dataset root>/meta/stats.json)
"""
import glob, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rel16_aux_masked as RM
RM.install()
sys.path.insert(0, "/home/bh-aiteam/c8old/xvla")
import rebot_fk_torch
import lerobot.policies.factory as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps

root, base, sdim = sys.argv[1], sys.argv[2], int(sys.argv[3]); fails = []
RENAME = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}


def need(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    if not ok: fails.append(msg)


def col(t, c):
    a = t.column(c).combine_chunks(); return a.storage if isinstance(a, pa.ExtensionArray) else a


FK_CALLS = [0]; _tcp = rebot_fk_torch.ReBotFKTorch.tcp
def _count(self, *a, **k): FK_CALLS[0] += 1; return _tcp(self, *a, **k)
rebot_fk_torch.ReBotFKTorch.tcp = _count

cfg = PreTrainedConfig.from_pretrained(base); cfg.pretrained_path = base
for k, v in dict(device="cpu", dtype="float32", chunk_size=16, n_action_steps=16, max_state_dim=sdim, max_action_dim=32, action_mode="auto", use_proprio=True,
                 freeze_vision_encoder=False, freeze_language_encoder=False, train_policy_transformer=True, train_soft_prompts=True).items(): setattr(cfg, k, v)
meta = LeRobotDatasetMetadata("rebot/preflight", root=root); dts = resolve_delta_timestamps(cfg, meta)
need("action" not in (dts or {}), f"P1 resolve_delta_timestamps leaves action unexpanded (delta keys {sorted((dts or {}).keys())})")
need(not any(k in meta.features for k in ("aux.q_t", "aux.fk_mask")) and all(k in meta.features for k in RM.MASK_KEYS), f"P2 features: no aux.q_t / aux.fk_mask, mask columns present")

A, M = [], []
for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet")):
    t = pq.read_table(f, columns=["action", "index"] + list(RM.MASK_KEYS))
    A.append(np.asarray(col(t, "action").to_pylist(), np.float32)); M.append(np.stack([col(t, k).to_numpy() for k in RM.MASK_KEYS], 1))
    IX = col(t, "index").to_numpy() if len(A) == 1 else IX
A0 = A[0]; A, M = np.concatenate(A), np.concatenate(M)
need(bool((M == 0).all()), f"P2 has_dq / has_fk == 0 on all {len(M)} rows")
need(bool(np.isfinite(A[..., :20]).all() and np.isnan(A[..., 20:]).all()), "P3 action dims 0:20 finite, dims 20:32 NaN on every row")

ds = LeRobotDataset("rebot/preflight", root=root, delta_timestamps=dts, video_backend="pyav")
items = [ds[i] for i in (0, len(ds) // 2)]
for x in items:
    j = int(x["index"]); stored = A0[np.flatnonzero(IX == j)[0]] if j in IX else None
    need(tuple(x["action"].shape) == (16, 32) and tuple(x["observation.state"].shape) == (sdim,), f"P1 sample {j}: action {tuple(x['action'].shape)} state {tuple(x['observation.state'].shape)}")
    if stored is not None:
        need(np.array_equal(x["action"].numpy(), stored, equal_nan=True), f"P1 sample {j}: loader action == stored parquet row (bit-for-bit, NaN-aware)")

policy = F.make_policy(cfg=cfg, ds_meta=meta, rename_map=RENAME); policy.train()
pre, _ = F.make_pre_post_processors(policy_cfg=cfg, pretrained_path=base, preprocessor_overrides={
    "device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": RENAME},
    "normalizer_processor": {"stats": meta.stats, "features": {**policy.config.input_features, **policy.config.output_features}, "norm_map": policy.config.normalization_mapping}})
batch = pre(torch.utils.data.default_collate(items))
need("aux.q_t" not in batch and all(k in batch for k in RM.MASK_KEYS), "P5 batch has mask keys and no aux.q_t")
loss, log = policy.forward(batch); loss.backward()
need(bool(torch.isfinite(loss)) and log["rel_loss"] > 0, f"P4 losses finite: {log}")
need(log["dq_loss"] == 0.0 and log["fk_loss"] == 0.0, "P4 dq_loss == fk_loss == 0 exactly")
need(policy.model.action_space._rel16_qt is None and FK_CALLS[0] == 0, f"P5 no q_t access (action space q_t None, FK calls {FK_CALLS[0]})")
dec = [m for n, m in policy.named_modules() if n.endswith("action_decoder")][0]
W = dec.fc.weight.grad.view(-1, dec.input_size, dec.output_size); Bg = dec.bias.weight.grad
need(bool((W[..., 20:32] == 0).all() and (Bg[:, 20:32] == 0).all() and (W[..., :20].abs() > 0).any()), "P6 action_decoder grad: dims 20:32 exactly 0 (all domains), dims 0:20 non-zero")
print("PREFLIGHT CARTONLY PASS" if not fails else f"PREFLIGHT CARTONLY FAILED: {fails}")
sys.exit(1 if fails else 0)
