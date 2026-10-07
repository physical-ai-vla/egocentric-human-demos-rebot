"""[2026-09-28] REL16-v2 pre-training preflight (CPU, no GPU): the action chunk is NOT re-expanded by the data loader.

The dataset stores the whole (16, 20) current-anchor chunk on each query row. lerobot_train would normally turn
action_delta_indices into delta_timestamps and stack 16 consecutive rows on top of it -> a nested (16, 16, 20) horizon
that trains without any error. The node's patched datasets/factory.py skips that expansion for a stored 2-D action whose
first axis equals the horizon. This runs the SAME code path lerobot_train uses (policy config from the base with the
launcher's overrides -> factory.resolve_delta_timestamps -> LeRobotDataset) and asserts:
  1 resolve_delta_timestamps() puts no expansion on "action"
  2 dataset samples give action shape (16, 20) (never (16, 16, 20)), batched by the default collate to [B, 16, 20]
  3 the sampled action equals the stored parquet row bit-for-bit
usage: preflight_rel16v2.py <dataset root> <base dir> <state dim> [action dim=20]   (v3d: 32 = REL16 20 + dq 12)
"""
import glob, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
import lerobot.policies.factory  # noqa: F401  registers the "xvla" config choice (draccus)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps

root, base, sdim = sys.argv[1], sys.argv[2], int(sys.argv[3]); adim = int(sys.argv[4]) if len(sys.argv) > 4 else 20
fails = []


def need(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        fails.append(msg)


cfg = PreTrainedConfig.from_pretrained(base)
for k, v in dict(chunk_size=16, n_action_steps=16, max_state_dim=sdim, max_action_dim=adim, action_mode="auto", use_proprio=True).items():
    setattr(cfg, k, v)
meta = LeRobotDatasetMetadata("rebot/preflight", root=root)
dts = resolve_delta_timestamps(cfg, meta)
need("action" not in (dts or {}), f"resolve_delta_timestamps leaves action unexpanded (delta keys: {sorted((dts or {}).keys())}, "
     f"policy action_delta_indices len {len(cfg.action_delta_indices or [])})")
ds = LeRobotDataset("rebot/preflight", root=root, delta_timestamps=dts, video_backend="pyav")
t = pq.read_table(sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))[0], columns=["action", "index"])
col = lambda c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
A = np.asarray(col("action").to_pylist(), np.float32); IX = np.asarray(col("index").to_numpy())
items = []
for i in (0, 1234, 20000):
    x = ds[i]; items.append(x)
    j = int(x["index"]); stored = A[np.flatnonzero(IX == j)[0]]
    need(tuple(x["action"].shape) == (16, adim), f"sample {i}: action shape {tuple(x['action'].shape)} == (16, {adim})")
    if adim == 32:
        need("aux.q_t" in x and tuple(x["aux.q_t"].shape) == (12,), f"sample {i}: aux.q_t present, shape (12,)")
    need(np.array_equal(x["action"].numpy(), stored), f"sample {i}: loader action == stored parquet row (bit-for-bit)")
    need(tuple(x["observation.state"].shape) == (sdim,), f"sample {i}: state shape {tuple(x['observation.state'].shape)} == ({sdim},)")
b = torch.utils.data.default_collate([{"action": it["action"]} for it in items])
need(tuple(b["action"].shape) == (3, 16, adim), f"collated batch action {tuple(b['action'].shape)} == [B, 16, {adim}]")
print("PREFLIGHT PASS" if not fails else f"PREFLIGHT FAILED: {fails}")
sys.exit(1 if fails else 0)
