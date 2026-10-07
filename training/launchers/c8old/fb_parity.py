#!/usr/bin/env python3
"""[2026-09-28] First-batch parity for the C-old B1-old recipe (CPU, no GPU needed). Same env flags as c8old_chain common_env.
Builds the dataset exactly the way lerobot_train does after humanik_delta.install() (LEAD 5 action window, validity keys,
[t, t+5..t+34] state window, measured EEF window), takes 16 FIXED valid chunk starts (evenly spaced over the humanik sampler's
allowed set, no RNG), converts them with humanik_delta's own model-unit conversion and prints sha256 of every tensor.
Run on the 4090 and on the 5090 against the same dataset: every hash must match.
Usage: <python> fb_parity.py <dataset_root> <repo_id> <out.json>"""
import hashlib, json, os, sys
for k, v in dict(HUMANIK_DELTA="1", HUMANIK_LEAD="5", HUMANIK_ROBOT="1", HUMANIK_TARGET="cmd", XVLA_EE_AUX="1", EE_AUX_SOURCE="state",
                 EE_AUX_SCALE="10", EE_AUX_LAMBDA="2.0", EE_FK_LAMBDA="20").items():
    os.environ.setdefault(k, v)
os.environ.setdefault("EEF_TARGET_SOURCE", "measured")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import humanik_delta; humanik_delta.install()
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import lerobot.datasets.factory as fac
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
root, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
sha = lambda t: hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]
meta = LeRobotDatasetMetadata(repo, root=root)
dt = fac.resolve_delta_timestamps(XVLAConfig(chunk_size=30, n_action_steps=30), meta)
ds = LeRobotDataset(repo, root=root, delta_timestamps=dt, video_backend="pyav")
allowed = humanik_delta._valid_start_indices(ds)
allowed = list(range(len(ds))) if allowed is None else list(allowed)
pick = [allowed[i] for i in np.linspace(0, len(allowed) - 1, 16).round().astype(int)]
items = [ds[i] for i in pick]
batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
st_raw = batch["observation.state"]; fut_raw = st_raw[:, 1:] if st_raw.ndim == 3 else None; st0 = st_raw[:, 0] if st_raw.ndim == 3 else st_raw
state, action = humanik_delta._to_model_units(st0, batch["action"])
tgt = action.clone(); tgt[..., humanik_delta.ARM] = action[..., humanik_delta.ARM] - state[:, None, humanik_delta.ARM]
res = dict(n_frames=len(ds), n_valid_starts=len(allowed), picked=pick, delta_keys={k: [round(v[0] * meta.fps), round(v[-1] * meta.fps), len(v)] for k, v in dt.items()},
           shapes={k: list(v.shape) for k, v in batch.items()}, sha={k: sha(v) for k, v in sorted(batch.items())},
           model_units=dict(state=sha(state), action=sha(action), target_dq=sha(tgt)),
           summary=dict(dq_arm_abs_p50=float(tgt[..., humanik_delta.ARM].abs().median()), dq_arm_abs_max=float(tgt[..., humanik_delta.ARM].abs().max()),
                        grip_state_uniq=sorted(set(np.round(state[..., humanik_delta.GRIP].flatten().tolist(), 4)))[:8],
                        grip_target_uniq=sorted(set(np.round(tgt[..., humanik_delta.GRIP].flatten().tolist(), 4)))[:8],
                        nan=int(sum(torch.isnan(v.float()).sum() for v in batch.values()))),
           versions=dict(torch=torch.__version__, python=sys.version.split()[0]))
json.dump(res, open(out, "w"), indent=1); print(json.dumps({k: res[k] for k in ("n_frames", "n_valid_starts", "model_units", "summary")}))
