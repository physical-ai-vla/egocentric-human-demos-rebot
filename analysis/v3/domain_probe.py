"""[2026-10-02] init-loss probe: which pretrained xvla-base slot is the best warm start for a new reBot domain.
Loads the D6 base + R312c with the training code path (REL-only plugin, processors as lerobot_train), then for each
candidate slot evaluates the REL loss with batch domain_id overridden. Same fixed samples + same noise seed for every slot."""
import os, sys, random
sys.path.insert(0, "${REMOTE_HOME}/umi_bridge/umi76")
import rel16_aux_relonly; rel16_aux_relonly.install()
import numpy as np, torch
import lerobot.policies.factory  # noqa
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy, make_pre_post_processors
BASE = "${REMOTE_HOME}/xvla_base_cart20v3_d6"
ROOT = "${REMOTE_HOME}/holobrain-data/lerobot/r312c_relcart20_rel16_v4"
RENAME = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}
cfg = PreTrainedConfig.from_pretrained(BASE)
for k, v in dict(chunk_size=16, n_action_steps=16, max_state_dim=20, max_action_dim=32, action_mode="auto", use_proprio=True, device="cuda", dtype="float32").items():
    setattr(cfg, k, v)
cfg.pretrained_path = BASE
ds = LeRobotDataset("rebot/r312c", root=ROOT, delta_timestamps=resolve_delta_timestamps(cfg, None if False else __import__("lerobot.datasets.lerobot_dataset", fromlist=["x"]).LeRobotDatasetMetadata("rebot/r312c", root=ROOT)), video_backend="pyav")
policy = make_policy(cfg=cfg, ds_meta=ds.meta, rename_map=RENAME).eval()
pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=BASE, dataset_stats=ds.meta.stats,
    preprocessor_overrides={"device_processor": {"device": "cuda"},
        "normalizer_processor": {"stats": ds.meta.stats, "features": {**policy.config.input_features, **policy.config.output_features}, "norm_map": policy.config.normalization_mapping},
        "rename_observations_processor": {"rename_map": RENAME}})
rng = np.random.default_rng(0); idx = rng.choice(len(ds), 128, replace=False)
batches = []
for i in range(0, 128, 4):
    batches.append(pre(torch.utils.data.default_collate([ds[int(j)] for j in idx[i:i+4]])))
print("batch domain_id from processor:", batches[0]["domain_id"].tolist() if "domain_id" in batches[0] else None, flush=True)
res = {}
for s in [0, 6, 10, 11, 15, 16, 17]:
    L = []
    for bi, b in enumerate(batches):
        b = dict(b); b["domain_id"] = torch.full_like(b["domain_id"], s)
        torch.manual_seed(1000 + bi)
        with torch.no_grad():
            out = policy.forward(b)
        loss = out[0] if isinstance(out, tuple) else out
        L.append(float(loss))
    res[s] = (np.mean(L), np.median(L))
    print(f"slot {s:2d}: mean loss {res[s][0]:.4f} median {res[s][1]:.4f}", flush=True)
print("BEST", min(res, key=lambda k: res[k][0]))
