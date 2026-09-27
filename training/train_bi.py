#!/usr/bin/env python3
"""X-VLA bimanual 3-stack finetuning (14-DOF, 3 cams global/left_wrist/right_wrist)."""
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import lerobot.scripts.lerobot_train as lerobot_train
import os as _os, torch as _torch
if _os.environ.get("XVLA_FUSED_ADAM") == "1":   # [2026-09-08] fused AdamW kernel (same math; fewer kernel launches)
    _AdamW = _torch.optim.AdamW
    class _FusedAdamW(_AdamW):
        def __init__(self, params, *a, **k): k.setdefault("fused", True); super().__init__(params, *a, **k)
    _torch.optim.AdamW = _FusedAdamW; print("[fused-adamw] torch.optim.AdamW -> fused=True", flush=True)
if _os.environ.get("XVLA_TF32") == "1":   # [2026-08-28] fp32 weights/optimizer/activations unchanged; only matmul/conv kernels use TF32 (10-bit mantissa, fp32 accumulate). ~1.5-2x on Ada/Blackwell.
    _torch.backends.cuda.matmul.allow_tf32 = True; _torch.backends.cudnn.allow_tf32 = True; _torch.backends.cudnn.benchmark = True
    print("[tf32] allow_tf32 enabled (matmul + cudnn)", flush=True)
import os; sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrastive_sampler; contrastive_sampler.install()
import keep_indices; keep_indices.install()
import aug_defaults; aug_defaults.install()
import lr_groups; lr_groups.install()
import humanik_delta; humanik_delta.install()
import eef_delta; eef_delta.install()   # [2026-09-17] EEF_DELTA=1 -> bimanual EEF-delta action space (D0/D250 main line); mutually exclusive with HUMANIK_DELTA   # [2026-09-05] HUMANIK_DELTA=1 -> t+5 cumulative-delta targets + validity-masked joint loss   # [2026-08-27] XVLA_LR_GROUPS → per-group LRs   # [2026-08-27] XVLA_AUG=1 → weak photometric aug (train only)   # [2026-08-27] XVLA_TARGET_HEAD=1 → keep episode/frame index through the preprocessor   # [2026-08-27] CONTRASTIVE=1 → same-scene batch sampler (needs set_map.json in dataset root)

DATASET_ROOT = "${HOME}/holobrain-data/lerobot/rebot_3stack_bimanual_60ep"
REPO_ID = "rebot/3stack_bimanual_60ep"
OUTPUT_DIR = "${HOME}/xvla_bi_out"

class TensorBoardLogger:
    def __init__(self, cfg):
        d = Path(cfg.output_dir) / "tensorboard"; d.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(d)); print("[tb]", d)
    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        for k, v in d.items():
            if isinstance(v,(int,float)) and not isinstance(v,bool):
                self.writer.add_scalar(f"{mode}/{k}", v, step)
        self.writer.flush()
    def __getattr__(self, _n): return lambda *a, **k: None

lerobot_train.WandBLogger = TensorBoardLogger

DEFAULT_ARGS = [
    f"--dataset.repo_id={REPO_ID}",
    f"--dataset.root={DATASET_ROOT}",
    "--policy.path=lerobot/xvla-base",
    "--policy.dtype=bfloat16",
    "--policy.action_mode=" + os.environ.get("XVLA_ACTION_MODE", "auto"),
    "--policy.max_action_dim=20",
    "--policy.device=cuda",
    "--policy.freeze_vision_encoder=false",
    "--policy.freeze_language_encoder=false",
    "--policy.train_policy_transformer=true",
    "--policy.train_soft_prompts=true",
    '--rename_map={"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}',
    f"--output_dir={OUTPUT_DIR}",
    "--job_name=xvla_bi_3stack",
    "--steps=20000",
    "--save_freq=2000",
    "--log_freq=100",
    "--wandb.enable=true",
    "--wandb.project=xvla_bi_3stack",
    "--policy.push_to_hub=false",
]

def main():
    ua = sys.argv[1:]; uk = {a.split("=",1)[0] for a in ua}
    merged = [] if "--config_path" in uk else [a for a in DEFAULT_ARGS if a.split("=",1)[0] not in uk]   # resume: saved train_config.json already holds every setting
    sys.argv = [sys.argv[0], *merged, *ua]
    lerobot_train.train()

if __name__ == "__main__":
    main()
