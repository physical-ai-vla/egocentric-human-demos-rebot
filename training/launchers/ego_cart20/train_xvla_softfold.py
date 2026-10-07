"""[2026-10-07 user: "same as X-VLA"] lerobot_train with the 2toinf/X-VLA train.py optimisation recipe, for the Soft-Fold
abs-EEF fine-tune (base lerobot/xvla-base, domain 5, action_mode ee6d = MSE pos/rot + BCE gripper, no normalisation).

Replaces lerobot's xvla-adamw optimizer / scheduler (3 groups, betas 0.9/0.99, cosine) by X-VLA build_optimizer +
update_group_lrs (README fine-tune command: lr 1e-4, learning_coef 0.1, freeze_steps 1000, no --use_cosine_decay):
  groups   vlm (lr * 0.1) | transformer_core (lr) | soft_prompts (lr * 0.1) | action_heads (encoder + decoder, lr)
  AdamW    betas (0.9, 0.95), weight decay XVLA_WD (default 0.01, paper App. H; the code default is 0.0)
  steps < XVLA_FREEZE (1000): vlm and transformer_core lr = 0  ->  only soft prompts + action heads train
  after   : constant lr per group (X-VLA default without --use_cosine_decay)
  grad clip XVLA_MAX_GRAD_NORM (1.0, X-VLA default)
and lerobot's image transforms by torchvision ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0), applied to
every camera of every training sample (X-VLA datasets/dataset.py) when the run passes --dataset.image_transforms.enable=true.
Resume: the LambdaLR state carries last_epoch, so the freeze / constant phases continue correctly.
"""
import os, runpy, sys

import torch

LR_COEF = float(os.environ.get("XVLA_LR_COEF", "0.1"))
FREEZE = int(os.environ.get("XVLA_FREEZE", "1000"))
WD = float(os.environ.get("XVLA_WD", "0.01"))
MAX_GRAD_NORM = float(os.environ.get("XVLA_MAX_GRAD_NORM", "1.0"))

import lerobot.optim.factory as OF
import lerobot.datasets.transforms as TF


def _group_of(name):
    n = name.removeprefix("model.")
    if n.startswith("vlm."): return "vlm"
    if n.startswith("transformer.soft_prompt_hub."): return "soft_prompts"
    if n.startswith("transformer.action_encoder.") or n.startswith("transformer.action_decoder."): return "action_heads"
    return "transformer_core"


def make_optimizer_and_scheduler(cfg, policy):
    lr = cfg.optimizer.lr
    cfg.optimizer.grad_clip_norm = MAX_GRAD_NORM
    groups = {k: [] for k in ("vlm", "transformer_core", "soft_prompts", "action_heads")}
    for name, p in policy.named_parameters():
        if p.requires_grad: groups[_group_of(name)].append(p)
    scale = {"vlm": LR_COEF, "transformer_core": 1.0, "soft_prompts": LR_COEF, "action_heads": 1.0}
    pg = [{"name": k, "params": v, "lr": lr * scale[k], "weight_decay": WD} for k, v in groups.items()]
    opt = torch.optim.AdamW(pg, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    frozen = {"vlm", "transformer_core"}
    lam = [(lambda s, k=k: 0.0 if (k in frozen and s < FREEZE) else 1.0) for k in groups]
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    n = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    print(f"[xvla-recipe] AdamW betas (0.9, 0.95) wd {WD} lr {lr} coef {LR_COEF} freeze {FREEZE} (vlm+core lr 0) then constant; "
          f"grad clip {MAX_GRAD_NORM}; params {n}", flush=True)
    assert n["action_heads"] > 0 and n["soft_prompts"] > 0 and n["vlm"] > 0, n
    return opt, sch


OF.make_optimizer_and_scheduler = make_optimizer_and_scheduler


class XVLAColorJitter(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        from torchvision.transforms import v2
        self.t = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
        print("[xvla-recipe] image transforms = ColorJitter(0.2, 0.2, 0.2, 0) on every camera", flush=True)

    def forward(self, x):
        return self.t(x)


TF.ImageTransforms = XVLAColorJitter
import lerobot.datasets.factory as DF  # noqa: E402  (it imported the name already)
DF.ImageTransforms = XVLAColorJitter

sys.argv[0] = os.environ.get("LEROBOT_TRAIN", "/srv/data/johann/relonly/code/lerobot-seeed/src/lerobot/scripts/lerobot_train.py")
runpy.run_path(sys.argv[0], run_name="__main__")
