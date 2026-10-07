#!/usr/bin/env python3
"""[2026-09-28] Resume entry for the C-old runs: imports the C-OLD train_bi (c8old/xvla, humanik_delta md5 6b63986b…) — NOT
workspace/xvla/resume_bi.py, whose train_bi pulls the workspace humanik_delta (different md5). All patches install at import;
arguments go straight to lerobot_train (e.g. --config_path=<ckpt>/pretrained_model/train_config.json --resume=true)."""
import sys
sys.path.insert(0, "/home/bh-aiteam/c8old/xvla")
import train_bi  # noqa: F401  (installs contrastive / keep_indices / aug / lr_groups / humanik_delta / eef_delta + TensorBoard logger)
import lerobot.scripts.lerobot_train as lt
if __name__ == "__main__":
    lt.train()
