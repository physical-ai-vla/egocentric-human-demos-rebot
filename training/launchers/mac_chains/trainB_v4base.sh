#!/bin/bash
# Track B v4-base launcher: stock lerobot_train on r150_umi_v4_3cam_s20, no X-VLA code changes.
#
# Everything policy-side is a CLI override of the pretrained lerobot/xvla-base config. Two of them are
# forced by our data contract and must not drift:
#   max_state_dim=20 / max_action_dim=20  -- the pretrained action_encoder is [., 72] = action 20 + proprio
#                                            20 + time 32; 38D makes it 90 and strict load fails.
#   action_mode=auto                      -- the pretrained default `ee6d` puts BCE on the gripper channels
#                                            (and zeroes them in the model input), which assumes a gripper in
#                                            [0,1]. Ours is a continuous width, MEAN_STD-normalised, so BCE is
#                                            the wrong loss. `auto` is plain MSE over the real 20 dims.
# chunk_size is free: no weight carries it (the 30 in the earlier shape error was num_domains).
#
# usage: trainB_v4base.sh <run-name> <steps> [episodes-json|all] [gpu] [extra lerobot args...]
set -euo pipefail

RUN=${1:?run name}
STEPS=${2:?steps}
EPISODES=${3:-all}
GPU=${4:-0}
shift 4 2>/dev/null || shift $#

PY=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python
SRC=/home/bh-aiteam/lerobot-seeed/src
# DS selects the dataset: s20 (frozen, gripper at t) or s20g (gripper at t+1). Same schedule either way.
DS=${DS:-r150_umi_v4_3cam_s20}
# BASE is the weights to start from: the stock pretrained model, or a surgically widened copy of it when the
# state contract changes (a duplicate --policy.path on the command line is ignored, so it has to come from here)
BASE=${BASE:-lerobot/xvla-base}
STATE_DIM=${STATE_DIM:-20}
DATA=/home/bh-aiteam/holobrain-data/lerobot/$DS
OUT=/home/bh-aiteam/holobrain-data/trainB/$RUN
RENAME='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'

EP_ARG=()
[ "$EPISODES" != "all" ] && EP_ARG=(--dataset.episodes="$EPISODES")

# lerobot_train refuses a pre-existing output_dir (resume guard), so only the parent is created and the
# log lives next to it rather than inside it.
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.log"
cd /home/bh-aiteam/umi_bridge

# fp32 + 3 views + full fine-tuning is tight on a 24 GB 4090: batch 8 OOMs, so BS is a parameter and
# expandable_segments keeps fragmentation from costing the last few hundred MB.
BS=${BS:-4}
# Under `ray job submit` Ray already allocates the GPU; overriding CUDA_VISIBLE_DEVICES here would fight it,
# so pass GPU=ray to leave the allocation alone.
GPU_ENV="CUDA_VISIBLE_DEVICES=$GPU"
[ "$GPU" = "ray" ] && GPU_ENV=""
env $GPU_ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC $PY $SRC/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=rebot/$DS \
  --dataset.root="$DATA" \
  "${EP_ARG[@]}" \
  --policy.path=$BASE \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.max_state_dim=$STATE_DIM \
  --policy.max_action_dim=20 \
  --policy.action_mode=auto \
  --policy.use_proprio=true \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --rename_map="$RENAME" \
  --steps="$STEPS" \
  --batch_size="$BS" \
  --num_workers=4 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUT" \
  "$@" 2>&1 | tee "$LOG"
