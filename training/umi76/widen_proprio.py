#!/usr/bin/env python3
"""Rebuild X-VLA's action encoder for a 76-D UMI proprio input, keeping the pretrained action and time
columns and REPLACING the proprio block.

Two facts drive every line here.

1. soft_transformer.py:381 builds the input as
       cat([action_with_noise(dim_action), proprio(dim_propio), time(dim_time)])
   so proprio sits in the MIDDLE. Appending the new columns at the end would leave the time embedding
   reading proprio columns -- no error, just a model that trains on scrambled input.

2. The old 20-D proprio was slim20, `[L prev_rel9, L width, R prev_rel9, R width]`. UMI76's first 20 dims
   are `robot0_eef_pos` history, `robot0_eef_pos_wrt1` history and part of `robot0_eef_rot_axis_angle`.
   Same count, unrelated meaning. Copying the old proprio weights onto them would carry a wrong prior, so
   the WHOLE 76-D proprio block is initialized fresh.

       new[..., 0:dim_action]                    <- old action columns
       new[..., dim_action : dim_action+76]      <- NEW init (scale matched to the pretrained layer)
       new[..., dim_action+76 : ]                <- old time columns

DomainAwareLinear stores the weight as nn.Embedding(num_domains, output_size * input_size), so it is
reshaped to (num_domains, output_size, input_size), rebuilt on the input axis, and flattened back.

usage: widen_proprio.py <src-pretrained_model-dir> <dst-dir> [--new-dim 76] [--seed 0]
"""
import argparse, json, pathlib, shutil, sys
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--new-dim", type=int, default=76)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)

cfg = json.loads((src / "config.json").read_text())
old_p = int(cfg.get("max_state_dim", 20))
dim_action = int(cfg.get("max_action_dim", 20))
dim_time = int(cfg.get("dim_time", 32))
new_p = a.new_dim
old_in, new_in = dim_action + old_p + dim_time, dim_action + new_p + dim_time
print(f"proprio {old_p} -> {new_p}   action_encoder input {old_in} -> {new_in}")
print(f"  old layout  action[0:{dim_action}] proprio[{dim_action}:{dim_action+old_p}] time[{dim_action+old_p}:{old_in}]")
print(f"  new layout  action[0:{dim_action}] proprio[{dim_action}:{dim_action+new_p}] time[{dim_action+new_p}:{new_in}]")

sd = load_file(str(src / "model.safetensors"))
KEY = [k for k in sd if k.endswith("action_encoder.fc.weight")]
if len(KEY) != 1:
    sys.exit(f"expected exactly one action_encoder.fc.weight, found {KEY}")
key = KEY[0]
W = sd[key]
num_domains = W.shape[0]
if W.shape[1] % old_in:
    sys.exit(f"{key} second dim {W.shape[1]} is not a multiple of {old_in}")
hidden = W.shape[1] // old_in
Wm = W.reshape(num_domains, hidden, old_in).float()

act_old = Wm[:, :, 0:dim_action]
time_old = Wm[:, :, dim_action + old_p:]
# Scale the fresh proprio block to the pretrained layer's own scale rather than a generic xavier: the
# stored weight is an Embedding of shape (num_domains, out*in), so torch's xavier would compute fan-in
# from that flattened shape and land on the wrong magnitude.
ref_std = torch.cat([act_old.reshape(-1), time_old.reshape(-1)]).std().item()
g = torch.Generator().manual_seed(a.seed)
prop_new = torch.randn(num_domains, hidden, new_p, generator=g) * ref_std

new_W = torch.empty(num_domains, hidden, new_in, dtype=torch.float32)
new_W[:, :, 0:dim_action] = act_old
new_W[:, :, dim_action:dim_action + new_p] = prop_new
new_W[:, :, dim_action + new_p:] = time_old
sd[key] = new_W.reshape(num_domains, hidden * new_in).to(W.dtype).contiguous()
print(f"  {key}: {tuple(W.shape)} -> {tuple(sd[key].shape)}  (num_domains {num_domains}, hidden {hidden})")
print(f"  pretrained column std {ref_std:.5f}; new proprio block drawn N(0, {ref_std:.5f}), seed {a.seed}")

# ---- gates. The old "zeros reproduce the original" check does not apply once proprio is re-initialized,
# so what is verified instead is that the two blocks that MUST survive did, exactly, and that the block
# that must NOT have been copied really was replaced.
chk = new_W
assert torch.equal(chk[:, :, 0:dim_action], act_old), "action columns changed"
assert torch.equal(chk[:, :, dim_action + new_p:], time_old), "time columns changed"
print("  gate PASS: action and time columns are bit-identical to the source")
if old_p <= new_p:
    overlap = (chk[:, :, dim_action:dim_action + old_p] - Wm[:, :, dim_action:dim_action + old_p]).abs().max()
    if overlap < 1e-9:
        sys.exit("GATE FAILED: the first 20 proprio columns still equal the old slim20 weights; "
                 "they must be re-initialized because the semantics differ")
    print(f"  gate PASS: proprio block replaced (max |new - old| over the first {old_p} cols = {overlap:.4f})")
n_other = sum(1 for k in sd if k != key)
print(f"  {n_other} other tensors carried over untouched")

dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "model.safetensors":
        shutil.copy2(f, dst / f.name)
cfg["max_state_dim"] = new_p
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
save_file(sd, str(dst / "model.safetensors"), metadata={"format": "pt"})
(dst / "WIDENING.md").write_text(
    f"# proprio widening provenance\n\n"
    f"source: {src}\n"
    f"max_state_dim {old_p} -> {new_p}; action_encoder input {old_in} -> {new_in}\n"
    f"layout [action {dim_action} | proprio {new_p} | time {dim_time}]\n\n"
    f"- action columns: copied from the source, bit-identical\n"
    f"- time columns:   copied from the source, bit-identical, moved to [{dim_action+new_p}:{new_in}]\n"
    f"- proprio block:  NOT copied. The old 20 dims were slim20 "
    f"([L prev_rel9, L width, R prev_rel9, R width]); UMI76's first 20 are robot0_eef_pos history, "
    f"robot0_eef_pos_wrt1 history and part of robot0_eef_rot_axis_angle. Same count, different meaning, "
    f"so the whole block is newly drawn N(0, {ref_std:.5f}) with seed {a.seed}.\n"
    f"- every other tensor is untouched.\n\n"
    f"Both REL32 and DELTA32 must start from THIS directory so their initialization is identical.\n")
print(f"written -> {dst}")
