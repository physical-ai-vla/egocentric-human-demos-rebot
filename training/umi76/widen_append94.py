#!/usr/bin/env python3
"""[2026-09-28] REL16-v2 twin B base: append 18 proprio columns to the twin-A base (xvla_base_umi76), so the twins
start from IDENTICAL weights except for the new columns.

    A  xvla_base_umi76     action_encoder input 128 = [action 20 | proprio76 | time 32]
    B  xvla_base_umi94     action_encoder input 146 = [action 20 | proprio76 (== A) | TCP18 (new) | time 32]

soft_transformer.py builds the encoder input as cat([action, proprio, time]), so the new columns go between
proprio76 and time and the time block moves by 18; appending after time would feed time weights TCP values.
The 18 new columns are drawn N(0, 0.01730) -- the same scale the proprio76 block was drawn at (WIDENING.md) --
with their own seed. Every other tensor is copied untouched, and the gates below check that.

usage: widen_append94.py <A-base-dir> <B-base-dir> [--seed 1]
"""
import argparse, hashlib, json, pathlib, shutil, sys
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
if dst.exists():
    sys.exit(f"{dst} exists, refusing to overwrite")
cfg = json.loads((src / "config.json").read_text())
OLD_P, ADD, DIM_A, DIM_T, STD = 76, 18, 20, int(cfg.get("dim_time") or 32), 0.01730
assert int(cfg["max_state_dim"]) == OLD_P, f"source max_state_dim {cfg['max_state_dim']} != 76"
old_in, new_in = DIM_A + OLD_P + DIM_T, DIM_A + OLD_P + ADD + DIM_T

sd = load_file(str(src / "model.safetensors"))
keys = [k for k in sd if k.endswith("action_encoder.fc.weight")]
assert len(keys) == 1, keys
key = keys[0]; W = sd[key]; nd = W.shape[0]
assert W.shape[1] % old_in == 0, f"{W.shape} not a multiple of {old_in}"
hid = W.shape[1] // old_in
Wm = W.reshape(nd, hid, old_in)
g = torch.Generator().manual_seed(a.seed)
new = torch.empty(nd, hid, new_in, dtype=W.dtype)
new[:, :, :DIM_A + OLD_P] = Wm[:, :, :DIM_A + OLD_P]
new[:, :, DIM_A + OLD_P:DIM_A + OLD_P + ADD] = (torch.randn(nd, hid, ADD, generator=g) * STD).to(W.dtype)
new[:, :, DIM_A + OLD_P + ADD:] = Wm[:, :, DIM_A + OLD_P:]
out = dict(sd); out[key] = new.reshape(nd, hid * new_in).contiguous()

# gates
assert torch.equal(new[:, :, :DIM_A], Wm[:, :, :DIM_A]), "action columns changed"
assert torch.equal(new[:, :, DIM_A:DIM_A + OLD_P], Wm[:, :, DIM_A:DIM_A + OLD_P]), "proprio76 columns changed"
assert torch.equal(new[:, :, DIM_A + OLD_P + ADD:], Wm[:, :, DIM_A + OLD_P:]), "time columns changed"
assert all(torch.equal(out[k], sd[k]) for k in sd if k != key), "a tensor other than the action encoder changed"
nstd = float(new[:, :, DIM_A + OLD_P:DIM_A + OLD_P + ADD].float().std())
print(f"{key}: {tuple(W.shape)} -> {tuple(out[key].shape)}  new-column std {nstd:.5f} (target {STD})")

dst.mkdir(parents=True)
for f in src.iterdir():
    if f.name not in ("model.safetensors", "config.json", "BASE_SHA256"):
        shutil.copy2(f, dst / f.name)
save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
cfg["max_state_dim"] = OLD_P + ADD
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
h = hashlib.md5(open(dst / "model.safetensors", "rb").read()).hexdigest()
(dst / "BASE_SHA256").write_text(h + "\n")
with open(dst / "WIDENING.md", "a") as fh:
    fh.write(f"\n\n# [2026-09-28] twin B: 76 -> 94 (widen_append94.py)\nsource: {src} (hash {(src / 'BASE_SHA256').read_text().strip()})\n"
             f"action_encoder input {old_in} -> {new_in}, layout [action 20 | proprio76 == A | TCP18 new | time 32]\n"
             f"action / proprio76 / time columns and every other tensor copied bit-identical; TCP18 N(0, {STD}) seed {a.seed}\n"
             f"model hash {h}\n")
print(f"wrote {dst}  hash {h}")
