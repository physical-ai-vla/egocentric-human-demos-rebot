#!/usr/bin/env python3
"""[2026-09-29] REL16-v3 base: widen lerobot/xvla-base CORRECTLY (input-major) for action 32 (REL16 20 + Δq 12) and state76.

DomainAwareLinear keeps fc as Embedding(num_domains, input*output) and uses `fc(d).view(B, input_size, output_size)`, i.e. the
flat vector is INPUT-major. The earlier widen_proprio.py / widen_append94.py reshaped it as (nd, hidden, in) -- output-major --
so their "copied" columns were in fact a permutation of the pretrained weights (cosine +0.001 vs the true rows). This script
works in the true layout and proves it with a functional test.

    action_encoder  in: xvla-base [action 20 | proprio 20 | time 32] = 72  ->  v3 [action 32 | proprio 76 | time 32] = 140
        rows action[0:20]  <- base action rows 0:20      (exact)
        rows action[20:32] <- NEW (Δq inputs), N(0, sd of the base action rows), seed
        rows proprio[0:76] <- NEW (base proprio was slim20, unrelated meaning), N(0, sd of base proprio rows), seed
        rows time[0:32]    <- base time rows 40:72       (exact)
    action_decoder  out: 20 -> 32:  outputs 0:20 <- base (exact), outputs 20:32 <- NEW N(0, sd of base outputs), bias 0
Functional gates (all domains): encoder(x_action20 ⊕ 0, proprio 0, time t) == base encoder(x_action20, proprio 0, t) exactly
(up to fp32 matmul order) and decoder(h)[:, :20] == base decoder(h). Every other tensor is copied untouched.
usage: widen_v3.py <xvla-base dir> <dst dir> [--seed 0]
"""
import argparse, hashlib, json, pathlib, shutil, sys
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
if dst.exists():
    sys.exit(f"{dst} exists, refusing to overwrite")
sd = load_file(str(src / "model.safetensors")); cfg = json.loads((src / "config.json").read_text())
EK, EB = "model.transformer.action_encoder.fc.weight", "model.transformer.action_encoder.bias.weight"
DK, DB = "model.transformer.action_decoder.fc.weight", "model.transformer.action_decoder.bias.weight"
H = int(cfg.get("hidden_size") or 1024); A0, P0, T = 20, 20, int(cfg.get("dim_time") or 32); A1, P1 = 32, 76
We, Wd, bd = sd[EK], sd[DK], sd[DB]; nd = We.shape[0]
assert We.shape[1] == (A0 + P0 + T) * H and Wd.shape[1] == H * A0, (We.shape, Wd.shape)
E0 = We.view(nd, A0 + P0 + T, H).float(); D0 = Wd.view(nd, H, A0).float()         # TRUE layout (input, output)
g = torch.Generator().manual_seed(a.seed)
E1 = torch.empty(nd, A1 + P1 + T, H)
E1[:, :A0] = E0[:, :A0]
E1[:, A0:A1] = torch.randn(nd, A1 - A0, H, generator=g) * E0[:, :A0].std()
E1[:, A1:A1 + P1] = torch.randn(nd, P1, H, generator=g) * E0[:, A0:A0 + P0].std()
E1[:, A1 + P1:] = E0[:, A0 + P0:]
D1 = torch.empty(nd, H, A1); D1[..., :A0] = D0; D1[..., A0:] = torch.randn(nd, H, A1 - A0, generator=g) * D0.std()
b1 = torch.zeros(nd, A1); b1[:, :A0] = bd.float()
out = dict(sd)
out[EK] = E1.reshape(nd, -1).to(We.dtype).contiguous(); out[DK] = D1.reshape(nd, -1).to(Wd.dtype).contiguous(); out[DB] = b1.to(bd.dtype).contiguous()

# functional gates, per domain, in the forward's own convention (x @ W + b with W = view(input, output))
torch.manual_seed(1)
x = torch.randn(nd, 16, A0); t = torch.randn(nd, 16, T); z0 = torch.zeros(nd, 16, P0); z1 = torch.zeros(nd, 16, P1)
eb = sd[EB].float()[:, None]
y0 = torch.bmm(torch.cat([x, z0, t], -1), E0) + eb
y1 = torch.bmm(torch.cat([x, torch.zeros(nd, 16, A1 - A0), z1, t], -1), out[EK].float().view(nd, A1 + P1 + T, H)) + eb
de = float((y0 - y1).abs().max())
h = torch.randn(nd, 16, H)
o0 = torch.bmm(h, D0) + bd.float()[:, None]; o1 = torch.bmm(h, out[DK].float().view(nd, H, A1)) + out[DB].float()[:, None]
dd = float((o0 - o1[..., :A0]).abs().max())
assert de < 1e-4 and dd < 1e-4, f"functional gate failed: encoder {de:.2e} decoder {dd:.2e}"
assert all(torch.equal(out[k], sd[k]) for k in sd if k not in (EK, DK, DB)), "a tensor other than the action encoder/decoder changed"
print(f"functional gates PASS: encoder(new ⊕ 0) == base encoder max|d| {de:.2e}; decoder[:20] == base decoder max|d| {dd:.2e}")
dst.mkdir(parents=True)
for f in src.iterdir():
    if f.is_file() and f.name not in ("model.safetensors", "config.json"):
        shutil.copy2(f, dst / f.name)
save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
cfg["max_state_dim"] = P1; cfg["max_action_dim"] = A1
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
hsh = hashlib.md5(open(dst / "model.safetensors", "rb").read()).hexdigest()
(dst / "BASE_SHA256").write_text(hsh + "\n")
(dst / "WIDENING.md").write_text(f"# REL16-v3 base (widen_v3.py, INPUT-major, 2026-09-29)\nsource {src}\n"
    f"encoder in 72 -> {A1+P1+T} [action {A1} | proprio {P1} | time {T}]: action[0:20] and time copied exactly, action[20:32] + proprio NEW (seed {a.seed})\n"
    f"decoder out 20 -> {A1}: outputs 0:20 exact, 20:32 NEW, bias 0\nfunctional gates: encoder {de:.2e}, decoder {dd:.2e}\nmodel md5 {hsh}\n")
print(f"wrote {dst} md5 {hsh}")
