#!/usr/bin/env python3
"""[2026-09-29] CART20 base from the REL16-v3 base (xvla_base_rel16v3, widen_v3.py): only the action-encoder PROPRIO rows change.

xvla_base_rel16v3 action_encoder input rows (INPUT-major, Embedding(nd, in*out).view(nd, in, out)):
    [action 32 | proprio 76 | time 32] = 140   ->   CART20: [action 32 | proprio 20 | time 32] = 84
    action rows 0:32 and time rows copied exactly; proprio rows 0:20 of the v3 base's (fresh N(0, sd), seed 0, no pretrained
    meaning) 76 are kept, rows 20:76 dropped. Decoder and every other tensor are bit-identical to the v3 base, so the two runs
    differ only in the state width and what the proprio rows see.
Functional gate (all domains): encoder_cart20(x_action32, proprio 0, t) == encoder_v3(x_action32, proprio 0, t).
usage: widen_cart20.py <xvla_base_rel16v3 dir> <dst dir>
"""
import hashlib, json, pathlib, shutil, sys
import torch
from safetensors.torch import load_file, save_file

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if dst.exists():
    sys.exit(f"{dst} exists, refusing to overwrite")
sd = load_file(str(src / "model.safetensors")); cfg = json.loads((src / "config.json").read_text())
EK, EB = "model.transformer.action_encoder.fc.weight", "model.transformer.action_encoder.bias.weight"
H = int(cfg.get("hidden_size") or 1024); A, P0, P1, T = 32, 76, 20, int(cfg.get("dim_time") or 32)
assert cfg["max_state_dim"] == P0 and cfg["max_action_dim"] == A, (cfg["max_state_dim"], cfg["max_action_dim"])
We = sd[EK]; nd = We.shape[0]
assert We.shape[1] == (A + P0 + T) * H, We.shape
E0 = We.view(nd, A + P0 + T, H)
E1 = torch.cat([E0[:, :A], E0[:, A:A + P1], E0[:, A + P0:]], 1)
out = dict(sd); out[EK] = E1.reshape(nd, -1).contiguous()
torch.manual_seed(1)
x = torch.randn(nd, 16, A); t = torch.randn(nd, 16, T); eb = sd[EB].float()[:, None]
y0 = torch.bmm(torch.cat([x, torch.zeros(nd, 16, P0), t], -1), E0.float()) + eb
y1 = torch.bmm(torch.cat([x, torch.zeros(nd, 16, P1), t], -1), E1.float()) + eb
de = float((y0 - y1).abs().max())
y0p = torch.bmm(torch.cat([x, torch.ones(nd, 16, P1), torch.zeros(nd, 16, P0 - P1), t], -1), E0.float())
y1p = torch.bmm(torch.cat([x, torch.ones(nd, 16, P1), t], -1), E1.float())
dp = float((y0p - y1p).abs().max())
assert de < 1e-4 and dp < 1e-3, f"functional gate failed: {de:.2e} / proprio rows {dp:.2e}"
assert all(torch.equal(out[k], sd[k]) for k in sd if k != EK), "a tensor other than the encoder fc changed"
print(f"functional gates PASS: encoder(proprio 0) == v3 base max|d| {de:.2e}; proprio rows 0:20 == v3 rows 0:20 max|d| {dp:.2e}")
dst.mkdir(parents=True)
for f in src.iterdir():
    if f.is_file() and f.name not in ("model.safetensors", "config.json", "BASE_SHA256", "WIDENING.md"):
        shutil.copy2(f, dst / f.name)
save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
cfg["max_state_dim"] = P1
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
hsh = hashlib.md5(open(dst / "model.safetensors", "rb").read()).hexdigest()
(dst / "BASE_SHA256").write_text(hsh + "\n")
(dst / "WIDENING.md").write_text(f"# CART20 base (widen_cart20.py, 2026-09-29)\nsource {src}\nencoder in {A+P0+T} -> {A+P1+T}: "
    f"action 0:32 + time exact, proprio rows 0:20 of the v3 base kept, 20:76 dropped; all else bit-identical\n"
    f"gates: encoder {de:.2e}, proprio rows {dp:.2e}\nmodel md5 {hsh}\n")
print(f"wrote {dst} md5 {hsh}")
