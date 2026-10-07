"""[2026-09-29] CPU unit test for rel16_aux: GT prediction -> fk_loss ~ 0; perturbed Δq -> fk_loss > 0 with gradients on both heads."""
import os, sys, json, torch, numpy as np
D = sys.argv[1]; os.environ["REL16_STATS"] = f"{D}/meta/stats.json"; os.environ["REL16_LOG_EVERY"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rel16_aux; rel16_aux.install()
import lerobot.policies.xvla.action_hub as H
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("rebot/v3d", root=D, video_backend="pyav")
st = json.load(open(os.environ["REL16_STATS"]))["action"]; mean, std = torch.tensor(st["mean"]), torch.tensor(st["std"])
idx = [100, 5000, 40000, 90000]
act = torch.stack([ds[i]["action"] for i in idx]); qt = torch.stack([ds[i]["aux.q_t"] for i in idx])
y = (act - mean) / (std + 1e-8)                                         # what the preprocessor feeds as the target
sp = H.AutoActionSpace(real_dim=32, max_dim=32); sp._rel16_qt = qt
o = sp.compute_loss(y.clone(), y.clone())
print("GT pred:", {k: float(v) for k, v in o.items()})
assert o["rel_loss"] == 0 and o["dq_loss"] == 0 and o["fk_loss"] < 1e-5, "GT must give ~0 losses"
p = y.clone(); p[..., 20:32] += 0.3; p.requires_grad_(True)
o2 = sp.compute_loss(p, y); o2["fk_loss"].backward()
g = p.grad
print("perturbed dq:", {k: float(v) for k, v in o2.items()}, "| grad norm REL pos dims", float(g[..., [0, 1, 2, 10, 11, 12]].norm()), "dq dims", float(g[..., 20:32].norm()))
assert o2["fk_loss"] > 1e-4 and g[..., [0, 1, 2]].norm() > 0 and g[..., 20:32].norm() > 0
print("REL16_AUX UNIT TEST PASS")
