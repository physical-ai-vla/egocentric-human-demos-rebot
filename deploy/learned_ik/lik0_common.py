"""[2026-09-29] LearnedIK-v0 shared pieces: FK (rebot_fk_torch = dataset TCP), tensor contract, model, loss.
Contract: LEARNEDIK_V0_CONTRACT.md. Input q_t (12, rad, follower frame) + rel (16, 18) = per arm pos3 + rot6d (first two ROWS
of R) of A_k = inv(T_t) T_{t+(k+1)dt}; output dq (16, 12) rad. No vision, language or gripper."""
import os, sys, json
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("REBOT_URDF", os.path.expanduser("~/holobrain-mac-model/reBot_B601_DM_dualarm.urdf"))
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))          # rebot_fk.py (URDF parser)
sys.path.insert(0, os.path.expanduser("~/c8/c8old"))                     # rebot_fk_torch.py, sha f85f8121 == node copy
import rebot_fk_torch  # noqa: E402

H, NJ = 16, 12
REL_COLS = list(range(0, 9)) + list(range(10, 19))                       # action dims -> rel18 (gripper 9, 19 dropped)
DQ_COLS = list(range(20, 32))
SPLIT = os.path.join(HERE, "LEARNEDIK_V0_SPLIT.json")
_FK = {}


def fk(dev, dtype=torch.float32):
    k = (str(dev), dtype)
    if k not in _FK:
        _FK[k] = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=dtype).to(dev, dtype)
    return _FK[k]


def rot6d_to_mat(r6):
    """rot6d = first two ROWS of R (umi pose_util.mat_to_rot6d); Gram-Schmidt like pose_util.rot6d_to_mat."""
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)                              # rows


def fk_rel(q_t, dq):
    """Relative TCP pose inv(T_t) T(q_t + dq_k): returns pos (B,16,2,3) in m and R (B,16,2,3,3)."""
    f = fk(q_t.device, q_t.dtype)
    Tt = f.tcp(q_t)                                                       # (B, 2, 4, 4)
    Tk = f.tcp(q_t[:, None] + dq)                                         # (B, 16, 2, 4, 4)
    Rt = Tt[..., :3, :3]
    p = torch.einsum("baji,bkaj->bkai", Rt, Tk[..., :3, 3] - Tt[:, None, :, :3, 3])
    R = torch.einsum("baji,bkajl->bkail", Rt, Tk[..., :3, :3])
    return p, R


def rel_split(rel):
    """rel (B,16,18) -> pos (B,16,2,3), R (B,16,2,3,3)."""
    r = rel.reshape(*rel.shape[:2], 2, 9)
    return r[..., :3], rot6d_to_mat(r[..., 3:9])


class Norm(nn.Module):
    def __init__(self, st):
        super().__init__()
        for k in ("q", "rel", "dq"):
            self.register_buffer(f"{k}_m", torch.tensor(st[k]["mean"], dtype=torch.float32))
            self.register_buffer(f"{k}_s", torch.tensor(st[k]["std"], dtype=torch.float32))


class LIK0(nn.Module):
    """Deterministic MLP 300 -> 1024 x3 -> 192 (LayerNorm + SiLU); takes and returns physical units."""

    def __init__(self, st, width=1024, depth=3):
        super().__init__()
        self.norm = Norm(st)
        layers, d = [], NJ + H * 18
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.SiLU()]; d = width
        layers.append(nn.Linear(d, H * NJ))
        self.net = nn.Sequential(*layers)

    def forward_norm(self, q, rel):
        n = self.norm
        x = torch.cat([(q - n.q_m) / n.q_s, ((rel - n.rel_m) / n.rel_s).flatten(1)], 1)
        return self.net(x).view(-1, H, NJ)                                # normalized dq

    def denorm(self, y):
        return y * self.norm.dq_s + self.norm.dq_m

    def forward(self, q, rel):
        return self.denorm(self.forward_norm(q, rel))                    # dq in rad


def loss_fn(model, q, rel, dq, lam_fk=1.0):
    """L = MSE(normalized dq) + lam_fk * mean ||R_t^T(FK(q_t+dq_hat).p - FK(q_t).p) - p_rel||^2 in cm^2 (position only)."""
    y = model.forward_norm(q, rel)
    n = model.norm
    l_dq = ((y - (dq - n.dq_m) / n.dq_s) ** 2).mean()
    p_fk, _ = fk_rel(q, model.denorm(y))
    p_rel, _ = rel_split(rel)
    l_fk = (((p_fk - p_rel) * 100.0) ** 2).sum(-1).mean()
    return l_dq + lam_fk * l_fk, l_dq, l_fk


DATA_DIR = os.path.join(HERE, os.environ.get("LIK_DATA_DIR", "data"))       # [2026-09-29] LIK-v3L uses data_v3L


def load_data(path=None):
    return dict(np.load(path or os.path.join(DATA_DIR, "lik0_data.npz")))
