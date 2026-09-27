"""[2026-09-08] Batched, differentiable reBot FK in torch (URDF-parsed via rebot_fk.ReBotFK). Used by humanik_delta for the
EE auxiliary target (C) and the FK-consistency loss (D). TCP = {side}_gripper_link @ T(TIP_X along +x) (pinch centre, TIP_X=0.07 m)."""
import os, sys, math, torch, numpy as np
sys.path.insert(0, "${HOME}/rebot_ee")
os.environ.setdefault("REBOT_URDF", "${HOME}/rebot_ee/reBot_B601_DM_dualarm.urdf")
import rebot_fk  # noqa: E402

TIP_X = float(os.environ.get("REBOT_TIP_X", "0.07"))

class ReBotFKTorch:
    def __init__(self, device="cpu", dtype=torch.float32):
        fk = rebot_fk.ReBotFK(os.environ["REBOT_URDF"]); self.dev, self.dt = device, dtype
        self.chains = {}
        for side in ("left", "right"):
            steps = []
            for n in fk.chains[side]:
                j = fk.joints[n]
                steps.append((torch.tensor(j["T"], dtype=dtype, device=device), j["type"] == "revolute",
                              torch.tensor(np.asarray(j["axis"], float) / (np.linalg.norm(j["axis"]) + 1e-12), dtype=dtype, device=device)))
            self.chains[side] = steps
        self.T_tip = torch.eye(4, dtype=dtype, device=device); self.T_tip[0, 3] = TIP_X

    def to(self, device, dtype=None):
        dtype = dtype or self.dt
        if device == self.dev and dtype == self.dt: return self
        n = ReBotFKTorch.__new__(ReBotFKTorch); n.dev, n.dt = device, dtype
        n.chains = {s: [(T.to(device=device, dtype=dtype), r, a.to(device=device, dtype=dtype)) for T, r, a in st] for s, st in self.chains.items()}
        n.T_tip = self.T_tip.to(device=device, dtype=dtype); return n

    @staticmethod
    def _rot(axis, th):
        """Rodrigues: axis (3,), th (N,) -> (N,4,4)."""
        N = th.shape[0]; c = torch.cos(th); s = torch.sin(th); C = 1 - c; x, y, z = axis
        R = torch.stack([
            torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
            torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
            torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)
        T = torch.eye(4, dtype=th.dtype, device=th.device).expand(N, 4, 4).clone(); T[:, :3, :3] = R; return T

    def side(self, q6, side):
        """q6 (N,6) rad -> (N,4,4) base->TCP."""
        N = q6.shape[0]; T = torch.eye(4, dtype=q6.dtype, device=q6.device).expand(N, 4, 4).clone(); k = 0
        for Tf, rev, axis in self.chains[side]:
            T = T @ Tf.to(q6.dtype)
            if rev: T = T @ self._rot(axis.to(q6.dtype), q6[:, k]); k += 1
        return T @ self.T_tip.to(q6.dtype)

    def tcp(self, q12):
        """q12 (..., 12) rad = [L6, R6] -> (..., 2, 4, 4)."""
        sh = q12.shape[:-1]; q = q12.reshape(-1, 12)
        return torch.stack([self.side(q[:, :6], "left"), self.side(q[:, 6:], "right")], 1).reshape(*sh, 2, 4, 4)

if __name__ == "__main__":   # self-test vs numpy FK
    fk = ReBotFKTorch(); npfk = rebot_fk.ReBotFK(os.environ["REBOT_URDF"]); rng = np.random.default_rng(0); err = 0
    for _ in range(50):
        q14 = np.zeros(14); q14[[0,1,2,3,4,5,7,8,9,10,11,12]] = rng.uniform(-1.5, 1.5, 12)
        ref = npfk.eef(q14, tip=True); T = fk.tcp(torch.tensor(q14[[0,1,2,3,4,5,7,8,9,10,11,12]], dtype=torch.float32)[None])[0]
        err = max(err, np.abs(T[0, :3, 3].numpy() - ref["left"]).max(), np.abs(T[1, :3, 3].numpy() - ref["right"]).max())
    print("torch FK vs numpy FK max |dpos| = %.2e m" % err)
