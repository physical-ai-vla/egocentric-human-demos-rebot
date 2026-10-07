"""[2026-09-29] REL16-v3 training plugin for lerobot-seeed X-VLA: Cartesian REL16 primary + joint Δq16 auxiliary + FK consistency.

Dataset contract (derive_v3d.py): action (16, 32) = [REL16 L pos3 rot6d grip | R pos3 rot6d grip] (20) + Δq12 (rad), and a
loss-only column aux.q_t (12, rad, follower q at t). X-VLA predicts the CLEAN action x0 in the normalized space
(modeling_xvla.forward: action_noisy = noise*t + action*(1-t), loss on pred vs action), so the loss can be split and a
geometric term applied to the unnormalized prediction:

    rel_loss = MSE(pred[..., :20], target[..., :20])                      (identical to the v1/v2 loss on those dims)
    dq_loss  = REL16_LAMBDA_Q  * MSE(pred[..., 20:32], target[..., 20:32])
    fk_loss  = REL16_LAMBDA_FK * mean_k,arm || R_t^T (FK(q_t + Δq̂_k).p − FK(q_t).p) − p̂_REL,k ||²   [m²]

Both Δq̂ and p̂_REL are the model's own predictions (unnormalized with the dataset action mean/std exactly as the
postprocessor does: x = y·std + mean), so the term ties the Cartesian head to the joint head; the TCP is
rebot_fk_torch.tcp, which equals the dataset TCP (0.000 mm / 0.00 deg, 2026-09-29), so no frame fix is needed.
Env: REL16_STATS (dataset meta/stats.json, required), REL16_LAMBDA_Q (1.0), REL16_LAMBDA_FK (20), REL16_LOG_EVERY (200).
Install before lerobot_train builds the policy: `import rel16_aux; rel16_aux.install()` (train_rel16aux.py does this).
"""
import json, os, sys
import torch

LAMBDA_Q = float(os.environ.get("REL16_LAMBDA_Q", "1.0"))
LAMBDA_FK = float(os.environ.get("REL16_LAMBDA_FK", "20"))
LOG_EVERY = int(os.environ.get("REL16_LOG_EVERY", "200"))
_S = {"n": 0, "fk_mm": []}


def install():
    sys.path.insert(0, "/home/bh-aiteam/c8old/xvla")
    import rebot_fk_torch
    import lerobot.policies.xvla.modeling_xvla as M
    import lerobot.policies.xvla.action_hub as H
    st = json.load(open(os.environ["REL16_STATS"]))["action"]
    mean = torch.tensor(st["mean"], dtype=torch.float32); std = torch.tensor(st["std"], dtype=torch.float32)
    assert mean.numel() == 32, f"REL16_STATS action dim {mean.numel()} != 32 (need the v3d dataset)"
    fk_cache = {}

    def fk_for(dev):
        if dev not in fk_cache:
            fk_cache[dev] = rebot_fk_torch.ReBotFKTorch(device=dev, dtype=torch.float32)
        return fk_cache[dev]

    # the LeRobot preprocessor rebuilds the batch from its features and drops unknown keys, so aux.q_t (loss-only, not a
    # feature) never reached forward() -- found in the first v3 GPU smoke. Wrap the preprocessor (C-old humanik_delta pattern):
    # take aux.q_t out before, put it back after, on the device the processed action lives on. lerobot_train imports
    # make_pre_post_processors by name, so this must run before lerobot_train is imported (train_rel16aux.py does).
    import lerobot.policies.factory as F
    orig_mk = F.make_pre_post_processors

    class _KeepAux:
        def __init__(self, pre): self._pre = pre
        def __call__(self, batch):
            q = batch.get("aux.q_t") if isinstance(batch, dict) else None
            out = self._pre(batch)
            if q is not None and isinstance(out, dict):
                ref = out.get("action", out.get("observation.state"))
                out["aux.q_t"] = q.to(ref.device) if ref is not None else q
            return out
        def __getattr__(self, n): return getattr(self._pre, n)

    def mk(*args, **kw):
        pre, post = orig_mk(*args, **kw)
        return _KeepAux(pre), post
    F.make_pre_post_processors = mk

    orig_forward = M.XVLAPolicy.forward

    def forward(self, batch):
        q = batch.get("aux.q_t")
        if q is None:
            raise RuntimeError("[rel16_aux] batch has no aux.q_t -- train on the v3d dataset")
        self.model.action_space._rel16_qt = q
        return orig_forward(self, batch)
    M.XVLAPolicy.forward = forward

    def compute_loss(self, pred, target):
        pred = self._pad_to_model_dim(pred); target = self._pad_to_model_dim(target)
        assert pred.shape[-1] >= 32 and target.shape[-1] >= 32 and self.real_dim == 32, \
            f"[rel16_aux] expected real_dim 32, got {self.real_dim} / {tuple(pred.shape)}"
        out = {"rel_loss": torch.mean((pred[..., :20] - target[..., :20]) ** 2) * self.JOINTS_SCALE,
               "dq_loss": LAMBDA_Q * torch.mean((pred[..., 20:32] - target[..., 20:32]) ** 2) * self.JOINTS_SCALE}
        qt = getattr(self, "_rel16_qt", None)
        if LAMBDA_FK > 0 and qt is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                dev = pred.device; m, s = mean.to(dev), std.to(dev)
                x = pred[..., :32].float() * s + m                                     # unnormalized prediction (B, 16, 32)
                qt32 = qt.to(dev).float()                                              # (B, 12)
                fk = fk_for(dev)
                with torch.no_grad():
                    Tt = fk.tcp(qt32)                                                  # (B, 2, 4, 4)
                Tk = fk.tcp(qt32[:, None] + x[..., 20:32])                             # (B, 16, 2, 4, 4), grad -> Δq̂
                Rt = Tt[..., :3, :3]; pt = Tt[..., :3, 3]
                p_fk = torch.einsum("baji,bkaj->bkai", Rt, Tk[..., :3, 3] - pt[:, None])  # R_t^T (p_k - p_t), (B, 16, 2, 3)
                p_rel = torch.stack([x[..., 0:3], x[..., 10:13]], dim=2)               # predicted REL16 position, (B, 16, 2, 3)
                d2 = ((p_fk - p_rel) ** 2).sum(-1)                                     # m²
                out["fk_loss"] = (LAMBDA_FK * d2.mean()).to(pred.dtype)
                with torch.no_grad():
                    _S["fk_mm"].append(d2.detach().sqrt().reshape(-1, 2).cpu() * 1000)
        _S["n"] += 1
        if _S["n"] % LOG_EVERY == 1 and _S["fk_mm"]:
            a = torch.cat(_S["fk_mm"]); _S["fk_mm"] = []
            print(f"[rel16_aux] batch {_S['n']}: rel {out['rel_loss'].item():.4f} dq {out['dq_loss'].item():.4f} fk {out.get('fk_loss', torch.tensor(0.)).item():.5f} "
                  f"| FK(dq) vs REL pos, predicted: L p50 {a[:, 0].median():.1f} p95 {torch.quantile(a[:, 0], .95):.1f} mm, "
                  f"R p50 {a[:, 1].median():.1f} p95 {torch.quantile(a[:, 1], .95):.1f} mm  (lambda_q {LAMBDA_Q}, lambda_fk {LAMBDA_FK})", flush=True)
        return out
    H.AutoActionSpace.compute_loss = compute_loss
    print(f"[rel16_aux] installed: REL16 + dq12 aux (lambda_q {LAMBDA_Q}) + FK consistency (lambda_fk {LAMBDA_FK}), stats {os.environ['REL16_STATS']}", flush=True)
