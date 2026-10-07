"""[2026-10-03 user] REL-only trainer + a per-sample, per-dim LOSS MASK over action dims 0:20 (right-arm-only ego tasks).

Copy-and-extend of rel16_aux_relonly.py (that file is sha-pinned by running launchers and is NOT modified).
Everything of the REL-only recipe stays: action dims 20:32 are hard-zeroed at the transformer input and output
(install_zero_mask from rel16_aux_relonly), no dq loss, no FK loss.  One addition:

    aux.loss_mask  (B, 20) float, 1 = supervise this action dim, 0 = no target exists (e.g. the dummy LEFT arm and a
                   constant right gripper of the HRA_red right-only approach task).  Same mask for all 16 horizon steps.

    rel_loss = sum_{b,k,d<20} m[b,d] * (pred - target)^2 / sum_{b,k,d<20} m[b,d]   * JOINTS_SCALE

  * a batch WITHOUT aux.loss_mask (every existing bimanual dataset) uses m = 1, which is exactly the REL-only loss
    (mean over B*16*20) -- tested against rel16_aux_relonly's expression
  * masked dims contribute exactly 0 loss and 0 gradient, also with a NaN placeholder target (target replaced first)
  * normalising by the number of SUPERVISED elements keeps the per-element weight of a right-only sample equal to a
    bimanual one when the two are mixed in one batch (cotrain)
Env: REL16_LOG_EVERY (200), RELONLY_PREFLIGHT=N (self-checks on the first N batches), RELONLY_ZERO (20:32).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rel16_aux_relonly as R   # noqa: E402  (install_zero_mask, the pinned zero contract)

MASK_KEY = "aux.loss_mask"
LOG_EVERY = int(os.environ.get("REL16_LOG_EVERY", "200"))
PREFLIGHT = int(os.environ.get("RELONLY_PREFLIGHT", "0"))
_S = {"n": 0, "masked_samples": 0, "samples": 0, "supervised_frac": []}


def masked_rel_loss(pred20, target20, m):
    """pred20/target20 (B, H, 20); m (B, 20) bool or None.  Returns the mean over supervised elements."""
    if m is None:
        return torch.mean((pred20 - target20) ** 2)
    mm = m[:, None, :].expand_as(pred20)
    # replace masked targets BEFORE the subtraction: torch.where alone still back-propagates 0 * NaN = NaN through the
    # unselected branch when a masked target is NaN (caught by the unit test)
    safe = torch.where(mm, target20, pred20.detach())
    se = torch.where(mm, (pred20 - safe) ** 2, torch.zeros_like(pred20))
    n = mm.sum()
    if n == 0:
        raise RuntimeError("[relonly-lossmask] a batch with no supervised action element")
    return se.sum() / n


def install():
    import lerobot.policies.factory as F
    import lerobot.policies.xvla.action_hub as H
    import lerobot.policies.xvla.modeling_xvla as M
    R.install_zero_mask()

    # the LeRobot preprocessor drops unknown keys: carry aux.loss_mask past it (same pattern as rel16_aux_masked)
    orig_mk = F.make_pre_post_processors

    class _KeepMask:
        def __init__(self, pre): self._pre = pre
        def __call__(self, batch):
            m = batch.get(MASK_KEY) if isinstance(batch, dict) else None
            out = self._pre(batch)
            if m is not None and isinstance(out, dict):
                ref = out.get("action", out.get("observation.state"))
                out[MASK_KEY] = m.to(ref.device) if ref is not None else m
            return out
        def __getattr__(self, n): return getattr(self._pre, n)

    def mk(*a, **kw):
        pre, post = orig_mk(*a, **kw)
        return _KeepMask(pre), post
    F.make_pre_post_processors = mk

    orig_forward = M.XVLAPolicy.forward

    def forward(self, batch):
        a = batch["action"]; B = a.shape[0]
        m = batch.get(MASK_KEY)
        if m is not None:
            m = m.reshape(B, -1).to(a.device)
            assert m.shape[1] == 20, f"[relonly-lossmask] {MASK_KEY} must be (B, 20), got {tuple(m.shape)}"
            m = m > 0.5
            _S["masked_samples"] += int((~m).any(1).sum()); _S["supervised_frac"].append(float(m.float().mean()))
        _S["dims_supervised"] = None if m is None else m.any(0).detach().cpu()        # read by the preflight gradient hook
        _S["samples"] += B
        self.model.action_space._lossmask = m
        return orig_forward(self, batch)
    M.XVLAPolicy.forward = forward

    def compute_loss(self, pred, target):
        pred = self._pad_to_model_dim(pred); target = self._pad_to_model_dim(target)
        assert pred.shape[-1] >= 32 and target.shape[-1] >= 32 and self.real_dim == 32, \
            f"[relonly-lossmask] expected real_dim 32, got {self.real_dim} / {tuple(pred.shape)}"
        m = getattr(self, "_lossmask", None)
        out = {"rel_loss": masked_rel_loss(pred[..., :20], target[..., :20], m) * self.JOINTS_SCALE}
        _S["n"] += 1
        if PREFLIGHT and _S["n"] <= PREFLIGHT:
            t2 = target.clone(); t2[..., 20:32] = torch.randn_like(t2[..., 20:32]) * 10
            if m is not None:     # masked targets randomised -> loss must not move
                mm = m[:, None, :].expand_as(t2[..., :20]); t2[..., :20] = torch.where(mm, t2[..., :20], torch.randn_like(t2[..., :20]) * 10)
            l2 = masked_rel_loss(pred[..., :20], t2[..., :20], m) * self.JOINTS_SCALE
            ok = torch.equal(l2.detach(), out["rel_loss"].detach())
            if m is not None:
                per_s = m.sum(1).tolist(); H = pred.shape[1]
                print(f"[relonly-lossmask-preflight] batch {_S['n']}: supervised_dims_per_sample {per_s} | effective_mask_sum "
                      f"{int(m.sum()) * H} (= sum dims x {H} horizon; the loss denominator) | full would be {m.shape[0] * 20 * H}", flush=True)
            print(f"[relonly-lossmask-preflight] batch {_S['n']}: {'PASS' if ok else 'FAIL'} loss invariant to masked + 20:32 targets | "
                  f"{'PASS' if pred[..., 20:32].abs().max().item() == 0 else 'FAIL'} pred 20:32 == 0 | "
                  f"mask {'absent (all ones)' if m is None else f'{m.float().mean().item():.3f} supervised, {int((~m).any(1).sum())}/{m.shape[0]} samples masked'} | "
                  f"loss {out['rel_loss'].item():.4f}", flush=True)
        if _S["n"] % LOG_EVERY == 1:
            sf = _S["supervised_frac"][-1] if _S["supervised_frac"] else 1.0
            print(f"[relonly-lossmask] batch {_S['n']}: rel {out['rel_loss'].item():.4f} | supervised dims {sf:.3f} | "
                  f"masked samples so far {_S['masked_samples']}/{_S['samples']}", flush=True)
        return out
    H.AutoActionSpace.compute_loss = compute_loss
    assert "rebot_fk_torch" not in sys.modules, "[relonly-lossmask] FK module must not be imported"
    print(f"[relonly-lossmask] installed: REL(0:20) loss with per-sample {MASK_KEY} (absent = all ones); dims 20:32 zeroed; no dq / FK", flush=True)
