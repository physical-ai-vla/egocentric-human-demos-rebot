"""[2026-09-30] REL-only ablation of the REL16-v3/v4 trainer: ONE variable, Δq + FK supervision ON -> OFF.

Same dataset bytes (r312c_relcart20_rel16_v4, action (16, 32) = REL20 + Δq12, aux.q_t present but unused), same base,
same architecture/decoder width (32), same optimizer/schedule/seed. Differences vs rel16_aux.py:
  loss        rel_loss only = MSE(pred[..., :20], target[..., :20]) * JOINTS_SCALE   (identical expression to v4's rel_loss,
              gripper dims 9/19 included exactly as in v4). No dq_loss, no fk_loss, no aux.q_t, no FK import.
  channels    action dims 20:32 are HARD-ZEROED at the transformer boundary, both ways, in training and at every
              inference denoising step (user, option 1):
                input  action_with_noise[..., 20:32] := 0  -> ground-truth Δq never reaches the network (no leakage,
                       train == infer), and the untrained 20:32 prediction never feeds back into the next flow step
                output pred[..., 20:32]            := 0  -> decoder rows 20:32 receive exactly zero task gradient
              (AdamW weight decay still shrinks those rows; they carry no task information -- do not read them as pretrained).
The same zero contract must be applied at deployment: call install_zero_mask() before the policy runs (infer side).
Env: RELONLY_ZERO (default "20:32"), REL16_LOG_EVERY (200), RELONLY_PREFLIGHT=N (self-checks on the first N batches).
"""
import inspect, os, sys
import torch

LOG_EVERY = int(os.environ.get("REL16_LOG_EVERY", "200"))
_lo, _hi = (int(x) for x in os.environ.get("RELONLY_ZERO", "20:32").split(":"))
PREFLIGHT = int(os.environ.get("RELONLY_PREFLIGHT", "0"))
_S = {"n": 0, "fwd_calls": 0, "fwd_in_nonzero": 0, "fwd_out_nonzero": 0}


def _mask_like(x):
    m = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    m[_lo:_hi] = 0
    return m


def install_zero_mask():
    """Zero action channels [_lo:_hi) at the SoftPromptedTransformer boundary (input and output). Train and infer."""
    import lerobot.policies.xvla.soft_transformer as ST
    if getattr(ST.SoftPromptedTransformer, "_relonly_patched", False):
        return
    orig = ST.SoftPromptedTransformer.forward
    sig = inspect.signature(orig)

    def forward(self, *args, **kw):
        b = sig.bind(self, *args, **kw)
        x = b.arguments["action_with_noise"]
        assert x.shape[-1] >= _hi, f"[relonly] action dim {x.shape[-1]} < {_hi}"
        b.arguments["action_with_noise"] = x * _mask_like(x)
        _S["fwd_calls"] += 1
        if b.arguments["action_with_noise"][..., _lo:_hi].abs().max().item() != 0.0:
            _S["fwd_in_nonzero"] += 1
        out = orig(*b.args, **b.kwargs)
        out = out * _mask_like(out)
        if out[..., _lo:_hi].abs().max().item() != 0.0:
            _S["fwd_out_nonzero"] += 1
        return out
    ST.SoftPromptedTransformer.forward = forward
    ST.SoftPromptedTransformer._relonly_patched = True
    print(f"[relonly] action channels {_lo}:{_hi} hard-zeroed at the transformer input and output (train + every infer step)", flush=True)


def install():
    import lerobot.policies.xvla.action_hub as H
    install_zero_mask()

    def compute_loss(self, pred, target):
        pred = self._pad_to_model_dim(pred); target = self._pad_to_model_dim(target)
        assert pred.shape[-1] >= 32 and target.shape[-1] >= 32 and self.real_dim == 32, \
            f"[relonly] expected real_dim 32, got {self.real_dim} / {tuple(pred.shape)}"
        out = {"rel_loss": torch.mean((pred[..., :20] - target[..., :20]) ** 2) * self.JOINTS_SCALE}
        _S["n"] += 1
        if PREFLIGHT and _S["n"] <= PREFLIGHT:
            # GT Δq randomised -> the loss must not change (the target's 20:32 is not used anywhere)
            t2 = target.clone(); t2[..., 20:32] = torch.randn_like(t2[..., 20:32]) * 10
            l2 = torch.mean((pred[..., :20] - t2[..., :20]) ** 2) * self.JOINTS_SCALE
            ok = torch.equal(l2.detach(), out["rel_loss"].detach())
            print(f"[relonly-preflight] batch {_S['n']}: {'PASS' if ok else 'FAIL'} loss invariant to GT dq randomisation | "
                  f"{'PASS' if pred[..., 20:32].abs().max().item() == 0 else 'FAIL'} pred 20:32 == 0 | "
                  f"{'PASS' if torch.isfinite(out['rel_loss']).item() else 'FAIL'} loss finite ({out['rel_loss'].item():.4f}) | "
                  f"transformer calls {_S['fwd_calls']}, input 20:32 non-zero {_S['fwd_in_nonzero']}, output non-zero {_S['fwd_out_nonzero']}", flush=True)
        if _S["n"] % LOG_EVERY == 1:
            print(f"[relonly] batch {_S['n']}: rel {out['rel_loss'].item():.4f} (dq loss OFF, FK loss OFF, channels {_lo}:{_hi} zeroed)", flush=True)
        return out
    H.AutoActionSpace.compute_loss = compute_loss
    assert "rebot_fk_torch" not in sys.modules, "[relonly] FK module must not be imported"
    print(f"[relonly] installed: REL(0:20, incl. gripper) loss only; no dq loss, no FK loss, aux.q_t unused", flush=True)
