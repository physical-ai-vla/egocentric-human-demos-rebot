"""[2026-09-30] rel16_aux with per-sample supervision masks, for the ego CART-ONLY pretrain (ego_relcart20task_rel16_v4_nopseudoq).

rel16_aux.py (sha-pinned by the running launchers) is NOT modified; this is a separate copy. Same architecture / schema:
action (16, 32) = REL20 (REL16 pos3 rot6d g per arm) + dq12, same rel / dq / FK terms and lambdas. Two optional loss-only columns:
    aux.has_dq_supervision  (B,)  1 = the dq12 target exists, 0 = no joint label (ego)
    aux.has_fk_supervision  (B,)  1 = aux.q_t exists and FK consistency applies, 0 = no q_t (ego)
Contract (generic, never by dataset name):
  * a column that is ABSENT means ones -> the loss is computed with the ORIGINAL rel16_aux expressions, so real R312c v4 FT is
    unchanged (tested bit-exact against rel16_aux on a real v4 batch: test_rel16_aux_masked.py)
  * aux.q_t may be absent only if every sample has has_fk_supervision == 0
  * masked dq target dims may be NaN (ego stores NaN: no value exists). Before the model runs they are set to 0 in the
    normalized space (the flow-matching input x_t = noise*t + action*(1-t) needs a finite value); NaN anywhere else, or in a
    dq dim of a sample with has_dq_supervision == 1, raises
  * masked samples contribute EXACTLY 0 to dq_loss / fk_loss and exactly 0 gradient (torch.where, not multiply). The mean is
    over the FULL batch (B * 16 * 12 for dq, B * 16 * 2 for FK), i.e. per-sample weight identical to the unmasked recipe
  * rel_loss (dims 0:20, gripper dims 9 / 19 included, as in the real v4 recipe) is never masked
Env: as rel16_aux (REL16_STATS, REL16_LAMBDA_Q, REL16_LAMBDA_FK, REL16_LOG_EVERY).
"""
import json, os, sys
import torch

LAMBDA_Q = float(os.environ.get("REL16_LAMBDA_Q", "1.0"))
LAMBDA_FK = float(os.environ.get("REL16_LAMBDA_FK", "20"))
LOG_EVERY = int(os.environ.get("REL16_LOG_EVERY", "200"))
MASK_KEYS = ("aux.has_dq_supervision", "aux.has_fk_supervision")
AUX_KEYS = ("aux.q_t",) + MASK_KEYS
_S = {"n": 0, "fk_mm": [], "masked_dq": 0, "masked_fk": 0, "samples": 0}


def _mask(batch, key, B, dev):
    m = batch.get(key)
    return None if m is None else (m.reshape(B).to(dev) > 0.5)


def sanitize_action(action, mdq):
    """normalized action (B, 16, 32): masked dq dims NaN -> 0; any other non-finite value raises"""
    if mdq is None:
        return action
    dq = action[..., 20:32]; keep = mdq[:, None, None]
    if not torch.isfinite(action[..., :20]).all():
        raise RuntimeError("[rel16_aux_masked] non-finite REL20 target")
    if not torch.isfinite(dq[mdq]).all():
        raise RuntimeError("[rel16_aux_masked] non-finite dq target on a sample with has_dq_supervision = 1")
    return torch.cat([action[..., :20], torch.where(keep, dq, torch.zeros_like(dq))], dim=-1)


def install():
    sys.path.insert(0, "/home/bh-aiteam/c8old/xvla")
    import rebot_fk_torch
    import lerobot.policies.xvla.modeling_xvla as M
    import lerobot.policies.xvla.action_hub as H
    st = json.load(open(os.environ["REL16_STATS"]))["action"]
    mean = torch.tensor(st["mean"], dtype=torch.float32); std = torch.tensor(st["std"], dtype=torch.float32)
    assert mean.numel() == 32, f"REL16_STATS action dim {mean.numel()} != 32 (need a REL20 + dq12 dataset)"
    fk_cache = {}

    def fk_for(dev):
        if dev not in fk_cache:
            fk_cache[dev] = rebot_fk_torch.ReBotFKTorch(device=dev, dtype=torch.float32)
        return fk_cache[dev]

    # the LeRobot preprocessor drops unknown keys: keep every loss-only aux column (rel16_aux kept only aux.q_t)
    import lerobot.policies.factory as F
    orig_mk = F.make_pre_post_processors

    class _KeepAux:
        def __init__(self, pre): self._pre = pre
        def __call__(self, batch):
            aux = {k: batch[k] for k in AUX_KEYS if isinstance(batch, dict) and k in batch}
            out = self._pre(batch)
            if aux and isinstance(out, dict):
                ref = out.get("action", out.get("observation.state"))
                for k, v in aux.items():
                    out[k] = v.to(ref.device) if ref is not None else v
            return out
        def __getattr__(self, n): return getattr(self._pre, n)

    def mk(*args, **kw):
        pre, post = orig_mk(*args, **kw)
        return _KeepAux(pre), post
    F.make_pre_post_processors = mk

    orig_forward = M.XVLAPolicy.forward

    def forward(self, batch):
        a = batch["action"]; B, dev = a.shape[0], a.device
        mdq, mfk = _mask(batch, MASK_KEYS[0], B, dev), _mask(batch, MASK_KEYS[1], B, dev)
        q = batch.get("aux.q_t")
        if q is None and (mfk is None or mfk.any()):
            raise RuntimeError("[rel16_aux_masked] batch has no aux.q_t but FK supervision is on for some sample")
        batch = dict(batch); batch["action"] = sanitize_action(a, mdq)
        sp = self.model.action_space; sp._rel16_qt, sp._rel16_mdq, sp._rel16_mfk = q, mdq, mfk
        return orig_forward(self, batch)
    M.XVLAPolicy.forward = forward

    def compute_loss(self, pred, target):
        pred = self._pad_to_model_dim(pred); target = self._pad_to_model_dim(target)
        assert pred.shape[-1] >= 32 and target.shape[-1] >= 32 and self.real_dim == 32, \
            f"[rel16_aux_masked] expected real_dim 32, got {self.real_dim} / {tuple(pred.shape)}"
        mdq, mfk = getattr(self, "_rel16_mdq", None), getattr(self, "_rel16_mfk", None)
        B = pred.shape[0]; _S["samples"] += B
        out = {"rel_loss": torch.mean((pred[..., :20] - target[..., :20]) ** 2) * self.JOINTS_SCALE}
        if mdq is None:                                                                # original rel16_aux expression
            out["dq_loss"] = LAMBDA_Q * torch.mean((pred[..., 20:32] - target[..., 20:32]) ** 2) * self.JOINTS_SCALE
        else:
            se = torch.where(mdq[:, None, None], (pred[..., 20:32] - target[..., 20:32]) ** 2, torch.zeros_like(pred[..., 20:32]))
            out["dq_loss"] = LAMBDA_Q * se.sum() / se.numel() * self.JOINTS_SCALE; _S["masked_dq"] += int((~mdq).sum())
        qt = getattr(self, "_rel16_qt", None)
        if LAMBDA_FK > 0 and mfk is not None and not mfk.any():                        # no FK label in this batch (ego)
            out["fk_loss"] = torch.zeros((), dtype=pred.dtype, device=pred.device); _S["masked_fk"] += B
        elif LAMBDA_FK > 0 and qt is not None:
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
                if mfk is None:
                    out["fk_loss"] = (LAMBDA_FK * d2.mean()).to(pred.dtype)
                else:
                    d2m = torch.where(mfk[:, None, None], d2, torch.zeros_like(d2))
                    out["fk_loss"] = (LAMBDA_FK * d2m.sum() / d2m.numel()).to(pred.dtype); _S["masked_fk"] += int((~mfk).sum())
                with torch.no_grad():
                    sel = d2 if mfk is None else d2[mfk]
                    if sel.numel():
                        _S["fk_mm"].append(sel.detach().sqrt().reshape(-1, 2).cpu() * 1000)
        _S["n"] += 1
        if _S["n"] % LOG_EVERY == 1:
            msg = f"[rel16_aux_masked] batch {_S['n']}: rel {out['rel_loss'].item():.4f} dq {out['dq_loss'].item():.4f} fk {out.get('fk_loss', torch.tensor(0.)).item():.5f}" \
                  f" | masked samples so far dq {_S['masked_dq']} / fk {_S['masked_fk']} of {_S['samples']}"
            if _S["fk_mm"]:
                a = torch.cat(_S["fk_mm"]); _S["fk_mm"] = []
                msg += f" | FK(dq) vs REL pos: L p50 {a[:, 0].median():.1f} R p50 {a[:, 1].median():.1f} mm"
            print(msg + f"  (lambda_q {LAMBDA_Q}, lambda_fk {LAMBDA_FK})", flush=True)
        return out
    H.AutoActionSpace.compute_loss = compute_loss
    print(f"[rel16_aux_masked] installed: REL16 + masked dq12 aux (lambda_q {LAMBDA_Q}) + masked FK (lambda_fk {LAMBDA_FK}), stats {os.environ['REL16_STATS']}", flush=True)
