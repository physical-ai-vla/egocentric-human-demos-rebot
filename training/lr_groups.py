"""Per-group learning rates by parameter name. env XVLA_LR_GROUPS="vlm=5e-6,proj=1e-5,align=2e-5,xytok=3e-5,encdec=1e-5,softnorm=1e-5,policy=1e-5"
vlm=model.vlm (vision/language LoRA) | proj=vlm_proj,aux_visual_proj | align=target_align.{q,k,out} | xytok=target_align.{xy_proj,tok_proj}
encdec=action_encoder,action_decoder | softnorm=soft_prompt_hub,norm,pos_emb | policy=transformer blocks (+anything else)."""
import os, torch
def _group(n):
    if "target_align" in n: return "xytok" if ("xy_proj" in n or "tok_proj" in n) else "align"
    if ".vlm." in n or n.startswith("model.vlm"): return "vlm"
    if "vlm_proj" in n or "aux_visual_proj" in n: return "proj"
    if "arm_head" in n: return "arm"
    if "action_encoder" in n: return "enc"
    if "action_decoder" in n: return "dec"
    if "soft_prompt_hub" in n or ".norm." in n or n.endswith(".norm.weight") or n.endswith(".norm.bias") or "pos_emb" in n: return "softnorm"
    return "policy"
def install():
    spec = os.environ.get("XVLA_LR_GROUPS", "")
    if not spec: return
    lrs = {k: float(v) for k, v in (kv.split("=") for kv in spec.split(","))}
    from lerobot.optim.optimizers import XVLAAdamWConfig
    def build(self, params):
        assert isinstance(params, dict); groups = {}
        for n, p in params.items():
            if p.requires_grad: groups.setdefault(_group(n), []).append(p)
        pg = [{"params": ps, "lr": lrs.get(g, self.lr), "weight_decay": self.weight_decay, "name": g} for g, ps in groups.items()]
        print("[lr_groups] " + " | ".join(f"{d['name']}: n={len(d['params'])} lr={d['lr']:.1e}" for d in pg), flush=True)
        return torch.optim.AdamW(pg, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
    XVLAAdamWConfig.build = build
