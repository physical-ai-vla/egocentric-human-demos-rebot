"""[2026-10-02] AdamW8bit variant of train_rel16_relonly.py: torch.optim.AdamW -> bitsandbytes AdamW8bit (bnb 0.45.5 from
${REMOTE_HOME}/pylib_bnb, NOT installed into the holobrain env), then runs the unchanged REL-only entry (runpy).
Checkpoint fix: bnb keeps state['step'] as a python int, which lerobot's safetensors optimizer save rejects
("Key state/0/step ... expected torch.Tensor"). state_dict() returns COPIES of the per-param state with step as a 0-d int64
tensor (the live state is not touched); load_state_dict() turns it back into an int before bnb loads it."""
import os, runpy, sys
sys.path.insert(0, "${REMOTE_HOME}/pylib_bnb")
import torch, bitsandbytes as bnb


class AdamW8bitShim(bnb.optim.AdamW8bit):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, **kw):
        assert not kw.get("amsgrad"), "amsgrad not supported by the 8bit shim"
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        print(f"[8bit] bitsandbytes {bnb.__version__} AdamW8bit replaces torch AdamW (groups {len(self.param_groups)})", flush=True)

    def state_dict(self):
        # also: bnb shares ONE qmap tensor across all params; safetensors refuses shared storage -> clone repeats
        sd = super().state_dict(); seen = set(); out = {}
        for k, v in sd["state"].items():
            d = {}
            for kk, vv in v.items():
                if kk == "step" and isinstance(vv, int):
                    vv = torch.tensor(vv, dtype=torch.int64)
                elif torch.is_tensor(vv):
                    ptr = vv.untyped_storage().data_ptr()
                    if ptr in seen:
                        vv = vv.clone()
                    else:
                        seen.add(ptr)
                d[kk] = vv
            out[k] = d
        sd["state"] = out
        return sd

    def load_state_dict(self, state_dict, *a, **kw):
        st = {k: {kk: (int(vv.item()) if kk == "step" and torch.is_tensor(vv) else vv) for kk, vv in v.items()}
              for k, v in state_dict["state"].items()}
        n = super().load_state_dict({**state_dict, "state": st}, *a, **kw)
        steps = {v.get("step") for v in self.state.values() if "step" in v}
        print(f"[8bit] optimizer state loaded: {len(self.state)} params, step values {sorted(steps)[:3]}", flush=True)
        return n


torch.optim.AdamW = AdamW8bitShim
sys.argv[0] = "${REMOTE_HOME}/umi_bridge/umi76/train_rel16_relonly.py"
runpy.run_path(sys.argv[0], run_name="__main__")
