"""[2026-10-03] lerobot_train with the REL-only + per-dim loss-mask plugin (rel16_relonly_lossmask.py). Same CLI as lerobot_train.py.
RELONLY_PREFLIGHT=N runs the plugin self-checks on the first N batches and adds a gradient hook on the action decoder that
asserts rows 20:32 get exactly zero task gradient and rows 0:20 a non-zero one."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rel16_relonly_lossmask
rel16_relonly_lossmask.install()

if int(os.environ.get("RELONLY_PREFLIGHT", "0")):
    import lerobot.policies.xvla.soft_transformer as ST
    _orig_init = ST.SoftPromptedTransformer.__init__
    _cnt = {"n": 0}

    def _init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        dec = self.action_decoder
        I, O = dec.input_size, dec.output_size

        def hook_w(g):
            if _cnt["n"] < int(os.environ["RELONLY_PREFLIGHT"]):
                gv = g.view(g.shape[0], I, O)
                z = gv[..., 20:32].abs().max().item(); nz = gv[..., :20].abs().max().item()
                print(f"[relonly-preflight] decoder weight grad: rows 20:32 max {z:.3e} -> {'PASS' if z == 0 else 'FAIL'} | "
                      f"rows 0:20 max {nz:.3e} -> {'PASS' if nz > 0 else 'FAIL'}", flush=True)
                sup = rel16_relonly_lossmask._S.get("dims_supervised")
                if sup is not None:                                   # dims no sample in this batch supervises: exactly zero grad
                    per = gv[..., :20].abs().amax(dim=(0, 1)).cpu()
                    off = [d for d in range(20) if not bool(sup[d])]; on = [d for d in range(20) if bool(sup[d])]
                    zo = max([per[d].item() for d in off], default=0.0); mn = min([per[d].item() for d in on], default=1.0)
                    print(f"[relonly-lossmask-preflight] masked dims {off}: grad max {zo:.3e} -> {'PASS' if zo == 0 else 'FAIL'} | "
                          f"supervised dims {on[0] if on else '-'}..{on[-1] if on else '-'}: min row grad {mn:.3e} -> {'PASS' if mn > 0 else 'FAIL'}", flush=True)
            return g

        def hook_b(g):
            if _cnt["n"] < int(os.environ["RELONLY_PREFLIGHT"]):
                z = g[:, 20:32].abs().max().item()
                print(f"[relonly-preflight] decoder bias grad rows 20:32 max {z:.3e} -> {'PASS' if z == 0 else 'FAIL'}", flush=True)
                _cnt["n"] += 1
            return g
        dec.fc.weight.register_hook(hook_w); dec.bias.weight.register_hook(hook_b)
        print(f"[relonly-preflight] gradient hooks on action_decoder ({I} -> {O})", flush=True)
    ST.SoftPromptedTransformer.__init__ = _init

from lerobot.scripts.lerobot_train import main
main()
