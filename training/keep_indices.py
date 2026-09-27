"""[target head] lerobot's preprocessor pipeline drops non-feature keys (episode_index/frame_index) that the target head needs to look up
labels. install() wraps make_pre_post_processors so the returned preprocessor re-injects those keys after processing."""
import os, torch
KEEP = ("episode_index", "frame_index", "index")
class _KeepWrap:
    def __init__(self, pre): self._pre = pre
    def __call__(self, batch):
        saved = {k: batch[k] for k in KEEP if isinstance(batch, dict) and k in batch}
        out = self._pre(batch)
        if isinstance(out, dict):
            for k, v in saved.items():
                if k not in out:
                    t = v if torch.is_tensor(v) else torch.as_tensor(v)
                    dev = next((x.device for x in out.values() if torch.is_tensor(x)), None); out[k] = t.to(dev) if dev is not None else t
        return out
    def __getattr__(self, n): return getattr(self._pre, n)
def install():
    if os.environ.get("XVLA_TARGET_HEAD", "0") != "1" and os.environ.get("XVLA_TARGET_ALIGN", "0") != "1": return
    import lerobot.scripts.lerobot_train as lt, lerobot.policies.factory as fac
    orig = fac.make_pre_post_processors
    def wrapped(*a, **k):
        pre, post = orig(*a, **k); print("[target_head] preprocessor wrapped to keep episode_index/frame_index", flush=True); return _KeepWrap(pre), post
    fac.make_pre_post_processors = wrapped
    if hasattr(lt, "make_pre_post_processors"): lt.make_pre_post_processors = wrapped
