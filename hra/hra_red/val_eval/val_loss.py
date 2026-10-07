"""[2026-10-04] Held-out validation loss of HRA-RIGHTONLY-LOSSMASK-D20-B8-300K checkpoints on Mac MPS.

Same loss as the training log: rel16_relonly_lossmask.install() (masked REL loss over dims 0:20 * JOINTS_SCALE, dims 20:32 zeroed),
checkpoint pre-processor (domain 20, rename map, normalizer stats from the ckpt), policy.forward(batch) -> loss.
Dataset is built like the node's patched lerobot-seeed factory: the stored (16, 32) action already is the horizon, so
delta_timestamps = None (no expansion); observation_delta_indices is None for xvla; image transforms OFF; float images.
Differences vs training (by design): policy.eval() (dropout / drop_path off), torch.no_grad, MPS instead of CUDA.

usage: val_loss.py [--steps 015000,030000] [--dtype bf16|fp32] [--max_batches N] [--seeds 0,1] [--out results.json]
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/umi_bridge/rel16_audit/relonly"))
import rel16_relonly_lossmask as LM  # noqa: E402

LM.install()    # BEFORE policy / processors are built

from lerobot.datasets.lerobot_dataset import LeRobotDataset            # noqa: E402
from lerobot.policies.factory import make_pre_post_processors           # noqa: E402  (patched by install)
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "ckpts")
VAL = os.path.expanduser("~/c8/hra_red/lerobot/ego_hra_red_rightonly_v1_val")
TRN = os.path.expanduser("~/c8/hra_red/lerobot/ego_hra_red_rightonly_v1_train")
DEV = "mps"


def build_ds(root, tc):
    d = tc["dataset"]
    return LeRobotDataset(d["repo_id"].rsplit("/", 1)[0] + "/" + os.path.basename(root), root=root, delta_timestamps=None,
                          image_transforms=None, video_backend=d.get("video_backend", "pyav"), tolerance_s=tc["tolerance_s"])


def run(policy, pre, ds, idx, seed, dtype, max_batches, bs, verbose=False):
    sub = torch.utils.data.Subset(ds, idx)
    dl = torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=False, num_workers=2, drop_last=False)
    torch.manual_seed(seed); np.random.seed(seed)
    losses, ns, sf = [], [], []
    for i, batch in enumerate(dl):
        if max_batches and i >= max_batches:
            break
        with torch.no_grad():
            b = pre(batch)
            if verbose and i < 2:
                m = b.get(LM.MASK_KEY)
                print(f"   batch {i}: keys has aux.loss_mask={m is not None}"
                      + (f" shape {tuple(m.shape)} supervised frac {float((m > 0.5).float().mean()):.3f}" if m is not None else ""), flush=True)
            if dtype == "bf16":
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    loss, _ = policy.forward(b)
            else:
                loss, _ = policy.forward(b)
        l = float(loss.detach().float().cpu())
        if not np.isfinite(l):
            raise RuntimeError(f"non-finite loss at batch {i}")
        losses.append(l); ns.append(len(batch["index"]))
        if LM._S["supervised_frac"]:
            sf.append(LM._S["supervised_frac"][-1])
        if verbose and i < 3:
            print(f"   batch {i}: loss {l:.4f}", flush=True)
    return {"mean_batch_loss": float(np.mean(losses)), "sample_weighted": float(np.average(losses, weights=ns)),
            "n_batches": len(losses), "n_samples": int(sum(ns)), "supervised_frac": float(np.mean(sf)) if sf else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="015000,030000,045000,050000,005000")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--train_n", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    tc0 = None
    for st in a.steps.split(","):
        ck = os.path.join(CK, st)
        tc = json.load(open(os.path.join(ck, "train_config.json")))
        assert tc["output_dir"].endswith("HRA-RIGHTONLY-LOSSMASK-D20-B8-300K") and tc["dataset"]["root"].endswith("ego_hra_red_rightonly_v1_train")
        if tc0 is None:
            tc0 = tc
            vds = build_ds(VAL, tc); tds = build_ds(TRN, tc)
            vidx = list(range(len(vds)))
            g = np.random.default_rng(12345)
            tidx = sorted(g.choice(len(tds), size=min(a.train_n, len(tds)), replace=False).tolist())
            print(f"[data] val {len(vds)} frames / {vds.num_episodes} eps; train subset {len(tidx)} of {len(tds)} (rng 12345)", flush=True)
        t0 = time.time()
        policy = XVLAPolicy.from_pretrained(ck)
        policy.to(DEV); policy.eval()
        pre, _post = make_pre_post_processors(policy.config, pretrained_path=ck,
                                              preprocessor_overrides={"device_processor": {"device": DEV},
                                                                      "rename_observations_processor": {"rename_map": tc["rename_map"]}},
                                              postprocessor_overrides={"device_processor": {"device": DEV}})
        dom = [s["config"]["domain_id"] for s in json.load(open(os.path.join(ck, "policy_preprocessor.json")))["steps"]
               if s.get("registry_name") == "xvla_add_domain_id"]
        print(f"[{st}] loaded in {time.time() - t0:.1f}s, domain {dom}, dtype {a.dtype}", flush=True)
        r = {"dtype": a.dtype, "domain": dom}
        for s in seeds:
            t1 = time.time(); r[f"val_seed{s}"] = run(policy, pre, vds, vidx, s, a.dtype, a.max_batches, a.bs, verbose=(s == seeds[0]))
            print(f"[{st}] val seed {s}: {r[f'val_seed{s}']} ({time.time() - t1:.0f}s)", flush=True)
        t1 = time.time(); r["train_seed0"] = run(policy, pre, tds, tidx, seeds[0], a.dtype, a.max_batches, a.bs)
        print(f"[{st}] train-subset seed {seeds[0]}: {r['train_seed0']} ({time.time() - t1:.0f}s)", flush=True)
        r["val_mean"] = float(np.mean([r[f"val_seed{s}"]["mean_batch_loss"] for s in seeds]))
        r["gap_val_minus_train"] = r["val_mean"] - r["train_seed0"]["mean_batch_loss"]
        r["runtime_s"] = time.time() - t0
        if not a.max_batches:
            res[st] = r
            json.dump(res, open(a.out, "w"), indent=2)
        print(f"[{st}] DONE val_mean {r['val_mean']:.4f} train {r['train_seed0']['mean_batch_loss']:.4f} gap {r['gap_val_minus_train']:+.4f} "
              f"runtime {r['runtime_s']:.0f}s", flush=True)
        del policy, pre, _post
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
