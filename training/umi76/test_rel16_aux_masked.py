"""[2026-09-30] acceptance test for rel16_aux_masked.py + the ego cart-only dataset (CPU only, node env).

usage: test_rel16_aux_masked.py unit  <ego root> <real v4 root>
       test_rel16_aux_masked.py model <ego root> <base dir>
unit : U1 real v4 batch, no mask columns -> masked plugin losses AND grads bit-identical to rel16_aux (real FT unchanged)
       U2 ego batch through LeRobotDataset (loader smoke) -> dq_loss == 0, fk_loss == 0 exactly, d/dpred[..., 20:32] == 0 exactly,
          rel_loss finite, rel grads non-zero; NaN dq dims sanitized; no aux.q_t / aux.fk_mask / pseudo columns
       U3 mixed batch (real, 2 of 4 samples masked) -> masked samples get exactly 0 dq/FK grad, unmasked ones non-zero
       U4 NaN in a has_dq = 1 sample / missing q_t with FK on -> raises
model: M1 full X-VLA (base, launcher overrides) forward+backward on 2 ego batches: total loss finite, dq_loss == fk_loss == 0,
          action_decoder weight / bias grads for output dims 20:32 exactly 0 in every domain, dims 0:20 non-zero
"""
import glob, importlib, json, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def need(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    if not ok: FAILS.append(msg)


def col(t, c):
    a = t.column(c).combine_chunks(); return a.storage if isinstance(a, pa.ExtensionArray) else a


def ego_loader(root, n):
    import lerobot.policies.factory  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    cfg = PreTrainedConfig.from_pretrained("/home/bh-aiteam/xvla_base_cart20v3")
    for k, v in dict(chunk_size=16, n_action_steps=16, max_state_dim=20, max_action_dim=32, action_mode="auto", use_proprio=True).items(): setattr(cfg, k, v)
    meta = LeRobotDatasetMetadata("rebot/ego", root=root); dts = resolve_delta_timestamps(cfg, meta)
    need("action" not in (dts or {}), "loader: action not re-expanded by delta timestamps")
    ds = LeRobotDataset("rebot/ego", root=root, delta_timestamps=dts, video_backend="pyav")
    idx = np.linspace(0, len(ds) - 1, n).astype(int); items = [ds[int(i)] for i in idx]
    x = items[0]
    need(tuple(x["action"].shape) == (16, 32) and tuple(x["observation.state"].shape) == (20,), f"loader: action {tuple(x['action'].shape)} state {tuple(x['observation.state'].shape)}")
    need(not any(k in x for k in ("aux.q_t", "aux.fk_mask")) and not any("pseudo" in k or k.startswith("q_") for k in x), f"QA5 loader keys have no pseudo-joint column: {sorted(k for k in x if k.startswith('aux'))}")
    need(all(float(it["aux.has_dq_supervision"]) == 0 and float(it["aux.has_fk_supervision"]) == 0 for it in items), "QA5 has_dq / has_fk == 0 on every sampled row")
    need(all(torch.isnan(it["action"][:, 20:]).all() and torch.isfinite(it["action"][:, :20]).all() for it in items), "loader: dq dims NaN, REL20 finite")
    need(all(it[k].shape[-3:] == (3, 224, 224) for it in items[:2] for k in it if k.startswith("observation.images")), "loader: 3 video streams decode at 224")
    return ds, items, meta


def unit(ego_root, real_root):
    import lerobot.policies.xvla.action_hub as H
    os.environ["REL16_STATS"] = f"{real_root}/meta/stats.json"
    import rel16_aux; rel16_aux.install(); co = H.AutoActionSpace.compute_loss
    import rel16_aux_masked as RM; RM.install(); cm = H.AutoActionSpace.compute_loss
    sp = H.AutoActionSpace(32, 32)
    st = json.load(open(f"{real_root}/meta/stats.json"))["action"]; mu, sd = torch.tensor(st["mean"]), torch.tensor(st["std"])
    t = pq.read_table(sorted(glob.glob(f"{real_root}/data/chunk-*/*.parquet"))[0], columns=["action", "aux.q_t"])
    rows = [0, 777, 5000, 20000]
    A = torch.tensor(np.asarray(col(t, "action").to_pylist(), np.float32)[rows]); Q = torch.tensor(np.asarray(col(t, "aux.q_t").to_pylist(), np.float32)[rows])
    tgt = (A - mu) / sd; g = torch.Generator().manual_seed(0); p0 = torch.randn(tgt.shape, generator=g) * 0.3 + tgt

    def run(fn, mdq, mfk, qt, target):
        p = p0.clone().requires_grad_(True); sp._rel16_qt, sp._rel16_mdq, sp._rel16_mfk = qt, mdq, mfk
        out = fn(sp, p, target); sum(out.values()).backward(); return out, p.grad
    # U1
    oo, go = run(co, None, None, Q, tgt); om, gm = run(cm, None, None, Q, tgt)
    need(all(torch.equal(oo[k], om[k]) for k in oo) and set(oo) == set(om), f"U1 real v4 losses bit-identical {({k: float(v) for k, v in om.items()})}")
    need(torch.equal(go, gm), "U1 real v4 grads bit-identical")
    # U2 ego (loader smoke) -- target normalized with the ego stats (dq dims: mean 0 / std 1 placeholder)
    ds, items, meta = ego_loader(ego_root, 4)
    es = json.load(open(f"{ego_root}/meta/stats.json"))["action"]
    need(es["mean"][20:] == [0.0] * 12 and es["std"][20:] == [1.0] * 12, "ego stats dq dims = identity placeholder (mean 0, std 1)")
    EA = torch.stack([it["action"] for it in items]); et = (EA - torch.tensor(es["mean"])) / torch.tensor(es["std"])
    mdq = torch.stack([it["aux.has_dq_supervision"] for it in items]).reshape(4) > 0.5; mfk = torch.stack([it["aux.has_fk_supervision"] for it in items]).reshape(4) > 0.5
    ets = RM.sanitize_action(et, mdq)
    need(torch.isfinite(ets).all() and (ets[..., 20:] == 0).all() and torch.equal(ets[..., :20], et[..., :20]), "U2 sanitize: NaN dq -> 0, REL20 untouched")
    oe, ge = run(cm, mdq, mfk, None, ets)
    need(oe["dq_loss"].item() == 0.0 and oe["fk_loss"].item() == 0.0, f"U2 ego dq_loss {oe['dq_loss'].item()} fk_loss {oe['fk_loss'].item()} exactly 0")
    need(torch.isfinite(oe["rel_loss"]) and oe["rel_loss"].item() > 0, f"U2 ego rel_loss finite {oe['rel_loss'].item():.4f}")
    need((ge[..., 20:32] == 0).all().item() and (ge[..., :20].abs() > 0).any().item(), "U2 ego grad dims 20:32 exactly 0, dims 0:20 non-zero")
    # U3 mixed
    mm = torch.tensor([True, False, True, False]); ox, gx = run(cm, mm, mm, Q, tgt)
    need((gx[~mm][..., 20:32] == 0).all().item() and (gx[mm][..., 20:32].abs() > 0).any().item(), f"U3 mixed: masked samples dq/FK grad exactly 0, unmasked non-zero (dq {ox['dq_loss'].item():.4f} fk {ox['fk_loss'].item():.5f})")
    # U4
    bad = tgt.clone(); bad[0, 0, 25] = float("nan")
    try: RM.sanitize_action(bad, torch.ones(4, dtype=torch.bool)); need(False, "U4 NaN on a supervised dq sample raises")
    except RuntimeError: need(True, "U4 NaN on a supervised dq sample raises")


def model(ego_root, base):
    os.environ["REL16_STATS"] = f"{ego_root}/meta/stats.json"
    import rel16_aux_masked as RM; RM.install()
    import lerobot.policies.factory as F
    from lerobot.configs.policies import PreTrainedConfig
    ds, items, meta = ego_loader(ego_root, 4)
    cfg = PreTrainedConfig.from_pretrained(base); cfg.pretrained_path = base
    for k, v in dict(device="cpu", dtype="float32", chunk_size=16, n_action_steps=16, max_state_dim=20, max_action_dim=32, action_mode="auto", use_proprio=True,
                     freeze_vision_encoder=False, freeze_language_encoder=False, train_policy_transformer=True, train_soft_prompts=True).items(): setattr(cfg, k, v)
    rename = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}
    policy = F.make_policy(cfg=cfg, ds_meta=meta, rename_map=rename); policy.train()
    pre, _ = F.make_pre_post_processors(policy_cfg=cfg, pretrained_path=base, preprocessor_overrides={
        "device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": rename},
        "normalizer_processor": {"stats": meta.stats, "features": {**policy.config.input_features, **policy.config.output_features}, "norm_map": policy.config.normalization_mapping}})
    dec = policy.model.transformer.action_decoder if hasattr(policy.model, "transformer") else [m for n, m in policy.named_modules() if n.endswith("action_decoder")][0]
    for bi in range(2):
        batch = torch.utils.data.default_collate(items[2 * bi:2 * bi + 2]); batch = pre(batch)
        need(all(k in batch for k in RM.MASK_KEYS), f"M1 batch {bi}: mask keys survive the preprocessor")
        policy.zero_grad(set_to_none=True); loss, log = policy.forward(batch); loss.backward()
        need(bool(torch.isfinite(loss)), f"M1 batch {bi}: total loss finite {log}")
        need(log["dq_loss"] == 0.0 and log["fk_loss"] == 0.0, f"M1 batch {bi}: dq_loss / fk_loss exactly 0")
        W = dec.fc.weight.grad.view(-1, dec.input_size, dec.output_size); Bg = dec.bias.weight.grad
        need(bool((W[..., 20:32] == 0).all() and (Bg[:, 20:32] == 0).all()), f"M1 batch {bi}: action_decoder grad dims 20:32 exactly 0 (all domains)")
        need(bool((W[..., :20].abs() > 0).any()), f"M1 batch {bi}: action_decoder grad dims 0:20 non-zero")


if __name__ == "__main__":
    torch.set_num_threads(8)
    {"unit": unit, "model": model}[sys.argv[1]](*sys.argv[2:])
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}"); sys.exit(1 if FAILS else 0)
