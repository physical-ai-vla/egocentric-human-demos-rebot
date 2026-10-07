#!/usr/bin/env python3
"""[2026-09-29] Train LearnedIK-v0 (LEARNEDIK_V0_CONTRACT.md) on the frozen P12 train split; val = P12 val episodes.
Test episodes are never loaded here. Recipe fixed before training (not tuned on results):
  AdamW lr 1e-3, wd 1e-4, cosine to 0 over STEPS=20000, warmup 500, batch 1024, fp32, lam_fk 0.1 (cm^2; 1.0 failed the smoke magnitude check), seed 0.
  val every 500 steps -> best.pt by val L_fk_pos; last.pt at the end.
--smoke: loss magnitudes at init (L_dq vs lam_fk*L_fk within 10x, the only allowed lam check) + 200 steps.
usage: train_lik0.py <out dir under runs/> [--smoke] [--device mps|cpu]"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lik0_common as C

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
a = ap.parse_args()
# lam_fk: contract default 1.0 failed the smoke magnitude check (init ratio 26.7, LIK0-P12-SMOKE, 2026-09-29) -> 0.1, the
# power of ten that puts the init ratio in [0.1, 10] (2.7). Set once at smoke, never from val results.
STEPS, WARM, BS, LR, WD, LAM, SEED = (200 if a.smoke else 20000), 500, 1024, 1e-3, 1e-4, float(os.environ.get("LIK_LAM_FK", "0.1")), 0
# [2026-09-29] LIK-v3L (intent labels): the smoke rule gave init ratio 12.7 at 0.1 -> LIK_LAM_FK=0.01 (ratio 1.27); LIK0 keeps 0.1
out = os.path.join(C.HERE, "runs", a.run)
if os.path.exists(out):
    sys.exit(f"{out} exists, refusing to overwrite")
os.makedirs(out)
torch.manual_seed(SEED); np.random.seed(SEED)
dev = torch.device(a.device)
D = C.load_data(); st = json.load(open(os.path.join(C.DATA_DIR, "lik0_stats.json")))
T = lambda m: {k: torch.from_numpy(D[k][m]).float().to(dev) for k in ("q", "rel", "dq")}
tr, va = T(D["split"] == 0), T(D["split"] == 1)
n = len(tr["q"]); print(f"train {n} tuples, val {len(va['q'])} tuples, device {dev}", flush=True)
model = C.LIK0(st).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
cfg = dict(steps=STEPS, warmup=WARM, batch=BS, lr=LR, wd=WD, lam_fk=LAM, seed=SEED, device=str(dev),
           params=sum(p.numel() for p in model.parameters()), split="P12", stats=os.path.relpath(os.path.join(C.DATA_DIR, "lik0_stats.json"), C.HERE))
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)


@torch.no_grad()
def evaluate(d):
    model.eval(); tot = np.zeros(3); mm = []
    for s in range(0, len(d["q"]), 4096):
        q, r, g = d["q"][s:s + 4096], d["rel"][s:s + 4096], d["dq"][s:s + 4096]
        l, ldq, lfk = C.loss_fn(model, q, r, g, LAM)
        tot += np.array([l.item(), ldq.item(), lfk.item()]) * len(q)
        p_fk, _ = C.fk_rel(q, model(q, r)); p_rel, _ = C.rel_split(r)
        mm.append(((p_fk - p_rel).norm(dim=-1)[:, 15] * 1000).cpu().numpy())      # k16, (B, 2)
    model.train(); mm = np.concatenate(mm)
    return tot / len(d["q"]), mm


with torch.no_grad():
    b = torch.randperm(n, device="cpu")[:BS].to(dev)
    _, ldq, lfk = C.loss_fn(model, tr["q"][b], tr["rel"][b], tr["dq"][b], LAM)
    ratio = (LAM * lfk.item()) / max(ldq.item(), 1e-12)
    print(f"[init] L_dq {ldq.item():.4f}  lam_fk*L_fk {LAM * lfk.item():.4f} (cm^2)  ratio {ratio:.2f}", flush=True)
    cfg["init_ratio"] = ratio; json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    if not (0.1 <= ratio <= 10):
        print("[init] loss magnitude check FAILED (outside 10x) -- stop and report, do not retune silently", flush=True)
        sys.exit(2)

best, t0, log = float("inf"), time.time(), open(os.path.join(out, "log.jsonl"), "w")
perm, pi = torch.randperm(n), 0
for step in range(1, STEPS + 1):
    if pi + BS > n:
        perm, pi = torch.randperm(n), 0
    b = perm[pi:pi + BS].to(dev); pi += BS
    l, ldq, lfk = C.loss_fn(model, tr["q"][b], tr["rel"][b], tr["dq"][b], LAM)
    opt.zero_grad(set_to_none=True); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 500 == 0 or step == STEPS or (a.smoke and step % 50 == 0):
        (vl, vdq, vfk), mm = evaluate(va)
        rec = dict(step=step, train=l.item(), train_dq=ldq.item(), train_fk=lfk.item(), val=vl, val_dq=vdq, val_fk=vfk,
                   val_k16_mm_p50=[float(np.median(mm[:, i])) for i in (0, 1)],
                   val_k16_mm_p95=[float(np.percentile(mm[:, i], 95)) for i in (0, 1)], lr=sched.get_last_lr()[0],
                   sec=round(time.time() - t0, 1))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"step {step:5d} train {l.item():.4f} (dq {ldq.item():.4f} fk {lfk.item():.4f}) | val dq {vdq:.4f} fk {vfk:.4f} "
              f"k16 p50 L {rec['val_k16_mm_p50'][0]:.2f} R {rec['val_k16_mm_p50'][1]:.2f} mm p95 L {rec['val_k16_mm_p95'][0]:.1f} "
              f"R {rec['val_k16_mm_p95'][1]:.1f} mm | {rec['sec']:.0f}s", flush=True)
        if vfk < best:
            best = vfk; torch.save(dict(model=model.state_dict(), step=step, stats=st, cfg=cfg), os.path.join(out, "best.pt"))
torch.save(dict(model=model.state_dict(), step=STEPS, stats=st, cfg=cfg), os.path.join(out, "last.pt"))
print(f"done: best val L_fk {best:.4f} cm^2, {out}", flush=True)
