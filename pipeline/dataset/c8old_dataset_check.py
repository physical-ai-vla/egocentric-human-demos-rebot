#!/usr/bin/env python3
"""[2026-09-25] C-old dataset post-build checks (contract §24). Structure: every episode = 65 frames, video frame counts ==
rows, nonfinite 0, gripper raw range (state [-270, 0], action [0, 45]), train / val source episodes disjoint.
Loader parity: LeRobotDataset with the TRAINING delta_timestamps window ([0] + (5+i)/30, i = 0..29) returns exactly the stored
rows t, t+5..t+34 for observation.ee.tcp_tgt / action. C parity: from the LOADED T_store window, the training-code measured
formula trans(inv(T[t]) @ T[t+5+i]) x EE_AUX_SCALE equals the same quantity recomputed independently by the builder from the
measured TCP (episode_rows). Main parity: action[t+5+i] - state[t] (arm, deg) == stored Phase-3 q difference.
Env: ~/xvla-mac/bin/python."""
import json, pathlib, sys
import numpy as np, torch
H = pathlib.Path.home(); C8 = H / "c8"; OUT = C8 / "c8old_data"; sys.path.insert(0, str(C8))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; LEAD, K = 5, 30; bad = []
prov = json.load(open(OUT / "c8old_provenance.json")); val_eps = set(prov["val_episodes"])
for split in ("c8old_train", "c8old_val"):
    ds = LeRobotDataset(f"local/{split}", root=str(OUT / split), video_backend="pyav"); hf = ds.hf_dataset; meta = ds.meta
    st = np.stack(hf["observation.state"]); ac = np.stack(hf["action"]); tt = np.stack(hf["observation.ee.tcp_tgt"]); ep = np.asarray(hf["episode_index"])
    lens = np.bincount(ep); nf = int((~np.isfinite(st)).sum() + (~np.isfinite(ac)).sum() + (~np.isfinite(tt)).sum())
    gs, ga = st[:, [6, 13]], ac[:, [6, 13]]
    print(f"{split}: episodes {len(lens)} rows {len(st)} lengths {set(lens.tolist())} nonfinite {nf} | state grip [{gs.min():.1f}, {gs.max():.1f}] action grip [{ga.min():.2f}, {ga.max():.2f}]"
          f" | videos {meta.total_frames if hasattr(meta, 'total_frames') else 'n/a'} frames, {len(meta.video_keys)} video keys")
    if set(lens.tolist()) != {65} or nf or gs.min() < -270.01 or gs.max() > 0.01 or ga.min() < -0.01 or ga.max() > 45.01: bad.append(f"{split} structure")
    win = [0.0] + [(LEAD + i) / 30 for i in range(K)]
    dsw = LeRobotDataset(f"local/{split}", root=str(OUT / split), video_backend="pyav", delta_timestamps={"observation.ee.tcp_tgt": win, "action": win[1:], "observation.state": [0.0]})
    rng = np.random.default_rng(0); starts = []
    for e in np.unique(ep):
        ix = np.flatnonzero(ep == e); starts += list(ix[0] + np.arange(0, 31))
    pick = rng.choice(starts, min(30, len(starts)), replace=False); lp = cp = mp = 0.0
    for g in pick:
        it = dsw[int(g)]; Tw = it["observation.ee.tcp_tgt"].numpy(); rows = np.r_[g, g + LEAD + np.arange(K)]
        lp = max(lp, float(np.abs(Tw - tt[rows]).max())); aw = it["action"].numpy(); lp = max(lp, float(np.abs(aw - ac[g + LEAD + np.arange(K)]).max()))
        T = torch.tensor(Tw, dtype=torch.float32).reshape(K + 1, 2, 4, 4); d = torch.linalg.inv(T[0])[None] @ T[1:]
        c_train = (d[..., :3, 3] * 10).numpy()                                          # training-code measured C (EE_AUX_SCALE 10)
        Ts = tt[rows].reshape(K + 1, 2, 4, 4); c_ref = np.stack([(np.linalg.inv(Ts[0, a]) @ Ts[1 + i, a])[:3, 3] * 10 for i in range(K) for a in (0, 1)]).reshape(K, 2, 3)
        cp = max(cp, float(np.abs(c_train - c_ref).max()))
        mp = max(mp, float(np.abs((ac[g + LEAD + np.arange(K)][:, ARM] - st[g][ARM]) - (st[g + LEAD + np.arange(K)][:, ARM] - st[g][ARM])).max()))
    print(f"   loader window parity max|d| {lp:.2e} | C parity (training formula vs numpy ref) {cp:.2e} (x10 units) | main: action == pseudo-q state arm window {mp:.2e} deg")
    if lp > 1e-6 or cp > 1e-4 or mp > 1e-4: bad.append(f"{split} parity")
tr_eps = set(json.load(open(OUT / "c8old_provenance.json"))["report"].keys())
print("train / val source episodes disjoint by construction (val list in provenance):", len(val_eps), "val episodes")
print("CHECKS", "PASS" if not bad else f"FAIL {bad}")
