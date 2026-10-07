#!/usr/bin/env python3
"""[2026-09-29] Re-infer saved real-robot cycles (~/v4_cycle_frames/<t>.json: the exact images + state + task of that cycle) with
N noise draws and report the base-frame predicted displacement per arm at k4 / k8 / k16: mean, std, and the fraction of draws with
the same sign as the mean on each axis. Same checkpoint as the rollout; nothing on the robot moves.
usage: V4_CKPT=<ckpt> V4_STATE_MODE=<umi76|relcart20> replay_frames.py <HH:MM:SS start> <HH:MM:SS end> <cycle,cycle,...> [--draws 10]
"""
import argparse, datetime, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ.setdefault("V4_ACTION_MODE", "umi")
import infer_core_v4 as IC

ap = argparse.ArgumentParser(); ap.add_argument("start"); ap.add_argument("end"); ap.add_argument("cycles"); ap.add_argument("--draws", type=int, default=10)
a = ap.parse_args(); want = [int(c) for c in a.cycles.split(",")]
fs = sorted(glob.glob(os.path.expanduser("~/v4_cycle_frames/*.json")))
day = datetime.datetime.fromtimestamp(float(os.path.basename(fs[-1])[:-5])).date()
ts = lambda s: datetime.datetime.combine(day, datetime.time(*map(int, s.split(":")))).timestamp()
ck = os.path.basename(os.environ["V4_CKPT"].rstrip("/"))
sel = {}
for f in fs:
    t = float(os.path.basename(f)[:-5])
    if ts(a.start) <= t <= ts(a.end):
        d = json.load(open(f))
        if d["ckpt"] == ck and d["cycle"] in want and d["cycle"] not in sel:
            sel[d["cycle"]] = d
print(f"{ck}: found cycles {sorted(sel)} of {want}")
INF = IC.V4Inferencer(os.environ["V4_CKPT"])
dim = INF.policy.model.dim_action
for c in sorted(sel):
    d = sel[c]; o = d["obs"][-1]
    m, _ = INF.tcp_now(o["joints14"])
    P = []
    for s in range(a.draws):
        g = torch.Generator().manual_seed(100 + s)
        act = INF.infer_raw(o["images"], np.asarray(d["state"], np.float32), task=d["task"], noise=torch.randn(1, INF.chunk, dim, generator=g))
        P.append([[m[r][:3, :3] @ act[k, r * 10:r * 10 + 3] * 1000 for k in (3, 7, 15)] for r in (0, 1)])
        G = locals().setdefault("G", []); G.append([[act[k, r * 10 + 9] for k in (0, 3, 7, 15)] for r in (0, 1)])
    P = np.array(P)                                                                   # (draws, arm, 3 k, 3 xyz) base frame mm
    Gc = np.array(G[-a.draws:]); G.clear()
    print(f"   grip g mean k1/4/8/16  L {np.round(Gc[:, 0].mean(0), 2)} (min over draws at k8 {Gc[:, 0, 2].min():.2f})  R {np.round(Gc[:, 1].mean(0), 2)} | draws with g8 < 0.6: L {np.mean(Gc[:, 0, 2] < 0.6):.1f} R {np.mean(Gc[:, 1, 2] < 0.6):.1f}")
    print(f"\n-- cycle {c}  TCP L {np.round(m[0][:3, 3] * 1000)} R {np.round(m[1][:3, 3] * 1000)}  task {d['task'][:60]}")
    for r, arm in ((0, "L"), (1, "R")):
        for j, k in enumerate((4, 8, 16)):
            x = P[:, r, j]; mu = x.mean(0); sd = x.std(0); same = (np.sign(x) == np.sign(mu)).mean(0)
            print(f"   {arm} k{k:2d} mean {np.round(mu).astype(int)} std {np.round(sd).astype(int)} |mean| {np.linalg.norm(mu):5.1f} | same-sign x/y/z {same.round(2)}")
