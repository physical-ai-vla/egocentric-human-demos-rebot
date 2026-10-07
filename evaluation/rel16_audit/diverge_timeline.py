#!/usr/bin/env python3
"""[2026-09-29] Cycle-wise divergence timeline from ~/v4_cycles.jsonl (logging only).
Per cycle, for one arm, in the BASE frame (mm): predicted displacement at the executed waypoint (IK target - TCP at inference,
= R_t * p_hat_k), actual displacement (TCP after - TCP at inference), cumulative sums, |pred|, IK residual, max limit excess.
Also: sign persistence of the dominant axis, and whether |pred| shrinks when the arm has stopped making progress.
usage: diverge_timeline.py <HH:MM:SS start> <HH:MM:SS end> <arm L|R> [--ckpt substring]
"""
import argparse, datetime, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("start"); ap.add_argument("end"); ap.add_argument("arm"); ap.add_argument("--ckpt", default="")
a = ap.parse_args(); ai = "LR".index(a.arm)
R = [json.loads(l) for l in open(os.path.expanduser("~/v4_cycles.jsonl"))]
day = datetime.datetime.fromtimestamp(R[-1]["t"]).date()
ts = lambda s: datetime.datetime.combine(day, datetime.time(*map(int, s.split(":")))).timestamp()
S = [r for r in R if ts(a.start) <= r["t"] <= ts(a.end) and a.ckpt in r["ckpt"]]
print(f"{len(S)} cycles {a.start}-{a.end} {S[0]['ckpt']} exec_k {S[0]['exec_k']} arm {a.arm}")
pred = np.array([np.array(r["tcp_target_mm"][ai]) - np.array(r["tcp_now_mm"][ai]) for r in S])
act = np.array([(np.array(r["tcp_after_mm"][ai]) if r["tcp_after_mm"] else np.array(r["tcp_now_mm"][ai])) - np.array(r["tcp_now_mm"][ai]) for r in S])
now = np.array([r["tcp_now_mm"][ai] for r in S])
lim = [max((r.get("limit_excess") or [0] * 12)[ai * 6:(ai + 1) * 6]) for r in S]
print(" cyc |  TCP x    y    z  | pred dx  dy  dz  |pred| | act dx  dy  dz | cum pred x y z      | IK res | lim | grip")
cp = np.zeros(3)
for i, r in enumerate(S):
    cp += pred[i]
    print(f" {r['cycle']:3d} | {now[i][0]:5.0f}{now[i][1]:5.0f}{now[i][2]:5.0f} | {pred[i][0]:5.0f}{pred[i][1]:5.0f}{pred[i][2]:5.0f} {np.linalg.norm(pred[i]):5.0f} | "
          f"{act[i][0]:5.0f}{act[i][1]:5.0f}{act[i][2]:5.0f} | {cp[0]:6.0f}{cp[1]:6.0f}{cp[2]:6.0f} | {r['ik_residual_mm']:6.1f} | {lim[i]:.2f} | {r['grip_cmd_sent'][ai]:.0f}")
tot_p, tot_a = pred.sum(0), act.sum(0); ax = int(np.argmax(np.abs(now[-1] - now[0])))
sg = np.sign(pred[:, ax]); run = max(len(list(g)) for k, g in __import__("itertools").groupby(sg))
print(f"\nnet TCP change {np.round(now[-1] - now[0])} mm; dominant axis {'xyz'[ax]}")
print(f"sum predicted {np.round(tot_p)} | sum actual {np.round(tot_a)} | actual/predicted on {'xyz'[ax]}: {tot_a[ax] / max(abs(tot_p[ax]), 1e-9) * np.sign(tot_p[ax]):.2f}")
print(f"predicted {'xyz'[ax]} sign: + in {np.mean(sg > 0):.2f} of cycles, longest same-sign run {run} cycles")
h = len(S) // 2
print(f"|pred| median first half {np.median(np.linalg.norm(pred[:h], axis=1)):.0f} mm, second half {np.median(np.linalg.norm(pred[h:], axis=1)):.0f} mm "
      f"(should shrink toward 0 near a target / at rest)")
