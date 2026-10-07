#!/usr/bin/env python3
"""[2026-09-29] Grasp-miss diagnostic: around each arm's first CLOSE command (open -> close transition of the SENT jaw cmd), print
+-3 cycles of ~/v4_cycles.jsonl: g_pred[1:16], selected g, sent cmd, jaw width, TCP now/target/after, IK residual, limit excess.
Reading (fixed before looking at the data):
  (a) early close   -- TCP still closing in on the object when g_sel < 0.6 (TCP moves > 10 mm/cycle in the cycles after close)
  (b) spatial miss  -- TCP had settled (< 10 mm/cycle) at close; the jaw closes to ~0 -> the object was not between the fingers
  (c) slip          -- width stays > 15 mm after close, then drops to ~0 during the lift
Cube position is not logged (no perception here): (a) vs (b) still needs the user's view / the video at t_close.
usage: close_window.py [--since HH:MM] [--log ~/v4_cycles.jsonl]
"""
import argparse, datetime, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--since", default=None); ap.add_argument("--log", default="~/v4_cycles.jsonl")
a = ap.parse_args()
R = [json.loads(l) for l in open(os.path.expanduser(a.log))]
if a.since:
    h, m = map(int, a.since.split(":")); d = datetime.datetime.fromtimestamp(R[-1]["t"])
    t0 = d.replace(hour=h, minute=m, second=0).timestamp(); R = [r for r in R if r["t"] >= t0]
print(f"{len(R)} cycles | ckpt {R[0]['ckpt'] if R else '-'} | exec_k {sorted({r['exec_k'] for r in R})}")
for arm, ai in (("L", 0), ("R", 1)):
    closes = [i for i in range(1, len(R)) if R[i]["grip_cmd_sent"][ai] < 20 <= R[i - 1]["grip_cmd_sent"][ai]]
    print(f"\n==== {arm}: {len(closes)} open->close transitions at cycles {[R[i]['cycle'] for i in closes]}")
    for c in closes:
        print(f"-- t_close = cycle {R[c]['cycle']} ({datetime.datetime.fromtimestamp(R[c]['t']).strftime('%H:%M:%S')})")
        for i in range(max(0, c - 3), min(len(R), c + 4)):
            r = R[i]; now = np.array(r["tcp_now_mm"][ai]); tgt = np.array(r["tcp_target_mm"][ai])
            aft = np.array(r["tcp_after_mm"][ai]) if r["tcp_after_mm"] else now
            g = r["g_pred"][ai]; ex = r["limit_excess"] or []
            exs = max(ex[ai * 6:(ai + 1) * 6]) if ex else float("nan")
            print(f"  {'>>' if i == c else '  '} cyc {r['cycle']:3d} g[k1,4,8,12,16] {g[0]:.2f} {g[3]:.2f} {g[7]:.2f} {g[11]:.2f} {g[15]:.2f} sel {r['g_sel'][ai]:.2f} "
                  f"sent {r['grip_cmd_sent'][ai]:4.0f} width {r['grip_width_now_mm'][ai]:5.1f} | tcp {now.round(0)} step {np.linalg.norm(aft - now):5.1f} "
                  f"tgt-now {np.linalg.norm(tgt - now):5.1f} z {now[2]:6.1f} | IK res {r['ik_residual_mm']:5.1f} limit_excess {exs:.3f}")
        post = R[c + 1:c + 4]
        if post:
            mv = np.median([np.linalg.norm(np.array(p["tcp_after_mm"][ai]) - np.array(p["tcp_now_mm"][ai])) for p in post if p["tcp_after_mm"]] or [np.nan])
            w = [p["grip_width_now_mm"][ai] for p in post]
            verdict = "(c) slip-like" if max(w) > 15 else ("(a) early-close-like" if mv > 10 else "(b) spatial-miss-like")
            print(f"   post-close: TCP step med {mv:.1f} mm/cycle, width {w} -> {verdict} (confirm with the view at t_close)")
