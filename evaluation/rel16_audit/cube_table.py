#!/usr/bin/env python3
"""[2026-09-29] Initial cube positions per episode, from the teleop data itself (no perception), for the prompt -> correct-cube metric.
Rule (fixed before any model output is read):
  grasp event = own jaw was open (> 80 mm) and within 14 rows settles to 20..75 mm (an object between the fingers) for >= 5 rows;
  table grasp = grasp event with TCP z < -10 mm (base frame, rebot_fk_torch.tcp(aux.q_t)); higher closes are re-grasps on the stack.
  The first three table grasps in time = the bottom, middle and top cubes of the episode's order (3-stack picks bottom first),
  at their INITIAL positions (each cube is picked from the table exactly once before it is stacked).
  The episode is valid only if it has >= 3 table grasps and the three are > 40 mm apart; others are excluded and reported.
Writes cube_table.json: {episode: {colour: {"arm": 0|1, "xyz_mm": [..], "row": frame}}, "_order": [bottom, middle, top]}.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch
os.environ.setdefault("REBOT_URDF", os.path.expanduser("~/holobrain-mac-model/reBot_B601_DM_dualarm.urdf"))
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model")); sys.path.insert(0, os.path.expanduser("~/c8/c8old"))
import rebot_fk_torch

R = os.path.expanduser("~/holobrain-data/lerobot/r180_umi76_rel16_v3d")
col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
ts = [pq.read_table(f, columns=["observation.state", "aux.q_t", "episode_index", "frame_index", "task_index"]) for f in sorted(glob.glob(R + "/data/chunk-*/*.parquet"))]
S = np.concatenate([np.asarray(col(t, "observation.state").to_pylist()) for t in ts]); Q = np.concatenate([np.asarray(col(t, "aux.q_t").to_pylist()) for t in ts])
E = np.concatenate([col(t, "episode_index").to_numpy() for t in ts]); F = np.concatenate([col(t, "frame_index").to_numpy() for t in ts])
TI = np.concatenate([col(t, "task_index").to_numpy() for t in ts])
with torch.no_grad():
    T = rebot_fk_torch.ReBotFKTorch(dtype=torch.float64).tcp(torch.tensor(Q)).numpy()
tasks = pd.read_parquet(R + "/meta/tasks.parquet"); TASK = {int(v): k for k, v in tasks["task_index"].items()}
W = S[:, [37, 75]] * 1000
out, bad = {}, {}
for e in np.unique(E):
    m = np.flatnonzero(E == e); ev = []
    for a in (0, 1):
        w = W[m, a]; i = 0
        while i < len(w) - 5:
            if w[i] > 80 and (w[i + 1:i + 15] < 75).any():
                j = i + 1 + int(np.argmax(w[i + 1:i + 15] < 75))
                if (w[j:j + 5] > 20).all() and (w[j:j + 5] < 75).all():
                    ev.append((j, a, T[m[j], a, :3, 3] * 1000))
                i = j + 5; continue
            i += 1
    ev = sorted([x for x in ev if x[2][2] < -10], key=lambda x: x[0])
    order = re.findall(r"(\w+) cube", TASK[int(TI[m[0]])])
    if len(ev) < 3 or len(order) != 3:
        bad[int(e)] = f"{len(ev)} table grasps"; continue
    P = np.stack([x[2] for x in ev[:3]])
    if min(np.linalg.norm(P[i] - P[j]) for i in range(3) for j in range(i + 1, 3)) < 40:
        bad[int(e)] = "table grasps < 40 mm apart"; continue
    out[str(int(e))] = {"_order": order, **{c: {"arm": int(ev[i][1]), "xyz_mm": [round(float(v), 1) for v in ev[i][2]], "row": int(F[m[ev[i][0]]])} for i, c in enumerate(order)}}
json.dump(dict(rule=__doc__.split("Rule")[1].split("Writes")[0].strip(), episodes=out, excluded=bad), open(os.path.expanduser("~/umi_bridge/rel16_audit/cube_table.json"), "w"), indent=1)
print(f"valid episodes {len(out)} / {len(np.unique(E))}; excluded {len(bad)}: {dict(list(bad.items())[:10])}")
n4 = sum(1 for e in np.unique(E) if str(int(e)) in out)
arms = np.array([[out[k][c]["arm"] for c in out[k]["_order"]] for k in out])
print("arm of the bottom cube: L", int((arms[:, 0] == 0).sum()), "R", int((arms[:, 0] == 1).sum()))
