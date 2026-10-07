"""[2026-09-28] Stage 3 fixture: HEAD training layouts for same-layout / conflicting-prompt robot tests. Frozen, read-only.

Cube positions come from the demonstrations themselves: a pick = an OPEN->CLOSE switch of the binary leader label whose
jaw stalls at 25-95 mm within 1 s; the TCP (base frame, state94 TCP18) at that moment is the cube's (x, y). Pick i is
the recorded order's i-th color, so each episode gives the colored layout. Positions carry ~5 cm pick noise, so only
clean layouts are kept: exactly 3 picks (no regrasp), cubes >= MIN_SEP apart, and a large NN margin = distance to the
nearest OTHER-order training layout (a layout that sits next to a different-order layout is ambiguous as a shortcut).

For each kept layout the recommended prompts are the recorded order plus orders whose FIRST cube is each of the two
other colors: same image, first target must move R -> B -> P with the prompt. A layout-copying model keeps going to the
recorded first cube.

Output: stage3_layouts.md (table) + stage3_layouts.png (overhead, base frame: x forward, y left).
"""
import glob, os, re
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DS = os.path.expanduser("~/holobrain-data/lerobot/r380_umi94_rel16_v2B")
OUTD = os.path.expanduser("~/umi_bridge/rel16_audit/r380")
MIN_SEP, TOP = 0.10, 8
COL = {"red": "R", "blue": "B", "purple": "P"}; RGB = {"R": "#d62728", "B": "#1f77b4", "P": "#9467bd"}


def col(t, c):
    a = t.column(c).combine_chunks(); return a.storage if isinstance(a, pa.ExtensionArray) else a


tabs = [pq.read_table(f) for f in sorted(glob.glob(f"{DS}/data/chunk-*/*.parquet"))]
D = {c: np.concatenate([np.asarray(col(t, c).to_pylist()) for t in tabs]) for c in
     ("observation.state", "action", "episode_index", "frame_index", "task_index")}
S, A, EP, FR, TK = (D[c] for c in ("observation.state", "action", "episode_index", "frame_index", "task_index"))
TS = {int(v): k for k, v in pd.read_parquet(f"{DS}/meta/tasks.parquet")["task_index"].items()}
ORDERS = sorted({"".join(COL[c] for c in re.findall(r"(\w+) cube", t)) for t in TS.values()})

lay = []
for e in np.unique(EP):
    m = np.flatnonzero(EP == e); m = m[np.argsort(FR[m])]
    order = "".join(COL[c] for c in re.findall(r"(\w+) cube", TS[int(TK[m[0]])]))
    picks = []
    for r in (0, 1):
        lab = A[m, 0, r * 10 + 9]; w = S[m, 37 if r == 0 else 75]
        for j in np.flatnonzero((lab[:-1] == 1) & (lab[1:] == 0)):
            k = min(j + 15, len(m) - 1)
            if 0.025 < w[k] < 0.095:
                picks.append((j, r, S[m[k], 76 + 9 * r:78 + 9 * r].copy()))
    picks.sort(key=lambda x: x[0])
    if len(picks) >= 3:
        pos = {order[i]: picks[i][2] for i in range(3)}
        lay.append(dict(e=int(e), pool="HEAD" if e < 180 else "FRONT", order=order, n=len(picks),
                        arms="".join("LR"[p[1]] for p in picks[:3]), pos=pos,
                        vec=np.concatenate([pos["R"], pos["B"], pos["P"]])))
V = np.stack([l["vec"] for l in lay])
rows = []
for i, l in enumerate(lay):
    if l["pool"] != "HEAD" or l["n"] != 3:
        continue
    sep = min(np.linalg.norm(l["pos"][a] - l["pos"][b]) for a, b in (("R", "B"), ("R", "P"), ("B", "P")))
    if sep < MIN_SEP:
        continue
    dist = np.linalg.norm(V - V[i], axis=1); dist[i] = np.inf
    other = np.array([x["order"] != l["order"] for x in lay])
    d_other = dist[other].min(); d_same = dist[~other].min() if (~other).any() else np.inf
    first = l["order"][0]
    conf = [o for o in ORDERS if o[0] != first]
    pick2 = [next(o for o in conf if o[0] == c) for c in "RBP" if c != first]
    rows.append(dict(layout_id=f"ep{l['e']:03d}", R=l["pos"]["R"], B=l["pos"]["B"], P=l["pos"]["P"], recorded_order=l["order"],
                     recorded_first=first, prompts=[l["order"]] + pick2, nn_other_cm=d_other * 100, nn_same_cm=d_same * 100,
                     min_sep_cm=sep * 100, arms=l["arms"]))
rows.sort(key=lambda r: -r["nn_other_cm"])
rows = rows[:TOP]
fmt = lambda p: f"({p[0]:.2f}, {p[1]:+.2f})"
lines = ["# Stage 3 same-layout / conflicting-prompt fixtures (HEAD, frozen 2026-09-28)", "",
         f"Clean HEAD layouts (exactly 3 picks, cubes >= {MIN_SEP*100:.0f} cm apart), ranked by NN margin = distance to the "
         "nearest training layout with a DIFFERENT order. Base frame metres, x forward, y left. Positions +-~5 cm.", "",
         "| layout_id | R(x,y) | B(x,y) | P(x,y) | recorded_order | recorded_first | prompts to run (same layout) | NN margin other-order cm | min cube sep cm | arms |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['layout_id']} | {fmt(r['R'])} | {fmt(r['B'])} | {fmt(r['P'])} | {r['recorded_order']} | {r['recorded_first']} | "
                 f"{' / '.join(r['prompts'])} | {r['nn_other_cm']:.1f} | {r['min_sep_cm']:.0f} | {r['arms']} |")
lines += ["", "Run all three prompts on the SAME placement. Pass = first reach goes to each prompt's first cube; a model that "
          "copies the layout keeps going to `recorded_first`."]
open(f"{OUTD}/stage3_layouts.md", "w").write("\n".join(lines) + "\n")
fig, axs = plt.subplots(2, (len(rows) + 1) // 2, figsize=(4 * ((len(rows) + 1) // 2), 8), squeeze=False)
for ax, r in zip(axs.ravel(), rows):
    for c in "RBP":
        x, y = r[c]
        ax.scatter(-y, x, s=420, c=RGB[c], marker="s", edgecolors="k", linewidths=3 if c == r["recorded_first"] else 0.5)
        ax.text(-y, x, c, ha="center", va="center", color="w", fontsize=12, weight="bold")
    ax.set_title(f"{r['layout_id']} rec {r['recorded_order']}\nrun {' / '.join(r['prompts'])}", fontsize=10)
    ax.set_xlim(-0.45, 0.45); ax.set_ylim(0.05, 0.55); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("robot-left  <-  -y (m)  ->  robot-right"); ax.set_ylabel("x forward (m)")
for ax in axs.ravel()[len(rows):]:
    ax.axis("off")
fig.suptitle("Stage 3 fixtures: thick border = recorded first cube (the layout-shortcut target)")
fig.tight_layout(); fig.savefig(f"{OUTD}/stage3_layouts.png", dpi=90)
print("\n".join(lines[:4 + len(rows) + 1]))
print(f"\nclean HEAD candidates before ranking: {sum(1 for l in lay if l['pool']=='HEAD' and l['n']==3)} with exactly 3 picks")
