"""[2026-09-28] Stage 2 fixture (ordinary layouts, prompt = the RECORDED order), frozen BEFORE any v2 checkpoint exists.

One HEAD training layout per order (6), taken from the same pick extraction as Stage 3, excluding the Stage 3 layouts,
chosen deterministically (seed 20260928) among clean candidates (exactly 3 picks, cubes >= 10 cm apart; relaxed to
>= 3 picks only if an order has no clean layout). 2 repeats each = 12 trials, order of trials shuffled. Runtime frozen:
REL16-v2B, V4_STATE_MODE=umi94, V4_GRIPPER=binary, exec_k 16, n_action 1, dwell 1.5 s, rot180/mirror off.
Output: r380/stage2_layouts.md, r380/stage2_trials.csv, r380/stage2_layouts.png
"""
import csv, os
import numpy as np
S3 = os.path.expanduser("~/umi_bridge/rel16_audit/stage3_layouts.py")
exec(open(S3).read().split("V = np.stack")[0])          # -> lay, ORDERS, RGB, OUTD, MIN_SEP (same extraction as Stage 3)
import matplotlib.pyplot as plt
s3 = {l.split("|")[1].strip() for l in open(f"{OUTD}/stage3_layouts.md") if l.startswith("| ep")}
rng = np.random.default_rng(20260928)


def sep(l):
    return min(np.linalg.norm(l["pos"][a] - l["pos"][b]) for a, b in (("R", "B"), ("R", "P"), ("B", "P")))


chosen = []
for o in ORDERS:
    cand = [l for l in lay if l["pool"] == "HEAD" and l["order"] == o and f"ep{l['e']:03d}" not in s3]
    clean = [l for l in cand if l["n"] == 3 and sep(l) >= MIN_SEP]
    pool_ = clean or [l for l in cand if sep(l) >= MIN_SEP] or cand
    l = pool_[int(rng.integers(len(pool_)))]
    chosen.append(dict(l, clean=bool(clean)))
trials = [(c, rep) for c in chosen for rep in (1, 2)]
trials = [trials[i] for i in rng.permutation(len(trials))]
fmt = lambda p: f"({p[0]:.2f}, {p[1]:+.2f})"
md = ["# Stage 2 ordinary-layout fixture (HEAD, prompt = recorded order, frozen 2026-09-28)", "",
      "Runtime: REL16-v2B, V4_STATE_MODE=umi94, V4_GRIPPER=binary, exec_k 16, n_action 1, dwell 1.5 s, rot180/mirror off.",
      "Base frame metres, x forward, y left, +-~5 cm.", "",
      "| layout_id | order (= prompt) | R(x,y) | B(x,y) | P(x,y) | expected first | clean (3 picks) |", "|---|---|---|---|---|---|---|"]
for c in chosen:
    md.append(f"| ep{c['e']:03d} | {c['order']} | {fmt(c['pos']['R'])} | {fmt(c['pos']['B'])} | {fmt(c['pos']['P'])} | {c['order'][0]} | {c['clean']} |")
md += ["", "Per trial record (stage2_trials.csv): reach, grasp, lift, stack_attempt, stack_success, wrong_target (first reach to a",
       "cube other than the prompt's first), execution_timeout. 2 repeats per layout, trial order shuffled (seed 20260928).",
       "Reset cubes to the table positions between trials. Stage 2 pass = the stacking chain on ordinary layouts; the headline",
       "is stack_success count and the furthest stage reached per trial. Layouts are fixed: do not swap in easier ones."]
open(f"{OUTD}/stage2_layouts.md", "w").write("\n".join(md) + "\n")
with open(f"{OUTD}/stage2_trials.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "layout_id", "repeat", "prompt", "expected_first", "reach", "grasp", "lift", "stack_attempt",
                "stack_success", "wrong_target", "execution_timeout", "notes"])
    for i, (c, rep) in enumerate(trials):
        w.writerow([i + 1, f"ep{c['e']:03d}", rep, c["order"], c["order"][0]] + [""] * 8)
fig, axs = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4), squeeze=False)
for ax, c in zip(axs.ravel(), chosen):
    for k in "RBP":
        x, y = c["pos"][k]
        ax.scatter(-y, x, s=420, c=RGB[k], marker="s", edgecolors="k", linewidths=3 if k == c["order"][0] else 0.5)
        ax.text(-y, x, k, ha="center", va="center", color="w", fontsize=12, weight="bold")
    ax.set_title(f"ep{c['e']:03d} prompt {c['order']}", fontsize=10)
    ax.set_xlim(-0.45, 0.45); ax.set_ylim(0.05, 0.55); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("robot-left <- -y (m) -> robot-right"); ax.set_ylabel("x forward (m)")
fig.suptitle("Stage 2 fixtures (thick border = prompt's first cube)"); fig.tight_layout(); fig.savefig(f"{OUTD}/stage2_layouts.png", dpi=90)
print("\n".join(md[5:5 + len(chosen) + 2]))
