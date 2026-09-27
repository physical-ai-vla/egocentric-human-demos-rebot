#!/usr/bin/env python3
"""Per-order / per-domain / per-phase breakdown of the existing C-old ego-val eval (no new inference).
Reads the raw per-(sample, arm, k) arrays written by ~/c8/c8old_mac_eval.py (<step>.npz, order = sample-major, arm-minor)
and joins each sample to its instruction, source episode, domain (main/early) and segment position in the episode.
Usage: per_order_breakdown.py <eval npz> <out json>
"""
import json, pathlib, sys
import numpy as np, pandas as pd
H = pathlib.Path.home(); C8 = H / "c8"; NPZ, OUT = sys.argv[1], sys.argv[2]
root = C8 / "c8old_data/c8old_val"
df = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index", "frame_index"]) for f in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
EP = df["episode_index"].to_numpy(); TI = df["task_index"].to_numpy()
starts = []
for e in np.unique(EP):
    ix = np.flatnonzero(EP == e); starts += list(ix[0] + np.arange(0, 31, 5))
starts = np.array(starts); assert np.array_equal(np.load(C8 / "c8old_runs/eval_cache_ego.npz")["starts"], starts)
tasks = pd.read_parquet(root / "meta/tasks.parquet"); T = {int(i): t for t, i in zip(tasks.index, tasks["task_index"])}
def code(s):   # "Stack the red cube on the bottom, blue cube in the middle, and purple ..." -> RBP (bottom->top)
    w = [x for x in s.replace(",", " ").split() if x in ("red", "blue", "purple")]; return "".join(c[0].upper() for c in w[:3])
# segment -> source episode via the per-episode row files (val segments are written in sorted-file, start order)
segs = []
for f in sorted((C8 / "c8old_rows").glob("*.npz")):
    z = np.load(f, allow_pickle=True)
    if bool(z["val"]):
        for s in range(len(z["starts"])): segs.append(dict(ep=f.stem, seg_start=int(z["starts"][s]), n_seg=len(z["starts"]), instr=str(z["instr"])))
dom = json.load(open(C8 / "c8_domain_manifest.json"))
n_ep = len(np.unique(EP)); assert n_ep == len(segs), (n_ep, len(segs))
z = np.load(NPZ); ks = (1, 4, 8, 16, 30); N = len(starts)
rows = []
for n, t in enumerate(starts):
    e = int(EP[t]); sg = segs[e]; assert code(sg["instr"]) == code(T[int(TI[t])])
    for a in (0, 1):
        r = dict(sample=n, arm="LR"[a], order=code(T[int(TI[t])]), src_ep=sg["ep"], seg_start=sg["seg_start"])
        for k in ks:
            i = 2 * n + a
            r.update({f"fk{k}": z[f"k{k}_fk"][i], f"mae{k}": z[f"k{k}_mae"][i], f"mae0_{k}": z[f"k{k}_mae0"][i], f"mt{k}": z[f"k{k}_mt"][i], f"cos{k}": z[f"k{k}_cos"][i]})
        rows.append(r)
R = pd.DataFrame(rows)
R["domain"] = R["src_ep"].map(lambda ep: "early" if dom[ep + "_left"]["early_calibration_domain"] else "main")
def summ(g):
    m = g["mt30"].astype(bool)
    return pd.Series(dict(n_samples=len(g) // 2, n_src_eps=g["src_ep"].nunique(), geo_all=np.mean([g[f"fk{k}"].median() for k in ks]),
                          fk30_p50=g["fk30"].median(), moving_frac=m.mean(), fk30_moving_p50=g.loc[m, "fk30"].median() if m.any() else np.nan,
                          cos30_moving_p50=g.loc[m, "cos30"].median() if m.any() else np.nan,
                          mae30_moving=g.loc[m, "mae30"].median() if m.any() else np.nan, mae0_30_moving=g.loc[m, "mae0_30"].median() if m.any() else np.nan))
out = dict(npz=pathlib.Path(NPZ).name, overall=summ(R).to_dict(), by_order=R.groupby("order").apply(summ).to_dict(orient="index"),
           by_domain=R.groupby("domain").apply(summ).to_dict(orient="index"), by_arm=R.groupby("arm").apply(summ).to_dict(orient="index"))
json.dump(out, open(OUT, "w"), indent=1, default=float)
pd.set_option("display.width", 200); print(R.groupby("order").apply(summ).round(2)); print(R.groupby("domain").apply(summ).round(2)); print(R.groupby("arm").apply(summ).round(2))
