"""[2026-09-28] R312-center FRONT selection (user decision): FRONT = explicit pan=center only (phase1 excluded), QA grade A,
exactly 22 episodes per order (PBR, the rarest, has exactly 22) -> FRONT132; nested half FRONT66 = 11 per order.

Deterministic, chosen before any training on it. Within each order, episodes are drawn round-robin over recording
SESSIONS (sorted by name) so no session dominates, and within a session round-robin over its contrastive SET, each
draw by a fixed RNG (seed 20260928). FRONT66 repeats the same procedure on FRONT132 (so R246 is a strict subset).
    R312 = HEAD180 + FRONT132    R246 = HEAD180 + FRONT66    R180 = HEAD180
Writes r312c_front132_v1.json / r246c_front66_v1.json in the converter's select= format (zarr_episodes = r675rbp
zarr numbering, i.e. the index into r675_rbp.json "episodes").
"""
import hashlib, json, os, re, collections
import numpy as np, pandas as pd

HERE = os.path.expanduser("~/umi_bridge/rel16_audit/r380"); OUT = os.path.expanduser("~/umi_bridge/umi76")
pool = pd.read_csv(f"{HERE}/front_pool_pan.csv")                  # zarr_ep, src_ep, pan, grade
sm = json.load(open(f"{HERE}/set_map.json")); SET, KEYS = sm["episode_sets"], sm["set_keys"]
import glob
t = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in
               glob.glob(os.path.expanduser("~/holobrain-data/lerobot/src_rebot_3stack_center675_s96/data/**/*.parquet"), recursive=True)])
ti = t.groupby("episode_index").task_index.first()
TS = {int(v): k for k, v in pd.read_parquet(os.path.expanduser("~/holobrain-data/lerobot/src_rebot_3stack_center675_s96/meta/tasks.parquet"))["task_index"].items()}
pool["order"] = ["".join(c[0].upper() for c in re.findall(r"(\w+) cube", TS[int(ti[s])])) for s in pool.src_ep]
pool["set_key"] = [KEYS[str(SET[s])] for s in pool.src_ep]
pool["session"] = pool.set_key.str.split("|").str[0]
C = pool[(pool.pan == "center") & (pool.grade == "A")].copy()
assert (C.set_key != "phase1").all()
rng = np.random.default_rng(20260928)


def pick(df, per_order):
    out = []
    for o in sorted(df.order.unique()):
        d = df[df.order == o]
        assert len(d) >= per_order, (o, len(d))
        # queue per session -> per set, each shuffled once by the fixed RNG
        sess = {}
        for s in sorted(d.session.unique()):
            sets = {k: list(rng.permutation(g.zarr_ep.values)) for k, g in sorted(d[d.session == s].groupby("set_key"))}
            sess[s] = sets
        chosen = []
        while len(chosen) < per_order:
            progressed = False
            for s in sorted(sess):
                if len(chosen) >= per_order:
                    break
                sets = sess[s]; live = [k for k in sorted(sets) if sets[k]]
                if not live:
                    continue
                k = live[int(rng.integers(len(live)))]
                chosen.append(int(sets[k].pop(0))); progressed = True
            assert progressed
        out += chosen
    return sorted(out)


f132 = pick(C, 22)
f66 = pick(C[C.zarr_ep.isin(f132)], 11)
assert set(f66) <= set(f132) and len(f132) == 132 and len(f66) == 66
rbp = json.load(open(f"{HERE}/r675_rbp.json"))["episodes"]
for name, sel, parent in (("r312c_front132_v1", f132, None), ("r246c_front66_v1", f66, "r312c_front132_v1.json")):
    S = C[C.zarr_ep.isin(sel)]
    doc = dict(schema=f"{name.split('_')[0]}/v1", created="2026-09-28",
               rule="pan=center (explicit, phase2_episodes.jsonl) only, phase1 excluded, QA-v2 grade A, exactly "
                    f"{len(sel)//6} per order, session round-robin then set round-robin, rng 20260928" + (f"; nested in {parent}" if parent else ""),
               global_transform="rot180", zarr="/home/bh-aiteam/umi_bridge/r675rbp_umi.zarr",
               zarr_episodes=[int(x) for x in sel], r675_pool_episodes=[int(rbp[x]) for x in sel],
               order_counts=S.order.value_counts().sort_index().to_dict(),
               session_counts=S.session.value_counts().to_dict(), n_sessions=int(S.session.nunique()), n_sets=int(S.set_key.nunique()),
               max_per_set=int(S.set_key.value_counts().max()), subset_of=parent)
    doc["zarr_episodes_sha256"] = hashlib.sha256(json.dumps(doc["zarr_episodes"]).encode()).hexdigest()
    json.dump(doc, open(f"{OUT}/{name}.json", "w"), indent=1)
    print(f"{name}: {len(sel)} eps  orders {doc['order_counts']}  sessions {doc['n_sessions']}  sets {doc['n_sets']}  max/set {doc['max_per_set']}")
cur = pd.read_csv(f"{HERE}/front200_pan.csv")
print(f"overlap with old FRONT200 center subset: {len(set(f132) & set(cur[cur.pan=='center'].zarr_ep))}/132")
print("pool per session available (center A):", C.session.value_counts().describe()[["count", "min", "max"]].to_dict())
