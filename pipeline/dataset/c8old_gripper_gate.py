"""[2026-09-26] C-old GRIPPER CONTRACT GATE (before the 300k launch). Source of truth = R150 B1 data + the B1 training preprocessing
(humanik_delta.py HUMANIK_ROBOT=1: state closed <= -135, cmd closed >= 27). Checks which ego mapping gives the SAME physical semantics:
  A) raw = -270 * open_fraction      B) raw = -270 * (1 - open_fraction)       (open_fraction: c8_grip, 0 = closed anchor, 1 = open anchor)
PHYSICAL TRUTH (from R150 wrist video, strips in gripper_gate/strip_*.png, both arms, ep 10/20/81/100): raw 0 = fingers
touching (rest, CLOSED); raw -270 = fully OPEN; cube held at -85..-151; cmd rises >= 27 while the fingers OPEN. So the B1 binary
label (state <= -135 / cmd >= 27 -> 1) means WIDE OPEN, not "closed" as the humanik_delta comment says (the label is only a label;
B1 is self-consistent). Required: ego f = 1 (open) -> B1 label 1 for state AND action; ego f = 0 (closed) -> label 0.
[2026-09-26 v1 of this gate assumed label 1 = closed from that comment and wrongly passed mapping B; corrected.]
v3 (2026-09-26): explicit anchor points f = 0 / 0.5 / 1 -> (0, 0) / (-135, 22.5) / (-270, 45) asserted exactly; ego anchor direction
per session side (c8_grip anchors + session_meta ticks_open / ticks_closed) logged and sign-checked; the WRITTEN dataset (final
aggregated splits if present, else all complete shards) compared bitwise with the cached rows; gripper distribution (open-label
fraction + p10/p50/p90 of state and action, train / val x left / right) against R150 recorded for provenance.
Env: ~/xvla-mac/bin/python. Writes ~/c8/c8old_runs/gripper_gate/{gate.json, *.png}."""
import json, pathlib, sys
import numpy as np, pyarrow.parquet as pq, av, cv2
H = pathlib.Path.home(); C8 = H / "c8"; R = C8 / "r150_ds"; OUT = C8 / "c8old_runs/gripper_gate"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(C8 / "c8old")); import humanik_delta as HD
SC, CC = HD.G_STATE_CLOSED, HD.G_CMD_CLOSED; GR = [6, 13]
t = pq.read_table(sorted((R / "data").rglob("*.parquet"))).to_pandas()
st = np.stack(t["observation.state"]); ac = np.stack(t["action"]); ep = t["episode_index"].to_numpy(); fi = t["frame_index"].to_numpy()
res = dict(b1_thresholds=dict(state_closed_le=SC, cmd_closed_ge=CC)); ok = True
for side, g in (("left", 6), ("right", 13)):
    s, c = st[:, g], ac[:, g]; start = fi == 0
    closed_cmd = c >= CC; open_cmd = c < CC
    r = dict(episode_start_state_p50=float(np.median(s[start])), episode_start_cmd_p50=float(np.median(c[start])),
             state_range=[float(s.min()), float(s.max())], cmd_range=[float(c.min()), float(c.max())],
             state_p50_when_cmd_closed=float(np.median(s[closed_cmd])), state_p50_when_cmd_open=float(np.median(s[open_cmd])),
             corr_cmd_state=float(np.corrcoef(c, s)[0, 1]),
             agree_binarized_state_vs_cmd=float(((s <= SC) == (c >= CC)).mean()),
             frac_cmd_closed=float(closed_cmd.mean()))
    # rest (episode start, video: fingers touching = CLOSED) must be label 0; opening command (cmd >= 27) drives state to label 1
    r["rest_is_label0_closed"] = bool(r["episode_start_state_p50"] > SC and r["episode_start_cmd_p50"] < CC)
    r["open_cmd_state_label1"] = bool(r["state_p50_when_cmd_closed"] <= SC)
    ok &= r["rest_is_label0_closed"] and r["open_cmd_state_label1"]; res[f"r150_{side}"] = r
    print(side, json.dumps(r))
# ---- synthetic ego values through each mapping + the B1 binarization (label 1 = OPEN, video-verified)
syn = {}
for name, fmap in (("A_-270*f", lambda f: -270 * f), ("B_-270*(1-f)", lambda f: -270 * (1 - f))):
    rows = []
    for f in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        raw = fmap(f); act = raw / -6
        rows.append(dict(open_fraction=f, state_raw=raw, action=act, b1_state_label=int(raw <= SC), b1_action_label=int(act >= CC)))
    syn[name] = rows
def sem(rows): return rows[-1]["b1_state_label"] == 1 and rows[-1]["b1_action_label"] == 1 and rows[0]["b1_state_label"] == 0 and rows[0]["b1_action_label"] == 0
res["synthetic"] = syn; res["A_matches_b1"] = sem(syn["A_-270*f"]); res["B_matches_b1"] = sem(syn["B_-270*(1-f)"])
print("anchor points (mapping A)   open_fraction  state_raw  action   B1 state/action label (1 = open)")
anchor_ok = True
for f, want_s, want_a in ((0.0, 0.0, 0.0), (0.5, -135.0, 22.5), (1.0, -270.0, 45.0)):
    raw = np.float32(-270.0) * np.float32(f); act = raw / np.float32(-6.0)
    good = abs(float(raw) - want_s) < 1e-6 and abs(float(act) - want_a) < 1e-6; anchor_ok &= good
    print(f"   {f:>24.1f}  {float(raw):>9.1f}  {float(act):>6.1f}   {int(raw <= SC)}/{int(act >= CC)}   {'OK' if good else 'MISMATCH'}")
res["anchor_points_A_exact"] = bool(anchor_ok)

# ---- ego anchor direction: f = (raw - closed) / (open - closed) -> f 0 at the closed anchor, 1 at the open anchor
census = json.load(open(C8 / "grip/grip_census.json"))["anchors"]; RAWD = H / "ego_collector/datasets/human_handumi_raw/Hpilot"
sessions = sorted({f.stem.rsplit("_", 1)[0] for f in (C8 / "c8old_rows").glob("*.npz")}); anc = {}; dir_ok = True
print("ego anchors  session_side  closed_raw  open_raw  direction  source  ticks_closed  ticks_open  f(closed) f(open)  state_raw(closed/open)")
for sess in sessions:
    for side in ("left", "right"):
        a = census[f"{sess}_{side}"]; cl, op, dr = a["closed"], a["open"], a["direction"]
        try: m = json.load(open(RAWD / f"Hpilot_{sess}" / "session_meta.json"))["devices"][f"gripper_{side}"]; tc, to = m.get("ticks_closed"), m.get("ticks_open")
        except Exception: tc = to = None
        f_cl = (cl - cl) / (op - cl); f_op = (op - cl) / (op - cl)
        good = np.sign(op - cl) == dr and f_cl == 0.0 and f_op == 1.0 and (tc is None or np.sign(to - tc) == dr)
        dir_ok &= bool(good); anc[f"{sess}_{side}"] = dict(closed=cl, open=op, direction=dr, source=a["source"], ticks_closed=tc, ticks_open=to, ok=bool(good))
        print(f"   {sess}_{side:<5} {cl:>10.1f} {op:>9.1f} {dr:>+4d}  {a['source']:<15} {str(tc):>6} {str(to):>6}   {f_cl:.0f} {f_op:.0f}   {-270 * f_cl:.0f}/{-270 * f_op:.0f}  {'OK' if good else 'SIGN MISMATCH'}")
res["ego_anchors"] = anc; res["ego_anchor_direction_ok"] = bool(dir_ok)

# ---- written dataset vs cached rows, bitwise (gripper dims + the full state / action for completeness)
def stats(sg, ag):
    return dict(n=int(len(sg)), open_label_state=float((sg <= SC).mean()), open_label_action=float((ag >= CC).mean()),
                state_p10_p50_p90=[round(float(x), 2) for x in np.percentile(sg, [10, 50, 90])], action_p10_p50_p90=[round(float(x), 3) for x in np.percentile(ag, [10, 50, 90])])
dist = {}; bit = {}
for split in ("c8old_train", "c8old_val"):
    want_val = split == "c8old_val"; npz = [f for f in sorted((C8 / "c8old_rows").glob("*.npz")) if bool(np.load(f)["val"]) == want_val]
    ref_s = np.concatenate([np.load(f)["st"].reshape(-1, 14) for f in npz]); ref_a = np.concatenate([np.load(f)["ac"].reshape(-1, 14) for f in npz])
    final = C8 / "c8old_data" / split
    if final.exists() and (C8 / "c8old_data" / f"{split}.complete").exists():
        src = "final"; d = pq.read_table(sorted((final / "data").rglob("*.parquet"))).to_pandas().sort_values("index")
    else:
        done = [f for f in npz if (C8 / "c8old_shards" / split / f"{f.stem}.complete").exists()]; src = f"shards {len(done)}/{len(npz)}"
        if not done: bit[split] = dict(source=src, ok=False); continue
        d = __import__("pandas").concat([pq.read_table(sorted((C8 / "c8old_shards" / split / f.stem / "data").rglob("*.parquet"))).to_pandas() for f in done])
        ref_s = np.concatenate([np.load(f)["st"].reshape(-1, 14) for f in done]); ref_a = np.concatenate([np.load(f)["ac"].reshape(-1, 14) for f in done])
    ws = np.stack(d["observation.state"]); wa = np.stack(d["action"])
    same = ws.shape == ref_s.shape and np.array_equal(ws[:, GR], ref_s[:, GR]) and np.array_equal(wa[:, GR], ref_a[:, GR])
    bit[split] = dict(source=src, rows=int(len(ws)), gripper_bitwise_equal=bool(same), full_state_action_bitwise_equal=bool(ws.shape == ref_s.shape and np.array_equal(ws, ref_s) and np.array_equal(wa, ref_a)))
    for k, gi in (("left", 6), ("right", 13)): dist[f"ego_{split}_{k}"] = stats(ws[:, gi], wa[:, gi])
    dist[f"ego_{split}_both"] = stats(ws[:, GR].ravel(), wa[:, GR].ravel())
    print(split, bit[split])
for k, gi in (("left", 6), ("right", 13)): dist[f"r150_{k}"] = stats(st[:, gi], ac[:, gi])
dist["r150_both"] = stats(st[:, GR].ravel(), ac[:, GR].ravel())
print("gripper distribution (open label = B1 label 1 = physically open)")
for k, v in dist.items():
    if "n" in v: print(f"   {k:<24} n {v['n']:>7}  open(state) {v['open_label_state']:.3f}  open(action) {v['open_label_action']:.3f}  state p10/50/90 {v['state_p10_p50_p90']}  action {v['action_p10_p50_p90']}")
# open/closed label crossings inside one 65-frame training window (ego segment vs R150 windows, stride 20), side level
def crossings(x): w = x <= SC; return (w[:, :-1] != w[:, 1:]).any(1)
eg = np.concatenate([crossings(np.load(f)["st"][..., g]) for f in sorted((C8 / "c8old_rows").glob("*.npz")) for g in GR])
rw = np.concatenate([crossings(np.stack([st[ep == e][i:i + 65, g] for i in range(0, (ep == e).sum() - 65, 20)])) for e in np.unique(ep) for g in GR])
dist["window_label_crossing_frac"] = dict(ego_segments=float(eg.mean()), ego_n=int(len(eg)), r150_windows=float(rw.mean()), r150_n=int(len(rw)))
print("   65-frame windows with an open/closed crossing: ego", round(float(eg.mean()), 3), "| R150", round(float(rw.mean()), 3))
res["written_vs_rows"] = bit; res["distribution"] = dist
bit_ok = all(b.get("gripper_bitwise_equal") for b in bit.values()) and "c8old_train" in bit
res["GATE"] = "PASS" if ok and res["A_matches_b1"] and not res["B_matches_b1"] and anchor_ok and dir_ok and bit_ok else "FAIL"
res["physical_evidence"] = sorted(str(p.name) for p in OUT.glob("strip_*.png") if not p.stem.endswith("_s"))
json.dump(res, open(OUT / "gate.json", "w"), indent=1); print("GRIPPER GATE", res["GATE"])
