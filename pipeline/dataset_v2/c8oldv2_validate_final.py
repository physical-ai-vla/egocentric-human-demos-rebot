#!/usr/bin/env python3
"""[2026-09-28] FINAL combined C-old v2 TR-only (old259 frozen || HRL80), after c8oldv2_final_append.py; then c8oldv2_append_invariant.py I0-I5.
(derived from the HRL80 validator:) HRL80 TR part (standalone, BEFORE append) -- same checks as c8oldv2_validate.py with HRL80 expectations + G2 gripper alignment.
Usage: c8oldv2_validate_hrl80.py   (originally: Post-build checks for C-old v2)
  L  train/val leakage by ORIGINAL ego episode = 0 (split is per source episode; checked on the written datasets)
  G  the build gate passed (FK(q_master) vs recomputed T_store < 15 mm / 5 deg) and rows == gated segments
  S  B1-old semantics identical to C-old v1: same feature names / shapes / dtypes / fps as c8old_train; gripper columns obey
     state = -270 * open_fraction in [-270, 0] and action gripper = state / -6; arm joints = master pseudo-q in degrees
     (LEAD5 / chunk 30 are formed at TRAIN time by humanik_delta from these absolute rows, so identical row semantics keep them)
"""
import glob, json, pathlib, sys
from collections import Counter
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
H = pathlib.Path.home(); C8 = H / "c8"; VAR = "final"
OUT = C8 / "c8oldv2_final_data"; ROWS = C8 / "c8oldv2_final_rows"; SPL = ("c8oldv2_final_train", "c8oldv2_final_val")
SPLIT = json.load(open(H / "umi_bridge/track_c/v2k/hrl80_split_frozen.json"))
fails = []
def check(n, ok, info=""):
    print(("PASS  " if ok else "FAIL  ") + n + (f"  {info}" if info else "")); (fails.append(n) if not ok else None)
def rd(f):
    t = pq.read_table(f); cols = [(c.combine_chunks().storage if isinstance(c.type, pa.ExtensionType) else c) for c in t.columns]
    return pa.table(cols, names=t.column_names).to_pandas(ignore_metadata=True)
# L: source episode of every written LeRobot episode comes from the rows index (one rows file = one source episode)
src = {}
for f in sorted(ROWS.glob("2*.npz")):
    z = np.load(f); src[f.stem] = (bool(z["val"]), len(z["starts"]))
tr_src = {e for e, (v, _) in src.items() if not v}; va_src = {e for e, (v, _) in src.items() if v}
check("L  no source episode in both splits", not (tr_src & va_src), f"train src {len(tr_src)}  val src {len(va_src)}")
hv, ht = set(SPLIT["val_source_episodes"]), set(SPLIT["train_source_episodes"]); v1_val = set(json.load(open(C8 / "c8old_rows/index.json"))["val_episodes"])
old_tr = set(json.load(open(C8 / "c8oldv2_data/MANIFEST_tr.json"))["train_source_episodes"])
check("L  val source episodes == C-old v1 26 + hrl80_split_frozen 6 (exact)", va_src == (v1_val | hv) and len(va_src) == 32, f"{len(va_src)} val src")
check("L  train source episodes == old259 TR train 233 + HRL80 frozen-train episodes with TR segments", tr_src == (old_tr | (tr_src & ht)) and len(old_tr) == 233 and not (tr_src - old_tr - ht), f"{len(tr_src)} = 233 + {len(tr_src & ht)}")
n_eps = {s: json.load(open(OUT / s / "meta/info.json"))["total_episodes"] for s in SPL}
n_rows_tr = sum(n for e, (v, n) in src.items() if not v); n_rows_va = sum(n for e, (v, n) in src.items() if v)
check("L  written LeRobot episodes == cached segments per split", n_eps[SPL[0]] == n_rows_tr and n_eps[SPL[1]] == n_rows_va, f"{n_eps}  vs rows {n_rows_tr}/{n_rows_va}")
# G
g1 = json.load(open(C8 / "c8oldv2_tr_gate.json")); g2 = json.load(open(C8 / "c8oldv2_tr_hrl80_gate.json"))
g = dict(pass_=g1["pass_"] and g2["pass_"], pos_mm_max=max(g1["pos_mm_max"], g2["pos_mm_max"]), rot_deg_max=max(g1["rot_deg_max"], g2["rot_deg_max"]), segments=g1["segments"] + g2["segments"], parts=[g1, g2])
check("G  build gate PASS (< 15 mm / 5 deg)", g["pass_"] and g["pos_mm_max"] < 15 and g["rot_deg_max"] < 5, f"pos max {g['pos_mm_max']:.3f} mm  rot max {g['rot_deg_max']:.3f} deg")
check("G  gated segments == written segments", g["segments"] == n_rows_tr + n_rows_va, f"{g['segments']} vs {n_rows_tr + n_rows_va}")
# S: features vs v1
i1 = json.load(open(C8 / "c8old_data/c8old_train/meta/info.json")); i2 = json.load(open(OUT / SPL[0] / "meta/info.json"))
f1 = {k: (v["dtype"], tuple(v["shape"])) for k, v in i1["features"].items()}; f2 = {k: (v["dtype"], tuple(v["shape"])) for k, v in i2["features"].items()}
check("S  feature names / dtypes / shapes identical to C-old v1", f1 == f2, "" if f1 == f2 else f"diff {set(f1.items()) ^ set(f2.items())}")
check("S  fps identical to C-old v1", i1["fps"] == i2["fps"], f"{i1['fps']} vs {i2['fps']}")
df = pq.read_table(sorted(glob.glob(str(OUT / SPL[0] / "data/*/*.parquet")))[0]); df = rd(sorted(glob.glob(str(OUT / SPL[0] / "data/*/*.parquet")))[0])
st = np.stack(df["observation.state"].to_numpy()); ac = np.stack(df["action"].to_numpy())
check("S  state/action width 14", st.shape[1] == 14 and ac.shape[1] == 14, f"{st.shape} {ac.shape}")
g_ok = ((st[:, [6, 13]] >= -270 - 1e-3) & (st[:, [6, 13]] <= 1e-3)).all() and np.allclose(ac[:, [6, 13]], st[:, [6, 13]] / -6, atol=1e-4)
check("S  gripper: state in [-270, 0] and action = state / -6 (B1-old contract)", bool(g_ok))
check("S  arm joints: action == state (absolute pseudo-q, deg)", np.allclose(ac[:, [0,1,2,3,4,5,7,8,9,10,11,12]], st[:, [0,1,2,3,4,5,7,8,9,10,11,12]], atol=1e-4))
tt = np.stack(df["observation.ee.tcp_tgt"].to_numpy())
check("S  tcp_tgt finite, 32 wide (2 x 4x4)", tt.shape[1] == 32 and np.isfinite(tt).all(), str(tt.shape))
# ---- R: row counts exact (every LeRobot episode = one 65-frame segment) and 31 valid chunk starts per segment (65 - (5+30-1))
def ep_lengths(split):
    L = []
    for f in sorted(glob.glob(str(OUT / split / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["episode_index"]).to_pandas(); L += t.groupby("episode_index").size().tolist()
    return L
lt, lv = ep_lengths(SPL[0]), ep_lengths(SPL[1])
check("R  every written episode has exactly 65 frames", set(lt) == {65} and set(lv) == {65}, f"train {len(lt)} eps, val {len(lv)} eps")
check("R  totals train 65 x segs / val 65 x segs", sum(lt) == 65 * n_rows_tr and sum(lv) == 65 * n_rows_va, f"{sum(lt)} / {sum(lv)} frames")
print(f"      chunk starts (LEAD 5, chunk 30): train {31 * len(lt)}  val {31 * len(lv)}")
# ---- Q: B1-old training target semantics on real rows: dq[i] = q_cmd[t+5+i] - q_state[t] (30 x 12, rad), grip = action grip
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
z0 = np.load(sorted(ROWS.glob("2*.npz"))[0]); stq, acq = z0["st"][0], z0["ac"][0]
dq = np.stack([np.radians(acq[5 + i, ARM] - stq[0, ARM]) for i in range(30)])
check("Q  Delta-q_cmd[k] = q_cmd[t+5+k] - q_state[t] forms a (30, 12) finite target", dq.shape == (30, 12) and np.isfinite(dq).all(), f"|dq| max {np.abs(dq).max():.3f} rad")
check("Q  valid chunk starts per 65-frame segment: t + 5 + 29 <= 64  ->  t in 0..30  = 31", len([t for t in range(65) if t + 5 + 29 <= 64]) == 31)
# written LeRobot rows vs cached rows: same dq target for random (segment, t)
full = pd_all = None
import pandas as pd
pd_all = pd.concat([rd(f) for f in sorted(glob.glob(str(OUT / SPL[0] / "data/*/*.parquet")))]).sort_values("index").reset_index(drop=True)
tr_files = [f for f in sorted(ROWS.glob("2*.npz")) if not bool(np.load(f)["val"])]; offsets = []; base = 0
for f in tr_files:
    n = len(np.load(f)["starts"]); offsets.append((f, base, n)); base += n
rng_q = np.random.default_rng(1); worst_q = 0.0
for _ in range(20):
    f, b0, n = offsets[rng_q.integers(len(offsets))]; k = int(rng_q.integers(n)); t = int(rng_q.integers(31)); z = np.load(f)
    ep = pd_all[pd_all.episode_index == b0 + k].sort_values("frame_index")
    S_w = np.stack(ep["observation.state"].to_numpy()); A_w = np.stack(ep["action"].to_numpy())
    dq_rows = np.radians(z["ac"][k][t + 5:t + 35][:, ARM] - z["st"][k][t, ARM]); dq_w = np.radians(A_w[t + 5:t + 35][:, ARM] - S_w[t, ARM])
    worst_q = max(worst_q, float(np.abs(dq_rows - dq_w).max()))
check("Q  written-parquet dq == cached-rows dq on 20 random (segment, t) (train split)", worst_q < 1e-5, f"max |diff| {worst_q:.2e} rad")
# ---- C: C/D recomputation parity + rotation tail (FK(q) from the written rows vs the recomputed T_store)
import os, sys, torch
os.environ.setdefault("REBOT_URDF", str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee/reBot_B601_DM_dualarm.urdf")); sys.path.insert(0, str(H / "holobrain-mac-model")); sys.path.insert(0, str(C8 / "c8old"))   # same rebot_fk as the builder (md5 766c59f4, all copies identical)
import rebot_fk_torch
FK = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float32)
rot, pos, where = [], [], []
for f in sorted(ROWS.glob("2*.npz")):
    z = np.load(f)
    for k in range(len(z["starts"])):
        q = torch.tensor(np.radians(z["st"][k][:, ARM]), dtype=torch.float32); Tf = FK.tcp(q).numpy(); Ts = z["tcp_tgt"][k].reshape(65, 2, 4, 4)
        dp = np.linalg.norm(Tf[..., :3, 3] - Ts[..., :3, 3], axis=-1) * 1e3
        dr = np.degrees(np.arccos(np.clip((np.einsum("...ij,...ij->...", Tf[..., :3, :3], Ts[..., :3, :3]) - 1) / 2, -1, 1)))
        rot.append(dr); pos.append(dp); where.append((f.stem, int(z["starts"][k])))
R = np.concatenate([r.ravel() for r in rot]); Pm = np.concatenate([p.ravel() for p in pos])
check("C  FK(q_written) vs recomputed T_store within 15 mm / 5 deg on every row", Pm.max() < 15 and R.max() < 5,
      f"pos p95 {np.percentile(Pm,95):.3f} p99 {np.percentile(Pm,99):.3f} max {Pm.max():.3f} mm | rot p95 {np.percentile(R,95):.3f} p99 {np.percentile(R,99):.3f} max {R.max():.3f} deg")
seg_max = np.array([r.max() for r in rot]); arm_max = np.array([r.max(0) for r in rot])     # (segments, 2)
hot = [(where[i], round(float(seg_max[i]), 3), "L" if arm_max[i, 0] >= arm_max[i, 1] else "R") for i in np.argsort(-seg_max)[:8]]
n48 = int((seg_max >= 4.8).sum()); n45 = int((seg_max >= 4.5).sum()); rows48 = int((R >= 4.8).sum()); n49 = int((seg_max >= 4.9).sum())
frame_of_max = np.array([int(np.unravel_index(np.argmax(r), r.shape)[0]) for r in rot])
arm_of = ["L" if a[0] >= a[1] else "R" for a in arm_max]
print(f"      rotation tail: segments with max >= 4.9 deg: {n49}, >= 4.8 deg: {n48} ({rows48} rows of {len(R)}), >= 4.5 deg: {n45}; worst: {hot}")
sel = seg_max >= 4.5
print(f"      >= 4.5 deg segments by arm {dict(Counter(np.array(arm_of)[sel].tolist()))}; frame-of-max histogram (0-64, bins of 13) {np.histogram(frame_of_max[sel], bins=[0,13,26,39,52,65])[0].tolist()}")
from collections import Counter
print(f"      >= 4.5 deg segments by source episode: {dict(Counter(w[0] for w, m in zip(where, seg_max) if m >= 4.5).most_common(6))}")
json.dump(dict(rot_deg=dict(p50=float(np.median(R)), p95=float(np.percentile(R, 95)), p99=float(np.percentile(R, 99)), max=float(R.max())),
               pos_mm=dict(p50=float(np.median(Pm)), p95=float(np.percentile(Pm, 95)), p99=float(np.percentile(Pm, 99)), max=float(Pm.max())),
               segments_rot_ge_4p9=n49, segments_rot_ge_4p8=n48, segments_rot_ge_4p5=n45, worst=hot,
               ge_4p5_by_arm=dict(Counter(np.array(arm_of)[sel].tolist())), ge_4p5_frame_of_max_hist=np.histogram(frame_of_max[sel], bins=[0,13,26,39,52,65])[0].tolist()), open(C8 / f"c8oldv2_{VAR}_gate_distribution.json", "w"), indent=1)
# ---- G2: gripper = mcap /<side>/gripper normalized, nearest sample per wrist video frame (extract_grip.align)
sys.path.insert(0, str(H / "umi_bridge")); from extract_grip import align as _align; from mcap.reader import make_reader as _mr
dts, miss, oor, nfr = [], 0, 0, 0
for f in sorted(ROWS.glob("2*.npz")):
    z = np.load(f); epd = pathlib.Path(str(z["epd"]))
    if "/HRL80/" not in str(epd): continue          # old259 gripper = ~/c8/grip (validated in MANIFEST_tr); G2 covers the HRL80 source
    for side, vk, col in (("left", "vfL", 6), ("right", "vfR", 13)):
        fr, gp, gnone = [], [], 0
        with open(epd / "sensors.mcap", "rb") as fh:
            for _, ch, msg in _mr(fh).iter_messages(topics=[f"/{side}_wrist/frame_meta", f"/{side}/gripper"]):
                d = json.loads(msg.data)
                if ch.topic.endswith("frame_meta"): fr.append((int(d["video_frame"]), int(d["capture_ns"])))
                elif d.get("normalized") is None: gnone += 1
                else: gp.append((int(d["sample_ns"]), float(d["normalized"]), 0, 0.0)); oor += int(not (0.0 <= float(d["normalized"]) <= 1.0))
        al = _align(fr, gp); lut = dict(zip(al["video_frame"].tolist(), al["dt_ms"].tolist())); miss += gnone
        for k in range(len(z["starts"])):
            vf = z[vk][k]; dts += [lut[int(v)] for v in vf]; nfr += len(vf)
            of = z["st"][k][:, col] / -270.0; oor += int(((of < -1e-6) | (of > 1 + 1e-6)).sum())
dts = np.array(dts); GRIP = dict(frames=int(nfr), dt_ms_p50=round(float(np.median(dts)), 2), dt_ms_p95=round(float(np.percentile(dts, 95)), 2), dt_ms_max=round(float(dts.max()), 2),
                                 frames_dt_gt_20ms=int((dts > 20).sum()), gripper_samples_normalized_missing=int(miss), range_violations=int(oor))
check("G2 gripper alignment: nearest-sample |dt| <= 20 ms on every row, no range violation", GRIP["frames_dt_gt_20ms"] == 0 and GRIP["range_violations"] == 0, str(GRIP))
# ---- V: frame-exact video parity, source mp4 frame (vf index) vs written LeRobot frame; first / middle / last + random
import av, cv2, torch as _t
# (no lerobot-seeed path: use the SAME lerobot the write stage used, whose pyav decode works)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dsv = LeRobotDataset(f"local/{SPL[1]}", root=str(OUT / SPL[1]), video_backend="pyav")
cams = [("observation.images.global", "head.mp4", "vfH"), ("observation.images.left_wrist", "left_wrist.mp4", "vfL"), ("observation.images.right_wrist", "right_wrist.mp4", "vfR")]
def src_frames(path, idxs):
    out = {}; want = set(idxs)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want: out[i] = cv2.resize(fr.to_ndarray(format="rgb24"), (640, 480), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            if i > max(want): break
    return out
val_files = [f for f in sorted(ROWS.glob("2*.npz")) if bool(np.load(f)["val"])][:3]; ep_i = 0; offs = []; vrec = []
for f in val_files:
    z = np.load(f); epd = pathlib.Path(str(z["epd"]))
    for k in range(min(2, len(z["starts"]))):
        e0 = dsv.meta.episodes[ep_i + k]["dataset_from_index"]
        for fi in (0, 32, 64):
            for key, mp4, vk in cams:
                vf = int(z[vk][k][fi]); src = src_frames(epd / mp4, range(max(0, vf - 2), vf + 3))
                got = dsv[int(e0 + fi)][key].permute(1, 2, 0).numpy()
                dd = {o: float(np.abs(got - src[vf + o]).mean()) for o in range(-2, 3) if (vf + o) in src}
                offs.append(min(dd, key=dd.get)); vrec.append(dict(src=f.stem, seg=k, frame=fi, cam=key, diff_by_offset=dd))
    ep_i += len(z["starts"])                                  # next source file starts after ALL of this file's segments
# + 4 random TRAIN segments (index via the train offsets table), frames 0 / 32 / 64
dst_ = LeRobotDataset(f"local/{SPL[0]}", root=str(OUT / SPL[0]), video_backend="pyav"); rr = np.random.default_rng(2)
for _ in range(4):
    f, b0, n = offsets[rr.integers(len(offsets))]; k = int(rr.integers(n)); z = np.load(f); epd = pathlib.Path(str(z["epd"]))
    e0 = dst_.meta.episodes[b0 + k]["dataset_from_index"]
    for fi in (0, 32, 64):
        for key, mp4, vk in cams:
            vf = int(z[vk][k][fi]); src = src_frames(epd / mp4, range(max(0, vf - 2), vf + 3)); got = dst_[int(e0 + fi)][key].permute(1, 2, 0).numpy()
            dd = {o: float(np.abs(got - src[vf + o]).mean()) for o in range(-2, 3) if (vf + o) in src}
            offs.append(min(dd, key=dd.get)); vrec.append(dict(src=f.stem, seg=k, frame=fi, cam=key, diff_by_offset=dd))
check("V  best-matching source frame is EXACTLY the expected one (argmin offset == 0) on every check (6 val + 4 train segs x first/mid/last x 3 cams)",
      all(o == 0 for o in offs), f"{len(offs)} checks, offsets {dict(Counter(offs))}; expected-frame mean|d| p50 {np.median([r['diff_by_offset'][0] for r in vrec]):.4f} vs nearest other offset p50 {np.median([min(v for o, v in r['diff_by_offset'].items() if o != 0) for r in vrec]):.4f}")
json.dump(vrec, open(C8 / f"c8oldv2_{VAR}_video_parity.json", "w"), indent=1)
# ---- M0: mapping truth -- every written episode's 65 state rows == its rows-cache segment (order of shards / segments)
pd_val = pd.concat([rd(f) for f in sorted(glob.glob(str(OUT / SPL[1] / "data/*/*.parquet")))]).sort_values("index").reset_index(drop=True)
bad_map = 0; nmap = 0
for spl, dfp in ((SPL[0], pd_all), (SPL[1], pd_val)):
    fl = [f for f in sorted(ROWS.glob("2*.npz")) if bool(np.load(f)["val"]) == (spl == SPL[1])]; idx = 0
    St = {e: np.stack(g.sort_values("frame_index")["observation.state"].to_numpy()) for e, g in dfp.groupby("episode_index")}
    for f in fl:
        z = np.load(f)
        for k in range(len(z["starts"])):
            nmap += 1; bad_map += int(not np.array_equal(St[idx], z["st"][k])); idx += 1
check("M0 source_episode -> segment -> LeRobot episode mapping exact (all 65 state rows bit-identical)", bad_map == 0, f"{nmap} episodes, {bad_map} mismatches")
# ---- M: manifest freeze
import hashlib
def hdir(d): hh = hashlib.sha256(); [hh.update(p.read_bytes()) for p in sorted(pathlib.Path(d).rglob("*")) if p.is_file() and p.suffix in (".parquet", ".json", ".jsonl")]; return hh.hexdigest()
man = dict(schema="c_old_v2_dataset_manifest/v1", variant=VAR, source_master=str(C8 / "robotlike/master_combined/MASTER.json"),
           source_master_sha256=hashlib.sha256((C8 / "robotlike/master_combined/MASTER.json").read_bytes()).hexdigest(),
           builder_sha256=dict(old259=hashlib.sha256((C8 / "c8oldv2_build_dataset.py").read_bytes()).hexdigest(), hrl80=hashlib.sha256((C8 / "c8oldv2_build_hrl80.py").read_bytes()).hexdigest(),
                               append=hashlib.sha256((C8 / "c8oldv2_final_append.py").read_bytes()).hexdigest()), grip_alignment_hrl80=GRIP,
           write_stage_sha256=hashlib.sha256((C8 / "write_stage.py").read_bytes()).hexdigest(),
           segments=[f"{e}@{s}" for e, s in where], train_source_episodes=sorted(tr_src), val_source_episodes=sorted(va_src),
           mapping=[dict(source_episode=f.stem, segment_id=f"{f.stem}@{int(st)}", source_start_frame=int(st), split=spl, lerobot_episode_index=idx)
                    for spl, fl in ((SPL[0], [f for f in sorted(ROWS.glob("2*.npz")) if not bool(np.load(f)["val"])]), (SPL[1], [f for f in sorted(ROWS.glob("2*.npz")) if bool(np.load(f)["val"])]))
                    for idx, (f, st) in enumerate([(f, st) for f in fl for st in np.load(f)["starts"]])],
           counts=dict(train_segments=n_rows_tr, val_segments=n_rows_va, train_chunk_starts=31 * n_rows_tr, val_chunk_starts=31 * n_rows_va),
           dataset_meta_sha256={s: hdir(OUT / s) for s in SPL}, gate=g, gate_distribution=f"c8oldv2_{VAR}_gate_distribution.json",
           contracts=["hybrid_frozen.json", "v2tr_frozen.json", "v2kr_frozen.json", "gripper: B1-old (state -270*open_fraction, action /-6)"],
           checks_failed=list(fails))
(OUT / f"MANIFEST_{VAR}.json").write_text(json.dumps(man, indent=1)); print(f"      manifest -> {OUT / f'MANIFEST_{VAR}.json'}")
print("\nALL C-OLD V2 CHECKS PASS" if not fails else f"\nFAILED: {fails}")
