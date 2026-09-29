#!/usr/bin/env python3
"""[2026-09-28] C-old v2 TR-only rows for the HRL80 part (append to the frozen old259 snapshot). Same row semantics as
c8oldv2_build_dataset.py (which stays frozen for old259); only the HRL80-specific INPUTS differ:
  source master   ~/c8/robotlike/hybrid_hrl80 (hybrid_merge on tr_hrl80 / r30_hrl80, frozen rules), segments with method TR
  segmentation    RLC.side_nogrip + the |dt| < 20 ms sync -- exactly the path v2k_retarget used for HRL80 (runs ~/c8/robotlike/runs,
                  export ~/c8/robotlike/export); every master segment start must be reproduced
  gripper         sensors.mcap /<side>/gripper 'normalized' (collector calibration: 0 closed .. 1 open) = open_fraction, looked up per
                  wrist VIDEO FRAME by extract_grip.align (nearest sample); B1-old contract state = -270 * open_fraction, action / -6
  split           hrl80_split_frozen.json (val = ep 49-54)
Unchanged: pseudo q = master "<start>_v2_0"; TR anchor = q_ref(0) of R30 window tr_window_id (build_windows on R30, stride 5);
T_store = FK(anchor) rel(t) @ diag(C^T, 1); gate FK(q) vs T_store < 15 mm / 5 deg on every row; instruction = episode_meta.
Usage: c8oldv2_build_hrl80.py gate|rows   (env ~/xvla-mac/bin/python)"""
import importlib.util, json, os, pathlib, sys
import numpy as np, torch
H = pathlib.Path.home(); C8 = H / "c8"; V2 = H / "umi_bridge/track_c/v2k"; RL = C8 / "robotlike"; MASTER = RL / "hybrid_hrl80"
RAWS = H / "ego_collector/datasets/human_handumi_raw/HRL80"; MODE = sys.argv[1]; assert MODE in ("gate", "rows")
ROWS = C8 / "c8oldv2_tr_hrl80_rows"
sys.path.insert(0, str(C8)); sys.path.insert(0, str(C8 / "c8old")); sys.path.insert(0, str(H / "umi_bridge"))
os.environ.setdefault("REBOT_URDF", str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee/reBot_B601_DM_dualarm.urdf"))
spec = importlib.util.spec_from_file_location("v2k", V2 / "v2k_retarget.py"); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
RLC, P3 = V.RLC, V.P3
P3.RUNS = RL / "runs"; _roots = [RL / "export"]
P3.export_dir = lambda tag: next((r / tag for r in _roots if (r / tag / "imu_data.json").exists()), _roots[0] / tag)
P3.init(); P = P3.P; V.P = P
from extract_grip import align
from mcap.reader import make_reader
import rebot_fk_torch
FK = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float32)
CT4 = np.eye(4); CT4[:3, :3] = P.C[:3, :3].T
d150 = np.load(H / "umi_bridge/track_c/data/r150_q_tcp.npz"); r30 = json.load(open(V2 / "r30_episodes.json"))["episodes"]
V.build_windows(d150, np.isin(d150["ep"], r30))
split = json.load(open(V2 / "hrl80_split_frozen.json")); VAL = set(split["val_source_episodes"])

segs = []
for p in sorted(MASTER.glob("2*.json")):
    r = json.load(open(p))
    for s in r.get("segments", []):
        if isinstance(s.get("v2"), dict) and s["v2"].get("retarget_method") == "TR": segs.append((r["episode"], s))
by_ep = {}
for e, s in segs: by_ep.setdefault(e, []).append(s)
print(f"[hrl80 TR] segments {len(segs)} from {len(by_ep)} source episodes; val episodes {len(VAL & set(by_ep))}/6 with {sum(e in VAL for e, _ in segs)} segments")


def grip_open(epd, side, vframes):
    fr, gp = [], []
    with open(epd / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=[f"/{side}_wrist/frame_meta", f"/{side}/gripper"]):
            d = json.loads(msg.data)
            if ch.topic.endswith("frame_meta"): fr.append((int(d["video_frame"]), int(d["capture_ns"])))
            elif d.get("normalized") is not None: gp.append((int(d["sample_ns"]), float(d["normalized"]), int(d.get("raw_position", 0)), 0.0))
    al = align(fr, gp); lut = dict(zip(al["video_frame"].tolist(), range(len(al["video_frame"]))))
    ix = np.array([lut[int(v)] for v in vframes])
    return np.clip(al["grip01"][ix], 0, 1).astype(np.float64), al["dt_ms"][ix]


def episode_rows(e, seglist):
    sess, num = e.rsplit("_", 1); epd = RAWS / f"HRL80_{sess}" / f"episode_{num}"
    segs_all, why = V.segments(epd); assert why is None, f"{e}: {why}"
    starts_all = {st for st, TL, _ in segs_all if TL is not None}
    tags = {s: RLC.tag_of(epd, s) for s in ("left", "right")}; S = {s: RLC.side_nogrip(tags[s], epd, s) for s in ("left", "right")}
    L, R = S["left"], S["right"]; fm = P3.frame_meta(epd)
    capL = np.array([c for _, c in fm["left_wrist"][L["off"]:L["off"] + L["n"]]], float); capR = np.array([c for _, c in fm["right_wrist"][R["off"]:R["off"] + R["n"]]], float)
    capH = np.array([c for _, c in fm["head"]], float); jR, jH = P3.nearest(capR, capL), P3.nearest(capH, capL)
    vfL = np.array([v for v, _ in fm["left_wrist"][L["off"]:L["off"] + L["n"]]]); vfR = np.array([v for v, _ in fm["right_wrist"][R["off"]:R["off"] + R["n"]]])
    vfH = np.array([v for v, _ in fm["head"]]); zq = np.load(MASTER / f"{e}.npz"); out = []
    for seg in seglist:
        start = seg["start"]; assert start in starts_all, f"{e}@{start}: master segment not reproduced by the retarget segmentation"
        s_ = np.arange(start, start + 65); q = zq[f"{start}_v2_0"].astype(np.float64)
        relL, relR = V.rel_traj(L["T_tcp"][s_]), V.rel_traj(R["T_tcp"][jR[s_]])
        T0 = V.fk_T(V.WIN["q"][seg["v2"]["tr_window_id"]][0]); tgL, tgR = T0[0][None] @ relL, T0[1][None] @ relR
        Ts = np.stack([tgL @ CT4, tgR @ CT4], 1); Tf = FK.tcp(torch.tensor(q, dtype=torch.float32)).numpy()
        dp = np.linalg.norm(Tf[..., :3, 3] - Ts[..., :3, 3], axis=-1) * 1e3
        dr = np.degrees(np.arccos(np.clip((np.einsum("...ij,...ij->...", Tf[..., :3, :3], Ts[..., :3, :3]) - 1) / 2, -1, 1)))
        fL, dtL = grip_open(epd, "left", vfL[s_]); fR, dtR = grip_open(epd, "right", vfR[jR[s_]]); qd = np.degrees(q)
        st = np.concatenate([qd[:, :6], -270 * fL[:, None], qd[:, 6:], -270 * fR[:, None]], 1).astype(np.float32)
        ac = st.copy(); ac[:, 6] /= -6; ac[:, 13] /= -6
        out.append(dict(start=start, st=st, ac=ac, tcp_tgt=Ts.reshape(65, 32).astype(np.float32), vfL=vfL[s_], vfR=vfR[jR[s_]], vfH=vfH[jH[s_]],
                        pos_mm_max=float(dp.max()), pos_mm_p50=float(np.median(dp)), rot_deg_max=float(dr.max()), rot_deg_p50=float(np.median(dr)),
                        grip_dt_ms_max=float(max(dtL.max(), dtR.max())), grip_nan=int(np.isnan(fL).sum() + np.isnan(fR).sum())))
    return epd, out


gate_f = C8 / "c8oldv2_tr_hrl80_gate.json"
if MODE == "gate":
    st = []
    for i, (e, sl) in enumerate(sorted(by_ep.items())):
        _, rows = episode_rows(e, sl); st += [(r["pos_mm_max"], r["pos_mm_p50"], r["rot_deg_max"], r["rot_deg_p50"], r["grip_dt_ms_max"], r["grip_nan"]) for r in rows]
        if i % 10 == 0: print(f"  gate {i}/{len(by_ep)}", flush=True)
    a = np.array(st); ok = (a[:, 0] < 15).all() and (a[:, 2] < 5).all() and (a[:, 5] == 0).all()
    print(f"HARD GATE [hrl80 TR] {len(a)} segments: pos max {a[:, 0].max():.3f} mm (p50 {np.median(a[:, 1]):.4f})  rot max {a[:, 2].max():.3f} deg (p50 {np.median(a[:, 3]):.4f})"
          f"  grip nearest-sample dt max {a[:, 4].max():.1f} ms  grip NaN {int(a[:, 5].sum())} ->", "PASS" if ok else "FAIL")
    json.dump(dict(part="HRL80", segments=len(a), pos_mm_max=float(a[:, 0].max()), rot_deg_max=float(a[:, 2].max()), grip_dt_ms_max=float(a[:, 4].max()),
                   grip_dt_ms_p99=float(np.percentile(a[:, 4], 99)), pass_=bool(ok)), open(gate_f, "w"), indent=1)
else:
    assert json.load(open(gate_f))["pass_"], "run the gate first"
    ROWS.mkdir(exist_ok=True)
    for e, sl in sorted(by_ep.items()):
        epd, rows = episode_rows(e, sl); instr = json.load(open(epd / "episode_meta.json"))["instruction"]
        np.savez(ROWS / f"{e}.npz", starts=np.array([r["start"] for r in rows]), st=np.stack([r["st"] for r in rows]), ac=np.stack([r["ac"] for r in rows]),
                 tcp_tgt=np.stack([r["tcp_tgt"] for r in rows]), vfL=np.stack([r["vfL"] for r in rows]), vfR=np.stack([r["vfR"] for r in rows]), vfH=np.stack([r["vfH"] for r in rows]),
                 method=np.array(["TR"] * len(rows)), epd=str(epd), instr=instr, val=e in VAL)
    json.dump(dict(part="HRL80", episodes=len(by_ep), segments=len(segs), val_episodes=sorted(VAL & set(by_ep)), master="robotlike/hybrid_hrl80"),
              open(ROWS / "index.json", "w"), indent=1)
    print(f"rows cached [hrl80 TR]: {len(by_ep)} episodes, {len(segs)} segments -> {ROWS}")
