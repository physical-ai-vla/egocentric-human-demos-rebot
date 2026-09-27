#!/usr/bin/env python3
"""[2026-09-25] C-old production dataset (contract §22b / §24): B1-compatible pseudo-joint pretraining set.
Source split, never mixed:
  main target  = the STORED Phase-3 pseudo q (phase3/<episode>.npz, fixed artefact; no IK re-run)
  C / D target = measured human TCP re-derived independently: c8_phase3.side() metric TCP -> Retarget(r150_active_median)
                 -> T_tgt (robot base frame, NO IK) -> T_store = T_tgt @ diag(C^T, 1) (rebot_fk_torch TCP convention)
Segments: bimanual E (first_fail None) of phase3_census.json minus the quarantined sessions (imu_camera_rotation_inconsistent_v1).
Row layout (B1 / V3 conventions): observation.state = [qL6 deg, -270 * open_L, qR6 deg, -270 * open_R]; action = state with
gripper raw / -6; observation.ee.tcp_tgt = [T_store_L (4x4), T_store_R (4x4)] flattened (32); 3 videos 640x480 (head ->
global, left wrist, right wrist); task = episode_meta instruction. One LeRobot episode = one 65-frame segment.
Split: ~10 % of SOURCE episodes -> c8old_val (default_rng(0)); never a segment-level split.
Gates (every row of every segment, before any write): jR / jH recomputed == stored in the npz; rebot_fk_torch FK(q_pseudo)
vs T_store: position < 15 mm AND rotation < 5 deg (the C^T convention check); any failure aborts.
Env: ~/xvla-mac/bin/python.  Usage: c8old_build_dataset.py gate | rows | write
  rows  (imports JAX via c8_phase3) -> c8old_rows/<episode>.npz;  write (NO JAX: a JAX process that forks for LeRobot's
  video / image writers deadlocked on 2026-09-25) -> the LeRobot datasets from the cached rows.
"""
import json, os, pathlib, shutil, sys
import numpy as np, torch, cv2, av
H = pathlib.Path.home(); C8 = H / "c8"; OUT = C8 / "c8old_data"; sys.path.insert(0, str(C8))
os.environ.setdefault("REBOT_URDF", str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee/reBot_B601_DM_dualarm.urdf"))
sys.path.insert(0, str(C8 / "c8old"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "gate"
ROWS = C8 / "c8old_rows"
if MODE == "write":
    import write_stage; raise SystemExit(write_stage.main())
import c8_phase3 as P3
P3.init(); P = P3.P
import rebot_fk_torch
FK = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float32)
CT4 = np.eye(4); CT4[:3, :3] = P.C[:3, :3].T
RT = P.Retarget(P3.ik.kin, P.ANCHORS["r150_active_median"])
d = json.load(open(C8 / "phase3_census.json"))["episodes"]; Q = set(json.load(open(C8 / "c8_quarantine.json"))["sessions"])
segs = [(e, s["start"]) for e, r in sorted(d.items()) if e.rsplit("_", 1)[0] not in Q for s in r.get("segments", []) if s["first_fail"] is None]
eps = sorted({e for e, _ in segs}); rng = np.random.default_rng(0); val_eps = set(rng.choice(eps, max(1, round(0.1 * len(eps))), replace=False).tolist())
print(f"segments {len(segs)} from {len(eps)} source episodes; val episodes {len(val_eps)} ({sum(e in val_eps for e, _ in segs)} segments)")


def decode(path, want):
    out = {}; want = set(int(x) for x in want)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want: out[i] = cv2.resize(fr.to_ndarray(format="rgb24"), (640, 480), interpolation=cv2.INTER_AREA)
            if i > max(want): break
    return out


def episode_rows(e, starts):
    """-> per segment: dict(q, st, ac, tcp_tgt, vf_L, vf_R, vf_H, gate stats)"""
    S = {s: P3.side(f"{e}_{s}") for s in ("left", "right")}; L, R = S["left"], S["right"]
    sess, ep = e.rsplit("_", 1); epd = P3.RAW / f"Hpilot_{sess}" / f"episode_{ep}"; fm = P3.frame_meta(epd)
    capL = np.array([c for _, c in fm["left_wrist"][L["off"]:L["off"] + L["n"]]], float); capR = np.array([c for _, c in fm["right_wrist"][R["off"]:R["off"] + R["n"]]], float)
    capH = np.array([c for _, c in fm["head"]], float); jR, jH = P3.nearest(capR, capL), P3.nearest(capH, capL)
    z = np.load(C8 / "phase3" / f"{e}.npz"); assert np.array_equal(jR, z["jR"]) and np.array_equal(jH, z["jH"]), f"{e}: jR/jH differ from Phase 3"
    vfL = np.array([v for v, _ in fm["left_wrist"][L["off"]:L["off"] + L["n"]]]); vfR = np.array([v for v, _ in fm["right_wrist"][R["off"]:R["off"] + R["n"]]])
    vfH = np.array([v for v, _ in fm["head"]])
    gL, gR = np.load(C8 / "grip" / f"{e}_left.npz"), np.load(C8 / "grip" / f"{e}_right.npz")
    out = []
    for start in starts:
        k = int(np.flatnonzero(z["start"] == start)[0]); assert bool(z["usable"][k]), f"{e}@{start} not usable in the npz"
        q = z["q"][k].astype(np.float64); s_ = np.arange(start, start + 65)
        tgL, tgR = RT.targets(L["T_tcp"][s_], 0), RT.targets(R["T_tcp"][jR[s_]], 1)
        Ts = np.stack([tgL @ CT4, tgR @ CT4], 1)                                # (65, 2, 4, 4) rebot_fk_torch convention
        Tf = FK.tcp(torch.tensor(q, dtype=torch.float32)).numpy()               # (65, 2, 4, 4)
        dp = np.linalg.norm(Tf[..., :3, 3] - Ts[..., :3, 3], axis=-1) * 1e3
        cosang = (np.einsum("...ij,...ij->...", Tf[..., :3, :3], Ts[..., :3, :3]) - 1) / 2; dr = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        fL, fR = gL["open_fraction"][s_], gR["open_fraction"][jR[s_]]; qd = np.degrees(q)
        st = np.concatenate([qd[:, :6], -270 * fL[:, None], qd[:, 6:], -270 * fR[:, None]], 1).astype(np.float32)
        ac = st.copy(); ac[:, 6] /= -6; ac[:, 13] /= -6
        out.append(dict(start=start, st=st, ac=ac, tcp_tgt=Ts.reshape(65, 32).astype(np.float32), vfL=vfL[s_], vfR=vfR[jR[s_]], vfH=vfH[jH[s_]],
                        pos_mm_max=float(dp.max()), pos_mm_p50=float(np.median(dp)), rot_deg_max=float(dr.max()), rot_deg_p50=float(np.median(dr))))
    return epd, out


by_ep = {}
for e, s in segs: by_ep.setdefault(e, []).append(s)
if MODE == "gate":
    stats = []
    for i, (e, starts) in enumerate(sorted(by_ep.items())):
        _, rows = episode_rows(e, starts); stats += [(r["pos_mm_max"], r["pos_mm_p50"], r["rot_deg_max"], r["rot_deg_p50"]) for r in rows]
        if i % 50 == 0: print(f"  gate {i}/{len(by_ep)} episodes", flush=True)
    a = np.array(stats); ok = (a[:, 0] < 15).all() and (a[:, 2] < 5).all()
    print(f"HARD GATE over {len(a)} segments x 65 rows x 2 arms: FK(q_pseudo) vs T_store  pos max {a[:,0].max():.3f} mm (segment p50 of p50 {np.median(a[:,1]):.4f})"
          f"  rot max {a[:,2].max():.3f} deg (p50 {np.median(a[:,3]):.4f})  jR/jH identical to Phase 3 for all episodes ->", "PASS" if ok else "FAIL")
    json.dump(dict(segments=len(a), pos_mm_max=float(a[:, 0].max()), rot_deg_max=float(a[:, 2].max()), pass_=bool(ok)), open(C8 / "c8old_gate.json", "w"), indent=1)
elif MODE == "rows":
    assert json.load(open(C8 / "c8old_gate.json"))["pass_"], "run the gate first"
    ROWS.mkdir(exist_ok=True)
    for e, starts in sorted(by_ep.items()):
        epd, rows = episode_rows(e, starts); instr = json.load(open(epd / "episode_meta.json"))["instruction"]
        np.savez(ROWS / f"{e}.npz", starts=np.array([r["start"] for r in rows]), st=np.stack([r["st"] for r in rows]), ac=np.stack([r["ac"] for r in rows]),
                 tcp_tgt=np.stack([r["tcp_tgt"] for r in rows]), vfL=np.stack([r["vfL"] for r in rows]), vfR=np.stack([r["vfR"] for r in rows]), vfH=np.stack([r["vfH"] for r in rows]),
                 epd=str(epd), instr=instr, val=e in val_eps)
    json.dump(dict(episodes=len(by_ep), segments=len(segs), val_episodes=sorted(val_eps)), open(ROWS / "index.json", "w"), indent=1)
    print(f"rows cached: {len(by_ep)} episodes, {len(segs)} segments -> {ROWS}")
