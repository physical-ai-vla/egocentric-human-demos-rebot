#!/usr/bin/env python3
"""[2026-09-28] C-old v2 ego pretraining set from the FROZEN hybrid master (~/c8/robotlike/hybrid_master, MASTER.json).

Identical to c8old_build_dataset.py (v1) except the two things the v2 retarget changed:
  main target  = the master's selected pseudo q  (<episode>.npz key "<start>_v2_0"), not phase3/<episode>.npz
  C / D target = measured human TCP -> relative motion rel(t) (Retarget.targets definition) attached to THAT segment's
                 anchor:  TR -> q_ref(0) of R30 window tr_window_id (deterministic build_windows, stride 5);
                 Kr_fallback -> seedbank_R30_q12[kr_seed_id];  T_tgt = FK(anchor) rel(t);  T_store = T_tgt @ diag(C^T, 1)
Everything else unchanged: B1-old gripper contract (state = -270 * open_fraction, action raw / -6), 65-frame segments, one
LeRobot episode per segment, 3 videos, instruction = episode_meta, val = the SAME 26 source episodes as C-old v1.
Gates before any write: jR / jH recomputed == master-run sync (and == Phase 3 where the episode has a Phase-3 npz);
FK(q_master) vs T_store: position < 15 mm AND rotation < 5 deg on every row of every segment.
Env: ~/xvla-mac/bin/python.  Usage: c8oldv2_build_dataset.py <tr|hybrid> gate|rows   (then write_stage.py with env, see RUN)
"""
import importlib.util, json, os, pathlib, sys
import numpy as np, torch
H = pathlib.Path.home(); C8 = H / "c8"; V2 = H / "umi_bridge/track_c/v2k"; MASTER = C8 / "robotlike/hybrid_master"
VARIANT = sys.argv[1]; MODE = sys.argv[2]; assert VARIANT in ("tr", "hybrid") and MODE in ("gate", "rows")
METHODS = {"tr": {"TR"}, "hybrid": {"TR", "Kr_fallback"}}[VARIANT]
ROWS = C8 / f"c8oldv2_{VARIANT}_rows"
sys.path.insert(0, str(C8)); sys.path.insert(0, str(C8 / "c8old"))
os.environ.setdefault("REBOT_URDF", str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee/reBot_B601_DM_dualarm.urdf"))
# reuse the v2 retarget module for its exact window library / FK helpers (R30 contract, frozen)
spec = importlib.util.spec_from_file_location("v2k", V2 / "v2k_retarget.py"); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
P3 = V.P3; P3.init(); P = P3.P; V.P = P
import rebot_fk_torch
FK = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float32)
CT4 = np.eye(4); CT4[:3, :3] = P.C[:3, :3].T
man = json.load(open(MASTER / "MASTER.json"))
d150 = np.load(H / "umi_bridge/track_c/data/r150_q_tcp.npz"); r30 = json.load(open(V2 / "r30_episodes.json"))["episodes"]
V.build_windows(d150, np.isin(d150["ep"], r30)); BANK = np.load(V2 / "seedbank_R30_q12.npy")
v1_val = set(json.load(open(C8 / "c8old_rows/index.json"))["val_episodes"])


def anchor(seg):
    v = seg["v2"]; return V.WIN["q"][v["tr_window_id"]][0] if v["retarget_method"] == "TR" else BANK[v["kr_seed_id"]]


segs = []
for p in sorted(MASTER.glob("2*.json")):
    r = json.load(open(p))
    for s in r.get("segments", []):
        if isinstance(s.get("v2"), dict) and s["v2"].get("retarget_method") in METHODS: segs.append((r["episode"], s))
eps = sorted({e for e, _ in segs})
print(f"[{VARIANT}] segments {len(segs)} ({', '.join(f'{m} {sum(s[1]['v2']['retarget_method'] == m for s in [x for x in segs])}' for m in sorted(METHODS))}) from {len(eps)} source episodes; "
      f"val episodes {len(v1_val & set(eps))} (same 26 as v1) with {sum(e in v1_val for e, _ in segs)} segments")


def episode_rows(e, seglist):
    S = {s: P3.side(f"{e}_{s}") for s in ("left", "right")}; L, R = S["left"], S["right"]
    sess, ep = e.rsplit("_", 1); epd = P3.RAW / f"Hpilot_{sess}" / f"episode_{ep}"; fm = P3.frame_meta(epd)
    capL = np.array([c for _, c in fm["left_wrist"][L["off"]:L["off"] + L["n"]]], float); capR = np.array([c for _, c in fm["right_wrist"][R["off"]:R["off"] + R["n"]]], float)
    capH = np.array([c for _, c in fm["head"]], float); jR, jH = P3.nearest(capR, capL), P3.nearest(capH, capL)
    ph = C8 / "phase3" / f"{e}.npz"
    if ph.exists():
        z3 = np.load(ph); assert np.array_equal(jR, z3["jR"]) and np.array_equal(jH, z3["jH"]), f"{e}: jR/jH differ from Phase 3"
    zq = np.load(MASTER / f"{e}.npz")
    vfL = np.array([v for v, _ in fm["left_wrist"][L["off"]:L["off"] + L["n"]]]); vfR = np.array([v for v, _ in fm["right_wrist"][R["off"]:R["off"] + R["n"]]])
    vfH = np.array([v for v, _ in fm["head"]])
    gL, gR = np.load(C8 / "grip" / f"{e}_left.npz"), np.load(C8 / "grip" / f"{e}_right.npz")
    out = []
    for seg in seglist:
        start = seg["start"]; s_ = np.arange(start, start + 65); q = zq[f"{start}_v2_0"].astype(np.float64)
        relL, relR = V.rel_traj(L["T_tcp"][s_]), V.rel_traj(R["T_tcp"][jR[s_]])
        T0 = V.fk_T(anchor(seg)); tgL, tgR = T0[0][None] @ relL, T0[1][None] @ relR
        Ts = np.stack([tgL @ CT4, tgR @ CT4], 1)
        Tf = FK.tcp(torch.tensor(q, dtype=torch.float32)).numpy()
        dp = np.linalg.norm(Tf[..., :3, 3] - Ts[..., :3, 3], axis=-1) * 1e3
        cosang = (np.einsum("...ij,...ij->...", Tf[..., :3, :3], Ts[..., :3, :3]) - 1) / 2; dr = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        fL, fR = gL["open_fraction"][s_], gR["open_fraction"][jR[s_]]; qd = np.degrees(q)
        st = np.concatenate([qd[:, :6], -270 * fL[:, None], qd[:, 6:], -270 * fR[:, None]], 1).astype(np.float32)
        ac = st.copy(); ac[:, 6] /= -6; ac[:, 13] /= -6
        out.append(dict(start=start, st=st, ac=ac, tcp_tgt=Ts.reshape(65, 32).astype(np.float32), vfL=vfL[s_], vfR=vfR[jR[s_]], vfH=vfH[jH[s_]],
                        method=seg["v2"]["retarget_method"], pos_mm_max=float(dp.max()), pos_mm_p50=float(np.median(dp)),
                        rot_deg_max=float(dr.max()), rot_deg_p50=float(np.median(dr))))
    return epd, out


by_ep = {}
for e, s in segs: by_ep.setdefault(e, []).append(s)
gate_f = C8 / f"c8oldv2_{VARIANT}_gate.json"
if MODE == "gate":
    stats = []
    for i, (e, sl) in enumerate(sorted(by_ep.items())):
        _, rows = episode_rows(e, sl); stats += [(r["pos_mm_max"], r["pos_mm_p50"], r["rot_deg_max"], r["rot_deg_p50"]) for r in rows]
        if i % 50 == 0: print(f"  gate {i}/{len(by_ep)} episodes", flush=True)
    a = np.array(stats); ok = (a[:, 0] < 15).all() and (a[:, 2] < 5).all()
    print(f"HARD GATE [{VARIANT}] over {len(a)} segments x 65 rows x 2 arms: FK(q_master) vs T_store  pos max {a[:,0].max():.3f} mm (p50 of p50 {np.median(a[:,1]):.4f})"
          f"  rot max {a[:,2].max():.3f} deg (p50 {np.median(a[:,3]):.4f}) ->", "PASS" if ok else "FAIL")
    json.dump(dict(variant=VARIANT, segments=len(a), pos_mm_max=float(a[:, 0].max()), rot_deg_max=float(a[:, 2].max()), pass_=bool(ok),
                   master_sha=man["files_sha256"].get("MASTER.json")), open(gate_f, "w"), indent=1)
else:
    assert json.load(open(gate_f))["pass_"], "run the gate first"
    ROWS.mkdir(exist_ok=True)
    for e, sl in sorted(by_ep.items()):
        epd, rows = episode_rows(e, sl); instr = json.load(open(epd / "episode_meta.json"))["instruction"]
        np.savez(ROWS / f"{e}.npz", starts=np.array([r["start"] for r in rows]), st=np.stack([r["st"] for r in rows]), ac=np.stack([r["ac"] for r in rows]),
                 tcp_tgt=np.stack([r["tcp_tgt"] for r in rows]), vfL=np.stack([r["vfL"] for r in rows]), vfR=np.stack([r["vfR"] for r in rows]), vfH=np.stack([r["vfH"] for r in rows]),
                 method=np.array([r["method"] for r in rows]), epd=str(epd), instr=instr, val=e in v1_val)
    json.dump(dict(variant=VARIANT, episodes=len(by_ep), segments=len(segs), val_episodes=sorted(v1_val & set(by_ep)), master="hybrid_master/MASTER.json"),
              open(ROWS / "index.json", "w"), indent=1)
    print(f"rows cached [{VARIANT}]: {len(by_ep)} episodes, {len(segs)} segments -> {ROWS}")
