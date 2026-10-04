"""Right-arm-only ego task (HRA_red, 2026-10-03): HandUMI right wrist -> X-VLA v4 [16,32] samples + a per-dim LOSS MASK.

User design (2026-10-03): do NOT force a single-arm approach take into the bimanual stacking dataset.  Keep the tensor
contract of ego CART20 v2 (state [20], action [16,32], same channel map, same UMI_DT / REL16 / rot6d / gripper polarity)
so the model and normalizer see the usual shapes, but supervise ONLY what was measured:

    action dims  0:10   left CART10       dummy (identity pose, gripper LEFT_DUMMY_G)   loss mask 0
                10:19   right xyz+rot6d   real (current-anchor REL16, raw-track interp)  loss mask 1
                19      right gripper     real value                                     loss mask 1 only if it moves
                20:32   AUX12             0                                              never in the REL-only loss
    state        0:9 left pose9 = identity, 18 gL = LEFT_DUMMY_G (input only, never a target)
                 9:18 right RELCART task anchor, 19 gR

The left arm is never taught "stand still": its target dims carry no loss.  A task/domain label (TASK_ID) travels with
every row.  Nothing in config.py / convert.py (the frozen bimanual contract) is changed; this module re-uses its pieces.

    PY=~/xvla-mac/bin/python
    $PY -m ego_cart20.right_only export  <raw HandUMI session dir> <raw_out>     # needs MASt3R ss1.csv for the right wrist
    $PY -m ego_cart20.right_only convert <raw_out> <processed_out> [--val-frac 0.1]
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from .config import ACT_GRIP, ACT_POS, ACT_ROT, CONTRACT, HORIZON, ST_GRIP, ST_POSE, Cart20Config
from .geometry.transforms import pose9, relative, pose_to_T
from .io.processed_writer import write_episode
from .labels.build_cart20 import target_times
from .labels.pack_action32 import check_action32, pack_cart20_to_action32
from .preprocessing.resample import PoseTrack, canonical_times
from .preprocessing.synchronize import nearest_index

TASK_ID = "HRA_RED_RIGHT_ONLY"
SCHEMA = "ego_cart20_right_only/v1"
LEFT_DUMMY_G = 0.0                       # input-only placeholder (old ego: the human jaw rests closed)
IDENTITY9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float64)   # pose9 of identity (rot6d = first two rows)
GRIP_SUPERVISE_MIN_SPAN = 0.05           # right gripper is supervised only if it actually moves in the episode
CAMERAS = ("head", "left_wrist", "right_wrist")


# ------------------------------------------------------------------------------------------------ loss mask
def loss_mask20(grip_supervised: bool) -> np.ndarray:
    m = np.zeros(20, np.float32)
    m[ACT_POS["right"]] = 1; m[ACT_ROT["right"]] = 1; m[ACT_GRIP["right"]] = 1.0 if grip_supervised else 0.0
    return m


# ------------------------------------------------------------------------------------------------ export (raw)
def export_session(session_dir: pathlib.Path, out_root: pathlib.Path, *, episodes: list[str] | None = None) -> list[dict]:
    """HandUMI right-only session -> raw_episode.{json,npz} with ONLY right_* arrays (+ cameras).  Pose = the same stack
    as HRL80 (robotlike_offline_check.side_nogrip: MASt3R-SLAM + IMU-VI metric scale + handumi_camera_tcp_v2) right-
    multiplied by the reBot tool frame X; gripper = the collector's normalized jaw, era HRL80_v014 caliper map."""
    from .sources import handumi_export as HX
    HX._init()
    HX.P3.RUNS = HX.RL / "runs"; HX.P3.export_dir = lambda tag: HX.RL / "export" / tag
    from .geometry.rotation6d import matrix_to_quaternion
    from .io.raw_episode_loader import save_raw_episode
    out_root.mkdir(parents=True, exist_ok=True); res = []; qc_dir = out_root / "_scale_qc"; qc_dir.mkdir(exist_ok=True)
    from . import cube_pnp as CP
    eps = sorted(p for p in session_dir.glob("episode_*") if (p / ".complete").exists())
    if episodes: eps = [p for p in eps if p.name in episodes]
    for epd in eps:
        eid = f"{session_dir.name}_{epd.name.split('_')[1]}"; out = out_root / eid
        if (out / "raw_episode.json").exists(): res.append(dict(episode=eid, status="exists")); continue
        try:
            tag = HX.RLC.tag_of(epd, "right")
            if not (HX.RL / "runs" / tag / "ss1.csv").exists():
                res.append(dict(episode=eid, status="no_pose", reason=f"no MASt3R ss1.csv for {tag}")); continue
            d = HX.RLC.side_nogrip(tag, epd, "right")            # IMU-VI scale: DIAGNOSTIC only from here on
            ex = HX.RL / "export" / tag
            pn = CP.episode_pnp_scale(ex / "raw_video.mp4", ex / "orbslam_setting.yaml", HX.RL / "runs" / tag / "ss1.csv")
            qc = dict(episode=eid, s_imu=float(d["s"]), imu_scale_valid=bool(d["scale_valid"]), **{k: v for k, v in pn.items() if k != "frames"})
            qc["r_pnp_over_imu"] = (pn["s_pnp"] / d["s"]) if pn.get("s_pnp") and d["s"] and np.isfinite(d["s"]) else None
            json.dump(dict(qc, frames=pn["frames"]), open(qc_dir / f"{eid}.json", "w"), default=float, indent=1)
            if not pn["valid"]:
                res.append(dict(qc, status="refused", reason=f"cube PnP scale invalid: {pn['reason']}")); print(json.dumps(res[-1], default=float), flush=True); continue
            d = metric_side(HX, tag, epd, d, pn["s_pnp"])           # re-scale the SAME SLAM trajectory with the cube scale
            fm = HX.P3.frame_meta(epd); off, n = d["off"], d["n"]; fr = fm["right_wrist"][off:off + n]; assert len(fr) == n
            vf = np.array([v for v, _ in fr], np.int64); t_ns = np.array([c for _, c in fr], np.int64)
            of = HX._hrl_grip(epd, "right", vf); g = HX.GCAL.G(np.clip(of, 0, 1), "R", "HRL80_v014")
            T = np.asarray(d["T_tcp"], np.float64) @ HX.X[None]; ok = np.asarray(d["ok"], bool) & np.isfinite(g)
            arrays = {"right_t_ns": t_ns, "right_position": T[:, :3, 3], "right_quaternion": matrix_to_quaternion(T[:, :3, :3]),
                      "right_valid": ok, "right_gripper": g, "right_open_fraction_raw": of, "right_video_frame": vf}
            for c in ("head", "right_wrist"):
                arrays[f"cam_{c}_t_ns"] = np.array([t for _, t in fm[c]], np.int64); arrays[f"cam_{c}_frame"] = np.array([v for v, _ in fm[c]], np.int64)
            em = json.load(open(epd / "episode_meta.json"))
            meta = dict(schema="ego_cart20_raw_episode/v1", episode_id=eid, source="human_egocentric_handumi", era="HRA_red_20261003",
                        task="approach", task_id=TASK_ID, arms=["right"], instruction=em["instruction"], stack_order="none",
                        episode_status=em.get("status"), quality=em.get("quality"), clock="capture_monotonic_ns",
                        videos={c: str(epd / f"{c}.mp4") for c in ("head", "right_wrist")},
                        tool_frame=dict(name="handumi_tcp_v2 @ F @ C @ CT4 (reBot dataset-TCP axes)", X=HX.X.tolist()),
                        gripper_calibration=dict(version="umi_aperture_cal_v1", era_map="HRL80_v014 (NOT re-measured for 10-03; "
                                                 "masked in the loss unless it moves)", convention="0 = closed, 1 = open, mm / 80 mm"),
                        provenance=dict(raw_dir=str(epd), pose_stack="robotlike_offline_check.side_nogrip trajectory, metric scale = cube PnP",
                                        mast3r_tag=tag, metric_scale=float(d["s"]), metric_scale_source="cube_pnp (edge %.3f m)" % CP.CUBE_EDGE_M,
                                        imu_vi_scale_diagnostic=qc["s_imu"], scale_qc=f"_scale_qc/{eid}.json", coverage=float(ok.mean()), exporter=str(pathlib.Path(__file__).resolve()),
                                        exported=time.strftime("%F %T")))
            save_raw_episode(out, meta, arrays)
            p_m = T[ok][:, :3, 3]
            res.append(dict(qc, status="ok", n=int(n), coverage=round(float(ok.mean()), 3), scale_used=float(d["s"]), scale_source="cube_pnp",
                            net_displacement_m=float(np.linalg.norm(p_m[-1] - p_m[0])) if len(p_m) else None))
        except Exception as ex:
            res.append(dict(episode=eid, status="error", reason=f"{type(ex).__name__}: {ex}"))
        print(json.dumps(res[-1]), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ convert (labels)
def metric_side(HX, tag, epd, d_imu, s_metric):
    """side_nogrip with the metric scale REPLACED: identical trajectory / ok mask / offset, translation x s_metric, then the
    same camera->TCP X.  (side_nogrip multiplies by the IMU-VI s; here that s is only kept as a diagnostic.)"""
    P3 = HX.P3; n = d_imu["n"]
    T, ok = P3.cp6["load_traj"](P3.STAGE / "runs" / tag / "ss1.csv", n)
    Tm = T.copy(); Tm[:, :3, 3] *= s_metric; Tt = Tm @ P3.cp6["X"]; Tt[~ok] = np.eye(4)
    return dict(d_imu, T_tcp=Tt, ok=ok, s=float(s_metric), s_imu=d_imu["s"])


def load_raw_right(ep_dir: pathlib.Path):
    meta = json.load(open(ep_dir / "raw_episode.json")); z = np.load(ep_dir / "raw_episode.npz")
    t0 = int(z["right_t_ns"][0]); t = (z["right_t_ns"].astype(np.int64) - t0) / 1e9
    T = pose_to_T(z["right_position"], z["right_quaternion"])
    cams = {c: ((z[f"cam_{c}_t_ns"].astype(np.int64) - t0) / 1e9, z[f"cam_{c}_frame"].astype(np.int64))
            for c in CAMERAS if f"cam_{c}_t_ns" in z.files}
    return meta, t, T, z["right_valid"].astype(bool), z["right_gripper"].astype(np.float64), cams, t0


def convert_right_only(ep_dir: pathlib.Path, cfg: Cart20Config = Cart20Config()) -> dict:
    meta, t, T, valid, g, cams, t0 = load_raw_right(ep_dir)
    tr = PoseTrack(t, T, valid, g, cfg.max_raw_gap_s)
    vt = tr.t[tr.valid]
    if len(vt) < 2: raise ValueError("right arm has < 2 valid raw samples")
    cand = vt; okc = tr.sample(cand)[2]; t_s = float(cand[np.flatnonzero(okc)[0]]); t_e = float(vt[-1])
    times = canonical_times(t_s, t_e, cfg.row_fps); N = len(times)
    Tr, gr, row_valid = tr.sample(times); anchor = Tr[0].copy(); assert row_valid[0]

    # state: left dummy (input only) | right RELCART task anchor
    state = np.zeros((N, 20), np.float32)
    state[:, ST_POSE["left"]] = IDENTITY9; state[:, ST_GRIP["left"]] = LEFT_DUMMY_G
    state[:, ST_POSE["right"]] = pose9(relative(anchor, Tr)); state[:, ST_GRIP["right"]] = gr
    state[~row_valid] = np.nan
    Tp, gp, okp = tr.sample(times - cfg.target_dt_s); prev_ok = okp & row_valid
    state_prevrel = np.zeros((N, 20), np.float32)
    state_prevrel[:, 0:9] = IDENTITY9; state_prevrel[:, 9] = LEFT_DUMMY_G
    state_prevrel[:, 10:19] = pose9(relative(Tr, np.where(prev_ok[:, None, None], Tp, Tr))); state_prevrel[:, 19] = gr
    state_prevrel[~row_valid] = np.nan

    cam_valid = np.zeros((N, len(CAMERAS)), bool); cameras = {}
    for ci, c in enumerate(CAMERAS):
        if c not in cams: continue
        ct, cf = cams[c]; j, dt = nearest_index(ct, times, cfg.image_tol_s); cam_valid[:, ci] = j >= 0
        cameras[c] = dict(frame_index=np.where(j >= 0, cf[np.maximum(j, 0)], -1).astype(np.int64),
                          video=meta.get("videos", {}).get(c), tol_s=cfg.image_tol_s)
    need = cam_valid[:, CAMERAS.index("head")].copy()

    rows, chunks = [], []
    for r in np.flatnonzero(row_valid & need):
        Tf, gf, okf = tr.sample(target_times(times[r], cfg.horizon, cfg.target_dt_s))
        if not okf.all(): continue
        c = np.zeros((cfg.horizon, 20))
        c[:, ACT_POS["left"]] = 0.0; c[:, ACT_ROT["left"]] = IDENTITY9[3:]; c[:, ACT_GRIP["left"]] = LEFT_DUMMY_G
        p9 = pose9(relative(Tr[r][None], Tf)); c[:, ACT_POS["right"]] = p9[:, :3]; c[:, ACT_ROT["right"]] = p9[:, 3:]
        c[:, ACT_GRIP["right"]] = gf
        rows.append(r); chunks.append(c.astype(np.float32))
    cart20 = np.stack(chunks) if chunks else np.zeros((0, HORIZON, 20), np.float32)
    action = pack_cart20_to_action32(cart20); check_action32(action, cart20)
    train_rows = np.asarray(rows, np.int64)

    gv = gr[row_valid]; span = float(np.nanmax(gv) - np.nanmin(gv)) if len(gv) else 0.0
    grip_sup = span >= GRIP_SUPERVISE_MIN_SPAN
    lm = np.tile(loss_mask20(grip_sup), (len(train_rows), 1))
    pose_l = np.full((N, 4, 4), np.nan); pose_l[:] = np.eye(4)
    pose_r = relative(anchor, Tr); pose_r[~row_valid] = np.nan
    md = dict(episode_id=meta["episode_id"], source=meta.get("source"), task=meta["task"], task_id=TASK_ID, arms=["right"],
              instruction=meta["instruction"], stack_order=meta["stack_order"], fps=cfg.row_fps, **CONTRACT, schema=SCHEMA,
              left_arm="dummy (identity pose, gripper %.1f) -- loss-masked, never a target" % LEFT_DUMMY_G,
              loss_mask=dict(layout="[20] over action dims 0:20; 20:32 always excluded by REL-only", per_row=lm[0].tolist() if len(lm) else None,
                             right_gripper_supervised=grip_sup, right_gripper_span=round(span, 4), rule=f"gripper supervised iff span >= {GRIP_SUPERVISE_MIN_SPAN}"),
              task_start_time_s=t_s, clock_offset_ns=int(t0), duration_s=float(times[-1] - times[0]), frames_raw=int(len(t)),
              frames_rows=int(N), rows_valid=int(row_valid.sum()), train_rows=int(len(train_rows)), cameras=list(cameras),
              camera_order=list(CAMERAS), camera_available={c: c in cameras for c in CAMERAS}, config=cfg.to_dict(),
              tool_frame=meta.get("tool_frame"), gripper_calibration=meta.get("gripper_calibration"), raw_provenance=meta.get("provenance"))
    return dict(timestamps=times, row_valid=row_valid, train_rows=train_rows, state=state, state_prevrel=state_prevrel,
                state_prev_valid=prev_ok.astype(np.float32), cart20=cart20, action=action, left_pose=pose_l, right_pose=pose_r,
                left_gripper=np.full(N, LEFT_DUMMY_G, np.float32), right_gripper=np.where(row_valid, gr, np.nan).astype(np.float32),
                stage_id=np.full(N, -1, np.int8), camera_valid=cam_valid, cameras=cameras, metadata=md, loss_mask=lm)


def split_episodes(ids: list[str], val_frac: float) -> dict:
    """deterministic sha1 bucket (as dataset.splits.hash_split) -- this task has no frozen split yet"""
    sp = {}
    for e in ids:
        u = int(hashlib.sha1(e.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF; sp[e] = "val" if u < val_frac else "train"
    return sp


def convert_tree(raw_root: pathlib.Path, out_root: pathlib.Path, *, val_frac: float = 0.1) -> dict:
    assert not out_root.exists(), f"{out_root} exists (processed trees are write-once)"
    (out_root / "episodes").mkdir(parents=True); res = []
    for rd in sorted(p for p in raw_root.iterdir() if (p / "raw_episode.json").exists()):
        try:
            ep = convert_right_only(rd); out = write_episode(out_root / "episodes" / rd.name, ep)
            np.save(out / "loss_mask.npy", ep["loss_mask"]); m = ep["metadata"]
            res.append(dict(episode_id=rd.name, status="ok", train_rows=m["train_rows"], rows_valid=m["rows_valid"], frames_rows=m["frames_rows"],
                            grip_supervised=m["loss_mask"]["right_gripper_supervised"], grip_span=m["loss_mask"]["right_gripper_span"],
                            instruction=m["instruction"], stack_order=m["stack_order"]))
        except Exception as ex:
            res.append(dict(episode_id=rd.name, status="error", reason=f"{type(ex).__name__}: {ex}"))
    ok = [r for r in res if r["status"] == "ok" and r["train_rows"] > 0]
    sp = split_episodes([r["episode_id"] for r in ok], val_frac)
    for s in ("train", "val", "test"):
        with open(out_root / f"{s}_manifest.jsonl", "w") as f:
            for r in ok:
                if sp[r["episode_id"]] == s: f.write(json.dumps(dict(r, split=s, path=f"episodes/{r['episode_id']}")) + "\n")
    cnt = {s: dict(episodes=sum(sp[r["episode_id"]] == s for r in ok), train_rows=sum(r["train_rows"] for r in ok if sp[r["episode_id"]] == s))
           for s in ("train", "val")}
    meta = dict(schema=SCHEMA, task_id=TASK_ID, created=time.strftime("%F %T"), raw_root=str(raw_root.resolve()), contract=CONTRACT,
                split=f"sha1 bucket val {val_frac}", counts=cnt, grip_supervised_episodes=sum(r["grip_supervised"] for r in ok),
                status=dict(collections.Counter(r["status"] for r in res)), conversion=res,
                left_arm="dummy, loss-masked", loss_mask_file="episodes/<id>/loss_mask.npy [train_rows, 20]")
    json.dump(meta, open(out_root / "metadata.json", "w"), indent=1)
    return meta


# [2026-10-03] per-episode PHYSICAL sanity gate, applied after the export (the old global gates "s_pnp p95/p5 <= 1.6"
# and "HOME->cube width <= 0.35 m" were dropped: the MASt3R map scale and the start pose legitimately vary per episode).
SANITY = dict(s_min=0.1, s_max=1.0, travel_ratio_min=0.8, travel_max_m=0.8)


def sanity_gate(export_log: pathlib.Path, raw_root: pathlib.Path) -> dict:
    """keep an exported episode only if  0.1 <= s_pnp <= 1.0,  travel_ratio = slam_metric_travel / pnp_observed_approach >= 0.8,
    slam_metric_travel <= 0.8 m.  Rejected raw episodes are MOVED to <raw>/_rejected_sanity/ (never deleted) and listed with every
    number in <raw>/sanity_rejected.jsonl; the accepted list + the travel_ratio distribution go to <raw>/sanity_report.json."""
    import shutil
    ex = json.loads(export_log.read_text()); ok = [e for e in ex if e.get("status") == "ok"]; rej, acc, ratios = [], [], []
    for e in ok:
        pa = float(e["dist_m_range"][1] - e["dist_m_range"][0]); tr = float(e["net_displacement_m"]); ratio = tr / max(pa, 1e-6)
        rec = dict(episode_id=e["episode"], s_pnp=e["s_pnp"], pnp_approach_distance_m=round(pa, 4), slam_metric_travel_m=round(tr, 4),
                   travel_ratio=round(ratio, 3), centre_residual_cm=round(e["centre_resid_m"] * 100, 3), valid_pnp_frames=e["n_valid_pnp"],
                   s_imu_diagnostic=e.get("s_imu"))
        why = []
        if not (SANITY["s_min"] <= e["s_pnp"] <= SANITY["s_max"]): why.append(f"s_pnp {e['s_pnp']:.4f} outside {SANITY['s_min']}-{SANITY['s_max']}")
        if ratio < SANITY["travel_ratio_min"]: why.append(f"travel_ratio {ratio:.2f} < {SANITY['travel_ratio_min']}")
        if tr > SANITY["travel_max_m"]: why.append(f"slam_metric_travel {tr:.2f} m > {SANITY['travel_max_m']} m")
        if why: rej.append(dict(rec, reject_reason="; ".join(why)))
        else: acc.append(rec); ratios.append(ratio)
    dst = raw_root / "_rejected_sanity"; dst.mkdir(exist_ok=True)
    for r in rej:
        src = raw_root / r["episode_id"]
        if src.exists(): shutil.move(str(src), str(dst / r["episode_id"]))
    with open(raw_root / "sanity_rejected.jsonl", "w") as f:
        for r in rej: f.write(json.dumps(r) + "\n")
    r = np.array(ratios); big = [a["episode_id"] for a in acc if a["travel_ratio"] > 3.0]
    rep = dict(gate=SANITY, exported_ok=len(ok), accepted=len(acc), rejected=len(rej), accepted_ids=[a["episode_id"] for a in acc],
               travel_ratio_p5_p50_p95=np.percentile(r, [5, 50, 95]).round(3).tolist() if len(r) else None,
               travel_ratio_gt3=big, note="travel_ratio > 3 is NOT a hard gate (user): listed for inspection only")
    (raw_root / "sanity_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("session", type=pathlib.Path); e.add_argument("out", type=pathlib.Path); e.add_argument("--episodes", nargs="*")
    c = sub.add_parser("convert"); c.add_argument("raw", type=pathlib.Path); c.add_argument("out", type=pathlib.Path); c.add_argument("--val-frac", type=float, default=0.1)
    sn = sub.add_parser("sanity"); sn.add_argument("raw", type=pathlib.Path); sn.add_argument("--export-log", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "sanity":
        log = a.export_log or sorted(a.raw.glob("export_log_*.json"))[-1]
        rep = sanity_gate(log, a.raw); print("SANITY", json.dumps({k: v for k, v in rep.items() if k != "accepted_ids"})); return 0
    if a.cmd == "export":
        res = export_session(a.session, a.out, episodes=a.episodes)
        log = a.out / f"export_log_{time.strftime('%Y%m%d_%H%M%S')}.json"; json.dump(res, open(log, "w"), indent=1)
        print("DONE", dict(collections.Counter(r["status"] for r in res)), "->", log); return 0
    m = convert_tree(a.raw, a.out, val_frac=a.val_frac)
    print("DONE", json.dumps(m["counts"]), m["status"], "grip supervised eps", m["grip_supervised_episodes"]); return 0


if __name__ == "__main__":
    sys.exit(main())
