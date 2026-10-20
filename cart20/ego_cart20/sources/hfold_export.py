#!/usr/bin/env python3
"""[2026-10-12] HandUMI laundry folding (Hfold, hardware handumi_rgbd) -> raw ego episodes in the ego_cart20 raw format.

Self-contained replacement for sources/handumi_export.py for NEW sessions: that adapter reads the frozen C8 stack under ~/c8
(census / grip / stage files), which is no longer on this Mac. The geometry is the same chain, re-assembled from the
consolidated repo (~/bh-egocentric/pipeline):

  pose   MASt3R-SLAM ss1.csv (RGB-only, CORI-indexed, ~/c8/robotlike/runs/<tag>/ss1.csv from robotlike_session_check_4090.sh)
         x metric scale s (imu_vi, cp6_scale.imu_vi re-implemented 1:1: SG window 15, no bias, Huber IRLS; validity = |g| in
         [8.81, 10.81] and the C8 degenerate guard) -> camera_tcp_v2 (X_cam_tcp) -> tool frame X_tool = F @ C @ CT4 (reBot
         dataset-TCP axes, identical to handumi_export).
         --scale-json overrides s per tag (e.g. Orbbec-derived metric scale); the source of every s is recorded.
  t_ns   wrist video capture_ns (sensors.mcap <side>_wrist/frame_meta) of video frame off + i, off = N_video - n_CORI
         (video_to_cori_index_v1).
  grip   Feetech raw ticks re-normalised per episode/side (the stored `normalized` is wrong for the right unit after the
         10-12 reboot: servo counter moved): closed = p2 of the episode's raw ticks (the jaw rests closed), arc = the side's
         calibrated arc length; then binary X-VLA rule width < 2 cm -> closed, with full open = 80 mm (gcal1 caliper scale):
         open iff g_norm >= 0.25. 1 = open, 0 = closed.
  cams   head = Orbbec colour head_depth.mp4, left_wrist / right_wrist fisheye mp4 (frame_meta capture_ns, video_frame).

usage (~/xvla-mac/bin/python or ego_collector/.venv/bin/python):
  hfold_export.py --sessions <session dir> ... --out <raw_root> [--scale-json scales.json]
"""
import argparse, csv, json, pathlib, re, sys

import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

H = pathlib.Path.home(); RL = H / "c8/robotlike"; REPO = H / "bh-egocentric/pipeline"
CAM_TCP_V2 = REPO / "calibration/camera_tcp/handumi_camera_tcp_v2.yaml"
SIDES = ("left", "right"); W_VI = 15; G_BAND = (8.81, 10.81); S_GUARD = (0.1 * 0.3049, 10 * 0.9178)
GRIP_ARC = {"left": 3232 - 2134, "right": 5677 - 4375}          # calibrated jaw arc (ticks), gripper_v014 (pre-reboot, both valid)
OPEN_FRAC_2CM = 0.02 / 0.08                                     # width < 2 cm closed; 80 mm full open (gcal1 caliper scale)
INSTRUCTION = "fold the cloth"


def _h(R=None, t=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if t is not None: T[:3, 3] = t
    return T


# pseudo_joint_pipeline.C / F (frozen) and handumi_export's X = F @ C @ CT4
C = _h(Rot.from_rotvec([0, np.pi / 2, 0]).as_matrix()); F = _h(Rot.from_rotvec([np.pi, 0, 0]).as_matrix())
CT4 = np.eye(4); CT4[:3, :3] = C[:3, :3].T
X_TOOL = F @ C @ CT4


def x_cam_tcp():
    p = CAM_TCP_V2 if CAM_TCP_V2.exists() else next(REPO.rglob("handumi_camera_tcp_v2.yaml"))
    v = yaml.safe_load(p.read_text()); return _h(np.asarray(v["rotation"]["R_camera_tcp"], float), v["translation"]["t_camera_tcp_m"]), str(p)


def load_traj(p, n):                                            # cp6_scale.load_traj
    T = np.tile(np.eye(4), (n, 1, 1)); ok = np.zeros(n, bool)
    for r in csv.DictReader(open(p)):
        i = int(r["frame_idx"])
        if i < n and r["is_lost"] == "false":
            q = [float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")]; xyz = [float(r["x"]), float(r["y"]), float(r["z"])]
            if not np.all(np.isfinite(q + xyz)): continue                       # nonfinite_pose_invalid_v1
            T[i, :3, :3] = Rot.from_quat(q).as_matrix(); T[i, :3, 3] = xyz; ok[i] = True
    return T, ok


def T_b_c(setting):
    txt = pathlib.Path(setting).read_text(); m = re.search(r"IMU\.T_b_c1:.*?data:\s*\[([^\]]+)\]", txt, re.S)
    return np.array([float(v) for v in m.group(1).split(",")]).reshape(4, 4)


def imu_vi(exp, T, ok, W=W_VI):
    """cp6_scale.imu_vi (no bias), on an already-loaded CORI-indexed trajectory."""
    d = json.load(open(exp / "imu_data.json"))["1"]["streams"]
    t_cam = np.array([s["cts"] for s in d["CORI"]["samples"]]) / 1e3
    t_imu = np.array([s["cts"] for s in d["ACCL"]["samples"]]) / 1e3
    f = np.array([s["value"] for s in d["ACCL"]["samples"]], float); n = len(t_cam)
    Tcb = np.linalg.inv(T_b_c(exp / "orbslam_setting.yaml")); R_cb, t_cb = Tcb[:3, :3], Tcb[:3, 3]
    fb = np.full((n, 3), np.nan)
    for i, t in enumerate(t_cam):
        sel = np.abs(t_imu - t) <= 0.5 / 30
        if sel.any(): fb[i] = f[sel].mean(0)
    good = ok & np.isfinite(fb).all(1); idx = np.flatnonzero(good)
    if len(idx) < 3 * W: return None
    br = np.flatnonzero(np.diff(idx) != 1); r = max(np.split(idx, br + 1), key=len)
    if len(r) < 3 * W: return None
    dt = float(np.median(np.diff(t_cam[r])))
    Rwc = T[r, :3, :3]; pc = T[r, :3, 3]; Rwb = Rwc @ R_cb
    a_cam = savgol_filter(pc, W, 3, deriv=2, delta=dt, axis=0)
    a_lev = savgol_filter(np.einsum("nij,j->ni", Rwc, t_cb), W, 3, deriv=2, delta=dt, axis=0)
    Fw = savgol_filter(np.einsum("nij,nj->ni", Rwb, fb[r]), W, 3, deriv=0, axis=0)
    h = W // 2; sl = slice(h, len(r) - h)
    y = (Fw - a_lev)[sl].reshape(-1); cols = [a_cam[sl].reshape(-1)]
    for k in range(3):
        e = np.zeros((len(r), 3)); e[:, k] = -1.0; cols.append(e[sl].reshape(-1))
    A = np.stack(cols, 1); w = np.ones(len(y))
    for _ in range(10):
        xs = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        res = A @ xs - y; sc = 1.4826 * np.median(np.abs(res)) + 1e-9
        w = np.clip(1.345 * sc / np.maximum(np.abs(res), 1e-9), 0, 1)
    return dict(s=float(xs[0]), g_norm=float(np.linalg.norm(xs[1:4])), n=int(len(y) // 3), resid_rms=float(np.sqrt(np.mean(res ** 2))),
                run_frames=int(len(r)))


def read_mcap(ep):
    from mcap.reader import make_reader
    fm = {"head_depth": [], "left_wrist": [], "right_wrist": []}; gr = {"left": [], "right": []}
    with open(ep / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            t = ch.topic
            if t.endswith("/frame_meta"):
                s = t.split("/")[1]
                if s in fm: d = json.loads(msg.data); fm[s].append((int(d["video_frame"]), int(d["capture_ns"])))
            elif t.endswith("/gripper"):
                d = json.loads(msg.data); gr[d["side"]].append((int(d["sample_ns"]), int(d["raw_position_mod"])))
    return {k: np.array(sorted(v), np.int64).reshape(-1, 2) for k, v in fm.items()}, {k: np.array(sorted(v), np.int64).reshape(-1, 2) for k, v in gr.items()}


def calib_dir_arc(ep, side):
    """[2026-10-13] opening DIRECTION and arc from the gripper calibration active when the episode was recorded
    (episode_meta.gripper_calibration_version). The right unit was re-assembled at 12:21 on 10-13 and now opens towards
    DECREASING ticks (v018/v019: closed 37556 > open 36654); assuming +1 inverted every right label of UMIalign 13:15."""
    import yaml as _y
    try:
        ver = json.loads((pathlib.Path(ep) / "episode_meta.json").read_text()).get("gripper_calibration_version")
        c = _y.safe_load(open(pathlib.Path.home() / "ego_collector/configs/calibration" / f"{ver}.yaml"))["sides"][side]
        d = float(c["ticks_open"]) - float(c["ticks_closed"])
        if abs(d) > 2048: d = d - np.sign(d) * 4096                    # shortest way round (pre-v015 seam convention)
        return (1.0 if d > 0 else -1.0), abs(d), ver
    except Exception:
        return 1.0, float(GRIP_ARC[side]), None


def gripper_binary(samples, side, t_ns, ep=None):
    """-> (binary at t_ns, g_norm at t_ns, info). Direction + arc from the episode's calibration; closed = the 2nd percentile
    on the CLOSED side of the raw ticks (the jaw rests closed), so a servo-counter offset after a power cycle does not matter."""
    t, raw = samples[:, 0], samples[:, 1].astype(float)
    assert raw.max() - raw.min() < 2048, f"{side}: gripper raw crosses the encoder seam ({raw.min()}..{raw.max()})"
    sign, arc, ver = calib_dir_arc(ep, side) if ep is not None else (1.0, float(GRIP_ARC[side]), None)
    closed = float(np.percentile(sign * raw, 2)); g = np.clip((sign * raw - closed) / arc, 0, 1); closed *= sign
    j = np.clip(np.searchsorted(t, t_ns), 1, len(t) - 1); j = np.where(np.abs(t[j - 1] - t_ns) < np.abs(t[j] - t_ns), j - 1, j)
    gn = g[j]; b = (gn >= OPEN_FRAC_2CM).astype(np.float64)
    info = dict(closed_ticks_p2=closed, arc_ticks=arc, direction=sign, calibration=ver, raw_min=float(raw.min()), raw_max=float(raw.max()),
                open_frac_threshold=OPEN_FRAC_2CM, open_fraction_of_samples=round(float(b.mean()), 4))
    return b, gn, info


def export_episode(ep, out_root, scales, Xct, xct_path):
    sess = ep.parent.name; ts = sess.split("_", 1)[1]; eno = ep.name.split("_")[1]
    eid = f"hfold_{ts}_{eno}"
    from ego_cart20.sources.trim import trim_end_ns, trim_info
    fm, gr = read_mcap(ep); arrays, arms_meta = {}, {}; t_end = trim_end_ns(ep)         # [2026-10-14] SPACE-reach end trim
    for side in SIDES:
        tag = f"{ts}_{eno}_{side}"; exp = RL / "export" / tag; csvp = RL / "runs" / tag / "ss1.csv"
        if not csvp.exists(): raise FileNotFoundError(f"{tag}: no MASt3R ss1.csv yet")
        n = len(json.load(open(exp / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
        vf = fm[f"{side}_wrist"]; off = len(vf) - n
        assert off >= 0, f"{tag}: fewer video frames ({len(vf)}) than CORI samples ({n})"
        T, ok = load_traj(csvp, n)
        vi = imu_vi(exp, T, ok)
        if tag in scales:
            s = float(scales[tag]["s"]); src = scales[tag].get("source", "scale_json"); sv = np.isfinite(s) and s > 0
        else:
            s = vi["s"] if vi else float("nan"); src = "imu_vi_W15_nobias"
            sv = bool(vi and np.isfinite(s) and s > 0 and G_BAND[0] <= vi["g_norm"] <= G_BAND[1] and S_GUARD[0] <= s <= S_GUARD[1])
        Tm = T.copy(); Tm[:, :3, 3] *= (s if np.isfinite(s) else 1.0)
        Tt = Tm @ Xct @ X_TOOL
        valid = ok & bool(sv)
        t_ns = vf[off:off + n, 1]
        if t_end is not None:
            k = t_ns <= t_end; t_ns, Tt, valid = t_ns[k], Tt[k], valid[k]
        if len(gr[side]):
            b, gn, ginfo = gripper_binary(gr[side], side, t_ns, ep=ep); gv = np.ones(len(t_ns), bool)
        else:                                                                    # [2026-10-14] servos removed: red-tape jaw in the wrist video
            from ego_cart20.sources.jaw_visual import visual_gripper
            b, gv, ginfo = visual_gripper(ep, side, t_ns)
        arrays[f"{side}_gripper_valid"] = gv
        q = Rot.from_matrix(Tt[:, :3, :3]).as_quat()                            # xyzw
        arrays[f"{side}_t_ns"] = t_ns.astype(np.int64); arrays[f"{side}_position"] = Tt[:, :3, 3].astype(np.float64)
        arrays[f"{side}_quaternion"] = q[:, [3, 0, 1, 2]].astype(np.float64); arrays[f"{side}_valid"] = valid
        arrays[f"{side}_gripper"] = b
        arms_meta[side] = dict(tag=tag, n_cori=n, video_offset=int(off), mast3r_valid_frac=round(float(ok.mean()), 4), scale=s,
                               scale_source=src, scale_valid=bool(sv), imu_vi=vi, gripper=ginfo)
    for cam, key in (("head", "head_depth"), ("left_wrist", "left_wrist"), ("right_wrist", "right_wrist")):
        c = fm[key] if t_end is None else fm[key][fm[key][:, 1] <= t_end]
        arrays[f"cam_{cam}_t_ns"] = c[:, 1]; arrays[f"cam_{cam}_frame"] = c[:, 0]
    meta = dict(episode_id=eid, source="handumi_hfold", task="laundry_fold", instruction=INSTRUCTION, stack_order="FOLD", era=sess,
                clock="capture_monotonic_ns",
                videos={"head": str(ep / "head_depth.mp4"), "left_wrist": str(ep / "left_wrist.mp4"), "right_wrist": str(ep / "right_wrist.mp4")},
                tool_frame=dict(name="handumi TCP (camera_tcp_v2) @ X = F @ C @ CT4 (reBot dataset-TCP axes)", X=X_TOOL.tolist(), camera_tcp_v2=xct_path),
                gripper_calibration=dict(rule="per-episode p2 closed + calibrated arc; binary width<2cm (g_norm>=0.25) open=1", arms={k: v["gripper"] for k, v in arms_meta.items()}),
                trim=trim_info(ep),
                provenance=dict(episode_dir=str(ep), arms=arms_meta, exporter="ego_cart20/sources/hfold_export.py 2026-10-12"))
    from ego_cart20.io.raw_episode_loader import save_raw_episode
    save_raw_episode(pathlib.Path(out_root) / eid, meta, arrays)
    return dict(episode_id=eid, **{f"{k}_scale": round(v["scale"], 4) if np.isfinite(v["scale"]) else None for k, v in arms_meta.items()},
                **{f"{k}_scale_valid": v["scale_valid"] for k, v in arms_meta.items()},
                **{f"{k}_open_frac": v["gripper"]["open_fraction_of_samples"] for k, v in arms_meta.items()})


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--scale-json"); a = ap.parse_args()
    scales = json.load(open(a.scale_json)) if a.scale_json else {}
    Xct, xp = x_cam_tcp(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); res = []
    for s in a.sessions:
        for ep in sorted(pathlib.Path(s).glob("episode_*")):
            if not (ep / ".complete").exists(): continue
            try: r = export_episode(ep, out, scales, Xct, xp); r["status"] = "ok"
            except Exception as ex: r = dict(episode_id=f"{pathlib.Path(s).name}/{ep.name}", status="error", reason=f"{type(ex).__name__}: {ex}")
            print(json.dumps(r, default=float), flush=True); res.append(r)
    json.dump(res, open(out / "export_log.json", "w"), indent=1, default=float)
