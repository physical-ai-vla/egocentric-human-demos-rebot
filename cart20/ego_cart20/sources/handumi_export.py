#!/usr/bin/env python3
"""HandUMI (old259 Hpilot + HRL80) -> raw ego episodes in the ego_cart20 raw format.  Source adapter ONLY: it reads the
frozen C8 pose stack and writes poses/gripper/timestamps; no labels are made here.

Per arm (its own wrist clock -- NOT paired to the other arm's frames):
  pose      c8_phase3.side (old259) / robotlike_offline_check.side_nogrip (HRL80): MASt3R-SLAM + IMU-VI metric scale +
            handumi_camera_tcp_v2 -> human TCP; then right-multiplied by the fixed tool frame X = F @ C @ CT4 so the hand TCP
            axes follow the reBot dataset-TCP convention (the same conjugation as ego_relcart20task: CT4^-1 C^-1 F^-1 rel F C CT4).
  valid     the stack's ok mask (lost / NaN pose invalid); the whole arm is refused if its IMU-VI scale is not valid
  t_ns      wrist frame capture_ns (sensors.mcap frame_meta) of each CORI pose sample
  gripper   gcal1 aperture (umi_aperture_cal_v1, caliper): g = G_era,side(open_fraction), 0 = closed, 1 = open; the SAME
            function object as rel16ego_derive_gcal1.G (imported, not re-implemented)
Cameras: head / left_wrist / right_wrist (capture_ns, video_frame) from frame_meta; mp4 paths recorded, never copied.

usage (env ~/xvla-mac/bin/python):
  handumi_export.py --out ~/c8/ego_cart20_v2_raw [--episodes ID ...] [--workers 4]
"""
import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

H = pathlib.Path.home(); C8 = H / "c8"; V2K = H / "umi_bridge/track_c/v2k"; RL = C8 / "robotlike"
RAW_OLD = H / "ego_collector/datasets/human_handumi_raw/Hpilot"; RAW_HRL = H / "ego_collector/datasets/human_handumi_raw/HRL80"
HRL_SESSION = "20260928_101010"
SIDES = ("left", "right")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _init():
    global V, P3, RLC, P, X, GCAL, OLD_RUNS, OLD_EXPORT
    sys.path[:0] = [str(C8), str(C8 / "c8old"), str(C8 / "rel16ego/rebot_ee"), str(H / "umi_bridge/umi76"), str(H / "umi_bridge")]
    os.environ.setdefault("REBOT_URDF", str(C8 / "rel16ego/rebot_ee/reBot_B601_DM_dualarm.urdf"))
    V = _load("v2k", V2K / "v2k_retarget.py"); RLC, P3 = V.RLC, V.P3; P3.init(); P = P3.P; V.P = P
    CT4 = np.eye(4); CT4[:3, :3] = P.C[:3, :3].T
    X = np.asarray(P.F, np.float64) @ np.asarray(P.C, np.float64) @ CT4
    GCAL = _load("gcal1", C8 / "rel16ego/rel16ego_derive_gcal1.py")
    OLD_RUNS, OLD_EXPORT = P3.RUNS, P3.export_dir


def source_list():
    """inclusion rule (fixed before any CART20 result): old = phase3 census first_fail in {None, no_segment} with BOTH
    IMU-VI scales valid; HRL80 = every episode of the frozen HRL80 split.  No workspace / IK / pseudo-q selection."""
    cen = json.load(open(C8 / "phase3_census.json"))["episodes"]
    old = sorted(e for e, v in cen.items() if v.get("first_fail") in (None, "no_segment")
                 and v.get("left", {}).get("scale_valid") and v.get("right", {}).get("scale_valid"))
    sp = json.load(open(V2K / "hrl80_split_frozen.json"))
    return old + sorted(sp["train_source_episodes"] + sp["val_source_episodes"])


def _episode_dir(e):
    sess, num = e.rsplit("_", 1)
    return (RAW_HRL / f"HRL80_{sess}" if sess == HRL_SESSION else RAW_OLD / f"Hpilot_{sess}") / f"episode_{num}"


def export_one(e, out_root):
    out = pathlib.Path(out_root) / e
    if (out / "raw_episode.json").exists(): return dict(episode=e, status="exists")
    epd = _episode_dir(e); hrl = e.startswith(HRL_SESSION)
    if hrl:
        P3.RUNS = RL / "runs"; P3.export_dir = lambda tag: RL / "export" / tag
        S = {s: RLC.side_nogrip(RLC.tag_of(epd, s), epd, s) for s in SIDES}; era = "HRL80_v014"
    else:
        P3.RUNS, P3.export_dir = OLD_RUNS, OLD_EXPORT
        S = {s: P3.side(f"{e}_{s}") for s in SIDES}; era = "old259_contract"
    bad = [s for s in SIDES if not S[s]["scale_valid"]]
    if bad: return dict(episode=e, status="refused", reason=f"IMU-VI scale invalid: {bad}")
    fm = P3.frame_meta(epd); arrays = {}; grip_src = {}
    for s in SIDES:
        d = S[s]; off, n = d["off"], d["n"]; fr = fm[f"{s}_wrist"][off:off + n]; assert len(fr) == n, (e, s, len(fr), n)
        vf = np.array([v for v, _ in fr], np.int64); t_ns = np.array([c for _, c in fr], np.int64)
        of = _hrl_grip(epd, s, vf) if hrl else np.load(C8 / "grip" / f"{e}_{s}.npz")["open_fraction"].astype(np.float64)
        assert len(of) == n
        g = GCAL.G(np.clip(of, 0, 1), s[0].upper(), era)
        T = np.asarray(d["T_tcp"], np.float64) @ X[None]; ok = np.asarray(d["ok"], bool) & np.isfinite(g)
        from ego_cart20.geometry.rotation6d import matrix_to_quaternion
        arrays.update({f"{s}_t_ns": t_ns, f"{s}_position": T[:, :3, 3], f"{s}_quaternion": matrix_to_quaternion(T[:, :3, :3]),
                       f"{s}_valid": ok, f"{s}_gripper": g, f"{s}_open_fraction_raw": of, f"{s}_video_frame": vf})
        grip_src[s] = dict(scale=float(d["s"]), coverage=float(ok.mean()))
    for c in ("head", "left_wrist", "right_wrist"):
        arrays[f"cam_{c}_t_ns"] = np.array([t for _, t in fm[c]], np.int64); arrays[f"cam_{c}_frame"] = np.array([v for v, _ in fm[c]], np.int64)
    em = json.load(open(epd / "episode_meta.json"))
    meta = dict(schema="ego_cart20_raw_episode/v1", episode_id=e, source="human_egocentric_handumi", era=era, task="cube_stacking",
                instruction=em["instruction"], stack_order=em["order"], episode_status=em.get("status"), clock="capture_monotonic_ns",
                videos={c: str(epd / f"{c}.mp4") for c in ("head", "left_wrist", "right_wrist")},
                tool_frame=dict(name="handumi_tcp_v2 @ F @ C @ CT4 (reBot dataset-TCP axes)", X=X.tolist()),
                gripper_calibration=dict(version="umi_aperture_cal_v1", era_map=era, convention="0 = closed, 1 = open, mm / 80 mm",
                                         source="old: ~/c8/grip/<tag>.npz open_fraction" if not hrl else "HRL80: sensors.mcap /<side>/gripper normalized via extract_grip.align"),
                provenance=dict(raw_dir=str(epd), pose_stack="c8_phase3.side" if not hrl else "robotlike_offline_check.side_nogrip",
                                arms=grip_src, exporter=str(pathlib.Path(__file__).resolve()), exported=time.strftime("%Y-%m-%d %H:%M:%S")))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from ego_cart20.io.raw_episode_loader import save_raw_episode
    save_raw_episode(out, meta, arrays)
    return dict(episode=e, status="ok", era=era, n={s: int(S[s]["n"]) for s in SIDES})


def _hrl_grip(epd, side, vframes):
    """c8oldv2_build_hrl80.grip_open, verbatim logic (that module runs its build at import time, so it cannot be imported)"""
    from extract_grip import align
    from mcap.reader import make_reader
    fr, gp = [], []
    with open(epd / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=[f"/{side}_wrist/frame_meta", f"/{side}/gripper"]):
            d = json.loads(msg.data)
            if ch.topic.endswith("frame_meta"): fr.append((int(d["video_frame"]), int(d["capture_ns"])))
            elif d.get("normalized") is not None: gp.append((int(d["sample_ns"]), float(d["normalized"]), int(d.get("raw_position", 0)), 0.0))
    al = align(fr, gp); lut = dict(zip(al["video_frame"].tolist(), range(len(al["video_frame"]))))
    ix = np.array([lut[int(v)] for v in vframes])
    return np.clip(al["grip01"][ix], 0, 1).astype(np.float64)


def _worker(args):
    e, out = args
    try: return export_one(e, out)
    except Exception as ex: return dict(episode=e, status="error", reason=f"{type(ex).__name__}: {ex}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--workers", type=int, default=1); a = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    eps = a.episodes or source_list(); pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    print(f"export {len(eps)} episodes -> {a.out}", flush=True)
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers, initializer=_init) as ex: res = list(ex.map(_worker, [(e, a.out) for e in eps]))
    else:
        _init(); res = []
        for e in eps: res.append(_worker((e, a.out))); print(json.dumps(res[-1]), flush=True)
    log = pathlib.Path(a.out) / f"export_log_{time.strftime('%Y%m%d_%H%M%S')}.json"; json.dump(res, open(log, "w"), indent=1)
    import collections; print("DONE", dict(collections.Counter(r["status"] for r in res)), "->", log)
