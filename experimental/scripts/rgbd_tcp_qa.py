#!/usr/bin/env python3
"""[2026-09-18] QA items 7-9: the metric grasp-TCP track the RGB-D teacher sensor is supposed to produce.

    head RGB-D  ->  MediaPipe 2D + aligned depth  ->  metric palm pose  ->  camera_to_tcp_v1  ->  grasp TCP
                                                                            Feetech encoder   ->  grip

    7  grasp-TCP xyz trajectory after camera_to_tcp_v1
    8  displacement p50/p95/p99 at 30 fps, lead=5, chunk=30
    9  share of chunks displacing more than 0.5 m / 1.0 m (non-physical)

TWO THINGS THIS REPORT IS HONEST ABOUT, because both change how the numbers must be read:

  * **The head camera moves.** Everything here is expressed in the head-depth camera frame, and on a head mount that
    frame is not the world: every head rotation appears as hand motion. `palm_pose.py` states this as a load-bearing
    assumption of its own source ("the camera must not move"), and it is violated by design in this rig. Head odometry
    (Stage C) is not built. So items 8 and 9 measure hand motion PLUS head motion, and that is exactly what makes them
    worth running: a non-physical chunk share that is large here is the size of the head-motion problem, measured.

  * **camera_to_tcp_v1 is a WRIST-CAMERA -> TCP offset, and what depth gives us is a PALM frame.** The transform
    palm -> wrist camera has never been calibrated. Applying v1 on top of the palm pose (as instructed) therefore
    carries an unknown constant rigid offset in the hand frame. A constant offset in the hand frame does not affect a
    displacement over a chunk except through the hand's ROTATION during that chunk, so item 8/9 stay meaningful while
    item 7's absolute xyz does not. The report says so on every run rather than in a commit message.

    .venv/bin/python scripts/rgbd_tcp_qa.py --session <session_dir> [--sides left right] [--write-ego16]
    .venv/bin/python scripts/rgbd_tcp_qa.py --episode <episode_dir>

`--write-ego16` additionally writes EPISODE/derived/humanik/ego16.npz in the schema `scripts/ego_ik_retarget.py`
already consumes, so the IK / FK / joint-limit / smoothness checks run on this track with no new IK code.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handumi_collector.pose.rgbd_io import RgbdEpisode                        # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                      # noqa: E402

TCP_CAL = ROOT / "configs" / "calibration" / "camera_tcp" / "handumi_camera_tcp_v1.yaml"
FPS, LEAD, CHUNK = 30.0, 5, 30
NONPHYSICAL_M = (0.5, 1.0)


def load_camera_tcp(path: Path = TCP_CAL) -> dict:
    """camera_to_tcp_v1, carried with its provenance. A calibration used without its provenance is a magic number,
    and this one has a caveat that has to travel with it (rotation was never measured)."""
    d = yaml.safe_load(path.read_text())
    if d.get("version") != "camera_to_tcp_v1":
        raise ValueError(f"{path}: expected version camera_to_tcp_v1, got {d.get('version')!r}")
    out = {"file": str(path), "version": d["version"], "created": d.get("created"),
           "frame_convention": d.get("frame_convention"), "provenance": d.get("provenance"), "sides": {}}
    for side, v in d["camera_to_grasp_tcp"].items():
        out["sides"][side] = dict(position=np.asarray(v["position"], np.float64),
                                  quaternion=np.asarray(v["quaternion"], np.float64), note=v.get("note", ""))
    return out


def head_calibration(ep: RgbdEpisode):
    """The episode's own intrinsics, as the hand3d stack wants them. Never a default: a guessed focal length is a
    silent metric-scale error in every label, which is the one failure this whole pilot exists to avoid."""
    from ego_teleop.hand3d.depth import CameraIntrinsics as HCI
    from ego_teleop.hand3d.head_camera import HeadRgbdCalibration
    k = ep.intrinsics
    if not (k.fx and k.fy):
        raise ValueError(f"{ep.path}: no device intrinsics -- refuse to produce metric labels on a guessed scale")
    ci = HCI(fx=k.fx, fy=k.fy, cx=k.cx, cy=k.cy, width=k.width, height=k.height)
    return HeadRgbdCalibration(color=ci, depth=ci, depth_scale_m=k.depth_unit_m, aligned=bool(k.aligned_to_rgb),
                               fps=ep.fps, model="orbbec_gemini_336", notes=f"from {k.source}")


def palm_track(ep: RgbdEpisode, side: str, calib) -> dict:
    """One decode pass -> per-frame palm position and basis in the head-depth camera frame."""
    from ego_teleop.hand3d.head_camera import RgbdFrame as HandFrame
    from ego_teleop.hand3d.palm_pose import RgbdPalmPoseEstimator, PalmPoseConfig
    from ego_teleop.hand3d.providers import RgbdHandPoseProvider, HandPoseProviderConfig

    prov = RgbdHandPoseProvider(HandPoseProviderConfig(side=side, selfie_mirrored=False))
    est = RgbdPalmPoseEstimator(prov, PalmPoseConfig())
    n = ep.n_frames
    P = np.full((n, 3), np.nan); R = np.full((n, 3, 3), np.nan); ok = np.zeros(n, bool)
    health = []
    for f in ep.iter_frames(0, n, 1):
        hf = HandFrame(timestamp_ns=int(f.t_ns), color_bgr=f.rgb, depth_m=f.depth_m, calib=calib, frame_index=f.index)
        est.push_image(int(f.t_ns), f.index, hf)
        p = est.last_palm
        health.append(p.health.value if p is not None else "lost")
        if p is not None and p.position_m is not None:
            P[f.index] = p.position_m
            ok[f.index] = True
            if p.R_palm is not None:
                R[f.index] = p.R_palm
    return dict(position=P, R=R, valid=ok, health=health, counters=dict(est.counters), reasons=dict(est.reasons))


def apply_camera_tcp(P: np.ndarray, R: np.ndarray, valid: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """T_world_tcp = T_world_camera @ T_camera_tcp, with the offset expressed in the camera's own frame.

    The offset is rotated by the frame's basis before it is added -- adding it in the camera frame instead would make
    the TCP a rigid copy of the palm and quietly delete every rotation-induced lever-arm motion, which is a large part
    of what a grasp TCP actually does."""
    T = np.full_like(P, np.nan)
    for i in np.nonzero(valid)[0]:
        Ri = R[i]
        T[i] = P[i] + (Ri @ offset if np.all(np.isfinite(Ri)) else offset)
    return T


def displacement_stats(T: np.ndarray, valid: np.ndarray) -> dict:
    """Item 8/9. Two different distances, because they answer different questions.

    `step`  frame to frame: the size of the increment a policy has to emit, and the thing a single bad depth sample
            corrupts.
    `chunk` from the lead frame to the end of the chunk: how far the EEF travels over the 1 s the action chunk covers.
            This is what "non-physical" is judged on -- a hand does not cross a metre of table in one second."""
    n = len(T)
    fin = valid & np.all(np.isfinite(T), 1)
    step = np.full(n - 1, np.nan)
    both = fin[:-1] & fin[1:]
    step[both] = np.linalg.norm(np.diff(T, axis=0)[both], axis=1)

    starts, spans = [], []
    for t in range(n - (LEAD + CHUNK)):
        a, b = t + LEAD, t + LEAD + CHUNK - 1
        if fin[a] and fin[b]:
            starts.append(t); spans.append(float(np.linalg.norm(T[b] - T[a])))
    spans = np.asarray(spans)

    def q(a, p):
        a = np.asarray(a); a = a[np.isfinite(a)]
        return float(np.percentile(a, p)) if a.size else float("nan")

    out = dict(
        n_frames=n, n_valid=int(fin.sum()), valid_share=float(fin.mean()),
        step_mm_p50=q(step, 50) * 1e3, step_mm_p95=q(step, 95) * 1e3, step_mm_p99=q(step, 99) * 1e3,
        step_mm_max=float(np.nanmax(step)) * 1e3 if np.isfinite(step).any() else float("nan"),
        n_chunks=int(spans.size),
        chunk_mm_p50=q(spans, 50) * 1e3, chunk_mm_p95=q(spans, 95) * 1e3, chunk_mm_p99=q(spans, 99) * 1e3,
        chunk_mm_max=float(spans.max()) * 1e3 if spans.size else float("nan"),
    )
    for lim in NONPHYSICAL_M:                                        # item 9
        out[f"chunks_over_{lim}m"] = int((spans > lim).sum()) if spans.size else 0
        out[f"chunks_over_{lim}m_share"] = float((spans > lim).mean()) if spans.size else 0.0
    # the same question asked of a single frame: a 30 fps hand cannot jump 10 cm between frames, so these are sensor
    # failures rather than fast motion, and they are what a median-filter stage would have to remove
    out["steps_over_50mm"] = int(np.nansum(step > 0.05))
    out["steps_over_100mm"] = int(np.nansum(step > 0.10))
    return out


def trajectory_summary(T: np.ndarray, valid: np.ndarray) -> dict:
    """Item 7. Absolute xyz is reported with its caveat (see the module docstring); extent and path length are what
    can be read without the missing palm->camera transform."""
    fin = valid & np.all(np.isfinite(T), 1)
    if fin.sum() < 2:
        return dict(n_valid=int(fin.sum()), note="too few valid frames for a trajectory")
    Q = T[fin]
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    return dict(n_valid=int(fin.sum()), valid_share=float(fin.mean()),
                x_range_mm=[float(Q[:, 0].min()) * 1e3, float(Q[:, 0].max()) * 1e3],
                y_range_mm=[float(Q[:, 1].min()) * 1e3, float(Q[:, 1].max()) * 1e3],
                z_range_mm=[float(Q[:, 2].min()) * 1e3, float(Q[:, 2].max()) * 1e3],
                bbox_diag_mm=float(np.linalg.norm(Q.max(0) - Q.min(0))) * 1e3,
                path_length_mm=float(d.sum()) * 1e3,
                mean_speed_mm_s=float(d.sum() / (len(Q) - 1) * FPS) * 1e3)


def grip_on_frames(ep_path: Path, t_ns: np.ndarray) -> dict:
    """Grip resampled onto the depth timeline, nearest sample within half a frame. Never interpolated across a gap:
    a grip label is a state, and a made-up value between two real ones is a state that never happened."""
    raw = RawEpisode.load(ep_path)
    out = {}
    for side, g in (raw.grip or {}).items():
        gt = np.asarray(g.t_ns, np.int64); gv = np.asarray(g.normalized, np.float64)
        if not len(gt):
            out[side] = dict(value=np.full(len(t_ns), np.nan), valid=np.zeros(len(t_ns), bool)); continue
        k = np.searchsorted(gt, t_ns).clip(1, len(gt) - 1)
        pick = np.where(np.abs(t_ns - gt[k - 1]) <= np.abs(gt[k] - t_ns), k - 1, k)
        gap = np.abs(t_ns - gt[pick])
        v = gv[pick]; ok = (gap <= int(0.5e9 / FPS)) & np.isfinite(v)
        out[side] = dict(value=np.where(ok, v, np.nan), valid=ok)
    return out


def write_ego16(ep_path: Path, t_ns, tracks: dict, grips: dict) -> Path:
    """The exact schema scripts/ego_ik_retarget.py reads, so the IK / FK / posture / smoothness checks need no new code.

    S16 = [L xyz qxyzw grip | R xyz qxyzw grip]. Poses stay in the HEAD-DEPTH CAMERA frame -- there is no robot-base
    alignment here, because the anchor ego16.py uses (the GO-frame gravity + optical axis) needs an IMU this profile
    does not have. The IK numbers are therefore about the TRACK's kinematic plausibility, not about a robot-ready
    trajectory, and meta.json says so."""
    from scipy.spatial.transform import Rotation
    n = len(t_ns)
    S = np.zeros((n, 16), np.float32); valid = {}
    for j, side in enumerate(("left", "right")):
        o = j * 8
        tr = tracks.get(side)
        if tr is None:
            S[:, o + 6] = 1.0; valid[side] = np.zeros(n, bool); continue
        T, R, ok = tr["tcp"], tr["R"], tr["valid"]
        S[:, o:o + 3] = np.nan_to_num(T, nan=0.0)
        good_R = ok & np.all(np.isfinite(R.reshape(n, 9)), 1)
        q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        if good_R.any():
            q[good_R] = Rotation.from_matrix(R[good_R]).as_quat()      # xyzw
        S[:, o + 3:o + 7] = q
        g = grips.get(side, {})
        S[:, o + 7] = np.nan_to_num(g.get("value", np.full(n, np.nan)), nan=0.0)
        valid[side] = ok & np.all(np.isfinite(T), 1)
    d = ep_path / "derived" / "humanik"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "ego16.npz", S16=S, valid_L=valid["left"], valid_R=valid["right"],
             grip_valid=np.stack([grips.get("left", {}).get("valid", np.zeros(n, bool)),
                                  grips.get("right", {}).get("valid", np.zeros(n, bool))], 1),
             t_ns=np.asarray(t_ns, np.int64), go_idx=0, stop_idx=n - 1)
    (d / "ego16_meta.json").write_text(json.dumps(dict(
        source="rgbd_tcp_qa.py", pose_source="head_rgbd_palm + camera_to_tcp_v1",
        frame="head-depth camera frame (MOVING: head-mounted, no head odometry)",
        rotation="palm basis from MediaPipe landmarks; camera_to_tcp_v1 carries no rotation",
        caveat="palm -> wrist-camera transform is UNCALIBRATED; absolute xyz carries an unknown constant hand-frame offset",
        fps=FPS, lead=LEAD, chunk=CHUNK), indent=1))
    return d / "ego16.npz"


def qa_episode(ep_path: Path, *, sides: list[str], stream: str | None, cal: dict, write: bool) -> dict:
    ep = RgbdEpisode.load(ep_path, stream=stream)
    calib = head_calibration(ep)
    grips = grip_on_frames(ep_path, ep.t_ns)
    r = {"episode": ep_path.name, "path": str(ep_path), "n_frames": ep.n_frames, "fps": ep.fps, "sides": {}}
    tracks = {}
    for side in sides:
        pt = palm_track(ep, side, calib)
        off = cal["sides"][side]["position"]
        tcp = apply_camera_tcp(pt["position"], pt["R"], pt["valid"], off)
        tracks[side] = dict(tcp=tcp, R=pt["R"], valid=pt["valid"])
        r["sides"][side] = dict(
            camera_to_tcp_mm=[float(x) * 1e3 for x in off],
            palm_health=pt["counters"], lost_reasons=dict(sorted(pt["reasons"].items(), key=lambda kv: -kv[1])[:6]),
            trajectory=trajectory_summary(tcp, pt["valid"]),
            displacement=displacement_stats(tcp, pt["valid"]),
            grip_valid_share=float(grips.get(side, {}).get("valid", np.zeros(1)).mean()),
        )
    if write:
        r["ego16"] = str(write_ego16(ep_path, ep.t_ns, tracks, grips))
    return r


def print_report(rows: list[dict], cal: dict) -> None:
    print("\n" + "=" * 112)
    print("RGB-D GRASP-TCP QA  --  items 7 (trajectory), 8 (chunk displacement), 9 (non-physical chunks)")
    print("=" * 112)
    print(f"\ncalibration  {cal['version']}  ({cal['file']})")
    print(f"  created    {cal['created']}")
    for s, v in cal["sides"].items():
        print(f"  {s:>5s}      [{v['position'][0]*1e3:+7.1f}, {v['position'][1]*1e3:+7.1f}, {v['position'][2]*1e3:+7.1f}] mm")
    p = cal.get("provenance") or {}
    print(f"  provenance {p.get('method','?')[:100]}")
    print(f"             accepted frames {p.get('accepted_frames')}  of raw {p.get('raw_detections')}  cube {p.get('cube_mm')} mm")
    print(f"  CAVEAT     {p.get('caveat','')}")
    print("\n  READ THIS BEFORE THE NUMBERS")
    print("    * poses are in the HEAD-DEPTH CAMERA frame, and the head moves -> items 8/9 contain head motion")
    print("    * camera_to_tcp_v1 is a WRIST-CAMERA offset applied to a PALM frame; palm->wrist-camera is uncalibrated,")
    print("      so absolute xyz (item 7) carries an unknown constant hand-frame offset. Displacements are unaffected")
    print("      except through hand rotation within the chunk.")

    print("\n[7] grasp-TCP trajectory")
    print(f"{'episode':>14s} {'side':>6s} {'valid':>7s} {'bbox diag':>10s} {'path':>10s} {'speed':>11s}  x/y/z range mm")
    for r in rows:
        for side, s in r["sides"].items():
            t = s["trajectory"]
            if "note" in t:
                print(f"{r['episode'][-11:]:>14s} {side:>6s} {t['n_valid']:7d}  {t['note']}"); continue
            print(f"{r['episode'][-11:]:>14s} {side:>6s} {t['valid_share']*100:6.1f}% {t['bbox_diag_mm']:9.0f}mm "
                  f"{t['path_length_mm']:9.0f}mm {t['mean_speed_mm_s']:8.0f}mm/s  "
                  f"[{t['x_range_mm'][0]:.0f},{t['x_range_mm'][1]:.0f}] "
                  f"[{t['y_range_mm'][0]:.0f},{t['y_range_mm'][1]:.0f}] "
                  f"[{t['z_range_mm'][0]:.0f},{t['z_range_mm'][1]:.0f}]")

    print(f"\n[8] displacement, 30 fps lead={LEAD} chunk={CHUNK}   (step = frame to frame, chunk = span of the action chunk)")
    print(f"{'episode':>14s} {'side':>6s} {'step p50':>9s} {'p95':>8s} {'p99':>8s} {'max':>8s} | "
          f"{'chunk p50':>10s} {'p95':>8s} {'p99':>8s} {'max':>8s} {'n':>6s}")
    for r in rows:
        for side, s in r["sides"].items():
            d = s["displacement"]
            print(f"{r['episode'][-11:]:>14s} {side:>6s} {d['step_mm_p50']:8.1f}m {d['step_mm_p95']:7.1f} {d['step_mm_p99']:7.1f} "
                  f"{d['step_mm_max']:7.1f} | {d['chunk_mm_p50']:9.1f} {d['chunk_mm_p95']:7.1f} {d['chunk_mm_p99']:7.1f} "
                  f"{d['chunk_mm_max']:7.1f} {d['n_chunks']:6d}")

    print("\n[9] non-physical chunks and frame-to-frame jumps")
    print(f"{'episode':>14s} {'side':>6s} {'>0.5 m':>14s} {'>1.0 m':>14s} {'steps>50mm':>11s} {'steps>100mm':>12s}")
    tot = {0.5: 0, 1.0: 0, "n": 0}
    for r in rows:
        for side, s in r["sides"].items():
            d = s["displacement"]
            tot[0.5] += d["chunks_over_0.5m"]; tot[1.0] += d["chunks_over_1.0m"]; tot["n"] += d["n_chunks"]
            print(f"{r['episode'][-11:]:>14s} {side:>6s} {d['chunks_over_0.5m']:6d} ({d['chunks_over_0.5m_share']*100:5.2f}%) "
                  f"{d['chunks_over_1.0m']:6d} ({d['chunks_over_1.0m_share']*100:5.2f}%) {d['steps_over_50mm']:11d} {d['steps_over_100mm']:12d}")
    if tot["n"]:
        print(f"\n  overall  {tot[0.5]}/{tot['n']} chunks over 0.5 m ({tot[0.5]/tot['n']*100:.2f}%), "
              f"{tot[1.0]}/{tot['n']} over 1.0 m ({tot[1.0]/tot['n']*100:.2f}%)")

    print("\n[palm tracking health]")
    for r in rows:
        for side, s in r["sides"].items():
            h = s["palm_health"]; n = max(sum(h.values()), 1)
            why = ", ".join(f"{k}={v}" for k, v in list(s["lost_reasons"].items())[:4])
            print(f"  {r['episode']:>18s} {side:>6s}  ok {h.get('ARM_POSE_OK',0)/n*100:5.1f}%"
                  f"  degraded {h.get('ARM_POSE_DEGRADED',0)/n*100:5.1f}%"
                  f"  lost {h.get('ARM_POSE_LOST',0)/n*100:5.1f}%   grip {s['grip_valid_share']*100:5.1f}%   {why}")

    print("\nNEXT: --write-ego16 then\n"
          "      .venv/bin/python scripts/ego_ik_retarget.py <episode> ...\n"
          "      for IK success rate, FK reconstruction error, branch switch, joint-limit occupancy, dq smoothness\n"
          "      and velocity/acceleration spikes on 2-3 of these episodes.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path)
    g.add_argument("--episode", type=Path)
    ap.add_argument("--sides", nargs="+", default=["left", "right"], choices=["left", "right"])
    ap.add_argument("--stream", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes of the session")
    ap.add_argument("--write-ego16", action="store_true", help="also write derived/humanik/ego16.npz for the IK checks")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    cal = load_camera_tcp()
    eps = [a.episode] if a.episode else sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if a.limit:
        eps = eps[:a.limit]
    if not eps:
        print("no episodes"); return 1
    rows = []
    for p in eps:
        try:
            rows.append(qa_episode(p, sides=a.sides, stream=a.stream, cal=cal, write=a.write_ego16))
            print(f"  tracked {p.name}", flush=True)
        except Exception as exc:
            print(f"  {p.name}: FAILED -- {exc}", flush=True)
    if rows:
        print_report(rows, cal)
    if a.json and rows:
        a.json.write_text(json.dumps(rows, indent=1, default=float)); print(f"\nfull record -> {a.json}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
