#!/usr/bin/env python3
"""[2026-09-18] HandUMI body track -> ego16.npz, so the existing continuity IK runs on the RGB-D translation source.

    body centroid (head-camera frame)  ->  anchor to the reBot home TCP  ->  ego16.npz  ->  ego_ik_retarget.py

Three choices, each forced by what this profile actually has:

  **Orientation is the robot's home quaternion, held fixed.** The rotation component of `camera_to_tcp_v1` was never
  measured and the body's PCA axes are not yet trusted as an orientation, so inventing one would make the IK numbers
  describe a fiction. This is exactly `ego16.py --orientation fixed`, the documented phase-1 arm-only mode.

  **`camera_to_tcp_v1` is applied as a constant offset, and at this stage that is a no-op by construction.** With no
  body orientation the offset cannot be rotated into the world, so it is a rigid translation of the whole trajectory:
  it moves absolute position and cannot change a single step, chunk, IK or smoothness statistic. It is applied anyway,
  with its provenance, so the plumbing is the same one orientation will arrive into.

  **The table plane replaces IMU gravity in the alignment.** `ego16.py` builds `R_align` from the first IMU static
  window; this profile has `imus: []`. The table normal is a better substitute than a guess and is measured on every
  episode (plane rms 2.1-2.5 mm): robot +z = the table normal, robot +x = the camera optical axis projected onto the
  table, which is where the operator -- and so the robot -- faces.

    .venv/bin/python scripts/body_to_ego16.py --session <session> [--limit 8]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from handumi_collector.pose.rgbd_io import RgbdEpisode                      # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                    # noqa: E402
import handumi_body_track as B                                              # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "handumi" / "retarget_shakedown.yaml").read_text())
TCP_CAL = ROOT / "configs" / "calibration" / "camera_tcp" / "handumi_camera_tcp_v1.yaml"
FPS = 30.0


def align_from_plane(nrm: np.ndarray) -> np.ndarray:
    """Rows are the robot axes expressed in camera coordinates, so `R_align @ v_camera` is v in the robot frame."""
    z = nrm / np.linalg.norm(nrm)                       # table normal = robot up
    optical = np.array([0.0, 0.0, 1.0])                 # camera looks down +z; the operator faces that way
    x = optical - (optical @ z) * z
    n = np.linalg.norm(x)
    if n < 1e-6:                                        # camera pointing straight down: any horizontal axis will do
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        n = np.linalg.norm(x)
    x /= n
    y = np.cross(z, x)
    x = np.cross(y, z)                                  # re-orthogonalise, as the anchor rule specifies
    R = np.stack([x, y, z])
    # A mirrored or upside-down frame poisons every label downstream and is invisible in the numbers, so it is
    # asserted here rather than hoped for.
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, f"R_align is not a rotation (det {np.linalg.det(R):.6f})"
    assert np.allclose(R @ z, [0, 0, 1], atol=1e-6), "table normal does not map to world +Z"
    return R


def grip_on_frames(ep_path: Path, t_ns: np.ndarray) -> dict:
    raw = RawEpisode.load(ep_path)
    out = {}
    for side, g in (raw.grip or {}).items():
        gt = np.asarray(g.t_ns, np.int64); gv = np.asarray(g.normalized, np.float64)
        if not len(gt):
            out[side] = dict(value=np.full(len(t_ns), np.nan), valid=np.zeros(len(t_ns), bool)); continue
        k = np.searchsorted(gt, t_ns).clip(1, len(gt) - 1)
        pick = np.where(np.abs(t_ns - gt[k - 1]) <= np.abs(gt[k] - t_ns), k - 1, k)
        ok = (np.abs(t_ns - gt[pick]) <= int(0.5e9 / FPS)) & np.isfinite(gv[pick])
        out[side] = dict(value=np.where(ok, gv[pick], np.nan), valid=ok)
    return out


def convert(ep_path: Path, stream: str | None, cal: dict, apply_tcp: bool = False) -> dict:
    ep = RgbdEpisode.load(ep_path, stream=stream)
    nrm, pd, plane = B.fit_table_plane(ep)
    R_align = align_from_plane(nrm)
    K = ep.intrinsics.K
    n = ep.n_frames

    xyz = {s: np.full((n, 3), np.nan) for s in ("left", "right")}
    valid = {s: np.zeros(n, bool) for s in ("left", "right")}
    tracks: dict = {}
    for f in ep.iter_frames(0, n, 1):
        dets = B.detect_bodies(f, K, nrm, pd)
        a = B.associate(tracks, dets, f.rgb.shape[1])
        B.update_tracks(tracks, dets, a)
        for side in ("left", "right"):
            if side in a:
                xyz[side][f.index] = dets[a[side]]["xyz"]; valid[side][f.index] = True

    home = {s: np.asarray(CFG["robot_home_tcp"][s], float) for s in ("left", "right")}
    grips = grip_on_frames(ep_path, ep.t_ns)
    S = np.zeros((n, 16), np.float32)
    out_valid = {}
    for j, side in enumerate(("left", "right")):
        o = j * 8
        P, v = xyz[side], valid[side]
        S[:, o + 3:o + 7] = home[side][3:7]                         # orientation held at the robot home quaternion
        if v.any():
            go = int(np.nonzero(v)[0][0])                           # GO = first frame the body was seen
            off = cal["sides"][side]["position"] if apply_tcp else np.zeros(3)
            rel = (P - P[go]) @ R_align.T
            S[:, o:o + 3] = np.where(v[:, None], home[side][:3] + rel + off, home[side][:3])
        else:
            S[:, o:o + 3] = home[side][:3]
        g = grips.get(side, {})
        S[:, o + 7] = np.nan_to_num(g.get("value", np.full(n, np.nan)), nan=0.0)
        out_valid[side] = v

    d = ep_path / "derived" / "humanik"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "ego16.npz", S16=S, valid_L=out_valid["left"], valid_R=out_valid["right"],
             grip_valid=np.stack([grips.get("left", {}).get("valid", np.zeros(n, bool)),
                                  grips.get("right", {}).get("valid", np.zeros(n, bool))], 1),
             t_ns=np.asarray(ep.t_ns, np.int64), go_idx=0, stop_idx=n - 1)
    (d / "ego16_meta.json").write_text(json.dumps(dict(
        source="body_to_ego16.py", pose_source="head RGB-D HandUMI body centroid",
        orientation="robot home quaternion, FIXED (body orientation not estimated; camera_to_tcp_v1 rotation unmeasured)",
        camera_to_tcp="applied as a constant offset -- a no-op for every displacement statistic at this stage",
        align="robot +z = table-plane normal (IMU gravity substitute; this profile has imus: []), "
              "+x = camera optical axis projected onto the table",
        plane=plane, anchor="reBot home TCP at the first frame the body was seen",
        camera_to_tcp_applied=bool(apply_tcp),
        R_align=R_align.round(6).tolist(), R_align_det=float(np.linalg.det(R_align)),
        table_normal_camera=nrm.round(6).tolist(),
        table_normal_in_world=(R_align @ nrm).round(6).tolist(),
        coverage={s: float(out_valid[s].mean()) for s in out_valid}), indent=1, default=float))
    return dict(episode=ep_path.name, plane=plane,
                coverage={s: float(out_valid[s].mean()) for s in out_valid}, path=str(d / "ego16.npz"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path); g.add_argument("--episode", type=Path)
    ap.add_argument("--stream", default=None); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--apply-camera-tcp", action="store_true",
                    help="add camera_to_tcp_v1 as a constant offset. OFF by default: v1 has no rotation, so it is not "
                         "a full SE(3) calibration and must not be treated as one. It cannot change any relative-motion "
                         "statistic anyway -- see the module docstring.")
    a = ap.parse_args()
    from rgbd_tcp_qa import load_camera_tcp
    cal = load_camera_tcp(TCP_CAL)
    eps = [a.episode] if a.episode else sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if a.limit:
        eps = eps[:a.limit]
    for p in eps:
        try:
            r = convert(p, a.stream, cal, a.apply_camera_tcp)
            print(f"  {r['episode']}  coverage L {r['coverage']['left']*100:.1f}% R {r['coverage']['right']*100:.1f}%"
                  f"   plane rms {r['plane']['rms_mm']:.2f} mm  -> {Path(r['path']).name}", flush=True)
        except Exception as exc:
            print(f"  {p.name}: FAILED -- {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
