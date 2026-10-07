#!/usr/bin/env python3
"""[2026-09-19] Export path: the RGB-D body tracker + the IMU orientation branch -> one npz per episode.

No new estimator lives here. Translation comes from `handumi_body_track.track_episode` exactly as it was
validated (DARK_MAX 70, identity association, trim radius BODY_RADIUS_M = 0.12, gap/valid mask), and rotation
comes from `imu_orientation` exactly as its selftest checks it. This file only joins them onto one timeline and
writes the result down, because a second copy of either estimator is a second thing to keep in agreement.

    body centroid (camera frame)  --R_align-->  world xyz          translation, from depth
    gyro+accel on device_us  --clock map-->  SLERP to depth t_ns   rotation, from the IMU

WORLD FRAME: +Z is the table normal, +X is the camera's optical axis projected onto the table. This is the same
`align_from_plane` the ego16 export uses, so the two cannot disagree.

TRACK IDs: the tracker keeps one track per side and re-seeds it from the left/right spatial prior after a gap
longer than 15 frames. The id here increments on exactly that event, so a consumer can tell "the same unit,
continuously" from "re-acquired, and the identity is only as good as the spatial prior". It is derived from the
tracker's own rule rather than invented.

    .venv/bin/python scripts/body_track_export.py --session <session>
    .venv/bin/python scripts/body_track_export.py --episode <ep>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.handumi_body_track import track_episode                           # noqa: E402
from scripts.body_to_ego16 import align_from_plane                             # noqa: E402
from scripts.imu_orientation import clock_map, integrate, resample             # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                       # noqa: E402
from scripts.provenance import stamp                                            # noqa: E402

RESEED_GAP = 15            # frames; the tracker's own re-seed threshold (associate(): `t["miss"] > 15`)


def track_ids(valid: np.ndarray) -> np.ndarray:
    """0 before the first detection, then an id that increments whenever the tracker would have re-seeded."""
    out = np.zeros(len(valid), np.int32)
    cur, gap, started = 0, 0, False
    for i, v in enumerate(valid):
        if v:
            if not started or gap > RESEED_GAP:
                cur += 1
            started, gap = True, 0
        else:
            gap += 1
        out[i] = cur
    return out


def imu_on_frames(ep_path: Path, t_ns: np.ndarray) -> dict:
    """Orientation for each side, integrated on its own device clock and interpolated onto the depth timeline."""
    raw = RawEpisode.load(ep_path)
    out = {}
    for side, im in (raw.imu or {}).items():
        d = np.asarray(im.device_us, np.int64)
        h = np.asarray(im.host_ns, np.int64)
        r = integrate(np.asarray(im.gyro, float), np.asarray(im.accel, float), d)
        cm = clock_map(d, h)
        # place each IMU sample on the host timeline THROUGH the clock map, not by using host_ns directly:
        # host_ns carries USB batching ties, and the map is the smooth version of the same relationship.
        t_imu = (cm["scale_ns_per_us"] * d.astype(np.float64) + cm["offset_ns"]).astype(np.int64)
        q, ok = resample(r["quat"], t_imu, t_ns)
        out[side] = dict(quat=q, valid=ok, clock=cm,
                         yaw_drift_deg_per_min=r["yaw_drift_deg_per_min"],
                         still_share=r["still_share"], gravity_share=r["quiet_share"])
    return out


def export_episode(ep_path: Path) -> dict:
    tr = track_episode(ep_path, None, 0)
    R_align = align_from_plane(tr["plane_normal"])
    t_ns = np.asarray(tr["t_ns"], np.int64)
    imu = imu_on_frames(ep_path, t_ns)

    out = dict(timestamp_ns=t_ns, R_align=R_align,
               plane_normal=tr["plane_normal"], plane_d=np.float64(tr["plane_d"]))
    out.update(stamp(source_session=ep_path.parent.name, source_episode=ep_path.name,
                     export_version="body_track_export_v1", world="table_normal",
                     rotation_world="imu_gravity"))
    summary = {"episode": ep_path.name, "n_frames": int(tr["n_frames"]), "sides": {}}
    for side in ("left", "right"):
        xyz_cam = tr["xyz"][side]
        v = tr["valid"][side]
        xyz_world = np.full_like(xyz_cam, np.nan)
        xyz_world[v] = xyz_cam[v] @ R_align.T
        ids = track_ids(v)
        out[f"{side}_body_xyz"] = xyz_world
        out[f"{side}_valid"] = v
        out[f"{side}_track_id"] = ids
        q = imu.get(side)
        out[f"{side}_quat_world_imu"] = q["quat"] if q else np.full((len(t_ns), 4), np.nan)
        out[f"{side}_quat_valid"] = q["valid"] if q else np.zeros(len(t_ns), bool)
        # scalars the freeze gate needs. They are printed in the summary anyway, but a gate that reads them out
        # of a log is a gate nobody runs -- so they travel with the data.
        out[f"{side}_identity_switches"] = np.int32(tr["switches"][side])
        out[f"{side}_clock_residual_ms_p95"] = np.float64(q["clock"]["residual_ms_p95"] if q else np.nan)
        out[f"{side}_clock_rate_error_ppm"] = np.float64(q["clock"]["rate_error_ppm"] if q else np.nan)
        out[f"{side}_yaw_drift_deg_per_min"] = np.float64(q["yaw_drift_deg_per_min"] if q else np.nan)
        summary["sides"][side] = dict(
            coverage=float(v.mean()), identity_switches=int(tr["switches"][side]),
            track_ids=int(ids.max()), max_gap=int(_max_gap(v)),
            quat_coverage=float(q["valid"].mean()) if q else 0.0,
            clock_residual_ms_p95=float(q["clock"]["residual_ms_p95"]) if q else float("nan"),
            clock_rate_error_ppm=float(q["clock"]["rate_error_ppm"]) if q else float("nan"),
            yaw_drift_deg_per_min=float(q["yaw_drift_deg_per_min"]) if q else float("nan"),
            still_share=float(q["still_share"]) if q else float("nan"))
    d = ep_path / "derived" / "body_track"
    d.mkdir(parents=True, exist_ok=True)
    # Downstream steps APPEND to this file (body_canonical_tracker writes *_body_xyz_rigid, cube_tcp_detect and
    # cube_center_planes write the TCP observations). Rewriting it from scratch silently drops them, and the next
    # fit then fails for a reason that has nothing to do with what changed. Say so instead.
    target = d / "track_export.npz"
    if target.exists():
        had = set(np.load(target).files) - set(out)
        if had:
            print(f"  NOTE {ep_path.name}: this export drops {len(had)} downstream field(s) -- "
                  f"{', '.join(sorted(had)[:4])}{' ...' if len(had) > 4 else ''}. "
                  f"Re-run body_canonical_tracker.py and the cube detectors before fitting.")
    np.savez(target, **out)
    summary["path"] = str(d / "track_export.npz")
    return summary


def _max_gap(v: np.ndarray) -> int:
    best = cur = 0
    for x in v:
        cur = 0 if x else cur + 1
        best = max(best, cur)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path); g.add_argument("--episode", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    eps = ([a.episode] if a.episode else
           sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_")))
    rows = []
    for p in eps:
        rows.append(export_episode(p))
        print(f"  exported {p.name}", flush=True)

    print(f"\n  {'episode':>14s} {'side':>6s} {'cover':>7s} {'switch':>7s} {'ids':>4s} {'maxgap':>7s} "
          f"{'quat':>7s} {'clk p95':>8s} {'ppm':>6s} {'yaw drift':>10s} {'still':>7s}")
    for r in rows:
        for side, s in r["sides"].items():
            print(f"  {r['episode'][-11:]:>14s} {side:>6s} {100*s['coverage']:6.1f}% {s['identity_switches']:7d} "
                  f"{s['track_ids']:4d} {s['max_gap']:7d} {100*s['quat_coverage']:6.1f}% "
                  f"{s['clock_residual_ms_p95']:7.2f}m {s['clock_rate_error_ppm']:+6.0f} "
                  f"{s['yaw_drift_deg_per_min']:+9.2f}d {100*s['still_share']:6.1f}%")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=float)); print(f"\nrecord -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
