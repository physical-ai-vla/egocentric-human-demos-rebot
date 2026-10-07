"""Is a cube face a usable anchor for wrist translation? Measures; decides nothing; writes PNP_FEASIBILITY_REPORT.md.

    python -m handumi_collector.tools.pnp_feasibility SESSION_DIR [--side right] [--out REPORT.md]

Everything here is relative. No trajectory is integrated, no world frame is built, and HOME-return closure and the
valid-ratio gate are not consulted -- they judge an absolute path this pipeline does not consume, and using them was
how `opencv_vo` came to be rejected for the wrong reason before being rejected for the right one.

The number that decides it is the STILL jitter: while the gyro says the hand is barely turning, an anchor must give
the same position. Everything else is context for that."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from ..pose.cube_anchor import CUBE_M, detect_faces, pair_distances, solve_face
from ..pose.episode_io import RawEpisode

STILL_DPS = 10.0            # gyro magnitude below which the hand is "not turning"; a still anchor must not move
MAX_GAP_S = 0.5             # two observations further apart than this are not consecutive
FACE_AXIS = np.array([0.0, 0.0, 1.0])


def _q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if a.size else float("nan")


def _angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _delta_angle_modulo_square(Ra, Rb) -> float:
    """Camera rotation between two observations of the same face, up to the square's 90 degree symmetry.

    IPPE_SQUARE cannot know which way round a square is, so the recovered R is ambiguous by a quarter turn about the
    face normal. Comparing raw R would report those quarter turns as camera motion. Take the smallest of the four."""
    best = 180.0
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        best = min(best, _angle_deg(Ra.T @ (Rb @ Rz)))
    return best


def load_intrinsics(side: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str]:
    p = Path("configs/calibration") / f"fisheye_{side}_v001.yaml"
    d = yaml.safe_load(p.read_text())
    return (np.asarray(d["K"], float).reshape(3, 3), np.asarray(d["D"], float).reshape(-1),
            tuple(int(x) for x in d["image_size"]), f"{p.stem} (rms {d.get('rms_px')})")


def episode_metrics(ep_dir: Path, side: str, K, D, cal_wh, *, step: int = 3) -> dict:
    ep = RawEpisode.load(ep_dir)
    stream = f"{side}_wrist"
    if stream not in ep.frames:
        return {}
    fm = ep.frames[stream]
    t_all = (fm.capture_ns.astype(np.int64) - ep.t_start_ns) / 1e9
    imu = ep.imu.get(side)
    gt = (np.asarray(imu.host_ns) - ep.t_start_ns) / 1e9 if imu else np.zeros(0)
    gyro = np.asarray(imu.gyro) if imu else np.zeros((0, 3))
    gmag = np.linalg.norm(gyro, axis=1) * 180 / np.pi if len(gyro) else np.zeros(0)

    cap = cv2.VideoCapture(str(ep_dir / f"{stream}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Ks = None
    per_frame: list[dict] = []
    for i in range(0, min(n, len(t_all)), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        if Ks is None:
            Ks = K.copy(); Ks[:2] *= img.shape[1] / cal_wh[0]      # intrinsics were measured at the calibration size
        poses = {}
        for c, f in detect_faces(img).items():
            s = solve_face(f, Ks, D)
            if s is not None:
                poses[c] = s
        per_frame.append(dict(t=float(t_all[i]), poses=poses, pairs=pair_distances(poses) if len(poses) > 1 else {}))
    cap.release()
    if not per_frame:
        return {}

    frames = len(per_frame)
    usable = sum(1 for f in per_frame if f["poses"])
    multi = sum(1 for f in per_frame if len(f["poses"]) > 1)
    rms = [p.rms_px for f in per_frame for p in f["poses"].values()]
    rng = [float(np.linalg.norm(p.t)) for f in per_frame for p in f["poses"].values()]

    dp_all, dr_all, dp_still, dr_still, dr_vs_gyro, switches = [], [], [], [], [], 0
    still_detail: list[tuple[float, float, float]] = []
    obs = [(p.rms_px, float(np.linalg.norm(p.t))) for f in per_frame for p in f["poses"].values()]
    prev_best = None
    for a, b in zip(per_frame, per_frame[1:]):
        if b["t"] - a["t"] > MAX_GAP_S:
            prev_best = None
            continue
        shared = set(a["poses"]) & set(b["poses"])
        best_now = min(b["poses"], key=lambda c: b["poses"][c].rms_px) if b["poses"] else None
        if prev_best is not None and best_now is not None and best_now != prev_best:
            switches += 1
        prev_best = best_now
        if not shared:
            continue
        c = min(shared, key=lambda c: a["poses"][c].rms_px + b["poses"][c].rms_px)
        pa, pb = a["poses"][c], b["poses"][c]
        dp_mm = float(np.linalg.norm(pb.t - pa.t)) * 1000
        dr_deg = _delta_angle_modulo_square(pa.R, pb.R)
        dp_all.append(dp_mm); dr_all.append(dr_deg)
        worst_rms = max(pa.rms_px, pb.rms_px); rng_m = float(np.linalg.norm(pb.t))
        m = (gt >= a["t"]) & (gt <= b["t"])
        w = float(np.mean(gmag[m])) if m.any() else float("nan")
        if np.isfinite(w):
            gyro_deg = float(np.trapezoid(gmag[m], gt[m])) if m.sum() > 1 else 0.0
            dr_vs_gyro.append(abs(dr_deg - gyro_deg))
            if w < STILL_DPS:
                dp_still.append(dp_mm); dr_still.append(dr_deg)
                still_detail.append((dp_mm, worst_rms, rng_m))

    pair_std = {}
    keys = {k for f in per_frame for k in f["pairs"]}
    for k in keys:
        v = np.array([f["pairs"][k] for f in per_frame if k in f["pairs"]])
        if len(v) >= 5:
            pair_std[k] = dict(n=int(len(v)), median_cm=round(100 * float(np.median(v)), 2),
                               std_mm=round(1000 * float(np.std(v)), 1),
                               p95_dev_mm=round(1000 * float(np.percentile(np.abs(v - np.median(v)), 95)), 1))
    return dict(
        frames=frames, usable=usable, usable_pct=round(100 * usable / frames, 1),
        multi_pct=round(100 * multi / frames, 1),
        rms_px=dict(median=round(_q(rms, 50), 2), p95=round(_q(rms, 95), 2)),
        range_cm=dict(median=round(100 * _q(rng, 50), 1), p95=round(100 * _q(rng, 95), 1)),
        dp_mm=dict(median=round(_q(dp_all, 50), 1), p95=round(_q(dp_all, 95), 1), n=len(dp_all)),
        dr_deg=dict(median=round(_q(dr_all, 50), 2), p95=round(_q(dr_all, 95), 2)),
        still=dict(n=len(dp_still), dp_median_mm=round(_q(dp_still, 50), 1), dp_p95_mm=round(_q(dp_still, 95), 1),
                   dp_max_mm=round(float(np.max(dp_still)) if dp_still else float("nan"), 1),
                   dr_median_deg=round(_q(dr_still, 50), 2), dr_p95_deg=round(_q(dr_still, 95), 2)),
        dr_vs_gyro_deg=dict(median=round(_q(dr_vs_gyro, 50), 2), p95=round(_q(dr_vs_gyro, 95), 2), n=len(dr_vs_gyro)),
        anchor_switches=switches, pair_distances=pair_std,
        _still_dp=still_detail,
        _kept=lambda mr, rg, _o=obs: sum(1 for r, d in _o if r <= mr and (rg is None or rg[0] <= d <= rg[1])),
    )


FILTERS = (("none", 1e9, None), ("reproj < 3 px", 3.0, None), ("reproj < 3 px, range 8-80 cm", 3.0, (0.08, 0.80)),
           ("reproj < 1.5 px, range 8-80 cm", 1.5, (0.08, 0.80)))


def filter_sweep(per_ep: dict) -> list[dict]:
    """What plain outlier rejection buys. Answer, on the pilot: nothing, and the direction is the finding.

    Tightening the reprojection cut makes the median jitter WORSE while collapsing coverage, because reprojection is
    anti-correlated with usefulness here -- a small, distant, clean face fits beautifully and has the worst depth
    conditioning, while a large near face carries more corner error and far better geometry."""
    out = []
    for label, max_rms, rng in FILTERS:
        dp = [d for m in per_ep.values() for d in m["_still_dp"]
              if d[1] <= max_rms and (rng is None or rng[0] <= d[2] <= rng[1])]
        tot = sum(m["frames"] for m in per_ep.values())
        kept = sum(m["_kept"](max_rms, rng) for m in per_ep.values())
        v = np.array([d[0] for d in dp])
        out.append(dict(filter=label, kept_pct=round(100 * kept / tot, 1), n=len(v),
                        dp_p50_mm=round(_q(v, 50), 1), dp_p95_mm=round(_q(v, 95), 1),
                        dp_max_mm=round(float(v.max()) if v.size else float("nan"), 1)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--side", default="right")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--out", default="PNP_FEASIBILITY_REPORT.md")
    a = ap.parse_args(argv)
    K, D, wh, note = load_intrinsics(a.side)
    eps = sorted(Path(a.session).expanduser().glob("episode_*"))
    print(f"intrinsics: {note} at {wh[0]}x{wh[1]}   cube {CUBE_M*1000:.0f} mm   side {a.side}")
    rows = {}
    for e in eps:
        m = episode_metrics(e, a.side, K, D, wh, step=a.step)
        if m:
            rows[e.name] = m
            print(f"  {e.name}: usable {m['usable_pct']}%  rms {m['rms_px']['median']}  "
                  f"still dp p50 {m['still']['dp_median_mm']} mm p95 {m['still']['dp_p95_mm']}")
    if not rows:
        print("no episodes measured"); return 1
    sweep = filter_sweep(rows)
    for r in rows.values():
        r.pop("_still_dp", None); r.pop("_kept", None)
    Path(a.out).write_text(render(rows, note, a.side, sweep))
    print(f"\nwrote {a.out}")
    return 0


def render(rows: dict, intr: str, side: str, sweep: list[dict]) -> str:
    L = [f"# Cube-anchor PnP feasibility — {side} wrist", "",
         f"Intrinsics `{intr}`, cube {CUBE_M*1000:.0f} mm, strict detections only.",
         "Relative measurements only: nothing is integrated into a trajectory and HOME closure is not used.", "",
         "## Per episode", "",
         "| ep | usable | multi | reproj px p50/p95 | range cm | Δp mm p50/p95 | ΔR deg p50/p95 | anchor switches |",
         "|---|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        L.append(f"| {k[-3:]} | {m['usable_pct']}% | {m['multi_pct']}% | "
                 f"{m['rms_px']['median']}/{m['rms_px']['p95']} | {m['range_cm']['median']} | "
                 f"{m['dp_mm']['median']}/{m['dp_mm']['p95']} | {m['dr_deg']['median']}/{m['dr_deg']['p95']} | "
                 f"{m['anchor_switches']} |")
    L += ["", "## The number that decides it", "",
          f"While the gyro reads under {STILL_DPS:g} deg/s the hand is barely turning, so an anchor must return the",
          "same position. This is that, per frame interval:", "",
          "| ep | n still | Δp mm p50 | p95 | max | ΔR deg p50 | p95 | ΔR vs gyro p50/p95 |", "|---|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        s, g = m["still"], m["dr_vs_gyro_deg"]
        L.append(f"| {k[-3:]} | {s['n']} | {s['dp_median_mm']} | {s['dp_p95_mm']} | {s['dp_max_mm']} | "
                 f"{s['dr_median_deg']} | {s['dr_p95_deg']} | {g['median']}/{g['p95']} |")
    L += ["", "## Would more cubes help", "",
          "The cubes do not move relative to each other between grasps, so each solved separation is a constant of the",
          "scene and its spread measures the solves. A stable separation means a joint solve has something to work",
          "with; an unstable one means it would only average noise.", "",
          "| ep | pair | n | median cm | std mm | p95 deviation mm |", "|---|---|---|---|---|---|"]
    for k, m in rows.items():
        for pair, d in sorted(m["pair_distances"].items()):
            L.append(f"| {k[-3:]} | {pair} | {d['n']} | {d['median_cm']} | {d['std_mm']} | {d['p95_dev_mm']} |")
    L += ["", "## Does outlier rejection rescue it", "",
          "| filter | observations kept | n still | Δp mm p50 | p95 | max |", "|---|---|---|---|---|---|"]
    for r in sweep:
        L.append(f"| {r['filter']} | {r['kept_pct']}% | {r['n']} | {r['dp_p50_mm']} | {r['dp_p95_mm']} | {r['dp_max_mm']} |")
    L += ["", "## Raw", "", "```json", json.dumps(rows, indent=1), "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
