"""Compare fisheye intrinsics candidates on the SAME still AprilGrid frames (2026-10-15: trackcam fx 680 (v001) vs 790 (stereo joint)).
Still frames = |gyro| < 0.1 rad/s within +-100 ms and >= 10 tags. Per candidate: board pose re-fitted per frame with the intrinsics
FROZEN (pose-only soft-L1 LM), then corner residuals: median / p95, edge region (radius > 0.7 of the half diagonal from the
principal point) median, and the radial pattern = mean SIGNED radial residual per radius bin (a model error shows as a curve, noise as ~0).
Frames used by a fit can be excluded with --skip-json (stereo_calib.json pairs are NOT stored, so pass --every to subsample instead).
    python scripts/intrinsics_compare.py <take dir> --cam left --cand old=340.0,339.2,491.7,276.2,0.0162,0.0232,-0.0259,0.0053 --cand new=...
    (intrinsics at the SAVED size: fx fy cx cy k1 k2 k3 k4)"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from stereo_calib import detect, pose_only           # noqa: E402
from fisheye_calib_robust import proj_kb4           # noqa: E402
from handumi_collector.pose.timing import fit_device_to_host   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--cam", default="left"); ap.add_argument("--cand", action="append", required=True); ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args(); E = Path(a.take); t = np.load(E / f"cam_{a.cam}_t_ns.npy")
    z = np.load(E / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]); wn = np.linalg.norm(z["gyro"], axis=1)
    dets, size = detect(E / f"cam_{a.cam}.mp4", len(t))
    still = [k for k in sorted(dets) if dets[k][2] >= 10 and (m := np.abs(ti - t[k]) < 100e6).any() and wn[m].max() < 0.1][:: a.every]
    print(f"{a.cam}: 정지 프레임 {len(still)}개", flush=True); out = {}
    for c in a.cand:
        name, vals = c.split("="); p = np.array([float(v) for v in vals.split(",")]); res, rad, sgn = [], [], []
        for k in still:
            o, im, _ = dets[k]; T, _ = pose_only(p, o, im, p[0], size); P = o @ T[:3, :3].T + T[:3, 3]; pr = proj_kb4(P, p); d = pr - im
            res.append(np.linalg.norm(d, axis=1)); v = im - p[2:4]; r = np.linalg.norm(v, axis=1); rad.append(r); sgn.append(np.sum(d * v / np.maximum(r[:, None], 1e-9), axis=1))
        e, r, s = np.concatenate(res), np.concatenate(rad), np.concatenate(sgn); hd = np.hypot(*size) / 2; edge = r > 0.7 * hd
        bins = np.linspace(0, r.max(), 7); pat = [round(float(s[(r >= lo) & (r < hi)].mean()), 2) if ((r >= lo) & (r < hi)).sum() > 30 else None for lo, hi in zip(bins[:-1], bins[1:])]
        out[name] = dict(n=int(len(e)), median_px=round(float(np.median(e)), 3), p95_px=round(float(np.percentile(e, 95)), 3), edge_n=int(edge.sum()),
                         edge_median_px=round(float(np.median(e[edge])), 3) if edge.any() else None, radial_signed_mean_by_radius=pat, radius_bins_px=[round(float(b)) for b in bins])
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
    (E / f"intrinsics_compare_{a.cam}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
