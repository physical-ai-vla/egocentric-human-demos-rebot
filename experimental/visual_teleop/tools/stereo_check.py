"""Stereo Phase S0 + S1 for the DIY tracker: two Arducam fisheyes captured together (2026-10-15).
S0  per camera: resolution, actual fps, dropped frames (host dt > 1.5x median), exposure locked through uvc-util (same value L/R);
    L/R skew = left host stamp - nearest right host stamp (median / p90 / p95 / max). Host receive stamps, no hardware sync.
S1  overlap: on frame pairs every ~1 s, SIFT on both images, ratio-test matches, fundamental-matrix RANSAC on the raw fisheye pixels
    (loose 3 px threshold: distortion is not removed, the right camera has no calibration yet) -> features L/R, raw matches,
    stereo inliers, inlier ratio, coverage = share of an 8x6 grid holding at least one inlier (left / right).
    PASS hint (spec): inliers > 60 (median > 100 very good), ratio > 0.5, spread over the image.
Writes datasets/stereo_check/<ts>/{stereo_check.json, pair_*.jpg (side by side, inliers drawn)}. No recording of video.
    .venv/bin/python -m visual_teleop.tools.stereo_check [--left-index 4 --left-location 0x00140000] [--right-index 2 --right-location 0x03110000]
                                                         [--seconds 20] [--width 1920 --height 1080] [--exposure 15]"""
from __future__ import annotations
import argparse, json, threading, time
from pathlib import Path
import cv2
import numpy as np
from visual_teleop.tools.wrist_frontend_check import lock_exposure

ROOT = Path(__file__).resolve().parents[2]


class Grab(threading.Thread):
    def __init__(self, index: int, w: int, h: int):
        super().__init__(daemon=True); self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.t: list[int] = []; self.frames: dict[int, np.ndarray] = {}; self.keep_every = 30; self.run_flag = True; self.shape = None; self.last = None
    def run(self):
        k = 0
        while self.run_flag:
            ok, f = self.cap.read(); t = time.monotonic_ns()
            if not ok: continue
            self.t.append(t); self.shape = f.shape; self.last = (t, f)
            k += 1
        self.cap.release()


def index_for_location(location: str, candidates=range(8), w=1920, h=1080) -> int:
    """OpenCV (AVFoundation) index of the camera at a USB location: AVFoundation order != uvc-util order and changes on replug, so
    darken that location through uvc-util (exposure 1 vs 30) and take the opened camera whose brightness drops the most."""
    import subprocess
    U = str(Path.home() / "robot-cockpit/bin/uvc-util"); lst = subprocess.run([U, "-d"], capture_output=True, text=True).stdout
    line = next((l for l in lst.splitlines() if location.lower() in l.lower()), None)
    if line is None: raise SystemExit(f"uvc-util: no camera at {location}")
    u = line.split()[0]; caps = {}
    for i in candidates:
        c = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if c.isOpened(): c.set(cv2.CAP_PROP_FRAME_WIDTH, w); c.set(cv2.CAP_PROP_FRAME_HEIGHT, h); caps[i] = c
    def means():
        for _ in range(8): fr = {i: c.read()[1] for i, c in caps.items()}
        return {i: float(f.mean()) if f is not None else 0.0 for i, f in fr.items()}
    for a in ("auto-exposure-mode=1", "exposure-time-abs=30"): subprocess.run([U, "-I", u, "-s", a], capture_output=True)
    drops = {i: 0.0 for i in caps}
    for _ in range(2):                                   # two dark/bright cycles; the right camera responds both times
        subprocess.run([U, "-I", u, "-s", "exposure-time-abs=30"], capture_output=True); hi = means()
        subprocess.run([U, "-I", u, "-s", "exposure-time-abs=1"], capture_output=True); lo = means()
        for i in caps: drops[i] += hi[i] - lo[i]
    for c in caps.values(): c.release()
    order = sorted(drops, key=drops.get, reverse=True); best, second = order[0], (order[1] if len(order) > 1 else None)
    if drops[best] < 20 or (second is not None and drops[second] > 0.7 * drops[best]):
        raise SystemExit(f"{location}: 카메라 인덱스를 확실히 못 찾음 (밝기 변화 {dict((i, round(v, 1)) for i, v in drops.items())})")
    return best


def overlap(gl: np.ndarray, gr: np.ndarray):
    sift = cv2.SIFT_create(2000); kl, dl = sift.detectAndCompute(gl, None); kr, dr = sift.detectAndCompute(gr, None)
    res = dict(features_left=len(kl), features_right=len(kr), raw_matches=0, stereo_inliers=0, inlier_ratio=0.0, coverage_left=0.0, coverage_right=0.0)
    if dl is None or dr is None or len(kl) < 8 or len(kr) < 8: return res, None
    m = [a for a, b in cv2.BFMatcher().knnMatch(dl, dr, k=2) if a.distance < 0.75 * b.distance]; res["raw_matches"] = len(m)
    if len(m) < 8: return res, None
    pl = np.float32([kl[x.queryIdx].pt for x in m]); pr = np.float32([kr[x.trainIdx].pt for x in m])
    F, inl = cv2.findFundamentalMat(pl, pr, cv2.FM_RANSAC, 3.0, 0.999)
    if F is None or inl is None: return res, None
    inl = inl.ravel().astype(bool); h, w = gl.shape
    cov = lambda p: len({(int(x * 8 / w), int(y * 6 / h)) for x, y in p}) / 48.0
    d = pr[inl] - pl[inl]; res.update(dy_median_px=round(float(np.median(d[:, 1])), 1), dy_abs_median_px=round(float(np.median(np.abs(d[:, 1] - np.median(d[:, 1])))), 1),
               dx_median_px=round(float(np.median(d[:, 0])), 1))
    res.update(stereo_inliers=int(inl.sum()), inlier_ratio=round(float(inl.mean()), 3), coverage_left=round(cov(pl[inl]), 3), coverage_right=round(cov(pr[inl]), 3))
    vis = np.hstack([cv2.cvtColor(gl, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gr, cv2.COLOR_GRAY2BGR)])
    for a, b in zip(pl[inl], pr[inl]): cv2.line(vis, tuple(map(int, a)), (int(b[0]) + w, int(b[1])), (0, 255, 0), 1)
    return res, vis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-index", type=int, default=-1); ap.add_argument("--left-location", default="0x03110000")
    ap.add_argument("--right-index", type=int, default=-1); ap.add_argument("--right-location", default="0x03140000")
    ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--exposure", type=int, default=15); a = ap.parse_args()
    out = ROOT / "datasets" / "stereo_check" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    if a.left_index < 0: a.left_index = index_for_location(a.left_location)
    if a.right_index < 0: a.right_index = index_for_location(a.right_location)
    print(f"[stereo] 왼쪽 {a.left_location} = index {a.left_index}, 오른쪽 {a.right_location} = index {a.right_index}", flush=True)
    L, R = Grab(a.left_index, a.width, a.height), Grab(a.right_index, a.width, a.height)
    expo = dict(left=lock_exposure(a.left_location, a.exposure), right=lock_exposure(a.right_location, a.exposure)); print("[stereo] 노출", expo, flush=True)
    L.start(); R.start(); time.sleep(1.5); L.t.clear(); R.t.clear(); L.frames.clear(); R.frames.clear()
    print(f"[stereo] {a.seconds:.0f}초 동시 캡처 중 (UMI를 천천히 움직여도 됩니다)", flush=True); snaps = []; te = time.monotonic() + a.seconds
    while time.monotonic() < te:
        time.sleep(1.0)
        if L.last and R.last: (tl_, fl), (tr_, fr) = L.last, R.last; snaps.append((tl_, cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), tr_, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
    L.run_flag = R.run_flag = False; L.join(2); R.join(2)
    S = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), exposure=expo, left=dict(index=a.left_index, location=a.left_location), right=dict(index=a.right_index, location=a.right_location))
    for name, G in (("left", L), ("right", R)):
        t = np.array(G.t, np.int64); dt = np.diff(t) / 1e6; med = float(np.median(dt))
        S[name].update(resolution=list(G.shape[1::-1]) if G.shape else None, frames=int(len(t)), fps=round(len(t) / a.seconds, 2), dt_ms_p50=round(med, 2),
                       dt_ms_p95=round(float(np.percentile(dt, 95)), 2), dropped=int((dt > 1.5 * med).sum()), mean_gray=round(float(np.mean([x[1 if name == "left" else 3].mean() for x in snaps])), 1) if snaps else None)
    tl, tr = np.array(L.t, np.int64), np.array(R.t, np.int64); j = np.clip(np.searchsorted(tr, tl), 1, len(tr) - 1)
    nn = np.where(np.abs(tr[j] - tl) < np.abs(tr[j - 1] - tl), tr[j], tr[j - 1]); sk = (tl - nn) / 1e6; ab = np.abs(sk)
    S["skew_left_minus_right_ms"] = dict(median=round(float(np.median(sk)), 2), abs_p50=round(float(np.median(ab)), 2), abs_p90=round(float(np.percentile(ab, 90)), 2),
                                         abs_p95=round(float(np.percentile(ab, 95)), 2), abs_max=round(float(ab.max()), 2))
    pairs = []
    for k, (t, gl, r, gr) in enumerate(snaps):          # latest frame of each camera grabbed at the same moment, once per second
        res, vis = overlap(gl, gr); res["pair_skew_ms"] = round((t - r) / 1e6, 2); pairs.append(res)

        if vis is not None and k % 5 == 0: cv2.imwrite(str(out / f"pair_{k:02d}.jpg"), cv2.resize(vis, (1920, int(1920 * vis.shape[0] / vis.shape[1]))))
    S["overlap_pairs"] = pairs
    if pairs:
        inl = [p["stereo_inliers"] for p in pairs]; rat = [p["inlier_ratio"] for p in pairs]
        S["overlap_summary"] = dict(pairs=len(pairs), inliers_median=float(np.median(inl)), inliers_min=int(min(inl)), inlier_ratio_median=round(float(np.median(rat)), 3),
                                    coverage_left_median=round(float(np.median([p["coverage_left"] for p in pairs])), 3),
                                    coverage_right_median=round(float(np.median([p["coverage_right"] for p in pairs])), 3),
                                    dy_median_px=float(np.median([p.get("dy_median_px", np.nan) for p in pairs])), dx_median_px=float(np.median([p.get("dx_median_px", np.nan) for p in pairs])),
                                    verdict="PASS" if np.median(inl) > 60 and np.median(rat) > 0.5 else "FAIL")
    sk_ok = S["skew_left_minus_right_ms"]["abs_p95"]
    S["skew_verdict"] = "ideal (<5 ms)" if sk_ok < 5 else "testable (<10 ms)" if sk_ok < 10 else "marginal (<15 ms)" if sk_ok < 15 else "bad (>15 ms)"
    (out / "stereo_check.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in S.items() if k != "overlap_pairs"}, indent=1, ensure_ascii=False)); print("[stereo] 저장", out)


if __name__ == "__main__":
    main()
