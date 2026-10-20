"""Intrinsics of the UMI TRACKING fisheye (2026-10-15, DIY wrist tracker step 1). Two Arducams share a product name, so this
opens the camera by OpenCV index + USB location (exposure locked like the frontend check), auto-captures diverse sharp ChArUco
views with a coverage HUD, then fits BOTH models (Kannala-Brandt fisheye and OpenCV pinhole/plumb_bob) on 80 % of the views and
scores them on the held-out 20 % (per-corner residual, centre vs edge). Gate (user 10-15): RMS < 0.5 px very good, 0.5-1.0 usable,
> 1 px re-take; a big edge residual = model or coverage problem.
    .venv/bin/python -m visual_teleop.tools.trackcam_calib [--index 4] [--location 0x00140000] [--views 40]
    .venv/bin/python -m visual_teleop.tools.trackcam_calib --images 'datasets/trackcam_calib/<ts>/*.png'     (offline re-fit)"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ego_collector.camera.calibration import CharucoSpec, detect_charuco   # noqa: E402
from .wrist_frontend_check import lock_exposure, say                        # noqa: E402

LATEST = {"jpg": None}


def corners_of(images, spec, min_corners=10):
    obj, img = [], []
    for im in images:
        d = detect_charuco(im, spec, min_corners=min_corners)
        if d is None: continue
        obj.append(d.object_points.reshape(-1, 3).astype(np.float64)); img.append(d.image_points.reshape(-1, 2).astype(np.float64))
    return obj, img


def fit(model, obj, img, size):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    if model == "fisheye":
        K, D = np.eye(3), np.zeros((4, 1))
        rms, K, D, _, _ = cv2.fisheye.calibrate([o.reshape(1, -1, 3) for o in obj], [i.reshape(1, -1, 2) for i in img], size, K, D,
                                                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW, criteria=crit)
    else:
        rms, K, D, _, _ = cv2.calibrateCamera([o.astype(np.float32) for o in obj], [i.reshape(-1, 1, 2).astype(np.float32) for i in img], size, None, None,
                                              flags=cv2.CALIB_RATIONAL_MODEL, criteria=crit)
    return float(rms), K, D.reshape(-1)


def held_out(model, K, D, obj, img, size):
    """Per-corner residual on views NOT used for the fit: pose by PnP on undistorted points, reproject with the model."""
    res, rad = [], []; c = np.array([K[0, 2], K[1, 2]]); rmax = np.hypot(*size) / 2
    for o, i in zip(obj, img):
        if model == "fisheye": u = cv2.fisheye.undistortPoints(i.reshape(-1, 1, 2), K, D.reshape(4, 1))
        else: u = cv2.undistortPoints(i.reshape(-1, 1, 2), K, D)
        ok, rv, tv = cv2.solvePnP(o, u.reshape(-1, 2), np.eye(3), None)
        if not ok: continue
        if model == "fisheye": p, _ = cv2.fisheye.projectPoints(o.reshape(-1, 1, 3), rv, tv, K, D.reshape(4, 1))
        else: p, _ = cv2.projectPoints(o, rv, tv, K, D)
        e = np.linalg.norm(p.reshape(-1, 2) - i, axis=1); res.append(e); rad.append(np.linalg.norm(i - c, axis=1) / rmax)
    if not res: return None
    e, r = np.concatenate(res), np.concatenate(rad)
    return dict(rms_px=round(float(np.sqrt(np.mean(e ** 2))), 3), p95_px=round(float(np.percentile(e, 95)), 3),
                centre_rms_px=round(float(np.sqrt(np.mean(e[r < 0.5] ** 2))), 3) if (r < 0.5).any() else None,
                edge_rms_px=round(float(np.sqrt(np.mean(e[r >= 0.75] ** 2))), 3) if (r >= 0.75).any() else None, n_corners=int(len(e)))


def calibrate(images, spec, out: Path) -> dict:
    size = (images[0].shape[1], images[0].shape[0]); obj, img = corners_of(images, spec)
    if len(obj) < 12: raise SystemExit(f"보드가 잡힌 장이 {len(obj)}장뿐 (12장 이상 필요)")
    rng = np.random.default_rng(0); idx = rng.permutation(len(obj)); nv = max(3, len(obj) // 5); va, tr = idx[:nv], idx[nv:]
    rep = dict(image_size=size, views=len(obj), train=len(tr), val=len(va))
    for model in ("fisheye", "pinhole"):
        try:
            rms, K, D = fit(model, [obj[k] for k in tr], [img[k] for k in tr], size)
            rep[model] = dict(train_rms_px=round(rms, 3), K=K.round(4).tolist(), D=D.round(6).tolist(), val=held_out(model, K, D, [obj[k] for k in va], [img[k] for k in va], size))
        except cv2.error as e: rep[model] = dict(error=str(e)[:200])
    ok = [m for m in ("fisheye", "pinhole") if rep.get(m, {}).get("val")]
    best = min(ok, key=lambda m: rep[m]["val"]["rms_px"]) if ok else None
    rep["chosen"] = best
    if best:
        try: rms, K, D = fit(best, obj, img, size); src = "all views"
        except cv2.error:                     # cv2.fisheye InitExtrinsics asserts on a degenerate view -> keep the train fit
            rms, K, D = rep[best]["train_rms_px"], np.array(rep[best]["K"]), np.array(rep[best]["D"]); src = "train views (all-view fit hit a degenerate view)"
        rep["final"] = dict(model=best, rms_px=round(float(rms), 3), K=np.asarray(K).round(4).tolist(), D=np.asarray(D).round(6).tolist(), fitted_on=src)
        v = rep[best]["val"]["rms_px"]; rep["grade"] = "VERY_GOOD" if v < 0.5 else ("USABLE" if v <= 1.0 else "RETAKE")
    (out / "intrinsics.json").write_text(json.dumps(rep, indent=1)); return rep


def capture(a, spec, out: Path) -> list:
    cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    print("[calib]", lock_exposure(a.location, a.exposure), flush=True)
    for _ in range(10): cap.read()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        if LATEST["jpg"]: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + LATEST["jpg"] + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError): return
            b = "<html><head><meta charset=utf-8><title>trackcam calib</title></head><body style='margin:0;background:#111'><img src='/stream' style='width:100vw'></body></html>".encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(b)
        def log_message(self, *x): pass
    srv = ThreadingHTTPServer(("127.0.0.1", 8744), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); subprocess.Popen(["open", "http://127.0.0.1:8744"])
    say("보드를 자리마다 일 초씩 멈췄다가 옮기세요. 뽁 소리가 나면 다음 자리로. 네 귀퉁이까지, 가깝게 멀게, 기울여서 보여주세요.")
    grid = np.zeros((3, 3), int); dist = {"near": 0, "mid": 0, "far": 0}; kept, sigs = [], []; t_last = 0.0; t0 = time.monotonic(); prev_ip = None
    # 10-15: 45 views taken while the board moved gave a UNIFORM 1.2 px residual (rolling shutter) -> capture only when still
    while len(kept) < a.views and time.monotonic() - t0 < a.max_seconds:
        ok, f = cap.read()
        if not ok: continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); d = detect_charuco(f, spec, min_corners=10); v = f.copy(); H_, W_ = g.shape
        msg = "board not found"
        if d is None: prev_ip = None
        if d is not None:
            ip = d.image_points.reshape(-1, 2); c = ip.mean(0); span = float(np.ptp(ip[:, 0]) / W_); sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            cell = (min(2, int(3 * c[1] / H_)), min(2, int(3 * c[0] / W_))); dbin = "near" if span > 0.35 else ("mid" if span > 0.2 else "far")
            sig = np.r_[c / [W_, H_], span, (ip[-1] - ip[0]) / W_]
            novel = all(np.linalg.norm(sig - s) > 0.06 for s in sigs)
            still = prev_ip is not None and prev_ip.shape == ip.shape and float(np.median(np.linalg.norm(ip - prev_ip, axis=1))) < a.still_px
            prev_ip = ip.copy()
            msg = f"corners {d.count} span {span:.2f} sharp {sharp:.0f} {'NEW' if novel else 'similar'} {'STILL' if still else 'moving'}"
            for q in ip: cv2.circle(v, (int(q[0]), int(q[1])), 3, (0, 255, 0), -1)
            if novel and still and sharp > a.min_sharp and d.count >= 12 and time.monotonic() - t_last > 0.4:
                kept.append(f.copy()); sigs.append(sig); grid[cell] += 1; dist[dbin] += 1; t_last = time.monotonic()
                cv2.imwrite(str(out / f"view_{len(kept):03d}.png"), f); subprocess.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])
        for r in range(3):
            for cc in range(3):
                col = (0, 200, 0) if grid[r, cc] >= 2 else ((0, 200, 255) if grid[r, cc] else (0, 0, 255))
                cv2.rectangle(v, (int(cc * W_ / 3) + 4, int(r * H_ / 3) + 4), (int((cc + 1) * W_ / 3) - 4, int((r + 1) * H_ / 3) - 4), col, 2)
        cv2.putText(v, f"views {len(kept)}/{a.views}  near {dist['near']} mid {dist['mid']} far {dist['far']}  | {msg}", (12, 34), 0, 0.8, (0, 255, 255), 2)
        LATEST["jpg"] = cv2.imencode(".jpg", cv2.resize(v, (W_ // 2, H_ // 2)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    cap.release(); say("촬영이 끝났습니다. 계산합니다."); srv.shutdown()
    print(f"[calib] views {len(kept)}, grid\n{grid}\n distance {dist}", flush=True); return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=4); ap.add_argument("--location", default="0x00140000"); ap.add_argument("--exposure", type=int, default=15)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720); ap.add_argument("--views", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=180); ap.add_argument("--min-sharp", type=float, default=80.0); ap.add_argument("--still-px", type=float, default=0.5)
    ap.add_argument("--charuco", default=str(ROOT / "configs/charuco_wrist.yaml")); ap.add_argument("--images", default=None); a = ap.parse_args()
    spec = CharucoSpec.from_yaml(Path(a.charuco))
    if a.images:
        files = sorted(glob.glob(a.images)); out = Path(files[0]).parent; imgs = [cv2.imread(f) for f in files]
    else:
        out = ROOT / "datasets" / "trackcam_calib" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True); imgs = capture(a, spec, out)
    rep = calibrate(imgs, spec, out)
    show = {k: rep[k] for k in ("image_size", "views", "train", "val", "chosen", "grade") if k in rep}
    for m in ("fisheye", "pinhole"): show[m] = {k: rep[m].get(k) for k in ("train_rms_px", "val", "error") if k in rep[m]}
    print(json.dumps(show, indent=1)); print(f"[calib] 저장 {out / 'intrinsics.json'}")


if __name__ == "__main__":
    main()
