"""[2026-10-06] global camera (bridge 'middle') intrinsics from a checkerboard: grab frames for up to 4 min, auto-detect the inner
corner grid (tries common sizes), keep a frame when its corner centroid / tilt differs from the kept ones, stop at 25 frames, then
cv2.calibrateCamera (k1, k2, p1, p2, k3). Progress -> checker_progress.txt; result -> global_cam_intrinsics.json."""
import json, pathlib, time
import cv2, numpy as np, requests
OUT = pathlib.Path.home() / "c8/hra_red/global_calib"; BR = "http://localhost:8021"
SIZES = [(9, 6), (8, 6), (7, 6), (8, 5), (7, 5), (6, 5), (6, 4), (5, 4), (10, 7), (9, 7), (11, 8), (6, 9), (5, 7)]
kept = []; pattern = None; t0 = time.time(); prog = OUT / "checker_progress.txt"


def detect(gray, size):
    ok, c = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    if not ok: return None
    return cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))


def desc(c):
    """position (px), size (px) and perspective: the two pairs of opposite outer-edge length ratios (x100) -> tilt shows up"""
    p = c.reshape(-1, 2); W_, H_ = pattern
    q = p.reshape(H_, W_, 2); e_top = np.linalg.norm(q[0, -1] - q[0, 0]); e_bot = np.linalg.norm(q[-1, -1] - q[-1, 0])
    e_l = np.linalg.norm(q[-1, 0] - q[0, 0]); e_r = np.linalg.norm(q[-1, -1] - q[0, -1])
    return np.r_[p.mean(0) / 2, np.ptp(p, 0).max(), 300 * np.log(e_top / e_bot), 300 * np.log(e_l / e_r)]


while time.time() - t0 < 300 and len(kept) < 30:
    try:
        img = cv2.imdecode(np.frombuffer(requests.get(BR + "/frame/middle", timeout=5).content, np.uint8), 1)
    except Exception:
        time.sleep(0.5); continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); c = None
    for s in ([pattern] if pattern else SIZES):
        c = detect(gray, s)
        if c is not None:
            pattern = s; break
    if c is not None:
        d = desc(c)
        if all(np.linalg.norm(d - desc(k[1])) > 60 for k in kept):
            kept.append((img, c)); cv2.imwrite(str(OUT / f"checker_{len(kept):02d}.jpg"), img)
    prog.write_text(f"t={time.time()-t0:.0f}s pattern={pattern} kept={len(kept)} detected_now={c is not None}\n")
    time.sleep(0.4)
res = dict(pattern=pattern, frames=len(kept))
if len(kept) >= 8:
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32); obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    rms, K, D, rv, tv = cv2.calibrateCamera([obj] * len(kept), [k[1] for k in kept], (640, 480), None, None)
    res.update(rms_px=float(rms), K=K.tolist(), dist=D.ravel().tolist(), fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))
json.dump(res, open(OUT / "global_cam_intrinsics.json", "w"), indent=1)
prog.write_text(prog.read_text() + "DONE " + json.dumps({k: v for k, v in res.items() if k not in ("K",)}) + "\n")
print(json.dumps({k: v for k, v in res.items() if k not in ("K",)}))
