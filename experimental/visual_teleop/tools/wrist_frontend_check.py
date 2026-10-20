"""DIY wrist 6DoF tracker -- Phase 1: can the TRACKING fisheye (UMI-mounted, facing the static room) feed a visual frontend?
(2026-10-15, spec "DIY Wrist 6DoF Tracker v1"). No intrinsics needed yet: geometry check = fundamental matrix + RANSAC.

Per frame (grayscale, 640x360): Shi-Tomasi corners (re-detected when < 150 tracked) -> pyramidal LK -> forward-backward check
(< 1 px) -> F-matrix RANSAC (1 px). Logged: tracked, inliers, inlier ratio, median flow, sharpness (Laplacian var), latency.
Exposure is locked (manual, short shutter) through uvc-util by USB location. A spoken 30 s protocol: still / translate / rotate /
fast / cloth reach. PASS (spec): while MOVING, tracked > 80, inliers > 40, ratio > 0.5, >= 30 Hz, no repeated collapse.
Live view http://127.0.0.1:8743.
    .venv/bin/python -m visual_teleop.tools.wrist_frontend_check [--index 4] [--location 0x00140000] [--exposure 15] [--seconds 30]"""
from __future__ import annotations
import argparse, csv, json, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTO = [(5, "가만히 들고 계세요"), (7, "유미를 앞뒤 좌우로 천천히 움직이세요"), (7, "손목을 돌리고 숙이세요"), (5, "빠르게 움직이세요"), (6, "천을 집으러 가듯이 작업대로 손을 뻗으세요")]
LATEST = {"jpg": None}


def say(s):
    try: subprocess.Popen(["say", "-v", "Yuna", s])
    except Exception: pass


def lock_exposure(location: str, exposure: int) -> str:
    U = Path.home() / "robot-cockpit/bin/uvc-util"
    lst = subprocess.run([str(U), "-d"], capture_output=True, text=True).stdout
    line = next((l for l in lst.splitlines() if location.lower() in l.lower()), None)
    if line is None: return f"uvc-util: no camera at {location} (exposure NOT locked)"
    i = line.split()[0]
    for a in ("auto-exposure-mode=1", f"exposure-time-abs={exposure}"): subprocess.run([str(U), "-I", i, "-s", a], capture_output=True)
    return f"uvc #{i} manual exposure {exposure}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=4); ap.add_argument("--location", default="0x00140000"); ap.add_argument("--exposure", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=30.0); ap.add_argument("--port", type=int, default=8743); a = ap.parse_args()
    out = ROOT / "datasets" / "wrist_frontend" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv2.CAP_PROP_FPS, 30)
    print("[frontend]", lock_exposure(a.location, a.exposure), flush=True)
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
            b = "<html><head><meta charset=utf-8><title>wrist frontend</title></head><body style='margin:0;background:#111'><img src='/stream' style='width:100vw'></body></html>".encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(b)
        def log_message(self, *x): pass
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); subprocess.Popen(["open", f"http://127.0.0.1:{a.port}"])
    sched, acc = [], 0.0
    for d, s in PROTO: sched.append((acc, s)); acc += d
    vw = cv2.VideoWriter(str(out / "tracking_cam.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 30.0, (640, 360)); rows = []
    prev, pts = None, None; lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    say("추적 카메라 점검을 시작합니다"); time.sleep(2.5); t0 = time.monotonic(); ph = -1; last_t = None
    while (el := time.monotonic() - t0) < a.seconds:
        k = max(i for i, (st, _) in enumerate(sched) if st <= el)
        if k != ph: ph = k; say(sched[k][1]); print(f"[frontend] {el:4.1f}s {sched[k][1]}", flush=True)
        ok, f = cap.read(); t_ns = time.monotonic_ns()
        if not ok: continue
        tA = time.perf_counter(); g = cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2GRAY)
        tracked = inl = 0; flow = np.nan
        if prev is not None and pts is not None and len(pts):
            p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **lk); p0b, s2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **lk)
            fb = np.linalg.norm((pts - p0b).reshape(-1, 2), axis=1); good = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < 1.0)
            a0, a1 = pts.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]; tracked = int(good.sum())
            if tracked >= 8:
                flow = float(np.median(np.linalg.norm(a1 - a0, axis=1)))
                if flow > 0.3:
                    F, m = cv2.findFundamentalMat(a0, a1, cv2.FM_RANSAC, 1.0, 0.99); inl = int(m.sum()) if m is not None else 0
                else: inl = tracked                                            # no parallax: the geometry test is meaningless
            pts = a1.reshape(-1, 1, 2)
        if pts is None or len(pts) < 150:
            new = cv2.goodFeaturesToTrack(g, 300, 0.01, 8)
            pts = new if pts is None or not len(pts) else (np.vstack([pts, new]) if new is not None else pts)
        lat = (time.perf_counter() - tA) * 1e3; sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        dt = (t_ns - last_t) / 1e6 if last_t else np.nan; last_t = t_ns
        rows.append(dict(t_s=round(el, 3), phase=k, tracked=tracked, inliers=inl, ratio=round(inl / tracked, 3) if tracked else 0.0, median_flow_px=round(flow, 2) if np.isfinite(flow) else "",
                         sharpness=round(sharp), latency_ms=round(lat, 2), camera_dt_ms=round(dt, 1) if np.isfinite(dt) else ""))
        prev = g; v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for q in pts.reshape(-1, 2)[:300]: cv2.circle(v, (int(q[0]), int(q[1])), 2, (0, 255, 0), -1)
        cv2.putText(v, f"{sched[k][1][:0]}tracked {tracked} inl {inl} flow {flow:.1f} {lat:.0f}ms", (8, 22), 0, 0.6, (0, 255, 255), 2); vw.write(v)
        LATEST["jpg"] = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    vw.release(); cap.release(); say("점검이 끝났습니다")
    with open(out / "frontend.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    R = rows[1:]; T = np.array([r["t_s"] for r in R]); fps = len(R) / (T[-1] - T[0])
    mov = [r for r in R if r["phase"] in (1, 2, 3, 4)]; tr = np.array([r["tracked"] for r in mov]); il = np.array([r["inliers"] for r in mov]); ra = np.array([r["ratio"] for r in mov])
    collapse = tr < 30; runs, cur = [], 0
    for c_ in collapse:
        if c_: cur += 1
        elif cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    S = dict(fps=round(fps, 1), moving_frames=len(mov), tracked_p10=int(np.percentile(tr, 10)), tracked_p50=int(np.median(tr)), inliers_p10=int(np.percentile(il, 10)),
             inliers_p50=int(np.median(il)), ratio_p10=round(float(np.percentile(ra, 10)), 2), ratio_p50=round(float(np.median(ra)), 2), collapses=len(runs), longest_collapse_frames=max(runs, default=0),
             latency_ms_p95=round(float(np.percentile([r["latency_ms"] for r in R], 95)), 1),
             per_phase={PROTO[p][1]: dict(tracked_p10=int(np.percentile([r["tracked"] for r in R if r["phase"] == p], 10)), inliers_p10=int(np.percentile([r["inliers"] for r in R if r["phase"] == p], 10))) for p in range(len(PROTO)) if any(r["phase"] == p for r in R)})
    S["PASS"] = bool(S["tracked_p10"] > 80 and S["inliers_p10"] > 40 and S["ratio_p10"] > 0.5 and S["fps"] >= 28 and S["collapses"] == 0)
    (out / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False)); print(json.dumps(S, indent=1, ensure_ascii=False)); print(f"[frontend] 저장 {out}", flush=True)


if __name__ == "__main__":
    main()
