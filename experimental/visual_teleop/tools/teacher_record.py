"""Teacher recording for the markerless UMI-keypoint tracker (2026-10-15). 10-15 user: position camera = the Orbbec (depth), the C922
is removed -> default mode --hub: the running sensor hub records Orbbec colour + depth + IMU (rec_start/rec_stop on the cmd port) and
this tool only speaks the protocol. The older stand-alone C922 mode is kept as --c922. The 45 mm AprilTag (tag36h11 id 21) on the UMI is
used ONLY here, to auto-label keypoints later; at run time the tracker is markerless (C922 keypoints + IMU orientation + UMI geometry).

Records, on one host clock:  C922 raw frames (focus locked, no undistortion) -> c922.mp4 + c922_t_ns.npy
                             right UMI IMU (Teensy) raw samples + AHRS orientation -> imu_right.npz
                             meta.json (camera index/size/focus, protocol timing)
A spoken Korean protocol (default 4 min) walks through: near/far, left/right/up/down, roll/pitch/yaw, carrying, cloth grasps, partial
occlusion of the UMI, fast and slow moves. A live view with the tag-detection rate is served at http://127.0.0.1:8741.
The sensor hub must be OFF (it holds the Teensy). The C922 live viewer (c922_live) must be OFF (it holds the camera).
    .venv/bin/python -m visual_teleop.tools.teacher_record [--index 3] [--width 1920 --height 1080] [--minutes 4]"""
from __future__ import annotations
import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np
from .. import config as vcfg
from ..imu.imu_reader import ImuReader, open_teensy

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = [   # (seconds, spoken instruction)
    (20, "유미를 카메라 가까이에서 천천히 움직이세요"), (20, "이번에는 카메라에서 멀리, 작업대 끝 쪽에서 움직이세요"),
    (20, "왼쪽 오른쪽, 위 아래로 크게 움직이세요"), (20, "제자리에서 집게 축을 중심으로 비트세요"), (20, "집게 끝을 위아래로 끄덕이세요"),
    (20, "집게 끝을 좌우로 돌리세요"), (25, "유미를 들고 작업대 위를 자유롭게 옮겨 다니세요"), (30, "천을 집었다 놓았다 하세요. 천을 끌어 보세요"),
    (20, "손이나 천으로 유미 일부를 가렸다가 보여주세요"), (15, "빠르게 움직이세요"), (15, "아주 천천히 움직이세요"), (15, "가만히 들고 계세요")]
LAYOUT = [   # 10-15: red-tape layout calibration (~1 min): every tape must face the Orbbec in some frames, tip visible
    (12, "유미를 카메라 쪽으로 보이게 들고 천천히 왼쪽 오른쪽으로 돌리세요"), (12, "이번에는 위아래로 천천히 기울이세요"),
    (12, "집게 축을 중심으로 천천히 비트세요"), (12, "작업대 위에서 천천히 이리저리 옮기세요"), (12, "가만히 들고 계세요")]
LATEST = {"jpg": None, "tag_rate": 0.0, "phase": ""}


def say(s: str) -> None:
    try: subprocess.Popen(["say", "-v", "Yuna", s])
    except Exception: pass


def lock_focus(cfg: dict) -> str:
    from ..vision.tag_camera import TagCamera
    dummy = TagCamera.__new__(TagCamera); dummy.cfg = cfg; return TagCamera._lock_focus(dummy, cfg.get("focus", "auto"))


def hub_mode(c, minutes: float, protocol=None) -> None:
    """Protocol voice + hub recording (Orbbec colour mp4 + depth PNGs + imu npz + pose/det logs land in the session dir)."""
    # 10-15: live_teleop (preview) owns the hand port, so record THROUGH its dashboard: /rec_start makes an episode dir with the
    # control ticks (human pose = teacher pose) + the top C922, and tells the hub to write Orbbec colour/depth + IMU into it.
    import requests
    D = "http://127.0.0.1:8730"; r = requests.post(D + "/rec_start", json={}, timeout=5).json()
    if not r.get("ok"): raise SystemExit(f"dashboard /rec_start failed: {r}")
    out = Path(r["dir"]); print(f"[teacher] episode dir {out}", flush=True)
    sched, acc = [], 0.0; protocol = protocol or PROTOCOL
    for d, s in protocol: sched.append((acc, s)); acc += d
    total = minutes * 60 if minutes > 0 else acc
    say("교사 녹화를 시작합니다. 유미가 카메라에 보이게 들고 계세요."); t0 = time.monotonic(); k = -1
    try:
        while (el := time.monotonic() - t0) < total:
            ph = max(i for i, (st, _) in enumerate(sched) if st <= el) if el < acc else len(sched) - 1
            if ph != k: k = ph; say(sched[ph][1]); print(f"[teacher] {el:5.1f}s {sched[ph][1]}", flush=True)
            time.sleep(0.1)
    finally:
        r = requests.post(D + "/rec_stop", json={}, timeout=60).json(); time.sleep(3.0)
        (out / "teacher_meta.json").write_text(json.dumps(dict(mode="dashboard_orbbec_tagless", teacher="kpt+depth position, IMU orientation", protocol=protocol,
                                                              seconds=round(time.monotonic() - t0, 1), rec_stop=r), indent=1, ensure_ascii=False, default=str))
        say("녹화가 끝났습니다."); print(f"[teacher] 저장: {out} -> {sorted(x.name for x in out.iterdir())}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c922", action="store_true", help="old stand-alone C922 mode (camera + Teensy opened here)")
    ap.add_argument("--layout", action="store_true", help="1-min red-tape layout calibration protocol instead of the 4-min one")
    ap.add_argument("--index", type=int, default=3); ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--minutes", type=float, default=0.0, help="0 = run the whole protocol (~4 min)"); ap.add_argument("--side", default="right")
    ap.add_argument("--tag-id", type=int, default=21); ap.add_argument("--port", type=int, default=8741)
    a = ap.parse_args(); c = vcfg.load()
    if not a.c922: return hub_mode(c, a.minutes, LAYOUT if a.layout else None)
    out = ROOT / "datasets" / "umi_teacher" / time.strftime("%Y%m%d_%H%M%S"); (out).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height); cap.set(cv2.CAP_PROP_FPS, 30)
    focus = lock_focus(dict(c.cameras["tag_cam"], focus="auto")); print("[teacher] C922", focus, flush=True)
    for _ in range(60):
        ok, f = cap.read()
        if ok and f.shape[1] == a.width: break
        time.sleep(0.05)
    else: raise SystemExit(f"C922 (OpenCV {a.index}) did not deliver {a.width}x{a.height}")
    dev = open_teensy(a.side, c.imu); imu = ImuReader(a.side, dev, c.imu["ahrs"], max_age_ms=c.imu["max_age_ms"], min_rate_hz=c.imu["min_rate_hz"]); imu.start()
    from pupil_apriltags import Detector
    det = Detector(families="tag36h11", nthreads=2, quad_decimate=2.0)
    vw = cv2.VideoWriter(str(out / "c922.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 30.0, (a.width, a.height)); ts, phases = [], []
    stop = threading.Event()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while not stop.is_set():
                        j = LATEST["jpg"]
                        if j: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError): return
            body = "<html><head><meta charset=utf-8><title>teacher rec</title></head><body style='margin:0;background:#111'><img src='/stream' style='width:100vw'></body></html>".encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)
        def log_message(self, *x): pass
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    subprocess.Popen(["open", f"http://127.0.0.1:{a.port}"])
    sched, acc = [], 0.0
    for d, s in PROTOCOL: sched.append((acc, s)); acc += d
    total = a.minutes * 60 if a.minutes > 0 else acc
    say("교사 녹화를 시작합니다. 태그가 카메라에 보이게 들고 계세요."); time.sleep(4)
    t0 = time.monotonic(); k = -1; hist = []; n = 0
    try:
        while (el := time.monotonic() - t0) < total:
            ph = max(i for i, (st, _) in enumerate(sched) if st <= el) if el < acc else len(sched) - 1
            if ph != k: k = ph; LATEST["phase"] = sched[ph][1]; say(sched[ph][1]); print(f"[teacher] {el:5.1f}s {sched[ph][1]}", flush=True)
            ok, f = cap.read(); t = time.monotonic_ns()
            if not ok: continue
            vw.write(f); ts.append(t); phases.append(ph); n += 1
            if n % 5 == 0:                                                   # live check only (labels are made offline)
                g = cv2.cvtColor(cv2.resize(f, (960, 540)), cv2.COLOR_BGR2GRAY)
                hit = any(d.tag_id == a.tag_id for d in det.detect(g)); hist = (hist + [hit])[-30:]
                v = cv2.resize(f, (960, 540)); cv2.putText(v, f"tag seen (last 5 s): {100 * np.mean(hist):.0f}%  t={el:.0f}/{total:.0f}s", (10, 30), 0, 0.8, (0, 255, 255), 2)
                LATEST["jpg"] = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    finally:
        stop.set(); vw.release(); cap.release(); imu.stop()
        S = list(imu.samples_for_recorder)
        try: dev.stop(); dev.close()
        except Exception: pass
        np.save(out / "c922_t_ns.npy", np.asarray(ts, np.int64)); np.save(out / "c922_phase.npy", np.asarray(phases, np.int16))
        if S:
            np.savez(out / f"imu_{a.side}.npz", device_us=np.array([x.device_timestamp_us for x in S], np.int64), host_ns=np.array([x.host_receive_ns for x in S], np.int64),
                     acc=np.array([[x.ax, x.ay, x.az] for x in S]), gyro=np.array([[x.gx, x.gy, x.gz] for x in S]))
        (out / "meta.json").write_text(json.dumps(dict(camera=dict(index=a.index, width=a.width, height=a.height, focus=focus, uvc_location=c.cameras["tag_cam"].get("uvc_location")),
                                                      tag=dict(family="tag36h11", id=a.tag_id, size_m=0.045), imu=dict(side=a.side, serial=c.imu[a.side]["serial_number"], n=len(S)),
                                                      frames=len(ts), protocol=PROTOCOL, camera_imu_offset_ms=c.time_sync["camera_imu_offset_ms"].get(a.side)), indent=1, ensure_ascii=False))
        say("녹화가 끝났습니다."); print(f"[teacher] 저장: {out}  프레임 {len(ts)}, IMU {len(S)}", flush=True)


if __name__ == "__main__":
    main()
