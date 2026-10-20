"""Stereo Phase recorder: two ceiling-facing Arducam fisheyes (left 0x00140000, right 0x03110000) + the right UMI Teensy IMU (2026-10-15).
Cameras are picked by USB LOCATION (AVFoundation indices change on replug), exposure locked to the same value on both (after an
auto->manual toggle: a camera can get stuck washed out otherwise). Each camera runs in its own thread with its own host receive stamps;
frames are saved at 1080p / save_scale as grey mp4 + <cam>_t_ns.npy, IMU as imu_right.npz (device_us, host_ns, gyro, acc).
Spoken Korean protocol (--proto):
  calib  (~80 s) still start, AprilGrid held/fixed in view of BOTH cameras: near/far, corners, tilts (intrinsics + stereo extrinsics),
                 then rotations and brisk translation (camera-IMU), still end
  still  (~82 s) 15x (move to a new pose 2.5 s -> HOLD STILL 2.5 s): calibration uses the still frames only (L/R skew irrelevant)
  eval   (~90 s) spec S7 evaluation take: still, +-10 cm X/Y/Z, yaw +-30, pitch/roll, still, fast, stop->resume
    .venv/bin/python -m visual_teleop.tools.stereo_imu_record [--proto calib|eval] [--save-scale 2] [--exposure 15]"""
from __future__ import annotations
import argparse, json, subprocess, threading, time
from pathlib import Path
import cv2
import numpy as np
from visual_teleop import config as vcfg
from visual_teleop.tools.stereo_check import index_for_location

ROOT = Path(__file__).resolve().parents[2]
U = str(Path.home() / "robot-cockpit/bin/uvc-util")
RULER_CM = 30   # 10-16 ruler length used by the "ruler" protocol
PROTOS = {
    "calib": [(6, "보드가 두 카메라에 다 보이게 들고, 완전히 가만히 계세요"), (12, "보드를 보면서 천천히 가깝게 그리고 멀게 움직이세요"),
              (12, "보드가 화면 네 귀퉁이에 차례로 오도록 천천히 움직이세요"), (12, "천천히 앞뒤 좌우로 기울이세요"),
              (12, "보드를 보면서 좌우로 돌리고 끄덕이고 비트세요, 천천히 그리고 빠르게"), (12, "앞뒤 좌우 위아래로 빠르게 왕복하세요. 보드는 화면 안에"),
              (8, "가만히 계세요")],
    "still": [(4, "보드가 두 카메라에 다 보이게 들고, 완전히 가만히 계세요")] + [x for k, hint in enumerate(
        ["가깝게", "멀게", "왼쪽 위 귀퉁이로", "오른쪽 위 귀퉁이로", "왼쪽 아래 귀퉁이로", "오른쪽 아래 귀퉁이로", "가운데 중간 거리로", "왼쪽으로 기울여서",
         "오른쪽으로 기울여서", "앞으로 기울여서", "뒤로 기울여서", "가깝게 귀퉁이로", "멀리 귀퉁이로", "비틀어서", "가운데 가깝게"])
        for x in ((2.5, f"{hint} 옮기세요"), (2.5, "멈추세요"))] + [(3, "끝났습니다, 가만히 계세요")],
    "rot": [(4, "가만히 들고 계세요"), (7, "좌우로 돌리세요, 천천히 그리고 빠르게"), (7, "위아래로 끄덕이세요, 천천히 그리고 빠르게"),
            (7, "손목을 비트세요, 천천히 그리고 빠르게"), (6, "여러 방향을 섞어서 흔드세요"), (3, "가만히 계세요")],
    "ruler": [(5, "유미를 자의 시작점에 놓고 완전히 가만히 계세요. 돌리지 말고 자를 따라 똑바로 미세요")] + [x for _ in range(3) for x in ((4, f"자를 따라 {RULER_CM}센티 앞으로 미세요"), (3, "멈추세요"), (4, "시작점으로 돌아오세요"), (3, "멈추세요"))]
             + [(3, "끝났습니다, 가만히 계세요")],   # 10-16 ruler scale check: known 15 cm, no rotation
    # 10-16 VIO acceptance after the camera-IMU joint calibration: pure axes with holds (board flat on the desk, in view of both cameras)
    "verify_x": [(6, "보드가 보이게 들고 완전히 가만히 계세요")]
                + [x for _ in range(3) for x in ((4, "앞으로 20센티 똑바로 미세요"), (3, "멈춘 채로 계세요"), (4, "제자리로 돌아오세요"), (3, "멈춘 채로 계세요"))]
                + [(10, "10초 동안 완전히 가만히 계세요"), (2, "끝났습니다")],   # 10-16 user: forward/back first
    "verify_z": [(6, "보드가 보이게 들고 완전히 가만히 계세요")]
                + [x for _ in range(3) for x in ((4, "위로 20센티 똑바로 올리세요"), (3, "멈춘 채로 계세요"), (4, "제자리로 내려오세요"), (3, "멈춘 채로 계세요"))]
                + [(10, "10초 동안 완전히 가만히 계세요"), (2, "끝났습니다")],   # 10-16 user: up/down second
    "verify_y": [(6, "보드가 보이게 들고 완전히 가만히 계세요")]
                + [x for _ in range(3) for x in ((4, "왼쪽으로 20센티 똑바로 옮기세요"), (3, "멈춘 채로 계세요"), (4, "제자리로 돌아오세요"), (3, "멈춘 채로 계세요"))]
                + [(10, "10초 동안 완전히 가만히 계세요"), (2, "끝났습니다")],   # 10-16 user: left/right third
    "verify": [(6, "보드가 보이게 들고 완전히 가만히 계세요")]
              + [x for _ in range(2) for x in ((4, "앞으로 20센티 똑바로 미세요"), (3, "멈춘 채로 계세요"), (4, "제자리로 돌아오세요"), (3, "멈춘 채로 계세요"))]
              + [x for _ in range(2) for x in ((4, "위로 20센티 똑바로 올리세요"), (3, "멈춘 채로 계세요"), (4, "제자리로 내려오세요"), (3, "멈춘 채로 계세요"))]
              + [(4, "왼쪽으로 20센티 똑바로 옮기세요"), (3, "멈춘 채로 계세요"), (4, "제자리로 돌아오세요"), (3, "멈춘 채로 계세요")]
              + [(5, "제자리에서 좌우로 돌리세요"), (3, "멈춘 채로 계세요"), (5, "제자리에서 위아래로 끄덕이세요"), (3, "멈춘 채로 계세요"),
                 (5, "제자리에서 손목을 비트세요"), (3, "멈춘 채로 계세요"), (30, "30초 동안 완전히 가만히 계세요"), (3, "끝났습니다")],
    "eval": [(5, "완전히 가만히 계세요"), (10, "앞쪽으로 10센티 옮기고 멈추세요"), (10, "다시 10센티 뒤로 돌아오세요"), (10, "왼쪽으로 10센티 옮기고 멈추세요"),
             (10, "위로 10센티 올리고 멈추세요"), (10, "좌우로 30도씩 돌리세요"), (10, "끄덕이고 비트세요"), (5, "가만히 계세요"),
             (10, "빠르게 마음대로 움직이세요"), (5, "멈추세요"), (5, "다시 움직이세요"), (5, "유미를 든 채로 가만히 계세요. 내려놓지 마세요")],
}


def say(t: str) -> None:
    subprocess.Popen(["say", "-v", "Yuna", t])


def lock(location: str, exposure: int) -> str:
    lst = subprocess.run([U, "-d"], capture_output=True, text=True).stdout
    line = next((l for l in lst.splitlines() if location.lower() in l.lower()), None)
    if line is None: return f"{location}: no uvc device"
    i = line.split()[0]
    # auto -> (settle 2 s) -> manual: a too-fast toggle left a camera stuck fully white (10-15 night, twice)
    subprocess.run([U, "-I", i, "-s", "auto-exposure-mode=8"], capture_output=True); time.sleep(2.0)
    for s in ("auto-exposure-mode=1", f"exposure-time-abs={exposure}", "gain=0", "brightness=0", "gamma=100"):
        subprocess.run([U, "-I", i, "-s", s], capture_output=True); time.sleep(0.3)
    return f"uvc #{i} manual {exposure}"


class Cam(threading.Thread):
    def __init__(self, name: str, index: int, out: Path, sc: int):
        super().__init__(daemon=True); self.name_ = name; self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.size = (1920 // sc, 1080 // sc); self.vw = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 30.0, self.size)
        self.t: list[int] = []; self.on = True; self.rec = False; self.out = out; self.last = None; self.tl = 0
        self.full = False; self.full_every = 3; self._n = 0; self.q: list = []; self.full_dir = out / f"{name}_full"; self.full_dir.mkdir(exist_ok=True)
        threading.Thread(target=self._writer, daemon=True).start()
    def _writer(self):                       # full-res PNG writes off the capture thread (still phases only)
        while self.on or self.q:
            if self.q: t, g = self.q.pop(0); cv2.imwrite(str(self.full_dir / f"{t}.png"), g)
            else: time.sleep(0.005)
    def run(self):
        while self.on:
            ok, f = self.cap.read(); t = time.monotonic_ns(); self.tl = t if ok else self.tl
            if not ok: continue
            g = cv2.cvtColor(cv2.resize(f, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); self.last = g
            if self.rec: self.vw.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)); self.t.append(t)
            self._n += 1
            if self.rec and self.full and self._n % self.full_every == 0: self.q.append((t, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
        self.cap.release(); self.vw.release(); np.save(self.out / f"{self.name_}_t_ns.npy", np.asarray(self.t, np.int64))


PREVIEW = {"jpg": None, "phase": "준비", "status": ""}


def start_preview(L, R, port: int) -> None:
    """MJPEG preview of BOTH cameras for the whole run (aiming, gate, recording): tag counts, L/R stereo matches + vertical offset dy
    (once a second), current spoken phase, and a red STALL banner if a camera stops delivering frames."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from visual_teleop.tools.stereo_check import overlap
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)); st = {"m": "-", "t": 0.0}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        if PREVIEW["jpg"]: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + PREVIEW["jpg"] + b"\r\n")
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError): return
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write("<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>".encode())
        def log_message(self, *a): pass

    def render():
        while L.on or R.on:
            if L.last is None or R.last is None: time.sleep(0.1); continue
            gl, gr = L.last, R.last; vis = []
            for name, g, c in (("LEFT", gl, L), ("RIGHT", gr, R)):
                cc, ids, _ = det.detectMarkers(g); n = 0 if ids is None else len(ids); v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                if ids is not None: cv2.aruco.drawDetectedMarkers(v, cc)
                cv2.putText(v, f"{name} tags {n}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0) if n >= 8 else (0, 0, 255), 3)
                if c.tl and time.monotonic_ns() - c.tl > 1e9: cv2.putText(v, "STALL", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                vis.append(v)
            if time.time() - st["t"] > 1.0:
                ov, _ = overlap(cv2.resize(gl, (640, 360)), cv2.resize(gr, (640, 360))); dy = ov.get("dy_median_px")
                st.update(m=f"stereo {ov['stereo_inliers']}  dy(1080p) {round(dy * 3, 1) if dy is not None else '-'} px", t=time.time())
            out = np.hstack(vis); cv2.rectangle(out, (0, out.shape[0] - 50), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
            cv2.putText(out, f"{st['m']}   |   {PREVIEW['status']}", (12, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
            PREVIEW["jpg"] = cv2.imencode(".jpg", cv2.resize(out, (1600, 450)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes(); time.sleep(0.1)

    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever, daemon=True).start(); threading.Thread(target=render, daemon=True).start()
    subprocess.Popen(["open", f"http://localhost:{port}/"]); print(f"[stereo] 미리보기 http://localhost:{port}/", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proto", default="calib", choices=sorted(PROTOS)); ap.add_argument("--save-scale", type=int, default=2); ap.add_argument("--exposure", type=int, default=15)
    ap.add_argument("--left-location", default="0x03110000"); ap.add_argument("--right-location", default="0x03140000"); ap.add_argument("--side", default="right")
    ap.add_argument("--left-index", type=int, default=-1); ap.add_argument("--right-index", type=int, default=-1)
    ap.add_argument("--port", type=int, default=8745); ap.add_argument("--gate-seconds", type=float, default=120.0, help="how long to wait for the board in BOTH cameras"); a = ap.parse_args()
    out = ROOT / "datasets" / "stereo_imu" / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.proto}"; out.mkdir(parents=True, exist_ok=True)
    il = a.left_index if a.left_index >= 0 else index_for_location(a.left_location); ir = a.right_index if a.right_index >= 0 else index_for_location(a.right_location)
    if il == ir: raise SystemExit("두 카메라 인덱스가 같게 잡혔습니다 — 연결 확인")
    print(f"[stereo] 왼쪽 {a.left_location}=index {il}, 오른쪽 {a.right_location}=index {ir}", flush=True)
    L, R = Cam("cam_left", il, out, a.save_scale), Cam("cam_right", ir, out, a.save_scale)
    expo = dict(left=lock(a.left_location, a.exposure), right=lock(a.right_location, a.exposure)); print("[stereo] 노출", expo, flush=True)
    L.start(); R.start(); time.sleep(1.5); start_preview(L, R, a.port); PREVIEW["status"] = "AIM: both tags 8+ -> auto start"
    # --- PRE-RECORDING GATE: both frames must show the AprilGrid and match each other as a stereo pair (otherwise wrong camera / not mounted)
    from visual_teleop.tools.stereo_check import overlap
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    gate_ok = a.proto not in ("calib", "still", "eval", "verify", "verify_x", "verify_y", "verify_z", "ruler"); good = 0
    if gate_ok: PREVIEW["status"] = "no board gate for this protocol"
    gate_ok_needed = a.proto in ("calib", "still", "eval", "verify", "verify_x", "verify_y", "verify_z", "ruler"); tg = time.monotonic()      # board gate only for board takes
    if gate_ok_needed: say("미리보기를 보면서 보드가 두 화면에 다 보이게 맞추세요")
    for attempt in range(10_000 if gate_ok_needed else 0):
        if time.monotonic() - tg > a.gate_seconds: break
        if L.last is None or R.last is None: time.sleep(0.5); continue
        gl, gr = L.last.copy(), R.last.copy(); nl = 0 if (x := det.detectMarkers(gl)[1]) is None else len(x); nr = 0 if (x := det.detectMarkers(gr)[1]) is None else len(x)
        ov, _ = overlap(gl, gr); dy = ov.get("dy_median_px")
        print(f"[stereo] 사전 점검 {attempt + 1}: 보드 태그 왼쪽 {nl} / 오른쪽 {nr}, 스테레오 정합 {ov['stereo_inliers']} (비율 {ov['inlier_ratio']}), 상하 어긋남 {dy} px", flush=True)
        cv2.imwrite(str(out / "pregate.jpg"), np.hstack([gl, gr]))
        dy1080 = abs(dy) * (1080 / gl.shape[0]) if dy is not None else 1e9
        ok_now = nl >= 8 and nr >= 8 and ov["stereo_inliers"] >= 60 and ov["inlier_ratio"] >= 0.4 and dy1080 < 20; good = good + 1 if ok_now else 0
        PREVIEW["status"] = f"AIM {'OK ' + str(good) + '/3' if ok_now else '...'}  (L {nl} / R {nr}, dy {dy1080:.0f}px{' HEIGHT MISMATCH' if dy1080 >= 20 else ''})"
        if good >= 3: gate_ok = True; break
        time.sleep(1.0)
    if not gate_ok:
        L.on = R.on = False; L.join(3); R.join(3); say("두 카메라 점검에 실패해서 녹화를 시작하지 않습니다")
        raise SystemExit(f"[stereo] 사전 점검 실패: 두 카메라 중 하나가 보드를 못 보거나 스테레오 짝이 아닙니다. 확인 이미지 {out / 'pregate.jpg'}")
    br = {n: round(float(c.last.mean()), 1) for n, c in (("left", L), ("right", R)) if c.last is not None}; print("[stereo] 밝기", br, flush=True)
    from .calib_right import open_imu
    try: dev = open_imu(vcfg.load(), a.side); dev.buffer.drain()
    except RuntimeError as e:
        if a.proto != "still": raise
        print(f"[stereo] IMU 없음 ({e}) -- still(스테레오 전용)은 IMU 없이 녹화", flush=True)
        class _NoImu:                  # 10-16: XIAO unplugged while re-gluing the mount; stereo stills need no IMU
            class buffer:
                @staticmethod
                def drain(): return []
            def stop(self): pass
            def close(self): pass
        dev = _NoImu()
    sched, acc = [], 0.0
    for d, s in PROTOS[a.proto]: sched.append((acc, s)); acc += d
    (out / "protocol.json").write_text(json.dumps([dict(t0=st, text=tx) for st, tx in sched], ensure_ascii=False))
    (out / "meta.json").write_text(json.dumps(dict(save_scale=a.save_scale, size=list(L.size), left=dict(location=a.left_location, index=il), right=dict(location=a.right_location, index=ir),
                                                   exposure=expo, brightness=br, proto=a.proto, imu_side=a.side,
                                                   calibration=dict(stereo="configs/calibration/stereo_trackcam_v005.yaml", cam_imu="configs/calibration/stereo_trackcam_imu_v004.yaml",
                                                                    imu_noise="configs/calibration/imu_noise_xiao_right_v001.yaml", imu_serial=vcfg.load().imu[a.side]["serial_number"]),
                                                   phases_file="protocol.json (t0 s from recording start)", board_gt="scripts/trackcam_board_lever.py --stereo-cam cam_left"), ensure_ascii=False, indent=1))
    PREVIEW["status"] = "starting in 3 s"; say("좋습니다. 3초 뒤 녹화를 시작합니다"); time.sleep(4); dev.buffer.drain(); S = []
    L.rec = R.rec = True; t0 = time.monotonic(); ph = -1; aborted = None
    while (el := time.monotonic() - t0) < acc:
        k = max(i for i, (st, _) in enumerate(sched) if st <= el)
        if k != ph: ph = k; say(sched[k][1]); print(f"[stereo] {el:4.1f}s {sched[k][1]}", flush=True)
        PREVIEW["status"] = f"REC {el:4.1f}/{acc:.0f}s  {'HOLD STILL' if ('멈추세요' in sched[k][1] or '가만히' in sched[k][1]) else 'MOVE'}"
        stale = [n for n, c in (("왼쪽", L), ("오른쪽", R)) if c.t and time.monotonic_ns() - c.t[-1] > 1.0e9]
        if stale: say(f"{stale[0]} 카메라가 멈췄습니다. 녹화를 중단합니다"); print(f"[stereo] 중단: {stale} 카메라 프레임이 1초 넘게 안 들어옴", flush=True); aborted = f"{stale} stalled at {el:.1f}s"; break
        still_phase = ("멈추세요" in sched[k][1] or "가만히" in sched[k][1]) and el - sched[k][0] > 0.8   # settle 0.8 s after the cue
        L.full = R.full = a.proto == "still" and still_phase
        S += dev.buffer.drain(); time.sleep(0.01)
    L.rec = R.rec = False; L.full = R.full = False; S += dev.buffer.drain(); dev.stop(); dev.close(); L.on = R.on = False; L.join(5); R.join(5)
    while L.q or R.q: time.sleep(0.1)
    say("녹화가 끝났습니다"); print(f"[stereo] 전체 해상도 정지 프레임: 왼쪽 {len(list(L.full_dir.glob('*.png')))} / 오른쪽 {len(list(R.full_dir.glob('*.png')))}", flush=True)
    np.savez(out / f"imu_{a.side}.npz", device_us=np.array([s.device_timestamp_us for s in S]), host_ns=np.array([s.host_receive_ns for s in S]),
             gyro=np.array([[s.gx, s.gy, s.gz] for s in S]), acc=np.array([[s.ax, s.ay, s.az] for s in S]))
    tl, tr = np.load(out / "cam_left_t_ns.npy"), np.load(out / "cam_right_t_ns.npy")
    j = np.clip(np.searchsorted(tr, tl), 1, len(tr) - 1); nn = np.where(np.abs(tr[j] - tl) < np.abs(tr[j - 1] - tl), tr[j], tr[j - 1]); sk = np.abs(tl - nn) / 1e6
    # --- POST-RECORDING CHECK: board detection rate in BOTH cameras (every 15th frame of the saved videos)
    rate = {}
    for nm in ("cam_left", "cam_right"):
        cap = cv2.VideoCapture(str(out / f"{nm}.mp4")); k = hit = n = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            if k % 15 == 0:
                n += 1; ids = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))[1]; hit += int(ids is not None and len(ids) >= 8)
            k += 1
        rate[nm] = round(hit / max(n, 1), 2)
    full_n = {nm: len(list((out / f"{nm}_full").glob("*.png"))) for nm in ("cam_left", "cam_right")}
    board_needed = a.proto in ("calib", "still", "eval")
    verdict = "FAIL" if aborted else "OK" if (min(rate.values()) >= 0.5 or not board_needed) and len(tl) > 0.9 * len(tr) and len(tr) > 0.9 * len(tl) else "FAIL"
    (out / "post_check.json").write_text(json.dumps(dict(board_rate=rate, frames=dict(left=int(len(tl)), right=int(len(tr))), full_frames=full_n, aborted=aborted, verdict=verdict), ensure_ascii=False))
    print(f"[stereo] 녹화 후 점검: 보드 검출률 왼쪽 {rate['cam_left']} / 오른쪽 {rate['cam_right']}, 전체해상도 정지 {full_n} → {verdict}", flush=True)
    print(f"[stereo] 저장 {out}: 왼쪽 {len(tl)} / 오른쪽 {len(tr)} 프레임, IMU {len(S)}, 좌우 시간차 p50 {np.median(sk):.1f} p95 {np.percentile(sk, 95):.1f} ms", flush=True)


if __name__ == "__main__":
    main()
