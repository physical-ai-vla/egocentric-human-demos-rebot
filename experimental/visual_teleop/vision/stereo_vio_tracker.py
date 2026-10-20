"""DIY stereo-inertial 6DoF tracker -> tracker_pose UDP (2026-10-16, Stereo Phase S10, user-approved "1번: 바로 통합").
Two ceiling-facing Arducam fisheyes (left C0 0x03110000, right 0x00140000, stereo_trackcam_v003) + the right UMI Teensy IMU
(stereo_trackcam_imu_v001) -> OpenVINS stereo-inertial through ~/vio/ov_bridge.feed_stereo (3.4 ms/frame) -> UDP JSON in the exact
tracker6dof format (seq, t_src_ns, t_send_ns, p, q_xyzw, valid, conf) + extra fields (state, gyro_dps, n_feat, holds, resets, rezeros).
Tracker body TB = the IMU frame (calib_tracker pivot/mount then gives TB -> jaw TCP). Tracker world TW = the FIRST VIO world (z up).

Fast-rotation guard (user spec 10-16; offline: OpenVINS/Basalt stereo hold 0.2-1.6 cm up to ~166 deg/s and break at ~300 deg/s):
  |w| < 160 deg/s            TRACKING  valid, conf 1.0
  160..200 deg/s             DEGRADED  valid, conf 0.5, speed_scale 0.5 (receiver side may slow the robot)
  > 200 deg/s                HOLD      valid=false (the teleop holds; on recovery relative_se3 RE-ANCHORS = re-zero, no catch-up)
  recovery: |w| < 140 deg/s for 250 ms, then 5 consecutive HEALTHY frames -> TRACKING (rezero_count += 1)
  healthy = VIO initialised, n_msckf + n_slam >= MIN_FEAT, VIO speed < 1.5 m/s, |w_vio - w_gyro| < 30 deg/s
  divergence (unhealthy for > 1.0 s after the gyro is calm, or VIO speed > 2.5 m/s while TRACKING): reset the VIO; OpenVINS re-initialises
  when the UMI is held still (voice cue); the new VIO world is stitched onto TW by the gyro-propagated orientation (yaw only) so the
  receiver's world never rotates; translation continuity is irrelevant because every recovery is a re-anchor.
Sources: live cameras + Teensy, or --replay <stereo_imu_record take> (real-time paced, for MockBridge tests without hardware).
    .venv/bin/python -m visual_teleop.vision.stereo_vio_tracker [--replay datasets/stereo_imu/<ts>_eval] [--port 47813] [--log out.jsonl]"""
from __future__ import annotations
import argparse, collections, json, socket, subprocess, sys, threading, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from stereo_openvins_eval import write_config                        # noqa: E402
import stereo_openvins_eval as _ove, stereo_basalt_eval as _bse         # noqa: E402
CAMIMU_FILE = "configs/calibration/stereo_trackcam_imu_v004.yaml"   # 10-16 15:40: bracket slipped 16 deg (pitch); v003 before
_ove.CAMIMU_FILE = _bse.CAMIMU_FILE = CAMIMU_FILE
STEREO_FILE = "configs/calibration/stereo_trackcam_v005.yaml"   # 10-16 18:56 new camera mount (baseline 4.39 cm, parallel); v004 = old mount
_ove.STEREO_FILE = _bse.STEREO_FILE = STEREO_FILE
from ego_teleop.tracking.backends.openvins import import_bridge     # noqa: E402

DEG_TRACK, DEG_HOLD, DEG_RECOVER, CALM_S, HEALTHY_N = 160.0, 200.0, 140.0, 0.25, 5
# 10-16 live: n_msckf + n_slam (features USED in this frame's update) dips to 3-11 during ordinary 60-150 deg/s rotation while the
# rotation check / speed stay healthy -> 24 of 25 HOLDs were false. Feature count now only fails when ~0 for FEAT_ZERO_S.
FEAT_ZERO_S = 0.2
# 10-16 world calib near a white table: features 0-5 for ~2 s while still "TRACKING" and the pose jumped 50 cm -> any 3 consecutive
# frames below FEAT_LOW now fail health (live normal motion: p5 15 features)
FEAT_LOW, FEAT_LOW_FRAMES, FEAT_LOW_V = 10, 3, 0.3
# 10-16 pivot on a white desk: features 0 for seconds, VIO drifted 0.5-0.9 m/s (9 m), guard flipped HOLD<->TRACKING -> recovery is now
# STRICTER than staying: 5 consecutive frames with >= RECOVER_FEAT features and VIO speed < V_RECOVER
RECOVER_FEAT, V_RECOVER = 10, 0.6
STEP_V_MAX = 1.5            # pose change between consecutive frames, m/s
MIN_FEAT, V_HEALTHY, V_DIVERGE, W_MISMATCH_DPS, DIVERGE_S, RESET_COOLDOWN_S = 3, 1.5, 2.5, 30.0, 3.0, 3.0


def say(t: str) -> None:
    subprocess.Popen(["say", "-v", "Yuna", t])


class Guard:
    def __init__(self):
        self.state = "INIT"; self.calm_since = None; self.healthy_run = 0; self.unhealthy_since = None
        self.holds = self.resets = self.rezeros = self.degraded = 0; self.max_dps = 0.0; self.last_reset = -1e9

    def step(self, t: float, dps: float, healthy: bool, v: float, inited: bool = True, recover_ok: bool | None = None, diverging: bool = True) -> str:
        """returns the action: '' | 'reset'. Updates self.state. A reset is only issued when the VIO IS initialised but unhealthy
        (an un-initialised VIO is waiting for stillness -- resetting it again would restart that wait forever), at most every RESET_COOLDOWN_S."""
        self.max_dps = max(self.max_dps, dps); act = ""
        if self.state in ("TRACKING", "DEGRADED"):
            if dps > DEG_HOLD or v > V_DIVERGE or not healthy:
                self.state = "HOLD"; self.holds += 1; self.calm_since = None; self.healthy_run = 0; self.unhealthy_since = None
            elif dps > DEG_TRACK:
                if self.state != "DEGRADED": self.degraded += 1
                self.state = "DEGRADED"
            else: self.state = "TRACKING"
        if self.state in ("HOLD", "INIT"):
            if dps < DEG_RECOVER: self.calm_since = self.calm_since or t
            else: self.calm_since = None; self.healthy_run = 0
            calm = self.calm_since is not None and t - self.calm_since >= CALM_S
            if calm:
                ok = healthy if recover_ok is None else (healthy and recover_ok)
                self.healthy_run = self.healthy_run + 1 if ok else 0
                # 10-16 live80: OpenVINS re-initialises only from stillness, so a reset during continuous motion = HOLD until the operator
                # stops (86 s). Only DIVERGENCE (not a feature dip) counts towards a reset; a feature dip just waits in HOLD.
                if not healthy and inited and diverging: self.unhealthy_since = self.unhealthy_since or t
                else: self.unhealthy_since = None
                if self.healthy_run >= HEALTHY_N:
                    if self.state == "HOLD": self.rezeros += 1
                    self.state = "TRACKING"; self.unhealthy_since = None
                elif self.unhealthy_since is not None and t - self.unhealthy_since > DIVERGE_S and self.state in ("HOLD", "INIT") and t - self.last_reset > RESET_COOLDOWN_S:   # 10-16: a bad INIT (moving at start) stayed stuck forever
                    act = "reset"; self.resets += 1; self.unhealthy_since = None; self.healthy_run = 0; self.last_reset = t
        return act


class LiveSource:
    """two camera threads (ring buffers of recent frames) + Teensy IMU; yields (t_left_ns, left, right, imu_samples_since_last)."""
    def __init__(self, il: int, ir: int, exposure: int, sc: int, lloc: str, rloc: str, off_ns: int = 0):
        from visual_teleop.tools.stereo_imu_record import lock
        from visual_teleop import config as vcfg
        from visual_teleop.tools.calib_right import open_imu
        self.color_left = None; self.off_ns = off_ns; self.imu_margin_ms = 0.0; self.imu_wait_ms = 0.0; self.sc = sc; self.size = (1920 // sc, 1080 // sc); self.buf = {0: collections.deque(maxlen=8), 1: collections.deque(maxlen=8)}; self.on = True; self.cv = threading.Condition()
        self.idx = (il, ir); self.caps = [self._open(i) for i in (il, ir)]; self.reopen = [False, False]
        print("[stereo-vio] 노출", lock(lloc, exposure), lock(rloc, exposure), flush=True)
        for k in (0, 1): threading.Thread(target=self._grab, args=(k,), daemon=True).start()
        self.imu = open_imu(vcfg.load(), "right"); self.imu.buffer.drain(); self.lat_win = collections.deque(maxlen=4000); self.last_left = 0
        self.phase_lock()            # AFTER the IMU is open: opening the Teensy disturbed the camera delivery and the phase (0.8 -> 18 ms)
        self.relocks = 0; self.skipped_run = 0

    @staticmethod
    def _open(i):
        c = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION); c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); return c

    def skew_ms(self, seconds: float = 1.0) -> float:
        """median |t_left - nearest t_right| over `seconds` of fresh frames."""
        t_end = time.monotonic() + seconds; sk = []; seen = 0
        while time.monotonic() < t_end:
            time.sleep(0.03)
            with self.cv:
                if self.buf[0] and self.buf[1] and self.buf[0][-1][0] != seen:
                    seen = self.buf[0][-1][0]; sk.append(min(abs(t - seen) for t, _ in self.buf[1]) / 1e6)
        return float(np.median(sk)) if sk else 1e9

    def phase_lock(self, target_ms: float = 4.0, tries: int = 20) -> float:
        """10-16: the two USB cameras free-run at 30 fps with a RANDOM phase fixed at stream start; stereo needs |skew| small. Re-open the
        RIGHT stream until the measured skew is below target (the phase then holds for minutes: 0.9 ms over a 90 s take)."""
        time.sleep(1.0); best = 1e9
        for k in range(tries):
            sk = self.skew_ms(1.0); best = min(best, sk); print(f"[stereo-vio] 좌우 위상차 {sk:.1f} ms (시도 {k + 1})", flush=True)
            if sk < target_ms: return sk
            self.reopen[1] = True
            while self.reopen[1]: time.sleep(0.05)
            time.sleep(0.8)
        print(f"[stereo-vio] 경고: 위상차를 {target_ms} ms 아래로 못 맞춤 (최소 {best:.1f} ms)", flush=True); return best

    def _grab(self, k):
        while self.on:
            if self.reopen[k]:
                self.caps[k].release(); time.sleep(0.2); self.caps[k] = self._open(self.idx[k])
                with self.cv: self.buf[k].clear()
                self.reopen[k] = False
            ok, f = self.caps[k].read(); t = time.monotonic_ns()
            if not ok: continue
            small = cv2.resize(f, self.size, interpolation=cv2.INTER_AREA); g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if k == 0: self.color_left = (t, small)                     # 10-16: jaw tape -> gripper (the wrist fingertip node is gone)
            with self.cv: self.buf[k].append((t, g)); self.cv.notify_all()

    def imu_samples(self):
        out = []
        for s in self.imu.buffer.drain():
            self.lat_win.append(s.host_receive_ns - s.device_timestamp_us * 1000)       # lower envelope = device->host clock offset
            out.append((s.device_timestamp_us * 1000, np.array([s.gx, s.gy, s.gz]), np.array([s.ax, s.ay, s.az])))
        base = min(self.lat_win) if self.lat_win else 0
        return [(t + base, w, a) for t, w, a in out]

    def frames(self):
        while self.on:
            with self.cv:
                self.cv.wait(timeout=0.2)
                if not self.buf[0] or not self.buf[1] or self.buf[0][-1][0] == self.last_left: continue
                tl, gl = self.buf[0][-1]
                # 10-16: when the right camera lags the left by ~1 ms, the newest left frame arrives BEFORE its partner -> wait (<= 25 ms)
                t_wait = time.monotonic()
                while self.buf[1][-1][0] < tl - 3e6 and time.monotonic() - t_wait < 0.025: self.cv.wait(timeout=0.005)
                tr, gr = min(self.buf[1], key=lambda x: abs(x[0] - tl))
            self.last_left = tl; self.last_skew_ms = abs(tr - tl) / 1e6
            if abs(tr - tl) > 12e6:
                self.skipped_run += 1
                if self.skipped_run >= 15:              # 0.5 s without a usable pair: the phase drifted -> re-lock (caller sees a frame gap = HOLD)
                    self.relocks += 1; print(f"[stereo-vio] 위상 틀어짐 ({self.last_skew_ms:.1f} ms) → 다시 맞춤 #{self.relocks}", flush=True)
                    self.phase_lock(); self.skipped_run = 0; self.imu_samples()
                continue
            self.skipped_run = 0
            # 10-16 live: OpenVINS propagates with the IMU it already has -- a frame fed before the IMU covering its time arrived made
            # features collapse ("트래커를 다시 잡습니다" over and over). Wait (<= 60 ms) until IMU (on the camera clock) reaches t + 5 ms.
            imu = self.imu_samples(); t_wait = time.monotonic()
            while (not imu or imu[-1][0] + self.off_ns < tl + 5e6) and time.monotonic() - t_wait < 0.06:
                time.sleep(0.002); imu += self.imu_samples()
            self.imu_margin_ms = ((imu[-1][0] + self.off_ns - tl) / 1e6) if imu else -1e3; self.imu_wait_ms = (time.monotonic() - t_wait) * 1000
            yield tl, gl, gr, imu

    def close(self):
        self.on = False; time.sleep(0.3); [c.release() for c in self.caps]; self.imu.stop(); self.imu.close()


class ReplaySource:
    """a stereo_imu_record take replayed at real-time pace (MockBridge tests without hardware)."""
    def __init__(self, take: Path, speed: float = 1.0):
        from stereo_basalt_eval import frames
        from handumi_collector.pose.timing import fit_device_to_host
        self.tl, self.tr = np.load(take / "cam_left_t_ns.npy"), np.load(take / "cam_right_t_ns.npy")
        self.FL, self.FR = frames(take / "cam_left.mp4", len(self.tl)), frames(take / "cam_right.mp4", len(self.tr))
        z = np.load(take / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64)
        o = np.argsort(ti); self.ti, self.gy, self.ac = ti[o], z["gyro"][o], z["acc"][o]; self.j = 0; self.speed = speed

    def frames(self):
        t_wall0 = time.monotonic(); t0 = self.tl[0]
        for k, t in enumerate(self.tl[:len(self.FL)]):
            due = t_wall0 + (t - t0) / 1e9 / self.speed
            if (d := due - time.monotonic()) > 0: time.sleep(d)
            i = int(np.argmin(np.abs(self.tr[:len(self.FR)] - t)))
            if abs(int(self.tr[i]) - int(t)) > 12e6: continue
            imu = []
            while self.j < len(self.ti) and self.ti[self.j] <= t + 40e6:          # IMU arrives ~ in step with the frames
                imu.append((int(self.ti[self.j]), self.gy[self.j], self.ac[self.j])); self.j += 1
            yield int(t), self.FL[k], self.FR[i], imu

    def close(self): pass


class BasaltLive:
    """Basalt stereo-inertial as a live subprocess (~/vio/basalt_live/basalt_live, built from source 10-16) with the SAME interface the
    tracker uses for the OpenVINS bridge: feed_imu / feed_stereo -> initialised / state() -> {T_G_I, v_IinG, n_msckf, n_slam} / reset().
    stdin: type 1 IMU (int64 t_ns, 6 doubles), type 2 STEREO (int64 t_ns, int32 w, int32 h, w*h left, w*h right), type 0 quit.
    stdout: 'P t_ns px py pz qx qy qz qw vx vy vz' per estimated state, optional 'L t_ns n_points'. Basalt has no mask input -> the
    self-view masks are painted mid-grey (same as the offline export). Basalt is asynchronous: feed_stereo waits <= WAIT_S for the state of
    that frame, else the latest state is used (the age is in state()['lag_ms'])."""
    BIN = Path.home() / "vio/basalt_live/build/basalt_live"
    WAIT_S = 0.0             # asynchronous: the newest available state is used (its age is state()['lag_ms'])

    def __init__(self, calib: Path, config: Path):
        self.calib, self.config = str(calib), str(config); self._start()

    def _start(self):
        import struct, subprocess as sp
        self.struct = struct
        self.p = sp.Popen([str(self.BIN), self.calib, self.config], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=0)
        self.latest = None; self.npts = {}; self.cv = threading.Condition(); self.alive = True; self.t_last = 0
        # 10-16 live: writing ~1 MB stereo frames into the pipe blocked when Basalt's input queue was full and the capture loop fell to
        # 5.8 Hz (+10 phase re-locks). A writer thread now owns the pipe; frames beyond MAX_PENDING_FRAMES are dropped (oldest first),
        # IMU messages are never dropped and keep their order relative to the frames.
        self.q = collections.deque(); self.qcv = threading.Condition(); self.dropped = 0
        threading.Thread(target=self._reader, daemon=True).start(); threading.Thread(target=self._writer, daemon=True).start()

    MAX_PENDING_FRAMES = 2

    def _writer(self):
        while self.alive:
            with self.qcv:
                while not self.q and self.alive: self.qcv.wait(timeout=0.1)
                if not self.q: continue
                kind, b = self.q.popleft()
            try: self.p.stdin.write(b)
            except (BrokenPipeError, OSError, ValueError): self.alive = False

    def _enqueue(self, kind: str, b: bytes):
        with self.qcv:
            if kind == "F" and sum(1 for k, _ in self.q if k == "F") >= self.MAX_PENDING_FRAMES:
                for i, (k, _) in enumerate(self.q):
                    if k == "F": del self.q[i]; self.dropped += 1; break
            self.q.append((kind, b)); self.qcv.notify()

    def _reader(self):
        for line in self.p.stdout:
            f = line.decode(errors="ignore").split()
            if not f: continue
            if f[0] == "P" and len(f) >= 12:
                t = int(f[1]); v = np.array([float(x) for x in f[2:12]])
                T = np.eye(4); T[:3, :3] = R.from_quat(v[3:7]).as_matrix(); T[:3, 3] = v[0:3]
                with self.cv: self.latest = (t, T, v[7:10]); self.cv.notify_all()
            elif f[0] == "L" and len(f) >= 3: self.npts[int(f[1])] = int(f[2])
        self.alive = False

    def feed_imu(self, t_s: float, w, a):
        self._enqueue("I", self.struct.pack("<Bq6d", 1, int(round(t_s * 1e9)), *np.asarray(w, float), *np.asarray(a, float)))

    def feed_stereo(self, t_s: float, l, r, ml, mr) -> bool:
        l = l.copy(); r = r.copy(); l[ml > 0] = 128; r[mr > 0] = 128; h, w = l.shape; t = int(round(t_s * 1e9))
        self._enqueue("F", self.struct.pack("<Bqii", 2, t, w, h) + l.tobytes() + r.tobytes()); self.t_last = t
        if self.WAIT_S > 0:
            with self.cv: self.cv.wait_for(lambda: self.latest is not None and self.latest[0] >= t, timeout=self.WAIT_S)
        return self.latest is not None

    def state(self) -> dict:
        t, T, vel = self.latest; n = self.npts.get(t, self.npts.get(max(self.npts) if self.npts else t, 99))
        return dict(T_G_I=T, v_IinG=vel, n_msckf=n, n_slam=0, lag_ms=(self.t_last - t) / 1e6, t_ns=t)

    def reset(self):
        self.alive = False
        try: self.p.stdin.write(bytes([0])); self.p.stdin.close(); self.p.wait(timeout=2)
        except Exception: self.p.kill()
        self._start()


def find_stereo_pair(max_index: int = 7) -> tuple[int, int]:
    """10-16: AVFoundation indices changed on EVERY replug and twice the tracker opened the MacBook camera as 'right'. Grab one frame from
    every 1080p camera, take the pair with the most stereo matches (stereo_check.overlap), then left/right by the orange jaw tape: the LEFT
    tracking camera (0x03110000) sees the jaws right of the image centre (x/w ~0.6), the RIGHT one near/left of it (~0.45)."""
    from visual_teleop.tools.stereo_check import overlap
    fr = {}
    for i in range(max_index):
        c = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not c.isOpened(): continue
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); ok, f = False, None
        for _ in range(6): ok, f = c.read()
        c.release()
        if ok and f is not None and f.shape[:2] == (1080, 1920): fr[i] = cv2.resize(f, (960, 540))
    best = None
    for a_ in fr:
        for b_ in fr:
            if a_ < b_:
                ov, _ = overlap(cv2.cvtColor(fr[a_], cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr[b_], cv2.COLOR_BGR2GRAY))
                if best is None or ov["stereo_inliers"] > best[0]: best = (ov["stereo_inliers"], a_, b_)
    if best is None or best[0] < 60: raise SystemExit(f"스테레오 카메라 쌍을 못 찾음 (최대 정합 {None if best is None else best[0]}) -- 두 트래킹 카메라 연결 확인")
    def jaw_x(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, (0, 120, 80), (22, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
        ys, xs = np.nonzero(m[int(img.shape[0] * 0.45):]); return xs.mean() / img.shape[1] if len(xs) > 500 else None
    xa, xb = jaw_x(fr[best[1]]), jaw_x(fr[best[2]])
    if xa is None or xb is None: raise SystemExit("집게 주황 테이프가 안 보여 좌우를 못 정함 -- 유미를 평소처럼 들고 다시")
    left, right = (best[1], best[2]) if xa > xb else (best[2], best[1])
    print(f"[stereo-vio] 스테레오 쌍 자동 선택: 정합 {best[0]}, 왼쪽 index {left} (집게 x {max(xa, xb):.2f}), 오른쪽 index {right} (집게 x {min(xa, xb):.2f})", flush=True)
    return left, right


PREVIEW = {"jpg": None}


def start_preview(port: int, src, info: dict) -> None:
    """MJPEG preview (http://localhost:<port>/): left | right tracking frames (latest), tracker state / features / HOLD counts."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        if PREVIEW["jpg"]: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + PREVIEW["jpg"] + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError): return
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>")
        def log_message(self, *a): pass

    def render():
        while True:
            time.sleep(0.12)
            try:
                with src.cv: l = src.buf[0][-1][1] if src.buf[0] else None; r = src.buf[1][-1][1] if src.buf[1] else None
                if l is None or r is None: continue
                v = np.hstack([cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)])
                col = {"TRACKING": (0, 200, 0), "DEGRADED": (0, 200, 255), "HOLD": (0, 0, 255)}.get(info.get("state"), (200, 200, 200))
                cv2.rectangle(v, (0, 0), (v.shape[1], 46), (0, 0, 0), -1)
                cv2.putText(v, f"LEFT idx {info.get('il')} | RIGHT idx {info.get('ir')} | {info.get('state')} | feat {info.get('feat')} | gyro {info.get('gyro', 0):.0f} dps | HOLD {info.get('holds')} reset {info.get('resets')} | {info.get('backend')}",
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                PREVIEW["jpg"] = cv2.imencode(".jpg", cv2.resize(v, (1600, 450)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            except Exception: pass

    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever, daemon=True).start(); threading.Thread(target=render, daemon=True).start()
    subprocess.Popen(["open", f"http://localhost:{port}/"]); print(f"[stereo-vio] 미리보기 http://localhost:{port}/", flush=True)


def jaw_metric(bgr: np.ndarray):
    """Jaw opening from the orange tape on the two jaw tips (LEFT tracking camera), metric = px distance scaled to 1920 width.
    10-16 v2: v1 (all orange components) also took the red/orange can behind the open jaws, skin and cardboard -> 56 % misses / wrong
    values when OPEN. Now only orange pixels with the BLACK jaw body directly below them (dark pixels within 25 px underneath) count;
    the two largest such blobs are the jaws (their centroid distance = opening); one blob = closed (its width as the metric)."""
    h, w = bgr.shape[:2]; hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    org = (cv2.inRange(hsv, (0, 100, 80), (22, 255, 255)) | cv2.inRange(hsv, (170, 100, 80), (180, 255, 255))) > 0
    dark = (hsv[:, :, 2] < 60).astype(np.uint8)
    k = max(3, int(25 * h / 540)); below = np.zeros_like(dark)
    below[:-k] = cv2.dilate(dark, np.ones((k, 1), np.uint8), anchor=(0, 0))[k:] if False else cv2.dilate(dark[k:], np.ones((k, 3), np.uint8))   # dark within k px BELOW
    seed = (org & (below > 0)).astype(np.uint8)
    seed[: int(h * 0.40)] = 0
    if seed.sum() < 50: return None
    # grow the seeds back to the full tape blobs (orange connected to a seed)
    n, lab, st, cen = cv2.connectedComponentsWithStats(org.astype(np.uint8))
    keep = np.unique(lab[seed > 0]); keep = [int(x) for x in keep if x > 0 and st[x, 4] > 150 * (w / 960) ** 2]
    if not keep: return None
    keep.sort(key=lambda x: -st[x, 4]); keep = keep[:2]
    if len(keep) == 1: return float(st[keep[0], 2] * 1920 / w) * 0.25            # one merged blob = closed (quarter of its width)
    (x1, y1), (x2, y2) = cen[keep[0]], cen[keep[1]]
    return float(np.hypot(x1 - x2, y1 - y2) * 1920 / w)

class JawCls:
    """10-16 learned jaw OPEN / CLOSED (tools/jaw_capture + jaw_train) on the same input the classifier was trained on: both gray
    tracking frames side by side resized to the 1600 x 450 preview, the fixed bottom ROI, 70 x 34, jaw-pixel mask, logistic regression.
    metric = pinch_closed + p_open * (pinch_open - pinch_closed), EMA-smoothed, so the controller's pinch thresholds keep working."""

    def __init__(self, path: str, closed: float = 233.0, opened: float = 340.0, ema: float = 0.5) -> None:
        z = np.load(path); self.M, self.mu, self.sd, self.w, self.b = z["mask"], z["mean"], z["scale"], z["coef"], float(z["intercept"])
        y0, y1, x0, x1 = (int(v) for v in z["roi"]); self.roi = (slice(y0, y1), slice(x0, x1)); self.ds = int(z["ds"]); self.size = tuple(int(v) for v in z["size"])
        self.lo, self.hi, self.ema, self.p = closed, opened, ema, None

    def __call__(self, l_gray: np.ndarray, r_gray: np.ndarray) -> float:
        v = cv2.resize(np.hstack([l_gray, r_gray]), (1600, 450), interpolation=cv2.INTER_AREA); f = []
        for view in (v[:, :800], v[:, 800:]):
            c = view[self.roi]; c = cv2.resize(c, None, fx=1 / self.ds, fy=1 / self.ds, interpolation=cv2.INTER_AREA)
            f.append(cv2.resize(c, self.size, interpolation=cv2.INTER_AREA).ravel())
        x = (np.concatenate(f).astype(np.float32) / 255)[self.M]; p = 1 / (1 + np.exp(-(((x - self.mu) / self.sd) @ self.w + self.b)))
        self.p = p if self.p is None else self.ema * p + (1 - self.ema) * self.p
        return float(self.lo + self.p * (self.hi - self.lo))


class JawCNN:
    """10-16 fine-tuned ResNet18 jaw OPEN / CLOSED (tools/jaw_finetune; held-out hard scenes 0.93 vs 0.64 live for the pixel model).
    Runs in its OWN thread on the latest gray pair (~10 Hz) so the tracking loop never waits; __call__ returns the latest metric.
    Same input as training: both gray frames -> 1600 x 450 preview geometry -> jaw_capture ROI crops -> jaw_finetune.to_input.
    Output metric = closed or open (binary) from the EMA of p with a hysteresis (0.35 / 0.65), so single-frame flips do not toggle."""

    def __init__(self, path: str, src, closed: float = 233.0, opened: float = 340.0) -> None:
        import torch
        from visual_teleop.tools.jaw_capture import crops
        from visual_teleop.tools.jaw_finetune import to_input
        torch.set_num_threads(2); self.torch, self.crops, self.to_input = torch, crops, to_input
        self.net = torch.jit.load(path).eval(); self.src, self.lo, self.hi = src, closed, opened; self.p, self.state, self.metric = None, None, None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            time.sleep(0.08)
            with self.src.cv: l = self.src.buf[0][-1][1] if self.src.buf[0] else None; r = self.src.buf[1][-1][1] if self.src.buf[1] else None
            if l is None or r is None: continue
            v = cv2.resize(np.hstack([l, r]), (1600, 450), interpolation=cv2.INTER_AREA); cl, cr = self.crops(v)
            with self.torch.no_grad(): p = float(self.torch.sigmoid(self.net(self.torch.tensor(self.to_input(cl, cr)[None]))))
            self.p = p if self.p is None else 0.5 * p + 0.5 * self.p
            if self.state is None or (self.state == 0 and self.p > 0.65) or (self.state == 1 and self.p < 0.35): self.state = int(self.p > 0.5)
            self.metric = self.hi if self.state else self.lo

    def __call__(self, *_):
        return self.metric


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaw-model", default=str(ROOT / "configs/calibration/jaw_resnet18_v004.pt"), help="learned jaw classifier (tools/jaw_train); '' = orange-tape rule")
    ap.add_argument("--replay", default=None); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--port", type=int, default=47813); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--left-index", type=int, default=-1); ap.add_argument("--right-index", type=int, default=-1); ap.add_argument("--exposure", type=int, default=10)
    ap.add_argument("--left-location", default="0x03110000"); ap.add_argument("--right-location", default="0x03140000")
    ap.add_argument("--t-ci", default="-0.0413,-0.0275,0.0137"); ap.add_argument("--preview-port", type=int, default=8746, help="live MJPEG preview of both tracking cameras (0 = off)"); ap.add_argument("--backend", default="openvins", choices=["openvins", "basalt", "airslam"]); ap.add_argument("--airslam-host", default="100.64.0.2"); ap.add_argument("--airslam-port", type=int, default=47900); ap.add_argument("--pinch-port", type=int, default=47811, help="jaw-tape opening -> controller pinch link (0 = off)"); ap.add_argument("--log", default=None); ap.add_argument("--quiet", action="store_true"); a = ap.parse_args()
    ci = yaml.safe_load((ROOT / "configs/calibration/stereo_trackcam_imu_v004.yaml").read_text()); off_ns = int(float(ci["offset_ms"]) * 1e6); sc = 2
    work = Path(a.log).with_suffix("") if a.log else ROOT / "datasets/stereo_vio_live" / time.strftime("%Y%m%d_%H%M%S"); work.mkdir(parents=True, exist_ok=True)
    # live 10-16: num_pts 200 -> 5 of 6 HOLDs were feature-count dips, so 350
    if a.backend == "airslam":      # 10-16: AirSLAM (PLNet+LightGlue/TensorRT) on the 4090 via TCP, stereo VO + IMU gravity alignment (airslam_live.py)
        from visual_teleop.vision.airslam_live import AirSLAMLive
        br = AirSLAMLive(a.airslam_host, a.airslam_port)
    elif a.backend == "basalt":       # 10-16 A/B on the same take: Basalt stayed bounded after fast motion (5 jumps > 2 m/s vs 385 for OpenVINS)
        from stereo_basalt_eval import calib as basalt_calib
        (work / "basalt").mkdir(exist_ok=True); cp, _ = basalt_calib(work / "basalt", [float(v) for v in a.t_ci.split(",")], sc)
        br = BasaltLive(cp, Path.home() / "vio/basalt_bin/release/data/euroc_config_lv5.json")
    else:
        cfg = write_config(work / "ov", [float(v) for v in a.t_ci.split(",")], sc, {"num_pts": 350})   # 10-16: init_dyn_use tried and REVERTED (no faster re-init on replay, NaN pose live)
        ovb = import_bridge(); br = ovb.Bridge(str(cfg), "WARNING")
    ml = (np.load(ROOT / "configs/calibration/stereo_selfmask_left_960.npy") * 255).astype(np.uint8); mr = (np.load(ROOT / "configs/calibration/stereo_selfmask_right_960.npy") * 255).astype(np.uint8)
    if a.replay: src = ReplaySource(Path(a.replay), a.speed)
    else:
        from visual_teleop.tools.stereo_check import index_for_location
        il = ir = None
        if a.left_index >= 0 and a.right_index >= 0: il, ir = a.left_index, a.right_index
        else:
            for attempt in range(6):                      # the UMI may be lying down (blank view / no tape): ask and retry ~30 s
                try: il, ir = find_stereo_pair(); break
                except SystemExit as e:
                    print(f"[stereo-vio] {e} (재시도 {attempt + 1}/6)", flush=True)
                    if attempt == 0: say("유미를 평소처럼 들고 방 쪽을 보게 해 주세요")
                    time.sleep(3)
            if il is None: raise SystemExit("스테레오 카메라 쌍을 끝내 못 찾음")
        print(f"[stereo-vio] 카메라 왼쪽 index {il}, 오른쪽 index {ir}", flush=True); src = LiveSource(il, ir, a.exposure, sc, a.left_location, a.right_location, off_ns)
    INFO = dict(backend=a.backend, il=None if a.replay else il, ir=None if a.replay else ir)
    if not a.replay and a.preview_port: start_preview(a.preview_port, src, INFO)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); logf = open(a.log or work / "tracker_log.jsonl", "w")
    G = Guard(); seq = 0; gyro_hist = collections.deque(maxlen=450)                # (t_ns, w rad/s) ~1 s at 416 Hz (async Basalt states lag the frames)
    # TW <- VIO world (stitched across VIO resets); last trusted IMU orientation in TW for the gyro bridge across a reset
    T_TW_V = np.eye(4); R_TW_I_last = None; T_TW_I_last = None; t_last_good = None; gyro_since_good = []; need_stitch = False
    if a.jaw_model.endswith(".pt") and Path(a.jaw_model).exists(): jaw_cls = JawCNN(a.jaw_model, src)
    else: jaw_cls = JawCls(a.jaw_model) if a.jaw_model and Path(a.jaw_model).exists() else None
    print(f"[stereo-vio] 그리퍼: {'학습 분류기 ' + Path(a.jaw_model).name if jaw_cls else '주황 테이프 규칙'}", flush=True)
    nan_states = 0; last_nan_reset = -1e9; last_jaw = None; jaw_seen = collections.deque(maxlen=30); prev = None; feat_low_since = None; feat_low_run = 0; t_start = time.monotonic(); n_frames = 0; proc = []; say_stillness = 0.0
    print(f"[stereo-vio] 시작: UDP {a.host}:{a.port}, 로그 {logf.name}", flush=True)
    if not a.replay: say("트래커 시작. 가만히 있다가 천천히 움직이세요")
    try:
        for t_ns, gl, gr, imu in src.frames():
            for ti, w, acc in imu:
                tc = ti + off_ns; br.feed_imu(tc * 1e-9, np.asarray(w, np.float64), np.asarray(acc, np.float64)); gyro_hist.append((tc, np.asarray(w, float)))
                if need_stitch or G.state == "HOLD": gyro_since_good.append((tc, np.asarray(w, float)))
            t0p = time.perf_counter(); inited = br.feed_stereo(t_ns * 1e-9, gl, gr, ml, mr); proc.append((time.perf_counter() - t0p) * 1000); n_frames += 1
            recent = [w for tt, w in gyro_hist if tt > t_ns - 50e6]; dps = float(np.degrees(max((np.linalg.norm(w) for w in recent), default=0.0)))
            st = br.state() if inited else None; healthy = False; recover_ok = False; v = 0.0; w_vio_err = None; T_V_I = None
            if st is not None and not (np.all(np.isfinite(np.asarray(st["T_G_I"]))) and np.all(np.isfinite(np.asarray(st["v_IinG"])))):
                # 10-16 live: a broken initialisation produced a NaN state and the tracker died in from_matrix -> treat as divergence
                nan_states += 1; st = None
                if G.state in ("TRACKING", "DEGRADED"): G.state = "HOLD"; G.holds += 1
                if time.monotonic() - last_nan_reset > RESET_COOLDOWN_S: br.reset(); prev = None; need_stitch = True; G.resets += 1; last_nan_reset = time.monotonic()
            t_pose = int(st.get("t_ns", t_ns)) if st is not None else t_ns       # async Basalt: the state belongs to an earlier frame
            if st is not None and prev is not None and t_pose <= prev[0]: continue                 # no new state since the last tick: publish nothing
            if st is not None:
                T_V_I = np.asarray(st["T_G_I"]); v = float(np.linalg.norm(st["v_IinG"])); nf = int(st["n_msckf"]) + int(st["n_slam"])
                step_v = 0.0
                if prev is not None and t_pose > prev[0]:
                    step_v = float(np.linalg.norm(T_V_I[:3, 3] - prev[1][:3, 3]) / ((t_pose - prev[0]) / 1e9))
                    dR = R.from_matrix(prev[1][:3, :3].T @ T_V_I[:3, :3]); w_vio = dR.as_rotvec() / ((t_pose - prev[0]) / 1e9)
                    w_g = np.mean([w for tt, w in gyro_hist if prev[0] < tt <= t_pose], axis=0) if any(prev[0] < tt <= t_pose for tt, _ in gyro_hist) else w_vio
                    w_vio_err = float(np.degrees(np.linalg.norm(w_vio - w_g)))
                feat_low_since = (feat_low_since or t_ns) if nf < MIN_FEAT else None
                feat_low_run = feat_low_run + 1 if nf < FEAT_LOW else 0
                # 10-16 robot run: 15/16 HOLDs were "< 10 features x3" at 0.01-0.6 m/s with a healthy rotation check (mostly a threshold flutter);
                # replay failures had feat 0-14 but VIO speed p50 0.5 m/s -> the low-feature rule now also needs v > FEAT_LOW_V
                feat_dead = (feat_low_since is not None and (t_ns - feat_low_since) / 1e9 > FEAT_ZERO_S) or (feat_low_run >= FEAT_LOW_FRAMES and v > FEAT_LOW_V)
                # 10-16 Basalt A/B: 7-20 cm pose steps in 33 ms (2-6 m/s) while its own velocity said 0.2 m/s and features were fine
                healthy = (not feat_dead) and v < V_HEALTHY and step_v < STEP_V_MAX and (w_vio_err is None or w_vio_err < W_MISMATCH_DPS)
                recover_ok = nf >= RECOVER_FEAT and v < V_RECOVER
                prev = (t_pose, T_V_I)
            diverging = st is not None and ((v > 1.0 and dps < DEG_RECOVER) or (w_vio_err is not None and w_vio_err > W_MISMATCH_DPS))
            act = G.step(t_ns / 1e9, dps, healthy, v if st is not None else 0.0, inited=st is not None, recover_ok=recover_ok if st is not None else False, diverging=diverging)
            if act == "reset":
                br.reset(); prev = None; need_stitch = True
                if time.monotonic() - say_stillness > 4: say("트래커를 다시 잡습니다. 잠깐 가만히 있다가 천천히 움직이세요"); say_stillness = time.monotonic()
            if need_stitch and G.state == "TRACKING" and T_V_I is not None and R_TW_I_last is not None:
                # orientation of the IMU in TW now = last trusted orientation * gyro integral since then; keep only the yaw of TW <- V_new
                Rg = R.identity()
                for k in range(1, len(gyro_since_good)): Rg = Rg * R.from_rotvec(gyro_since_good[k - 1][1] * (gyro_since_good[k][0] - gyro_since_good[k - 1][0]) / 1e9)
                R_TW_I_now = R.from_matrix(R_TW_I_last) * Rg; R_TW_V = R_TW_I_now * R.from_matrix(T_V_I[:3, :3]).inv()
                yaw = R_TW_V.as_euler("zyx")[0]; T_TW_V = np.eye(4); T_TW_V[:3, :3] = R.from_euler("z", yaw).as_matrix()
                p_last = T_TW_I_last[:3, 3]; T_TW_V[:3, 3] = p_last - T_TW_V[:3, :3] @ T_V_I[:3, 3]; need_stitch = False
            valid = G.state in ("TRACKING", "DEGRADED") and T_V_I is not None and not need_stitch
            T_TW_I = T_TW_V @ T_V_I if T_V_I is not None else None
            if valid:
                R_TW_I_last = T_TW_I[:3, :3].copy(); T_TW_I_last = T_TW_I.copy(); t_last_good = t_ns; gyro_since_good = []
            msg = dict(kind="tracker_pose", side="right", seq=seq, t_src_ns=int(t_ns), t_send_ns=time.monotonic_ns(), valid=bool(valid),
                       conf=1.0 if G.state == "TRACKING" else 0.5 if G.state == "DEGRADED" else 0.0, state=G.state, gyro_dps=round(dps, 1),
                       speed_scale=0.5 if G.state == "DEGRADED" else 1.0, holds=G.holds, resets=G.resets, rezeros=G.rezeros)
            if T_TW_I is not None:
                msg.update(p=[float(x) for x in T_TW_I[:3, 3]], q_xyzw=[float(x) for x in R.from_matrix(T_TW_I[:3, :3]).as_quat()])
            else: msg.update(p=[0.0, 0.0, 0.0], q_xyzw=[0.0, 0.0, 0.0, 1.0])
            if st is not None: msg.update(n_feat=int(st["n_msckf"]) + int(st["n_slam"]), vio_speed=round(v, 3), step_speed=round(step_v, 3), w_mismatch_dps=None if w_vio_err is None else round(w_vio_err, 1))
            sock.sendto(json.dumps(msg).encode(), (a.host, a.port)); seq += 1
            INFO.update(state=G.state, feat=msg.get("n_feat"), gyro=dps, holds=G.holds, resets=G.resets)
            cl = getattr(src, "color_left", None)
            if cl is not None and seq % 2 == 0 and a.pinch_port:            # ~15 Hz jaw opening -> the controller's pinch link
                if jaw_cls is not None:
                    with src.cv: lg = src.buf[0][-1][1] if src.buf[0] else None; rg = src.buf[1][-1][1] if src.buf[1] else None
                    jm = jaw_cls(lg, rg) if lg is not None and rg is not None else None
                else: jm = jaw_metric(cl[1])
                last_jaw = jm; jaw_seen.append(jm is not None)
                sock.sendto(json.dumps(dict(kind="pinch", side="right", t_ns=int(cl[0]), metric=jm, conf=1.0 if jm is not None else 0.0, thumb=None, index=None)).encode(), (a.host, a.pinch_port))
            logf.write(json.dumps(dict(msg, jaw=last_jaw, proc_ms=round(proc[-1], 2), skew_ms=round(getattr(src, "last_skew_ms", 0.0), 1), imu_margin_ms=round(getattr(src, "imu_margin_ms", 0.0), 1), imu_wait_ms=round(getattr(src, "imu_wait_ms", 0.0), 1))) + "\n"); logf.flush()
            if not a.quiet and seq % 30 == 0:
                print(f"[stereo-vio] {time.monotonic() - t_start:6.1f}s {G.state:9s} gyro {dps:5.0f}°/s feat {msg.get('n_feat', 0):3d} p {np.round(msg['p'], 3).tolist()} HOLD {G.holds} reset {G.resets} rezero {G.rezeros} jaw {last_jaw if last_jaw is None else round(last_jaw)} (검출 {sum(jaw_seen)}/{len(jaw_seen)}) proc {np.median(proc[-30:]):.1f}ms skew {getattr(src, 'last_skew_ms', 0.0):.1f}ms", flush=True)
    except KeyboardInterrupt: pass
    finally:
        src.close(); logf.close()
        if hasattr(br, "close"): br.close()
        summ = dict(frames=n_frames, holds=G.holds, resets=G.resets, rezeros=G.rezeros, degraded_entries=G.degraded, max_gyro_dps=round(G.max_dps, 1),
                    proc_ms_p50=round(float(np.median(proc)), 2) if proc else None, proc_ms_p95=round(float(np.percentile(proc, 95)), 2) if proc else None)
        (work / "summary.json").write_text(json.dumps(summ, indent=1)); print("[stereo-vio] 요약", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
