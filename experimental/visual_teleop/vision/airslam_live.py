"""AirSLAM live backend client (2026-10-16) for stereo_vio_tracker --backend airslam.

AirSLAM (sair-lab, PLNet + LightGlue on TensorRT) runs on the 4090 as a long-running ray job (~/vio/airslam_bench/ray_live.sh,
TCP 100.64.0.2:47900). Same interface as BasaltLive: feed_imu / feed_stereo -> initialised / state() -> {T_G_I, v_IinG, n_msckf, n_slam,
lag_ms, t_ns} / reset(). Never blocks the capture loop: a writer thread owns the socket (self-view mask -> mid-grey, KB4 fisheye ->
rectified pinhole 960x540 HFOV 110 deg with the AirSLAM stereo rectification, sent as 640x360 JPEG q80 by default), at most MAX_PENDING_FRAMES frames are queued
(oldest dropped, IMU never dropped); the server also processes newest-first.

Mode (server side): default STEREO VO (use_imu 0). 10-16 offline on take 20261016_125927: AirSLAM stereo-INERTIAL diverged (15 m extent,
13 m jumps, 21 % lost frames) while stereo-only stayed bounded (1.2 m extent, 3.6 % lost). The IMU is therefore only used here to make
the world gravity-aligned: at the first pose after a (re)start the mean accelerometer of the last 0.5 s (hold still!) fixes the world's
roll/pitch (z up). Pose = IMU body T_G_I (AirSLAM T_w_c * T_c_b with the same v004 + imu_v003 calibration as OpenVINS/Basalt).
"""
from __future__ import annotations
import collections, socket, struct, threading, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

BENCH = Path.home() / "vio/airslam_bench"


def _rect_maps(cam_yaml: Path, rect_npz: Path):
    y = yaml.safe_load(cam_yaml.read_text().replace("%YAML:1.0", ""))
    z = np.load(rect_npz); P = z["P"]; W, H = y["image_width"], y["image_height"]; maps = []
    for k, Rr in ((0, z["R0"]), (1, z["R1"])):
        f = y[f"cam{k}"]["intrinsics"]; K = np.array([[f[0], 0, f[2]], [0, f[1], f[3]], [0, 0, 1.0]])
        D = np.array(y[f"cam{k}"]["distortion_coeffs"][:4], float).reshape(4, 1)
        maps.append(cv2.fisheye.initUndistortRectifyMap(K, D, Rr, P[:3, :3], (W, H), cv2.CV_16SC2))
    return maps, (W, H)


class AirSLAMLive:
    MAX_PENDING_FRAMES = 2
    JPEG_Q = 80

    def __init__(self, host: str = "100.64.0.2", port: int = 47900, cam_yaml: Path = BENCH / "airslam_rig_960.yaml",
                 rect_npz: Path = BENCH / "airslam_rig_rect_h110_rect.npz", connect_timeout_s: float = 20.0,
                 send_size: tuple[int, int] = (640, 360)):
        # send_size must match the server mode: (640, 360) <-> ray_live.sh noimu640 (default), (960, 540) <-> noimu / imu
        self.host, self.port = host, port
        self.maps, self.size = _rect_maps(cam_yaml, rect_npz); self.send_size = tuple(send_size)
        self.acc = collections.deque(maxlen=2000)            # (t_ns, accel IMU frame) for the gravity alignment
        self.lat = collections.deque(maxlen=300); self.srv_ms = collections.deque(maxlen=300); self.t_print = time.monotonic()
        self.dropped = 0; self.n_pose = self.n_pose_last = 0; self.t_enq = {}; self.connect_timeout_s = connect_timeout_s; self._start()

    # ---------------------------------------------------------------- connection
    def _start(self):
        t_end = time.monotonic() + self.connect_timeout_s; err = None
        while time.monotonic() < t_end:
            try:
                self.s = socket.create_connection((self.host, self.port), timeout=3.0); break
            except OSError as e:
                err = e; time.sleep(1.0)
        else:
            raise SystemExit(f"[airslam] 4090 AirSLAM 서버 {self.host}:{self.port} 에 연결 못 함 ({err}) -- ray_live.sh 잡이 떠 있는지 확인")
        self.s.settimeout(None); self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.latest = None; self.prev = None; self.v = np.zeros(3); self.R_align = None; self.ready = False
        self.alive = True; self.t_last = 0; self.q = collections.deque(); self.qcv = threading.Condition(); self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start(); threading.Thread(target=self._writer, daemon=True).start()
        print(f"[airslam] 서버 연결됨 {self.host}:{self.port}", flush=True)

    def _writer(self):
        while self.alive:
            with self.qcv:
                while not self.q and self.alive: self.qcv.wait(timeout=0.1)
                if not self.q: continue
                kind, item = self.q.popleft()
            try:
                if kind == "F":
                    t, l, r, ml, mr = item; jp = []
                    for img, m, (m1, m2) in ((l, ml, self.maps[0]), (r, mr, self.maps[1])):
                        img = img.copy(); img[m > 0] = 128
                        rect = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128)
                        if self.send_size != self.size: rect = cv2.resize(rect, self.send_size, interpolation=cv2.INTER_AREA)
                        jp.append(cv2.imencode(".jpg", rect, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_Q])[1].tobytes())
                    self.s.sendall(b"F" + struct.pack("<qii", t, len(jp[0]), len(jp[1])) + jp[0] + jp[1])
                else:
                    self.s.sendall(item)
            except (OSError, ValueError):
                self.alive = False

    def _enqueue(self, kind: str, item):
        with self.qcv:
            if kind == "F" and sum(1 for k, _ in self.q if k == "F") >= self.MAX_PENDING_FRAMES:
                for i, (k, _) in enumerate(self.q):
                    if k == "F": del self.q[i]; self.dropped += 1; break
            self.q.append((kind, item)); self.qcv.notify()

    def _reader(self):
        f = self.s.makefile("rb")
        for line in f:
            p = line.decode(errors="ignore").split()
            if not p: continue
            if p[0] == "READY": self.ready = True; print("[airslam] 서버 준비 완료 (AirSLAM 맵빌더 시작)", flush=True); continue
            if p[0] != "P" or len(p) < 17 or not self.ready: continue          # not ready = poses of the map builder before a reset
            now = time.monotonic_ns(); t = int(p[1]); v = np.array([float(x) for x in p[2:12]])
            inl, ftype, srv_ms = int(p[12]), int(p[13]), float(p[15])
            T = np.eye(4); T[:3, :3] = R.from_quat(v[3:7]).as_matrix(); T[:3, 3] = v[0:3]
            with self.lock:
                if self.R_align is None: self.R_align = self._gravity_align(t, T[:3, :3])
                A = np.eye(4); A[:3, :3] = self.R_align; TG = A @ T
                if self.prev is not None and t > self.prev[0]:
                    dt = (t - self.prev[0]) / 1e9; vel = (TG[:3, 3] - self.prev[1][:3, 3]) / dt
                    self.v = 0.6 * self.v + 0.4 * vel if inl > 10 else self.v
                self.prev = (t, TG); self.latest = (t, TG, self.v.copy(), inl)
            # latency = pose received - frame handed to feed_stereo (live: capture -> feed is the IMU wait, <= a few ms, see imu_wait_ms)
            te = self.t_enq.get(t); self.lat.append((now - te) / 1e6 if te else (now - t) / 1e6); self.srv_ms.append(srv_ms); self.n_pose += 1
            if time.monotonic() - self.t_print > 5.0 and self.lat:
                el = time.monotonic() - self.t_print; rate = (self.n_pose - self.n_pose_last) / el; self.n_pose_last = self.n_pose
                self.t_print = time.monotonic(); L = np.array(self.lat)
                print(f"[airslam] 지연(프레임 입력→포즈 수신) p50 {np.median(L):.0f} ms p90 {np.percentile(L, 90):.0f} ms | 서버 처리 p50 {np.median(self.srv_ms):.0f} ms | "
                      f"포즈 {rate:.1f} Hz | 인라이어 {inl} | 버린 프레임 {self.dropped}", flush=True)
        self.alive = False

    def _gravity_align(self, t_ns: int, R_wb: np.ndarray) -> np.ndarray:
        a = [x for tt, x in self.acc if t_ns - 5e8 <= tt <= t_ns]
        if not a:
            print("[airslam] 경고: 정렬용 IMU 가속도 없음 -- 중력 정렬 없이 진행", flush=True); return np.eye(3)
        A = np.array(a); m = A.mean(0); still = float(np.linalg.norm(A.std(0)))
        if still > 0.3: print(f"[airslam] 경고: 시작 순간 움직임 (가속도 std {still:.2f}) -- 중력 정렬이 부정확할 수 있음", flush=True)
        up = R_wb @ (m / np.linalg.norm(m))                   # accelerometer at rest points UP (specific force), in the VO world
        ax = np.cross(up, [0, 0, 1.0]); s = np.linalg.norm(ax); c = float(up[2])
        if s < 1e-9: return np.eye(3) if c > 0 else R.from_rotvec([np.pi, 0, 0]).as_matrix()
        return R.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()

    # ---------------------------------------------------------------- tracker interface
    def feed_imu(self, t_s: float, w, a):
        t = int(round(t_s * 1e9)); a = np.asarray(a, float); self.acc.append((t, a))
        self._enqueue("I", b"I" + struct.pack("<q6d", t, *np.asarray(w, float), *a))

    def feed_stereo(self, t_s: float, l, r, ml, mr) -> bool:
        t = int(round(t_s * 1e9)); self.t_enq[t] = time.monotonic_ns(); self._enqueue("F", (t, l, r, ml, mr)); self.t_last = t
        if len(self.t_enq) > 300: [self.t_enq.pop(k) for k in sorted(self.t_enq)[:100]]
        if not self.alive: raise SystemExit("[airslam] 서버 연결 끊김")
        return self.latest is not None

    def state(self) -> dict:
        with self.lock: t, T, vel, inl = self.latest
        return dict(T_G_I=T, v_IinG=vel, n_msckf=inl, n_slam=0, lag_ms=(self.t_last - t) / 1e6, t_ns=t)

    def reset(self):
        with self.qcv: self.q = collections.deque([(k, x) for k, x in self.q if k == "I"]); self.q.append(("R", b"R")); self.qcv.notify()
        with self.lock: self.latest = None; self.prev = None; self.v = np.zeros(3); self.R_align = None; self.ready = False
        print("[airslam] 리셋 요청 (서버가 새 맵빌더 시작)", flush=True)

    def close(self):
        try: self.s.sendall(b"Q"); self.s.close()
        except OSError: pass
        self.alive = False
        if self.lat:
            L = np.array(self.lat); print(f"[airslam] 종료: 지연 p50 {np.median(L):.0f} ms p90 {np.percentile(L, 90):.0f} ms, 버린 프레임 {self.dropped}", flush=True)
