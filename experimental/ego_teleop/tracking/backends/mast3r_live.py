"""MASt3R-Fusion as a LIVE wrist visual-inertial frontend, over a socket (multi-sensor spec sections 2, 10, 14).

    wrist Arducam ─┐                      ┌── this Mac ──┐            ┌────── the GPU box ──────┐
                   ├─► Mast3rLiveBackend ─┤  JPEG + IMU  ├──TCP──────►│ scripts/mast3r_live_    │
    wrist IMU  ────┘   (a PoseEstimator)  │  pose back   │◄───────────│ server.py → MASt3R-     │
                                          └──────────────┘            │ Fusion FrameTracker     │
                                                                      └─────────────────────────┘

Why a socket and not an import: MASt3R-Fusion is CUDA + torch + gtsam and lives at `~/vio_bakeoff/MASt3R-Fusion` on
the 5090 box; the teleoperation loop runs on the Mac that holds the cameras. Section 10 also says not to copy MASt3R
into this repo, so the model stays where it is and this file is the adapter in front of it. The server script is
`scripts/mast3r_live_server.py` — OUR code, deployed into THEIR checkout, importing their modules. Nothing of
MASt3R-Fusion is vendored here; the wire format is `docs/ego_teleop/MAST3R_LIVE_PROTOCOL.md`.

Three properties are what make this usable for teleoperation rather than for a bake-off:

**1. Latest frame, never a queue (sections 6, 17, 25).** At most one frame is in flight. A frame that arrives while
the server is still working on the previous one is DROPPED and counted, and `push_image` returns the newest pose the
reader thread has — carrying ITS OWN timestamp, so `TrackingSupervisor` grades it by real age and a stalled server
becomes DEGRADED and then LOST on schedule. A backlog would instead keep returning fresh-looking poses of where the
hand was half a second ago, which is the failure §25 is written against: "a 30 FPS tracker with 400 ms backlog".

**2. Global corrections are announced, not smuggled.** MASt3R-Fusion runs loop closure and global BA, so a pose it
emits can move the whole past trajectory. The server compares the re-read pose of the previously emitted frame
against what it actually sent; when the world moved it sets `map_update` on the next pose, and
`LocalPoseContinuity` absorbs it into an offset instead of handing the arm a jump (section 3). An UNFLAGGED jump
stays a jump and is graded LOST — a tracking failure and a loop closure look identical in the pose alone.

**3. Nothing is faked.** No server, a refused handshake or a protocol mismatch raises `BackendUnavailable` at
construction; a dropped connection makes every subsequent estimate LOST. The backend never invents a pose, never
extrapolates from IMU on its own (that is the estimator's job on the far side), and never reports a stale pose as
fresh.

    fused_wrist.yaml:
        vi_backend: mast3r_live
        vi_backend_options: {host: gpu-5090, port: 5577}

Smoke test without the GPU box, from the repo root:

    .venv/bin/python -m ego_teleop.tracking.backends.mast3r_live --selftest
"""
from __future__ import annotations
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from handumi_collector.pose.estimator import BackendInfo, BackendUnavailable, PoseEstimate, PoseEstimator, TrackingState

log = logging.getLogger("ego_teleop.mast3r_live")

MAGIC = b"M3RF"
PROTOCOL_VERSION = 1
HEAD = struct.Struct("<4sII")            # magic, header_len, payload_len
MAX_PLAUSIBLE_CAPTURE_LAG_MS = 2000.0    # above this, the frame timestamp is not on the host monotonic clock

STATE_MAP = {"init": TrackingState.INITIALIZING, "initializing": TrackingState.INITIALIZING,
             "tracking": TrackingState.TRACKING, "degraded": TrackingState.DEGRADED,
             "reloc": TrackingState.DEGRADED,     # relocalising IS a pose of reduced trust, not an absent one
             "lost": TrackingState.LOST, "terminated": TrackingState.LOST}


# ---- framing ------------------------------------------------------------------------------------------------
def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    h = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(HEAD.pack(MAGIC, len(h), len(payload)) + h + payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(n - len(buf))
        if not b: raise ConnectionError("peer closed mid-message")
        buf += b
    return bytes(buf)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    magic, hl, pl = HEAD.unpack(_recv_exactly(sock, HEAD.size))
    if magic != MAGIC: raise ConnectionError(f"bad magic {magic!r}: not a MASt3R-live stream")
    if hl > 1 << 20 or pl > 1 << 26: raise ConnectionError(f"implausible message ({hl} / {pl} bytes)")
    return json.loads(_recv_exactly(sock, hl)), _recv_exactly(sock, pl)


# ---- the backend --------------------------------------------------------------------------------------------
@dataclass
class _Link:
    """Everything the reader thread and the caller share. Guarded by `lock`; both sides hold it only to swap."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    est: PoseEstimate | None = None        # newest pose the server has returned
    served_seq: int = -1
    inflight: bool = False
    error: str = ""
    closed: bool = False


class Mast3rLiveBackend(PoseEstimator):
    INFO = BackendInfo(
        name="mast3r_live", uses_imu=True, metric_scale=True, online_capable=True,
        provides=("confidence", "num_features"),
        notes="MASt3R-Fusion driven live over a socket by scripts/mast3r_live_server.py; latest-frame (never queued); "
              "global/loop-closure corrections arrive flagged as extra['map_update'] and are absorbed downstream")
    info = INFO

    def __init__(self, host: str = "127.0.0.1", port: int = 5577, *, connect_timeout_s: float = 5.0,
                 jpeg_quality: int = 82, max_imu_batch: int = 4096, request_timeout_s: float = 2.0,
                 **_ignored) -> None:
        self.rect = None                 # KB -> pinhole remap, built in initialize() when the model needs it
        self.host, self.port = str(host), int(port)
        self.connect_timeout_s, self.request_timeout_s = float(connect_timeout_s), float(request_timeout_s)
        self.jpeg_quality, self.max_imu_batch = int(jpeg_quality), int(max_imu_batch)
        self.sock: socket.socket | None = None
        self.server: dict = {}
        self._link = _Link()
        self._reader: threading.Thread | None = None
        self._imu: list[tuple[int, float, float, float, float, float, float]] = []
        self._imu_lock = threading.Lock()
        self._seq = 0
        self._last_returned_seq = -1
        self.counters = dict(frames_sent=0, frames_dropped_busy=0, imu_sent=0, imu_dropped_overflow=0,
                             poses=0, map_updates=0, server_ms_last=0.0, link_ms_last=0.0,
                             cap_to_send_ms_last=0.0, net_ms_last=0.0, frame_clock_implausible=0)
        self._sent_at: dict[int, float] = {}

    # ---- lifecycle ------------------------------------------------------------------------------------------
    def initialize(self, *, intrinsics: dict | None = None, T_camera_imu: np.ndarray | None = None,
                   imu_noise: dict | None = None, image_size: tuple[int, int] | None = None) -> None:
        if intrinsics is None:
            raise BackendUnavailable("mast3r_live needs camera intrinsics: MASt3R-Fusion runs `use_calib` and an "
                                     "uncalibrated monocular stream has no metric scale to give the arm")
        # MASt3R-Fusion's Intrinsics.from_calib knows pinhole(+radtan) and mei only. The wrist Arducam is
        # Kannala-Brandt, so it is rectified HERE, with the same call and the same `balance` the EuRoC export uses
        # (ego_teleop/transforms/fisheye.py) — otherwise the live tracker and the offline bake-off would be looking
        # at two different cameras and their numbers could not be compared. The server is told the PINHOLE model.
        intrinsics = dict(intrinsics)
        if str(intrinsics.get("model", "pinhole")).lower() in ("kannala_brandt", "kb", "equidistant", "fisheye"):
            from ...transforms.fisheye import DEFAULT_BALANCE, make_rectifier
            try:
                self.rect = make_rectifier(np.asarray(intrinsics["K"], float), np.asarray(intrinsics["D"], float),
                                           intrinsics.get("image_size") or (intrinsics["width"], intrinsics["height"]),
                                           downscale=int(intrinsics.get("downscale", 1)),
                                           balance=float(intrinsics.get("balance", DEFAULT_BALANCE)))
            except (KeyError, TypeError, ValueError) as exc:
                raise BackendUnavailable(f"fisheye intrinsics for mast3r_live are incomplete ({exc}); need K, D and "
                                         f"image_size from configs/calibration/fisheye_<side>") from None
            intrinsics = dict(self.rect.intrinsics(), rectified_from=str(intrinsics.get("model")),
                              balance=self.rect.balance)
            image_size = self.rect.size
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        except OSError as exc:
            raise BackendUnavailable(
                f"no MASt3R live server at {self.host}:{self.port} ({exc}). Start it in the MASt3R-Fusion checkout: "
                f"`python mast3r_live_server.py --config config/base.yaml --port {self.port}` "
                f"(see docs/ego_teleop/MAST3R_LIVE_PROTOCOL.md). No pose source is substituted.") from None
        s.settimeout(self.connect_timeout_s)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)     # a 30 Hz pose stream must not wait for Nagle
        try:
            send_msg(s, dict(type="hello", version=PROTOCOL_VERSION, intrinsics=_jsonable(intrinsics),
                             image_size=list(image_size) if image_size else None,
                             T_camera_imu=None if T_camera_imu is None else np.asarray(T_camera_imu, float).reshape(16).tolist(),
                             imu_noise=_jsonable(imu_noise or {}), encoding="jpeg"))
            head, _ = recv_msg(s)
        except (OSError, ConnectionError, ValueError) as exc:
            s.close(); raise BackendUnavailable(f"MASt3R live handshake failed: {exc}") from None
        if head.get("type") != "ready":
            s.close(); raise BackendUnavailable(f"MASt3R live server refused the session: {head.get('error') or head}")
        if int(head.get("version", 0)) != PROTOCOL_VERSION:
            s.close(); raise BackendUnavailable(f"protocol mismatch: server v{head.get('version')} != client v{PROTOCOL_VERSION}")
        self.sock, self.server = s, head
        s.settimeout(None)
        self._link = _Link()
        self._reader = threading.Thread(target=self._read_loop, name="mast3r-live-rx", daemon=True)
        self._reader.start()
        log.info("mast3r_live: connected to %s:%s (%s)", self.host, self.port, head.get("backend", "?"))

    def close(self) -> None:
        s, self.sock = self.sock, None
        with self._link.lock: self._link.closed = True
        if s is not None:
            try: send_msg(s, dict(type="bye"))
            except OSError: pass
            try: s.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            s.close()
        if self._reader is not None: self._reader.join(2.0); self._reader = None

    def reset(self) -> None:
        """A new session on the far side: the map, the VI initialisation and the world frame all start over."""
        with self._imu_lock: self._imu.clear()
        self._seq = 0; self._last_returned_seq = -1; self._sent_at.clear()
        with self._link.lock:
            self._link.est = None; self._link.inflight = False; self._link.served_seq = -1
        if self.sock is not None:
            try: send_msg(self.sock, dict(type="reset"))
            except OSError as exc: self._fail(f"reset failed: {exc}")

    # ---- feeding --------------------------------------------------------------------------------------------
    def push_imu(self, t_ns: int, gyro_rad_s, accel_m_s2) -> None:
        """Buffered, and sent with the next frame. 416 Hz of one-sample TCP writes would spend more time in the
        kernel than in the estimator, and the far side consumes IMU per frame interval anyway (its `IMUPool` is
        queried between two frame stamps)."""
        g, a = np.asarray(gyro_rad_s, float), np.asarray(accel_m_s2, float)
        with self._imu_lock:
            if len(self._imu) >= self.max_imu_batch:
                # the link is down or the server has stopped asking; dropping the OLDEST keeps the batch that will
                # actually be integrated next, and the count says how much history was lost
                self._imu.pop(0); self.counters["imu_dropped_overflow"] += 1
            self._imu.append((int(t_ns), float(g[0]), float(g[1]), float(g[2]), float(a[0]), float(a[1]), float(a[2])))

    def push_image(self, t_ns: int, frame_index: int, image_bgr: np.ndarray) -> PoseEstimate:
        t_ns, frame_index = int(t_ns), int(frame_index)
        if self.sock is None:
            return self._lost(t_ns, frame_index, "not connected")
        with self._link.lock:
            busy, err = self._link.inflight, self._link.error
        if err:
            return self._lost(t_ns, frame_index, err)
        if busy:
            # LATEST-FRAME: drop this one rather than queue it. The returned pose keeps its own (older) timestamp,
            # so the supervisor sees the real age instead of a fresh-looking stale pose.
            self.counters["frames_dropped_busy"] += 1
            return self._latest_or_lost(t_ns, frame_index)
        blob = self._encode(image_bgr)
        if blob is None:
            return self._lost(t_ns, frame_index, "jpeg encode failed")
        with self._imu_lock:
            imu, self._imu = self._imu, []
        self._seq += 1
        with self._link.lock:
            self._link.inflight = True
        t_send = time.monotonic_ns()
        self._sent_at[self._seq] = t_send / 1e9
        # capture -> send: JPEG encode, rectification and however long this frame waited for the previous answer.
        # It is the one stage of the chain that lives entirely on this side, so it is the one we can act on.
        #
        # `t_ns` is documented as host monotonic (devices/base.py: "host monotonic right after grab()"). If it is
        # not, this difference is garbage — and so is every camera<->IMU pairing in the fusion, because they are
        # timed against the same clock. So an implausible value is reported as NaN and COUNTED rather than
        # clamped: a silent 0 here would hide a broken timeline, which is the one thing spec section 10 is about.
        cap_ms = (t_send - t_ns) / 1e6
        if -1.0 <= cap_ms <= MAX_PLAUSIBLE_CAPTURE_LAG_MS:
            self.counters["cap_to_send_ms_last"] = cap_ms
        else:
            self.counters["cap_to_send_ms_last"] = float("nan")
            self.counters["frame_clock_implausible"] += 1
        try:
            send_msg(self.sock, dict(type="frame", seq=self._seq, t_ns=t_ns, frame_index=frame_index,
                                     imu=[list(x) for x in imu]), blob)
        except OSError as exc:
            with self._imu_lock: self._imu = imu + self._imu      # nothing was consumed; keep it for the retry
            with self._link.lock: self._link.inflight = False
            self._fail(f"send failed: {exc}")
            return self._lost(t_ns, frame_index, str(exc))
        self.counters["frames_sent"] += 1; self.counters["imu_sent"] += len(imu)
        return self._latest_or_lost(t_ns, frame_index)

    # ---- the reader thread ----------------------------------------------------------------------------------
    def _read_loop(self) -> None:
        sock = self.sock
        while sock is not None:
            try:
                head, _ = recv_msg(sock)
            except (OSError, ConnectionError, ValueError) as exc:
                with self._link.lock:
                    if not self._link.closed: self._link.error = f"link lost: {exc}"
                    self._link.inflight = False
                return
            if head.get("type") != "pose":
                if head.get("type") == "error":
                    with self._link.lock: self._link.error = str(head.get("error", "server error"))
                continue
            est = self._to_estimate(head)
            seq = int(head.get("seq", -1))
            sent = self._sent_at.pop(seq, None)
            srv_ms = float(head.get("server_ms", 0.0) or 0.0)
            if sent is not None:
                link = (time.monotonic() - sent) * 1000.0
                self.counters["link_ms_last"] = link
                # what the wire costs, once the GPU's own time is taken out. Split because they are fixed by
                # different things: `net` by the link and the JPEG size, `server` by the model and the GPU.
                self.counters["net_ms_last"] = max(link - srv_ms, 0.0)
                est.extra.update(link_ms=link, net_ms=max(link - srv_ms, 0.0),
                                 cap_to_send_ms=self.counters["cap_to_send_ms_last"])
            self.counters["server_ms_last"] = srv_ms
            est.extra["recv_mono_ns"] = time.monotonic_ns()
            self.counters["poses"] += 1
            if est.extra.get("map_update"): self.counters["map_updates"] += 1
            with self._link.lock:
                self._link.est = est; self._link.served_seq = seq; self._link.inflight = False

    def _to_estimate(self, h: dict) -> PoseEstimate:
        T = h.get("T_world_camera")
        state = STATE_MAP.get(str(h.get("state", "")).lower(), TrackingState.UNINITIALIZED)
        M = None if T is None else np.asarray(T, float).reshape(4, 4)
        if M is not None and not np.all(np.isfinite(M)): M, state = None, TrackingState.LOST
        extra = dict(backend="mast3r_live", keyframe=bool(h.get("keyframe", False)),
                     server_ms=float(h.get("server_ms", 0.0) or 0.0), server_state=h.get("state"),
                     server_dropped=int(h.get("dropped", 0) or 0))
        # The first-hardware-run gate needs all three poses in the log, not just the one the robot follows:
        #   T_visual_raw    the tracker's answer BEFORE the global step   (extra["visual_raw_*"])
        #   T_map           the pose on the wire, post-optimisation        (= M, logged as vi_map_* downstream)
        #   T_local_control LocalPoseContinuity's output                   (logged as vi_* downstream)
        # `map` may move; `local_control` may not. With only the first two it is impossible to tell a loop closure
        # from the tracker genuinely losing the hand.
        raw = h.get("T_visual_raw")
        if raw is not None:
            R = np.asarray(raw, float).reshape(4, 4)
            if np.all(np.isfinite(R)):
                extra.update(visual_raw_x=float(R[0, 3]), visual_raw_y=float(R[1, 3]), visual_raw_z=float(R[2, 3]))
        extra["d_visual_raw_m"] = float(h.get("d_visual_raw_m", 0.0) or 0.0)
        if h.get("map_update"):
            # LocalPoseContinuity looks for exactly this key (fused_wrist.yaml provider.local.map_update_keys) and
            # absorbs the step into an offset, so a loop closure corrects the map and not the arm.
            extra["map_update"] = True
            extra["map_update_m"] = float(h.get("map_update_m", 0.0) or 0.0)
        return PoseEstimate(int(h.get("t_ns", 0)), int(h.get("frame_index", 0)), M, state,
                            confidence=_opt_float(h.get("confidence")), num_features=_opt_int(h.get("num_features")),
                            reprojection_error=_opt_float(h.get("reprojection_error")), extra=extra)

    # ---- helpers --------------------------------------------------------------------------------------------
    def _latest_or_lost(self, t_ns: int, frame_index: int) -> PoseEstimate:
        with self._link.lock:
            est = self._link.est
        if est is None: return self._lost(t_ns, frame_index, "no pose yet")
        return est

    def _lost(self, t_ns: int, frame_index: int, reason: str) -> PoseEstimate:
        return PoseEstimate(int(t_ns), int(frame_index), None, TrackingState.LOST,
                            extra=dict(backend="mast3r_live", reason=reason))

    def _fail(self, msg: str) -> None:
        with self._link.lock: self._link.error = msg
        log.warning("mast3r_live: %s", msg)

    def _encode(self, image_bgr: np.ndarray) -> bytes | None:
        import cv2
        a = np.asarray(image_bgr)
        if self.rect is not None:
            if a.shape[1::-1] != self.rect.size:
                self._fail(f"frame is {a.shape[1]}x{a.shape[0]} but the rectifier was built for "
                           f"{self.rect.size[0]}x{self.rect.size[1]} (check vi_backend_options.downscale)")
                return None
            a = self.rect(a)
        if a.ndim == 2: a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return buf.tobytes() if ok else None

    def get_pose(self) -> PoseEstimate | None:
        with self._link.lock: return self._link.est

    def get_quality(self) -> dict:
        with self._link.lock:
            err, inflight, seq = self._link.error, self._link.inflight, self._link.served_seq
        with self._imu_lock: pending = len(self._imu)
        return dict(backend="mast3r_live", host=f"{self.host}:{self.port}", server=self.server.get("backend", ""),
                    error=err, inflight=inflight, served_seq=seq, imu_pending=pending,
                    rectified=self.rect is not None, **self.counters)

    def latency_ms(self) -> dict:
        """The four stages of spec section 25, separately, because they are fixed by different things. The fifth
        number — motion -> fused pose — is the sum plus the fusion tick, and only the fusion loop can measure it."""
        c = self.counters
        return dict(capture_to_send=c["cap_to_send_ms_last"], network_rtt=c["net_ms_last"],
                    gpu_inference=c["server_ms_last"], link_total=c["link_ms_last"])


def _opt_float(v):
    try: return None if v is None else float(v)
    except (TypeError, ValueError): return None


def _opt_int(v):
    try: return None if v is None else int(v)
    except (TypeError, ValueError): return None


def _jsonable(d):
    out = {}
    for k, v in dict(d or {}).items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


# ---- a server that needs no GPU, for protocol tests and for --selftest -------------------------------------
class EchoPoseServer:
    """A stand-in for the real server: same wire format, a pose that is a pure function of the frame index. It exists
    so the framing, the latest-frame drop rule, the map-update path and the failure modes can be tested on a laptop.
    It is NOT a tracker and must never be used to produce a number about the hardware."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, delay_s: float = 0.0,
                 map_update_every: int = 0, drop_after: int | None = None, version: int | None = None) -> None:
        self.version = PROTOCOL_VERSION if version is None else int(version)
        self.srv = socket.socket(); self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port)); self.srv.listen(1)
        self.host, self.port = self.srv.getsockname()
        self.delay_s, self.map_update_every, self.drop_after = delay_s, map_update_every, drop_after
        self.frames = 0; self.imu = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._serve, name="mast3r-echo", daemon=True); self._t.start()

    def _serve(self) -> None:
        try: conn, _ = self.srv.accept()
        except OSError: return
        with conn:
            try:
                head, _ = recv_msg(conn)
                if head.get("type") != "hello":
                    send_msg(conn, dict(type="error", error="expected hello")); return
                send_msg(conn, dict(type="ready", version=self.version, backend="echo", img_size=512))
                while not self._stop.is_set():
                    h, _ = recv_msg(conn)
                    if h.get("type") == "bye": return
                    if h.get("type") == "reset": self.frames = 0; continue
                    if h.get("type") != "frame": continue
                    self.frames += 1; self.imu += len(h.get("imu", []))
                    if self.drop_after is not None and self.frames > self.drop_after: return
                    if self.delay_s: time.sleep(self.delay_s)
                    T = np.eye(4); T[0, 3] = 0.01 * self.frames
                    mu = bool(self.map_update_every and self.frames % self.map_update_every == 0)
                    send_msg(conn, dict(type="pose", seq=h["seq"], t_ns=h["t_ns"], frame_index=h.get("frame_index", 0),
                                        T_world_camera=T.reshape(16).tolist(), state="tracking", confidence=0.9,
                                        num_features=200, keyframe=False, server_ms=self.delay_s * 1000.0,
                                        **(dict(map_update=True, map_update_m=0.03) if mu else {})))
            except (OSError, ConnectionError, ValueError):
                return

    def close(self) -> None:
        self._stop.set()
        try: self.srv.close()
        except OSError: pass


def _selftest() -> int:
    srv = EchoPoseServer(delay_s=0.0, map_update_every=5)
    b = Mast3rLiveBackend(host=srv.host, port=srv.port)
    b.initialize(intrinsics=dict(fx=300.0, fy=300.0, cx=320.0, cy=240.0, model="pinhole"),
                 T_camera_imu=np.eye(4), image_size=(640, 480))
    img = np.zeros((480, 640, 3), np.uint8)
    got = 0
    for i in range(20):
        b.push_imu(i * 1_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 9.81))
        est = b.push_image(i * 33_000_000, i, img)
        for _ in range(200):                       # the pose comes back on the reader thread
            if b.counters["poses"] > got: break
            time.sleep(0.001)
        got = b.counters["poses"]
        est = b.get_pose()
    q = b.get_quality()
    print(json.dumps(q, indent=1))
    b.close(); srv.close()
    ok = q["poses"] >= 15 and q["map_updates"] >= 2 and not q["error"]
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="round-trip the protocol against the in-process echo server")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.selftest else ap.print_help() or 0)
