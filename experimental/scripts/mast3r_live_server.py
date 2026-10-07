#!/usr/bin/env python3
"""Drive MASt3R-Fusion live, over a socket, for wrist teleoperation (multi-sensor spec sections 2, 10, 14).

THIS FILE RUNS ON THE GPU BOX, INSIDE THE MASt3R-Fusion CHECKOUT — it imports `mast3r_fusion.*`, which only exists
there. It is versioned in ego_collector because it is OUR code and it has to stay in step with the client
(`ego_teleop/tracking/backends/mast3r_live.py`) and with the wire format
(`docs/ego_teleop/MAST3R_LIVE_PROTOCOL.md`). Nothing of MASt3R-Fusion is copied into ego_collector; this goes the
other way.

    # on the GPU box, once per session
    scp scripts/mast3r_live_server.py gpu-5090:~/vio_bakeoff/MASt3R-Fusion/
    ssh gpu-5090 'cd ~/vio_bakeoff/MASt3R-Fusion && python mast3r_live_server.py --config config/base_euroc.yaml --port 5577'

    # on the Mac
    vi_backend: mast3r_live
    vi_backend_options: {host: gpu-5090, port: 5577}

It is `main.py`'s loop with the dataset replaced by the socket, and four differences that matter:

**1. The frame source is live and lossy.** `SocketStream` holds exactly one pending frame. If a newer frame arrives
while the tracker is still working, the older one is DROPPED and counted. The client already sends at most one
frame in flight, so this is the second line of defence; between them there is nowhere for a backlog to form, which
is what §25 is about. `main.py` reads a directory and can never be late.

**2. IMU arrives incrementally.** `FactorGraph` builds an `IMUPool` from a whole file at startup and bisects into
it; `AppendableImuPool` keeps the same `get_records(t0, t1)` contract over a buffer that grows. `poses_stamps` is
bound to the stream's `timestamps` LIST, which the stream appends to, so the factor graph keeps seeing new frame
stamps without being rebuilt.

**3. Global corrections are announced.** After each optimisation step the pose of the last emitted keyframe is
re-read; if the optimiser moved it, the next pose goes out with `map_update` and how far the world shifted. The
client's `LocalPoseContinuity` then absorbs it instead of handing the arm a jump (section 3). Without this flag a
loop closure and a tracking failure are indistinguishable downstream, and the safe reading of the pair — grade it
LOST — would throw away every loop closure.

**4. No viewer, no result.txt, no h5.** `--no-viz` is implied; the trajectory leaves over the socket. The canonical
record of what the robot got is the client's `raw/fused_wrist_pose_live_<side>.parquet`, not a file here.

Timestamps: the client sends host-monotonic ns. They are converted to seconds against the FIRST frame of the
session, so the floats MASt3R-Fusion works in stay small and IMU and camera stay on one clock (the client has
already applied the camera<->IMU offset). `IMUPool` demands strictly increasing time, so out-of-order or duplicate
IMU samples are dropped and counted rather than being allowed to raise mid-session.
"""
from __future__ import annotations
import argparse
import bisect
import json
import socket
import struct
import sys
import threading
import time
import traceback
import numpy as np

MAGIC = b"M3RF"
PROTOCOL_VERSION = 1
HEAD = struct.Struct("<4sII")


# ---- framing (byte-identical to the client; kept inline so this file can be scp'd on its own) ----------------
def send_msg(sock, header: dict, payload: bytes = b"") -> None:
    h = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(HEAD.pack(MAGIC, len(h), len(payload)) + h + payload)


def _recv_exactly(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(n - len(buf))
        if not b: raise ConnectionError("peer closed mid-message")
        buf += b
    return bytes(buf)


def recv_msg(sock) -> tuple[dict, bytes]:
    magic, hl, pl = HEAD.unpack(_recv_exactly(sock, HEAD.size))
    if magic != MAGIC: raise ConnectionError(f"bad magic {magic!r}")
    if hl > 1 << 20 or pl > 1 << 26: raise ConnectionError(f"implausible message ({hl} / {pl})")
    return json.loads(_recv_exactly(sock, hl)), _recv_exactly(sock, pl)


# ---- IMU: the same contract as mast3r_fusion.geoFunc.data_utils.IMUPool, but appendable ----------------------
class AppendableImuPool:
    """`get_records(t0, t1)` over a growing buffer, in the units the offline pool produces.

    The offline pool converts gyro to DEGREES/s internally (`degree=False` multiplies by 180/pi), so the same
    conversion happens here — the factor graph downstream reads `self.data` expecting degrees. Getting that wrong
    produces a rotation error of 57x, which looks like a broken camera rather than a unit bug.
    """
    DEG = 180.0 / np.pi

    def __init__(self, capacity: int = 1 << 20) -> None:
        self.time: list[float] = []
        self.rows: list[np.ndarray] = []
        self.capacity = capacity
        self.dropped_nonmonotonic = 0
        self.lock = threading.Lock()

    def append(self, t_s: float, gyro_rad_s, accel_m_s2) -> None:
        with self.lock:
            if self.time and t_s <= self.time[-1]:
                self.dropped_nonmonotonic += 1        # IMUPool's invariant; a repeated stamp would poison bisect
                return
            self.time.append(float(t_s))
            g = np.asarray(gyro_rad_s, float) * self.DEG
            self.rows.append(np.concatenate([g, np.asarray(accel_m_s2, float)]))
            if len(self.time) > self.capacity:
                del self.time[:len(self.time) - self.capacity]; del self.rows[:len(self.rows) - self.capacity]

    @property
    def data(self) -> np.ndarray:
        with self.lock: return np.asarray(self.rows, float) if self.rows else np.zeros((0, 6))

    def span(self) -> tuple[float, float]:
        with self.lock: return (self.time[0], self.time[-1]) if self.time else (float("nan"), float("nan"))

    def get_records(self, t0, t1):
        """Same walk as the offline pool: successive sub-intervals [cur, next) each carrying the sample that spans
        them. Raises if the buffer does not yet cover `t1` — the caller must not integrate over IMU it does not
        have, and silently truncating would quietly bias the preintegration."""
        with self.lock:
            time, rows = self.time, self.rows
            if not time or t1 > time[-1] + 1e-9:
                raise LookupError(f"IMU does not cover [{t0:.4f}, {t1:.4f}] (have up to "
                                  f"{time[-1] if time else float('nan'):.4f})")
            dd, cur = [], t0
            while True:
                idx = bisect.bisect(time, cur + 0.001)
                if idx >= len(time): raise LookupError(f"IMU ran out inside [{t0:.4f}, {t1:.4f}]")
                if time[idx] >= t1 - 1e-6:
                    dd.append([cur, t1, rows[idx]]); break
                dd.append([cur, time[idx], rows[idx]]); cur = time[idx]
            return dd


# ---- the live frame source, shaped like a MonocularDataset ---------------------------------------------------
def make_socket_stream(base_cls, img_size: int):
    """Built at run time so this file imports cleanly (for `--dry-run`) without MASt3R-Fusion installed."""

    class SocketStream(base_cls):
        """One pending frame, never a queue. `timestamps` is the list `FactorGraph.poses_stamps` is bound to."""

        def __init__(self) -> None:
            super().__init__()
            self.img_size = img_size
            self.save_results = False
            self.use_calibration = True
            self.timestamps = []
            self._cv = threading.Condition()
            self._pending = None                     # (t_s, rgb, seq, frame_index)
            self._closed = False
            self.dropped_stale = 0
            self.received = 0
            self.served: list[tuple[int, int]] = []  # (seq, frame_index) in the order the tracker consumed them

        # -- producer (the connection thread)
        def offer(self, t_s: float, rgb: np.ndarray, seq: int, frame_index: int) -> None:
            with self._cv:
                if self._pending is not None:
                    self.dropped_stale += 1          # the tracker is behind: keep the NEWEST, latency over FPS
                self._pending = (t_s, rgb, seq, frame_index)
                self.received += 1
                self._cv.notify()

        def close(self) -> None:
            with self._cv:
                self._closed = True; self._cv.notify_all()

        # -- consumer (the tracker loop), through the MonocularDataset interface
        def __len__(self) -> int: return 1 << 30     # a live stream has no length; the loop stops on `closed`

        def read_img(self, idx: int) -> np.ndarray:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait(0.5)
                if self._pending is None: raise EOFError("stream closed")
                t_s, rgb, seq, frame_index = self._pending
                self._pending = None
            # timestamps is appended in CONSUMPTION order and indexed by frame id, which is what poses_stamps needs
            self.timestamps.append(float(t_s))
            self.served.append((int(seq), int(frame_index)))
            return rgb

        def get_timestamp(self, idx: int) -> float: return self.timestamps[idx]

        def has_calib(self) -> bool: return self.camera_intrinsics is not None

        def subsample(self, *a, **k) -> None: pass   # a live stream is not subsampled; the client paces it

    return SocketStream


# ---- the session ---------------------------------------------------------------------------------------------
class Session:
    """One client, one MASt3R-Fusion world. Everything CUDA happens on the main thread; the socket has its own."""

    def __init__(self, conn, args) -> None:
        self.conn, self.args = conn, args
        self.t0_ns: int | None = None
        self.imu = AppendableImuPool()
        self.stream = None
        self.stop = threading.Event()
        self.hello: dict = {}
        self.stats = dict(frames=0, poses=0, map_updates=0, imu=0)
        self._send_lock = threading.Lock()
        self._prev_raw = None                 # previous T_visual_raw, for the pre-optimisation frame-to-frame delta

    def t_s(self, t_ns: int) -> float:
        if self.t0_ns is None: self.t0_ns = int(t_ns)
        return (int(t_ns) - self.t0_ns) / 1e9

    def send(self, header: dict) -> None:
        with self._send_lock:
            send_msg(self.conn, header)

    # -- the socket thread: decode frames and IMU, hand them to the stream ------------------------------------
    def rx_loop(self) -> None:
        import cv2
        try:
            while not self.stop.is_set():
                head, payload = recv_msg(self.conn)
                kind = head.get("type")
                if kind == "bye": break
                if kind == "reset":
                    self.send(dict(type="error", error="reset mid-session is not supported; reconnect")); break
                if kind != "frame": continue
                for s in head.get("imu", ()):
                    self.imu.append(self.t_s(int(s[0])), s[1:4], s[4:7]); self.stats["imu"] += 1
                img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self.send(dict(type="error", error=f"undecodable frame seq={head.get('seq')}")); continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.stream.offer(self.t_s(int(head["t_ns"])), rgb, int(head.get("seq", -1)),
                                  int(head.get("frame_index", 0)))
                self.stats["frames"] += 1
        except (OSError, ConnectionError, ValueError) as exc:
            print(f"[rx] {exc}")
        finally:
            self.stop.set()
            if self.stream is not None: self.stream.close()

    # -- the main (CUDA) thread -------------------------------------------------------------------------------
    def run(self) -> None:
        import lietorch
        import torch
        import yaml
        from scipy.spatial.transform import Rotation
        from mast3r_fusion.config import config, load_config
        from mast3r_fusion.dataloader import Intrinsics, MonocularDataset
        from mast3r_fusion.frame import Mode, SharedKeyframes, SharedStates, create_frame
        from mast3r_fusion.global_opt import FactorGraph
        from mast3r_fusion.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
        from mast3r_fusion.tracker import FrameTracker
        import torch.multiprocessing as mp

        head, _ = recv_msg(self.conn)
        if head.get("type") != "hello" or int(head.get("version", 0)) != PROTOCOL_VERSION:
            self.send(dict(type="error", error=f"expected hello v{PROTOCOL_VERSION}, got {head}")); return
        self.hello = head
        intr = head.get("intrinsics") or {}
        if str(intr.get("model", "pinhole")).lower() != "pinhole":
            self.send(dict(type="error", error=f"this server takes PINHOLE frames; got model={intr.get('model')!r}. "
                                               f"The client rectifies fisheye itself (transforms/fisheye.py).")); return

        load_config(self.args.config)
        config["use_calib"] = True
        if head.get("imu_noise"):
            n = head["imu_noise"]
            noise = [float(n["accelerometer_noise_density"]), float(n["gyroscope_noise_density"]),
                     float(n["accelerometer_random_walk"]), float(n["gyroscope_random_walk"])]
            config.setdefault("ms_opt", {})["imu_noise"] = noise
            config.setdefault("global_opt", {})["imu_noise"] = noise
            config["ms_opt"]["imu_format"] = "custom_rad"

        device = self.args.device
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        manager = mp.Manager()

        W, H = int(intr["width"]), int(intr["height"])
        Stream = make_socket_stream(MonocularDataset, self.args.img_size)
        self.stream = Stream()
        self.stream.camera_intrinsics = Intrinsics.from_calib(
            self.stream.img_size, W, H,
            [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]), *intr.get("distortion", [0.0] * 4)],
            True, "pinhole", 1, None)
        h, w = self.stream.camera_intrinsics.img_size, W          # frame grid, as main.py derives it
        probe = np.zeros((H, W, 3), np.float32)
        from mast3r_fusion.image import resize_img
        h = resize_img(probe, self.stream.img_size)["img"][0].shape[1:]

        threading.Thread(target=self.rx_loop, name="m3r-rx", daemon=True).start()

        keyframes = SharedKeyframes(manager, h[0], h[1])
        states = SharedStates(manager, h[0], h[1])
        model = load_mast3r(device=device); model.share_memory()
        K = torch.from_numpy(self.stream.camera_intrinsics.K_frame).to(device, dtype=torch.float32)
        keyframes.set_intrinsics(K)
        tracker = FrameTracker(model, keyframes, device)
        fg = FactorGraph(model, keyframes, K, device, _FgArgs(self.args))
        fg.poses_stamps = self.stream.timestamps          # the SAME list; it grows as frames are consumed
        fg.imu_pool = self.imu                            # appendable, same get_records contract
        retrieval = load_retriever(model)
        self.send(dict(type="ready", version=PROTOCOL_VERSION, backend="mast3r_fusion",
                       img_size=self.stream.img_size, grid=[int(h[0]), int(h[1])], device=device,
                       config=self.args.config))
        print(f"[server] ready: {W}x{H} -> grid {h}, device {device}", flush=True)

        i = 0
        last_kf: tuple[int, np.ndarray] | None = None     # (frame_id, T_WC as emitted) for map-update detection
        pending_map_update = 0.0
        while not self.stop.is_set():
            try:
                t_start = time.monotonic()
                try:
                    timestamp, img = self.stream[i]
                except EOFError:
                    break
                seq, frame_index = self.stream.served[-1]
                mode = states.get_mode()
                T_WC = _identity_sim3(lietorch, Rotation) if i == 0 else states.get_frame().T_WC
                frame = create_frame(i, img, T_WC, img_size=self.stream.img_size, device=device)

                if mode == Mode.INIT:
                    X, C = mast3r_inference_mono(model, frame)
                    frame.update_pointmap(X, C)
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)
                    states.set_mode(Mode.TRACKING); states.set_frame(frame)
                    self._emit(seq, frame_index, timestamp, frame, "init", t_start, keyframe=True,
                               T_visual_raw=_mat(frame.T_WC))
                    i += 1
                    continue

                add_new_kf = False
                if mode == Mode.TRACKING:
                    add_new_kf, _match, try_reloc = tracker.track(frame)
                    if try_reloc: states.set_mode(Mode.RELOC)
                    states.set_frame(frame)
                    state_name = "reloc" if try_reloc else "tracking"
                else:
                    X, C = mast3r_inference_mono(model, frame)
                    frame.update_pointmap(X, C)
                    states.set_frame(frame); states.queue_reloc()
                    state_name = "reloc"

                # T_visual_raw: the tracker's own answer for THIS frame, read BEFORE the global optimisation.
                # It has to be captured here and not after, because `_backend_step` moves the keyframes and, when
                # this frame IS a keyframe, moves `frame.T_WC` with them — so the pose emitted after the step is a
                # globally optimised pose on keyframes and a tracked pose on the rest. That alternation is exactly
                # what makes `map_update` alone impossible to verify, so both poses go on the wire.
                T_visual_raw = _mat(frame.T_WC)

                if add_new_kf:
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)

                self._backend_step(states, keyframes, fg, retrieval, config)

                # section 3: did the optimiser move the world under the poses already sent?
                kf = keyframes.last_keyframe()
                T_now = kf.T_WC.data.cpu().numpy()[0][:3].copy()
                if last_kf is not None and last_kf[0] == kf.frame_id:
                    shift = float(np.linalg.norm(T_now - last_kf[1]))
                    if shift > self.args.map_update_eps_m: pending_map_update = shift
                last_kf = (kf.frame_id, T_now)

                d_raw = 0.0 if self._prev_raw is None else float(np.linalg.norm(T_visual_raw[:3, 3] - self._prev_raw[:3, 3]))
                self._prev_raw = T_visual_raw
                self._emit(seq, frame_index, timestamp, frame, state_name, t_start,
                           keyframe=add_new_kf, map_update_m=pending_map_update,
                           T_visual_raw=T_visual_raw, d_visual_raw_m=d_raw)
                pending_map_update = 0.0

                if len(keyframes) > 30: keyframes.roll_up(15)
                i += 1
            except Exception as exc:                  # one bad frame must not end the operator's session
                traceback.print_exc()
                try: self.send(dict(type="error", error=f"{type(exc).__name__}: {exc}"))
                except OSError: break
                i += 1
        print(f"[server] session over: {self.stats}", flush=True)

    def _backend_step(self, states, keyframes, fg, retrieval, config) -> None:
        """`main.py`'s `run_backend`, inline, minus the result.txt writing."""
        from mast3r_fusion.frame import Mode
        if states.get_mode() == Mode.INIT or states.is_paused(): return
        with states.lock:
            idx = states.global_optimizer_tasks[0] if states.global_optimizer_tasks else -1
        if idx == -1: return
        frame = keyframes[idx]
        kf_idx = [idx - 1] if idx >= 1 else []
        retrieval_inds = retrieval.update(frame, add_after_query=True, k=config["retrieval"]["k"],
                                          min_thresh=config["retrieval"]["min_thresh"])
        kf_idx += [k for k in _valid_numbers(idx, retrieval_inds) if abs(idx - k) < 20]
        kf_idx = [k for k in set(kf_idx) if k != idx]
        if kf_idx:
            fg.add_factors(kf_idx, [idx] * len(kf_idx), config["local_opt"]["min_match_frac"])
        with states.lock:
            states.edges_ii[:] = fg.ii.cpu().tolist(); states.edges_jj[:] = fg.jj.cpu().tolist()
        try:
            fg.solve_GN_calib(config["use_calib"])
            if getattr(fg, "init_vi_signal", False):
                fg.solve_GN_calib(config["use_calib"]); fg.init_vi_signal = False
                states.T_WC[:] = fg.frames.last_keyframe().T_WC[:].data
        except LookupError as exc:
            # the VI factor wanted IMU the stream has not delivered yet: skip THIS optimisation, keep the visual
            # pose. Never integrate over IMU that is not there.
            print(f"[backend] deferred: {exc}")
            return
        with states.lock:
            if states.global_optimizer_tasks: states.global_optimizer_tasks.pop(0)

    def _emit(self, seq, frame_index, timestamp, frame, state, t_start, *, keyframe=False, map_update_m=0.0,
              T_visual_raw=None, d_visual_raw_m=0.0) -> None:
        """Three poses go on the wire, per the first-hardware-run gate:

            T_world_camera  the pose the client follows   (post-optimisation: the backend's best current answer)
            T_visual_raw    the tracker's own answer for this frame, BEFORE the global step
            map_update_m    how far the optimiser moved the world under the poses already sent

        The client turns the first into `T_local_control` through `LocalPoseContinuity` and logs all of them. The
        gate is then directly measurable rather than assumed: **the map may move, T_local_control may not.**"""
        T = _mat(frame.T_WC)
        h = dict(type="pose", seq=int(seq), frame_index=int(frame_index),
                 t_ns=int(self.t0_ns + round(timestamp * 1e9)), state=state,
                 T_world_camera=None if T is None else T.reshape(16).tolist(),
                 T_visual_raw=None if T_visual_raw is None else np.asarray(T_visual_raw, float).reshape(16).tolist(),
                 d_visual_raw_m=float(d_visual_raw_m),
                 keyframe=bool(keyframe), server_ms=(time.monotonic() - t_start) * 1000.0,
                 dropped=int(self.stream.dropped_stale), num_features=None, confidence=None)
        if map_update_m: h.update(map_update=True, map_update_m=float(map_update_m)); self.stats["map_updates"] += 1
        self.send(h); self.stats["poses"] += 1


class _FgArgs:
    """FactorGraph reads a handful of argparse attributes. The IMU ones are unused here: `fg.imu_pool` is replaced
    by the appendable pool before the first optimisation, so nothing is ever loaded from a file."""
    def __init__(self, a) -> None:
        self.imu_path = ""; self.imu_dt = 0.0; self.save_h5 = False
        self.save_as = "live"; self.config = a.config; self.dataset = "live"


def _identity_sim3(lietorch, Rotation):
    """The camera-to-IMU-ish bootstrap pose `main.py` starts from, kept identical so the world convention matches."""
    T = lietorch.Sim3.Identity(1, device="cpu")
    M = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1], float).reshape(4, 4)
    q = Rotation.from_matrix(M[:3, :3]).as_quat()
    for k in range(3): T[0].data[k] = M[k, 3]
    for k in range(4): T[0].data[3 + k] = q[k]
    return T


def _mat(T_WC):
    """lietorch Sim3/SE3 -> a plain 4x4, or None. One helper so the raw and the post-optimisation reads cannot
    diverge in how they extract the pose."""
    if T_WC is None or not hasattr(T_WC, "matrix"): return None
    return np.asarray(T_WC.matrix()[0].detach().cpu().numpy(), float)


def _valid_numbers(a, b):
    """`main.py`'s find_valid_numbers, unchanged."""
    out = []
    for i, c in enumerate(b):
        if abs(c - a) <= 1: continue
        close = [j for j, d in enumerate(b) if abs(d - c) <= 20]
        if i == min(close) or c == a - 2: out.append(c)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5577)
    ap.add_argument("--config", default="config/base_euroc.yaml", help="MASt3R-Fusion config (relative to its root)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--map-update-eps-m", type=float, default=0.002,
                    help="a re-read keyframe pose that moved more than this = a global correction, announced to the "
                         "client so LocalPoseContinuity absorbs it instead of the arm")
    ap.add_argument("--once", action="store_true", help="serve one client and exit")
    ap.add_argument("--dry-run", action="store_true", help="check the wire format and exit; imports no CUDA")
    a = ap.parse_args(argv)

    if a.dry_run:
        print(json.dumps(dict(protocol=PROTOCOL_VERSION, magic=MAGIC.decode(), header=HEAD.format,
                              config=a.config, port=a.port), indent=1))
        return 0

    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port)); srv.listen(1)
    print(f"[server] MASt3R live on {a.host}:{a.port} (protocol v{PROTOCOL_VERSION})", flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[server] client {addr}", flush=True)
            with conn:
                try: Session(conn, a).run()
                except Exception:
                    traceback.print_exc()
            if a.once: break
    except KeyboardInterrupt:
        print("\n[server] stopped")
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
