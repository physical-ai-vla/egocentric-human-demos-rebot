"""Dedicated 6DoF tracker as a pose source (2026-10-15, user: "tracker path"; Vive / T265-class / XVisio / Mighty ...).
The teleop core never sees the tracker: this provider turns tracker poses into the same HandEstimate the RGB-D+IMU path produces.

Wire format (UDP JSON on fusion.tracker.port, one datagram per pose; any vendor bridge only has to emit this):
    {"kind": "tracker_pose", "side": "right", "seq": <int, +1 per pose>, "t_src_ns" (alias "t_ns"): <tracker's own sample time, optional>,
     "t_send_ns": <bridge send time, host monotonic>, "p": [x,y,z] (m, tracker world), "q_xyzw": [x,y,z,w], "valid": true, "conf": 0..1}
    Staleness uses OUR receive time (one clock for everything downstream); t_src / t_send are kept for latency logs.
    seq: gaps are counted as losses, an older seq than the last accepted one is dropped (reorder).
Frames:  T_W_TCP = T_W_TW . T_TW_TB . T_TB_TCP
    T_W_TW   tracker world -> W (gravity-up). Presets: identity (z-up trackers), steamvr_y_up (SteamVR / OpenVR worlds are y-up).
    T_TB_TCP tracker body -> UMI jaw-tip TCP: translation from the pivot calibration, rotation from the two-pose gravity mount
             (tools/calib_tracker.py pivot / mount). T_W_TW from calib_tracker world (jaw tip touched on 3 table points).
             Unmeasured => refused by the arm gate (config.unmeasured).
Quality: TRACKING (fresh, valid, conf >= warn) / DEGRADED (conf in [min, warn) or age in (stale_warn, hold]) / LOST (else)."""
from __future__ import annotations
import json
import socket
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from ..fusion.wrist_pose_fusion import HandEstimate, HandStatus

PRESETS = {"identity": np.eye(3), "steamvr_y_up": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}   # y-up -> z-up (x kept)


def make_T(R, p) -> np.ndarray:
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T


@dataclass
class TrackerPose:
    t_ns: int                       # our receive time (host monotonic)
    T_TW_TB: np.ndarray
    valid: bool
    conf: float
    seq: int = -1
    t_src_ns: int | None = None
    t_send_ns: int | None = None
    why: str = ""


@dataclass
class Tracker6DoFProvider:
    port: int
    T_W_TW: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_TB_TCP: np.ndarray = field(default_factory=lambda: np.eye(4))
    stale_warn_ms: float = 50.0
    hold_ms: float = 300.0
    min_conf: float = 0.5
    warn_conf: float = 0.8
    v_max_m_s: float = 2.0          # jump gate: a pose farther than v_max*dt + margin from the last ACCEPTED one is rejected
    margin_m: float = 0.03
    _latest: dict = field(default_factory=dict)
    _accepted: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    n_rx: int = 0
    stats: dict = field(default_factory=lambda: dict(lost_seq=0, reordered=0, invalid=0, jump_rejected=0))
    log: list | None = None         # set to [] to record (t_rx, seq, t_src, t_send, valid, conf, p_TW(3), q(4)) per datagram

    @classmethod
    def from_cfg(cls, tc: dict) -> "Tracker6DoFProvider":
        T_W_TW = make_T(PRESETS[tc["world_preset"]], np.zeros(3))
        if tc.get("T_W_TW") and tc["T_W_TW"].get("measured"): T_W_TW = np.asarray(tc["T_W_TW"]["matrix"], np.float64)   # explicit 4x4 wins
        T_tcp = np.asarray(tc["T_TB_TCP"]["matrix"], np.float64)
        return cls(int(tc["port"]), T_W_TW, T_tcp, tc["stale_warn_ms"], tc["hold_ms"], tc["min_conf"], tc["warn_conf"], tc.get("v_max_m_s", 2.0), tc.get("margin_m", 0.03))

    def ingest(self, msg: dict, t_rx_ns: int | None = None) -> None:
        if msg.get("kind") != "tracker_pose": return
        side = msg.get("side", "right"); t_rx = int(t_rx_ns if t_rx_ns is not None else time.monotonic_ns()); seq = int(msg.get("seq", -1))
        with self._lock:
            prev = self._latest.get(side); self.n_rx += 1
            if prev is not None and seq >= 0 and prev.seq >= 0:
                # 10-16 live: a restarted tracker counts from 0 again and every datagram was dropped as "reordered" (stuck invalid, INIT refused).
                # A lower seq with a NEWER send time is a tracker restart, not a reorder (robot8: a 300-seq threshold missed a short-lived tracker).
                t_new, t_old = msg.get("t_send_ns"), prev.t_send_ns
                restarted = t_new is not None and t_old is not None and int(t_new) > int(t_old)
                if seq <= prev.seq and not restarted: self.stats["reordered"] += 1; return    # older / duplicate datagram: drop
                if seq > prev.seq: self.stats["lost_seq"] += seq - prev.seq - 1
                else: self.stats["restarts"] = self.stats.get("restarts", 0) + 1
        p = np.asarray(msg.get("p", [np.nan] * 3), np.float64); q = np.asarray(msg.get("q_xyzw", [np.nan] * 4), np.float64)
        n = float(np.linalg.norm(q)) if np.all(np.isfinite(q)) else 0.0; why = ""
        if not bool(msg.get("valid", True)): why = "tracker_invalid_flag"
        elif not np.all(np.isfinite(p)): why = "tracker_nan_position"
        elif not (0.9 <= n <= 1.1): why = f"tracker_bad_quat:{n:.2f}"
        T = make_T(Rotation.from_quat(q / n).as_matrix(), p) if not why else None
        if T is not None:                                                                    # velocity-aware jump gate
            acc = self._accepted.get(side)
            if acc is not None:
                dt = max(0.0, (t_rx - acc[0]) / 1e9); allow = self.v_max_m_s * dt + self.margin_m
                if np.linalg.norm(T[:3, 3] - acc[1][:3, 3]) > allow:
                    with self._lock: self.stats["jump_rejected"] += 1
                    why = f"tracker_jump:{np.linalg.norm(T[:3, 3] - acc[1][:3, 3]) * 100:.0f}cm"; T = None
        if why and why != "" and not why.startswith("tracker_jump"): self.stats["invalid"] += 1
        t_src = msg.get("t_src_ns", msg.get("t_ns"))
        tp = TrackerPose(t_rx, T, not why, float(msg.get("conf", 1.0)), seq, t_src, msg.get("t_send_ns"), why)
        with self._lock:
            if self.log is not None:
                self.log.append((t_rx, seq, -1 if t_src is None else int(t_src), -1 if msg.get("t_send_ns") is None else int(msg["t_send_ns"]),
                                 int(not why), tp.conf, *p.tolist(), *(q.tolist() if q.shape == (4,) else [np.nan] * 4)))
            self._latest[side] = tp
            if not why: self._accepted[side] = (t_rx, T)

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.sock.bind(("127.0.0.1", self.port)); self.sock.settimeout(0.1)
        self._stop = threading.Event(); threading.Thread(target=self._run, daemon=True, name="tracker6dof").start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try: data, _ = self.sock.recvfrom(65535)
            except socket.timeout: continue
            except OSError: break
            try: self.ingest(json.loads(data))
            except (ValueError, KeyError, TypeError): pass

    def stop(self) -> None:
        self._stop.set()
        try: self.sock.close()
        except Exception: pass

    def latest_raw(self, side: str) -> TrackerPose | None:
        with self._lock: return self._latest.get(side)

    def estimate(self, side: str, t_ns: int) -> tuple[HandEstimate, str]:
        """-> (HandEstimate in W at the UMI TCP, quality)."""
        tp = self.latest_raw(side)
        if tp is None: return HandEstimate(side, t_ns, HandStatus.HOLD, ["tracker_no_data"], None, None, None, 0.0, "tracker", np.inf), "INIT"
        with self._lock: acc = self._accepted.get(side)
        if not tp.valid and acc is not None and (t_ns - acc[0]) / 1e6 <= self.hold_ms:
            # a bad sample (jump / bad quat / invalid flag): keep the last ACCEPTED pose for up to hold_ms, flagged DEGRADED
            age = (t_ns - acc[0]) / 1e6; T = self.T_W_TW @ acc[1] @ self.T_TB_TCP
            return HandEstimate(side, t_ns, HandStatus.WARN, [tp.why, f"held:{age:.0f}ms"], T, T[:3, 3].copy(), T[:3, :3].copy(), tp.conf, "tracker", age), "DEGRADED"
        age = (t_ns - tp.t_ns) / 1e6
        if not tp.valid or tp.conf < self.min_conf or age > self.hold_ms:
            why = tp.why if not tp.valid else (f"tracker_conf:{tp.conf:.2f}" if tp.conf < self.min_conf else f"tracker_stale:{age:.0f}ms")
            return HandEstimate(side, t_ns, HandStatus.HOLD, [why], None, None, None, tp.conf, "tracker", age), "LOST"
        T = self.T_W_TW @ tp.T_TW_TB @ self.T_TB_TCP
        warn = [f"tracker_age:{age:.0f}ms"] if age > self.stale_warn_ms else []
        if tp.conf < self.warn_conf: warn.append(f"tracker_conf:{tp.conf:.2f}")
        st = HandStatus.WARN if warn else HandStatus.TRACKING
        return HandEstimate(side, t_ns, st, warn, T, T[:3, 3].copy(), T[:3, :3].copy(), tp.conf, "tracker", age), ("DEGRADED" if warn else "TRACKING")


def pivot_calibration(T_list) -> tuple[np.ndarray, np.ndarray, float]:
    """Tracker body poses T_TW_TB while the UMI jaw tip rests on ONE fixed point and the UMI is rotated about it.
    Solve R_i t_tcp + p_i = p_pivot (least squares) -> (t_tcp in tracker body, p_pivot in tracker world, rms residual m)."""
    A, b = [], []
    for T in T_list: A.append(np.hstack([T[:3, :3], -np.eye(3)])); b.append(-T[:3, 3])
    A, b = np.vstack(A), np.concatenate(b); x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = A @ x - b; return x[:3], x[3:], float(np.sqrt(np.mean(res.reshape(-1, 3) ** 2) * 3))


def table_frame_from_points(P0, P1, P2, extra=()) -> tuple[np.ndarray, float]:
    """W from jaw-tip touches on the (level) table, in tracker-world coords: P0 = origin, P1 = a point FORWARD of it (robot +x
    direction), P2 = a point to its LEFT (+y side); extra = more table points for the plane fit. -> (T_W_TW, plane rms m).
    z = table normal (sign: P0->P1 x P0->P2), x = P0->P1 projected into the plane, y = z x x."""
    P = np.vstack([P0, P1, P2, *extra]).astype(np.float64); c = P.mean(0)
    n = np.linalg.svd(P - c)[2][2]; up = np.cross(P1 - P0, P2 - P0)
    if np.linalg.norm(up) < 1e-4: raise ValueError("points are colinear")
    if n @ up < 0: n = -n
    x = (P1 - P0) - ((P1 - P0) @ n) * n; x /= np.linalg.norm(x); y = np.cross(n, x)
    R_TW_W = np.column_stack([x, y, n]); T_TW_W = make_T(R_TW_W, P0)
    rms = float(np.sqrt(np.mean(((P - c) @ n) ** 2)))
    return np.linalg.inv(T_TW_W), rms


def R_TB_TCP_from_gravity(R_W_TB_A: np.ndarray, R_W_TB_B: np.ndarray) -> tuple[np.ndarray, float]:
    """Two still poses (same convention as calib_right imu-mount --tool umi):
    A  UMI level, jaws pointing forward, jaws side by side  -> gravity = +x_TCP;   B  jaws pointing straight down -> gravity = +z_TCP.
    R_W_TB_* = tracker body orientation in W (after T_W_TW). -> (R_TB_TCP, angle between the two gravity directions, deg)."""
    g = np.array([0, 0, -1.0]); dA = R_W_TB_A.T @ g; dB = R_W_TB_B.T @ g
    ang = float(np.degrees(np.arccos(np.clip(dA @ dB, -1, 1))))
    x = dA; z = dB - (dB @ x) * x; z /= np.linalg.norm(z); y = np.cross(z, x)
    return np.column_stack([x, y, z]), ang


def rotation_spread_deg(T_list) -> float:
    """Largest rotation angle between any pose and the mean-ish first pose (pivot excitation check)."""
    R0 = T_list[0][:3, :3]
    return float(max(np.degrees(np.linalg.norm(Rotation.from_matrix(R0.T @ T[:3, :3]).as_rotvec())) for T in T_list))
