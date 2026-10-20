"""RGB-D residual monitor for the 6DoF tracker source (2026-10-15, FastUMI-style: tracker primary, RGB-D secondary).

The tracker gives the UMI jaw-tip TCP in its calibrated world (T_W_TW, table-touch frame); the markerless keypoint gives the SAME
physical jaw tip from the fixed Orbbec, in the camera's gravity-levelled W. Both worlds are z-up but differ by a yaw + offset, so
the monitor first LEARNS that 4-DoF map from paired samples (needs >= min_spread_m of motion so yaw is observable), freezes it,
then reports the residual |p_rgbd - A(p_tracker)| per frame:

    learning   not enough spread yet (no verdict)
    ok         residual <= warn_m
    warn       residual >  warn_m              -> caller marks DEGRADED (tracker drift or RGB-D outlier, worth a look)
    hold       residual >  hold_m for >= hold_s -> caller HOLDs (gross disagreement: tracker drifted / slipped on the UMI)
    no_rgbd    no fresh RGB-D sample (occlusion) -> no verdict, the tracker carries on alone

A single RGB-D outlier never HOLDs (hold needs hold_s of persistent disagreement)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def fit_yaw_translation(P_src: np.ndarray, P_dst: np.ndarray) -> tuple[np.ndarray, float]:
    """dst ~ Rz(yaw) src + t  (least squares; xy by 2-D Kabsch, z by mean offset). -> (4x4, rms m)."""
    a, b = np.asarray(P_src, np.float64), np.asarray(P_dst, np.float64)
    ca, cb = a.mean(0), b.mean(0); A, B = a[:, :2] - ca[:2], b[:, :2] - cb[:2]
    U, _, Vt = np.linalg.svd(A.T @ B); d = np.sign(np.linalg.det(Vt.T @ U.T)); R2 = Vt.T @ np.diag([1, d]) @ U.T
    T = np.eye(4); T[:2, :2] = R2; T[:3, 3] = cb - T[:3, :3] @ ca
    res = b - (a @ T[:3, :3].T + T[:3, 3]); return T, float(np.sqrt(np.mean(np.sum(res ** 2, 1))))


@dataclass
class ResidualMonitor:
    learn_n: int = 60
    min_spread_m: float = 0.10
    warn_m: float = 0.03
    hold_m: float = 0.08
    hold_s: float = 0.5
    max_pair_dt_ms: float = 60.0
    T_rgbd_trk: np.ndarray | None = None
    fit_rms_m: float | None = None
    _pairs: list = field(default_factory=list)
    _bad_since: int | None = None
    last: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, m: dict) -> "ResidualMonitor":
        return cls(**{k: m[k] for k in ("learn_n", "min_spread_m", "warn_m", "hold_m", "hold_s", "max_pair_dt_ms") if k in m})

    def reset(self) -> None:
        self.T_rgbd_trk = None; self.fit_rms_m = None; self._pairs = []; self._bad_since = None

    def update(self, t_ns: int, p_trk: np.ndarray | None, rgbd: tuple | None) -> dict:
        """p_trk: tracker TCP position (tracker W) or None; rgbd: (t_ns, p_W) of the latest keypoint observation or None."""
        if p_trk is None or rgbd is None or abs(t_ns - rgbd[0]) / 1e6 > self.max_pair_dt_ms:
            self._bad_since = None; self.last = dict(state="no_rgbd" if self.T_rgbd_trk is not None else "learning"); return self.last
        p_trk, p_rgbd = np.asarray(p_trk, np.float64), np.asarray(rgbd[1], np.float64)
        if self.T_rgbd_trk is None:
            if not self._pairs or np.linalg.norm(p_trk - self._pairs[-1][0]) > 0.005: self._pairs.append((p_trk, p_rgbd))   # spread, not dwell
            P = np.array([x[0] for x in self._pairs]); spread = float(np.ptp(P[:, :2], 0).max()) if len(P) > 1 else 0.0
            if len(self._pairs) >= self.learn_n and spread >= self.min_spread_m:
                T, rms = fit_yaw_translation(P, np.array([x[1] for x in self._pairs]))
                self.T_rgbd_trk, self.fit_rms_m = T, rms
            self.last = dict(state="learning", n=len(self._pairs), spread_m=round(spread, 3)); return self.last
        r = float(np.linalg.norm(p_rgbd - (self.T_rgbd_trk[:3, :3] @ p_trk + self.T_rgbd_trk[:3, 3])))
        if r > self.hold_m: self._bad_since = self._bad_since if self._bad_since is not None else t_ns
        else: self._bad_since = None
        state = "hold" if self._bad_since is not None and (t_ns - self._bad_since) / 1e9 >= self.hold_s else ("warn" if r > self.warn_m else "ok")
        self.last = dict(state=state, residual_m=round(r, 4), fit_rms_m=round(self.fit_rms_m, 4)); return self.last
