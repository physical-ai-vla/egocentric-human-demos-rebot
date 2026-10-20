"""UMI jaw-tip position from tape + tag + depth (2026-10-15, user: "tag+orangetape+depth 세개 같이 쓰자 -- 빨간게 옷을 짚으면 없어져").

The orange tape sits ON the jaw tips (the point we control); the AprilTag sits on the SIDE of the body. They are different points,
so the tag is only usable after the rigid offset r_H (tag centre -> tape tip, in the UMI frame H) is known. It is LEARNED, not
assumed: while both are visible and the IMU orientation is valid, r_H samples = R_W_H^T (p_tape - p_tag); after `learn_n` the
median is frozen (reset at every INIT, the IMU yaw changes then).
    tape visible                 -> p = p_tape                       source "tape"
    tape hidden, tag + r_H known -> p = p_tag + R_W_H r_H            source "tag"   (e.g. jaws closed on cloth)
    otherwise                    -> no observation (HandFusion holds, then HOLD)"""
from __future__ import annotations
import numpy as np


class UmiPositionFusion:
    def __init__(self, learn_n: int = 30, max_offset_m: float = 0.25, tag_conf: float = 0.8) -> None:
        self.learn_n, self.max_offset, self.tag_conf = int(learn_n), float(max_offset_m), float(tag_conf)
        self.r_H: np.ndarray | None = None; self._samples: list = []; self.last_source = "none"

    def reset(self) -> None: self.r_H = None; self._samples = []

    def update(self, tape, tag, R_W_H) -> tuple[np.ndarray | None, float, str]:
        """tape: TapeObservation | None, tag: TagObservation | None, R_W_H: IMU-predicted UMI orientation (None before INIT)."""
        if tape is not None and tag is not None and R_W_H is not None and self.r_H is None:
            r = np.asarray(R_W_H).T @ (tape.p_W - tag.T_W_tag[:3, 3])
            if np.linalg.norm(r) <= self.max_offset: self._samples.append(r)
            if len(self._samples) >= self.learn_n: self.r_H = np.median(np.asarray(self._samples), axis=0); self._samples = []
        if tape is not None: self.last_source = "tape"; return tape.p_W, tape.conf, "tape"
        if tag is not None and self.r_H is not None and R_W_H is not None:
            self.last_source = "tag"; return tag.T_W_tag[:3, 3] + np.asarray(R_W_H) @ self.r_H, self.tag_conf, "tag"
        self.last_source = "none"; return None, 0.0, "none"
