"""Pinch state -> smooth gripper target (2026-10-14, spec 9).

Canonical g: 0 = CLOSED, 1 = OPEN.  pinch TRUE -> 0, pinch FALSE -> 1.  Never a binary jump to the servo:
    g_rate  = rate-limited move toward the binary target (rate_limit_per_s, g units/s)
    g       = low-pass of g_rate (alpha)
Robot jaw command = 45 * g via ego_teleop.robot.rebot_client.gripper_cmd_from_g (mapping A, video-verified polarity:
raw 0 = closed, -270 = open). TRANSACTIONAL like the arm: propose() is pure, commit() after the command was accepted."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ego_teleop.robot.rebot_client import gripper_cmd_from_g


def target_from_pinch(pinch: bool) -> float: return 0.0 if pinch else 1.0


@dataclass
class GripperMapper:
    rate_limit_per_s: float = 2.0
    lowpass_alpha: float = 0.3
    g_rate: float | None = None         # executed state (rate-limited)
    g: float | None = None              # executed state (low-passed) = what was sent

    @classmethod
    def from_cfg(cls, g: dict) -> "GripperMapper": return cls(g["rate_limit_per_s"], g["lowpass_alpha"])

    def reset(self, g_now: float) -> None: self.g_rate = self.g = float(np.clip(g_now, 0.0, 1.0))

    def propose(self, g_target: float | None, dt_s: float) -> tuple[float, float]:
        """-> (g_rate_next, g_next). g_target None = hold (no new intent)."""
        if self.g is None: raise RuntimeError("GripperMapper.reset(g_observed) first")
        if g_target is None or not np.isfinite(g_target): return self.g_rate, self.g
        step = self.rate_limit_per_s * max(float(dt_s), 0.0)
        gr = self.g_rate + float(np.clip(float(g_target) - self.g_rate, -step, step))
        return gr, float(np.clip(self.g + self.lowpass_alpha * (gr - self.g), 0.0, 1.0))

    def commit(self, proposal: tuple[float, float]) -> None: self.g_rate, self.g = float(proposal[0]), float(proposal[1])

    @staticmethod
    def jaw_command(g: float) -> float: return gripper_cmd_from_g(g)
