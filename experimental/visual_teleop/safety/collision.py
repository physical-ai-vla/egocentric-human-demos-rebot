"""Inter-arm safety (2026-10-14, spec 23). Checked on the PAIR of candidate commands; any violation => BOTH arms HOLD (never
modify one arm alone -- that would silently break bimanual coordination).

Geometry (RB frame, metres): TCP points and a wrist point 0.08 m behind each TCP along -z of the TCP frame. pyroki rebot_b601
IK/FK TCP (urdf {side}_tcp = gripper_link + 0.07 x, rpy 0 90 0): z = tool/approach (wrist -> fingertips), y = jaw opening axis;
measured 2026-10-14: j6 rotates about body z, the j5 axis lies 0.25 m behind the TCP along -z. This is a coarse proxy;
self-collision with full link geometry is not implemented."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

WRIST_BACK_M = 0.08


def _seg_dist(p0, p1, q0, q1) -> float:
    """Minimum distance between segments p0p1 and q0q1."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c, d, e = u @ u, u @ v, v @ v, u @ w, v @ w
    den = a * c - b * b
    s = 0.0 if den < 1e-12 else np.clip((b * e - c * d) / den, 0, 1)
    t = np.clip((b * s + e) / c, 0, 1) if c > 1e-12 else 0.0
    s = np.clip((b * t - d) / a, 0, 1) if a > 1e-12 else 0.0
    return float(np.linalg.norm((p0 + s * u) - (q0 + t * v)))


@dataclass
class InterArmCheck:
    min_tcp_separation_m: float = 0.10
    min_wrist_separation_m: float = 0.12
    forbid_crossing: bool = True
    crossing_margin_m: float = 0.02

    @classmethod
    def from_cfg(cls, d: dict) -> "InterArmCheck": return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def check(self, T_left: np.ndarray, T_right: np.ndarray) -> list[str]:
        pl, pr = T_left[:3, 3], T_right[:3, 3]
        wl, wr = pl - WRIST_BACK_M * T_left[:3, 2], pr - WRIST_BACK_M * T_right[:3, 2]
        out = []
        d_tcp = float(np.linalg.norm(pl - pr))
        if d_tcp < self.min_tcp_separation_m: out.append(f"tcp_separation:{d_tcp*1e3:.0f}mm")
        d_seg = _seg_dist(wl, pl, wr, pr)
        if d_seg < self.min_wrist_separation_m: out.append(f"wrist_separation:{d_seg*1e3:.0f}mm")
        if self.forbid_crossing and not (pl[1] > pr[1] + self.crossing_margin_m): out.append(f"crossing:yL={pl[1]:.3f},yR={pr[1]:.3f}")
        return out
