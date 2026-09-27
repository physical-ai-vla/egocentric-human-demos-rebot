#!/usr/bin/env python3
"""[2026-09-25] E_collision v1 (contract §23): inter-arm clearance from URDF link NAMES (v0 used joint-index lists as link
indices -> it measured left-arm / fixed-mount distances and is void). Exclusion rule by link semantics, not by distance:
`base` and `*_base_link` are fixed pedestal mounts -> excluded. Per-arm chain of segments:
  link1-link2, link2-link3, link3-link4, link4-link5, link5-link6, link6-gripper_link, gripper_link-tcp,
  gripper_link-gripper_left, gripper_link-gripper_right                         (9 segments per arm, 81 cross pairs)
Exact segment-segment distance (closest points, clamped). Returns (d_min, argmin pair by link names)."""
import numpy as np
CHAIN = [("link1", "link2"), ("link2", "link3"), ("link3", "link4"), ("link4", "link5"), ("link5", "link6"), ("link6", "gripper_link"),
         ("gripper_link", "tcp"), ("gripper_link", "gripper_left"), ("gripper_link", "gripper_right")]


def segseg(p0, p1, q0, q1):
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0; a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12: return float(np.linalg.norm(r))
    if a <= 1e-12: sc, tc = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= 1e-12: sc, tc = np.clip(-c / a, 0, 1), 0.0
        else:
            b = d1 @ d2; den = a * e - b * b; sc = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            tc = (b * sc + f) / e
            if tc < 0: tc, sc = 0.0, np.clip(-c / a, 0, 1)
            elif tc > 1: tc, sc = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p0 + sc * d1) - (q0 + tc * d2)))


class Clearance:
    def __init__(self, kin):
        self.kin, self.s = kin, kin.solver; names = list(self.s.robot.links.names); self.ix = {n: i for i, n in enumerate(names)}
        self.links = sorted({f"{side}_{n}" for side in ("left", "right") for pair in CHAIN for n in pair}); self.li = [self.ix[n] for n in self.links]

    def points(self, q12):
        q = self.kin._q(self.kin._svc(q12)).astype(np.float32); P = np.asarray(self.s.link_positions(q, self.li)); return dict(zip(self.links, P))

    def dmin(self, q12):
        P = self.points(q12); best = (np.inf, None)
        for a, b in CHAIN:
            for c, d in CHAIN:
                v = segseg(P[f"left_{a}"], P[f"left_{b}"], P[f"right_{c}"], P[f"right_{d}"])
                if v < best[0]: best = (v, f"L:{a}-{b}|R:{c}-{d}")
        return best
