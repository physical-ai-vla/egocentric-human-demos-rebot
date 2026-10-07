"""Aero Hand Open embodiment adapter (Aero-v1).

Canonical human hand action -> Aero Hand Open actuator commands.

Design (mirrors the arm-side split we already have: canonical human action first,
embodiment adapter second):

    HaWoR 21-joint skeleton
      -> HUMAN SEMANTIC 7D  (thumb_abd, thumb_flex, thumb_curl, index/middle/ring/pinky flex)
      -> Aero compact 7 joints (deg)   [SDK "compact joint representation"]
      -> Aero 16 joints (deg)          [underactuated expansion]
      -> Aero 7 actuator commands (deg of motor rotation)

The human semantic 7D is *not* Aero-specific: it is a description of the human
hand state. Aero happens to have exactly the same 7 controllable dimensions, but
a different 5-finger hand would consume the same canonical features.

Why the finger weights are not free parameters
----------------------------------------------
Each Aero finger has ONE tendon driving MCP+PIP+DIP. The real tendon model is

    tendon_travel = c_mcp*q_mcp + c_pip*q_pip + c_dip*q_dip        (mm, q in rad)

so the tendon-equivalent single angle for a human finger is the
coefficient-weighted mean of the human MCP/PIP/DIP flexions. That makes the
"w1,w2,w3" of a naive synergy mapping hardware-derived rather than hand-tuned.
Coefficients are taken from the official SDK (joints_to_actuations.py).

Units note (verified against the SDK's own limit table, see tests): the SDK
docstrings say "degrees", but the coefficients are mm/radian and the published
actuation limits only reproduce if joint angles enter the formulas in RADIANS
and the resulting motor rotation is then reported in degrees. We do that.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Aero hardware constants (from TetherIA/aero-hand-open, Apache-2.0)
# --------------------------------------------------------------------------- #

MOTOR_PULLEY_RADIUS = 9.000  # mm

# mm/radian
FINGER_COEFFS = (12.4912, 7.3211, 9.0000)  # mcp_flex, pip, dip
THUMB_FLEX_CMC_ABD = 2.5000
THUMB_FLEX_CMC_FLEX = 12.4931
THUMB_IP_CMC_ABD = 2.5000
THUMB_IP_CMC_FLEX = 2.5000
THUMB_IP_MCP = 9.4372
THUMB_IP_IP = 12.5000

# Compact joint representation: 3 thumb values + 1 per finger.
COMPACT_NAMES = (
    "thumb_abd",
    "thumb_flex",
    "thumb_curl",  # thumb MCP/IP combined
    "index_flex",
    "middle_flex",
    "ring_flex",
    "pinky_flex",
)
# Aero joint limits (deg) for the compact dims, from AeroHandConstants.
COMPACT_UPPER_DEG = (100.0, 55.0, 90.0, 90.0, 90.0, 90.0, 90.0)
COMPACT_LOWER_DEG = (0.0,) * 7

ACTUATION_NAMES = (
    "thumb_cmc_abd_act",
    "thumb_cmc_flex_act",
    "thumb_tendon_act",
    "index_tendon_act",
    "middle_tendon_act",
    "ring_tendon_act",
    "pinky_tendon_act",
)
ACTUATION_LOWER_DEG = (0.0, 0.0, -15.2789, 0.0, 0.0, 0.0, 0.0)
ACTUATION_UPPER_DEG = (100.0, 104.1250, 247.1500, 288.1603, 288.1603, 288.1603, 288.1603)

AERO_JOINT_NAMES = (
    "thumb_cmc_abd",
    "thumb_cmc_flex",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp_flex",
    "index_pip",
    "index_dip",
    "middle_mcp_flex",
    "middle_pip",
    "middle_dip",
    "ring_mcp_flex",
    "ring_pip",
    "ring_dip",
    "pinky_mcp_flex",
    "pinky_pip",
    "pinky_dip",
)

# --------------------------------------------------------------------------- #
# HaWoR 21-joint skeleton layout (OpenPose ordering, verified against the
# separately exported {side}_thumb_tip / {side}_index_tip: idx 4 and 8)
# --------------------------------------------------------------------------- #

WRIST = 0
FINGER_IDX = {
    "thumb": (1, 2, 3, 4),  # CMC, MCP, IP, TIP
    "index": (5, 6, 7, 8),  # MCP, PIP, DIP, TIP
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Human joint-angle ranges (deg) used to map human anatomy onto Aero's range.
# Fixed anatomical constants -- deliberately NOT per-episode percentiles (same
# policy as the global aperture calibration).
HUMAN_RANGE_DEG = {
    "thumb_abd": (20.0, 70.0),
    "thumb_flex": (0.0, 45.0),
    "thumb_curl": (0.0, 70.0),
    "index_flex": (5.0, 85.0),
    "middle_flex": (5.0, 85.0),
    "ring_flex": (5.0, 85.0),
    "pinky_flex": (5.0, 85.0),
}


# --------------------------------------------------------------------------- #
# Human skeleton -> human semantic 7D
# --------------------------------------------------------------------------- #


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b of the a-b-c chain, per frame. Returns radians."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba, axis=-1)
    nbc = np.linalg.norm(bc, axis=-1)
    denom = np.maximum(nba * nbc, 1e-9)
    cos = np.clip(np.einsum("...i,...i->...", ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cos)


def _flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Flexion angle at b (0 = straight chain)."""
    return np.pi - _angle_at(a, b, c)


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    nu = np.maximum(np.linalg.norm(u, axis=-1), 1e-9)
    nv = np.maximum(np.linalg.norm(v, axis=-1), 1e-9)
    cos = np.clip(np.einsum("...i,...i->...", u, v) / (nu * nv), -1.0, 1.0)
    return np.arccos(cos)


def human_joint_angles(joints: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame human hand joint angles (radians) from a (T,21,3) skeleton.

    Returns the raw anatomical angles, before any Aero-specific mapping:
      thumb_abd, thumb_cmc_flex, thumb_mcp, thumb_ip,
      {finger}_mcp, {finger}_pip, {finger}_dip
    """
    J = np.asarray(joints, dtype=np.float64)
    if J.ndim != 3 or J.shape[1] < 21 or J.shape[2] != 3:
        raise ValueError(f"expected (T,21,3) skeleton, got {J.shape}")
    w = J[:, WRIST]
    out: dict[str, np.ndarray] = {}

    t_cmc, t_mcp, t_ip, t_tip = (J[:, i] for i in FINGER_IDX["thumb"])
    i_mcp = J[:, FINGER_IDX["index"][0]]
    # Thumb abduction: 3D angle between the thumb metacarpal (CMC->MCP) and the
    # index metacarpal (wrist->index MCP). View-independent, unlike the
    # official 2D-projection variant.
    out["thumb_abd"] = _angle_between(t_mcp - t_cmc, i_mcp - w)
    out["thumb_cmc_flex"] = _flexion(w, t_cmc, t_mcp)
    out["thumb_mcp"] = _flexion(t_cmc, t_mcp, t_ip)
    out["thumb_ip"] = _flexion(t_mcp, t_ip, t_tip)

    for finger in ("index", "middle", "ring", "pinky"):
        mcp, pip, dip, tip = (J[:, i] for i in FINGER_IDX[finger])
        out[f"{finger}_mcp"] = _flexion(w, mcp, pip)
        out[f"{finger}_pip"] = _flexion(mcp, pip, dip)
        out[f"{finger}_dip"] = _flexion(pip, dip, tip)
    return out


def human_semantic7(joints: np.ndarray) -> np.ndarray:
    """(T,21,3) skeleton -> (T,7) human semantic hand state, in RADIANS.

    Columns follow COMPACT_NAMES. Finger/thumb-curl dims are the tendon-weighted
    means of the underlying human joint flexions, so they are the single angle
    an Aero-style single-tendon finger would need to match the human's tendon
    travel.
    """
    a = human_joint_angles(joints)
    c_mcp, c_pip, c_dip = FINGER_COEFFS
    c_sum = c_mcp + c_pip + c_dip
    t_sum = THUMB_IP_MCP + THUMB_IP_IP

    cols = [
        a["thumb_abd"],
        a["thumb_cmc_flex"],
        (THUMB_IP_MCP * a["thumb_mcp"] + THUMB_IP_IP * a["thumb_ip"]) / t_sum,
    ]
    for finger in ("index", "middle", "ring", "pinky"):
        cols.append(
            (
                c_mcp * a[f"{finger}_mcp"]
                + c_pip * a[f"{finger}_pip"]
                + c_dip * a[f"{finger}_dip"]
            )
            / c_sum
        )
    return np.stack(cols, axis=1)


def semantic7_normalized(sem7_rad: np.ndarray) -> np.ndarray:
    """Human semantic 7D (rad) -> [0,1] per dim using fixed anatomical ranges.

    0 = open/neutral, 1 = fully flexed/abducted. This is the canonical,
    embodiment-independent hand action vector.
    """
    s = np.asarray(sem7_rad, dtype=np.float64)
    lo = np.array([HUMAN_RANGE_DEG[n][0] for n in COMPACT_NAMES])
    hi = np.array([HUMAN_RANGE_DEG[n][1] for n in COMPACT_NAMES])
    deg = np.degrees(s)
    return np.clip((deg - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Canonical [0,1] -> Aero joints / actuations
# --------------------------------------------------------------------------- #


def normalized_to_compact_deg(u: np.ndarray) -> np.ndarray:
    """Canonical [0,1] hand action -> Aero compact 7 joints (deg)."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    lo = np.array(COMPACT_LOWER_DEG)
    hi = np.array(COMPACT_UPPER_DEG)
    return lo + u * (hi - lo)


def compact_to_full16_deg(compact_deg: np.ndarray) -> np.ndarray:
    """Aero compact 7 -> 16 joints (deg), using the underactuated expansion.

    Thumb: (abd, flex, curl, curl) -- MCP and IP share the tendon.
    Finger: (q, q, q)              -- MCP/PIP/DIP share the tendon.
    (Same assumption as the SDK's ActuationsToJointsModelCompact.)
    """
    c = np.atleast_2d(np.asarray(compact_deg, dtype=np.float64))
    out = np.zeros((c.shape[0], 16))
    out[:, 0] = c[:, 0]  # thumb_cmc_abd
    out[:, 1] = c[:, 1]  # thumb_cmc_flex
    out[:, 2] = c[:, 2]  # thumb_mcp
    out[:, 3] = c[:, 2]  # thumb_ip
    for k in range(4):
        out[:, 4 + 3 * k : 7 + 3 * k] = c[:, 3 + k][:, None]
    return out.reshape(np.asarray(compact_deg).shape[:-1] + (16,))


def full16_to_actuations_deg(joints_deg: np.ndarray) -> np.ndarray:
    """Aero 16 joints (deg) -> 7 actuator commands (deg of motor rotation).

    Angles enter the tendon model in radians; the motor rotation that comes out
    is reported in degrees (this exactly reproduces the SDK's published
    actuation limit table -- see tests).
    """
    q = np.atleast_2d(np.asarray(joints_deg, dtype=np.float64))
    r = np.radians(q)
    abd, flex = r[:, 0], r[:, 1]
    mcp, ip = r[:, 2], r[:, 3]

    act = np.zeros((q.shape[0], 7))
    act[:, 0] = np.degrees(abd)  # direct mapping
    act[:, 1] = np.degrees(
        (THUMB_FLEX_CMC_ABD * abd + THUMB_FLEX_CMC_FLEX * flex) / MOTOR_PULLEY_RADIUS
    )
    act[:, 2] = np.degrees(
        (
            THUMB_IP_CMC_ABD * abd
            - THUMB_IP_CMC_FLEX * flex
            + THUMB_IP_MCP * mcp
            + THUMB_IP_IP * ip
        )
        / MOTOR_PULLEY_RADIUS
    )
    c_mcp, c_pip, c_dip = FINGER_COEFFS
    for k in range(4):
        f = r[:, 4 + 3 * k : 7 + 3 * k]
        act[:, 3 + k] = np.degrees(
            (c_mcp * f[:, 0] + c_pip * f[:, 1] + c_dip * f[:, 2]) / MOTOR_PULLEY_RADIUS
        )
    act = np.clip(act, np.array(ACTUATION_LOWER_DEG), np.array(ACTUATION_UPPER_DEG))
    return act.reshape(np.asarray(joints_deg).shape[:-1] + (7,))


def canonical_to_aero(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical [0,1] hand action -> (compact7 deg, full16 deg, actuations deg)."""
    compact = normalized_to_compact_deg(u)
    full16 = compact_to_full16_deg(compact)
    act = full16_to_actuations_deg(full16)
    return compact, full16, act


# --------------------------------------------------------------------------- #
# Aero-v0: grasp scalar -> pinch preset (no full skeleton needed)
# --------------------------------------------------------------------------- #

# Aero-v0 uses the arm-side grasp scalar only: thumb+index pinch closes with the
# grasp value, middle/ring/pinky stay relaxed. Lets the existing 20/40-episode
# wrist+grasp dataset drive the Aero hand with no re-recording.
PINCH_OPEN = np.array([0.55, 0.10, 0.05, 0.05, 0.15, 0.15, 0.15])
PINCH_CLOSED = np.array([0.35, 0.70, 0.85, 0.80, 0.25, 0.25, 0.25])


def grasp_to_canonical(grasp: np.ndarray) -> np.ndarray:
    """Aero-v0: grasp scalar in [0,1] -> canonical 7D pinch synergy."""
    g = np.clip(np.asarray(grasp, dtype=np.float64), 0.0, 1.0)[..., None]
    return PINCH_OPEN + g * (PINCH_CLOSED - PINCH_OPEN)


# --------------------------------------------------------------------------- #
# Minimal URDF FK (revolute + fixed only) for visualisation / QC
# --------------------------------------------------------------------------- #


def _rpy_to_mat(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass
class _Joint:
    name: str
    jtype: str
    parent: str
    child: str
    xyz: np.ndarray
    rot: np.ndarray
    axis: np.ndarray


class AeroHandFK:
    """Forward kinematics of the Aero Hand Open from its URDF kinematic tree.

    Only link origins are used (meshes are not needed), which is enough to draw
    the hand as a skeleton and to read fingertip positions.
    """

    def __init__(self, urdf_path: str | Path, side: str = "right"):
        root = ET.parse(str(urdf_path)).getroot()
        self.side = side
        self.joints: list[_Joint] = []
        for j in root.findall("joint"):
            o = j.find("origin")
            a = j.find("axis")
            xyz = np.array(
                [float(v) for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
            )
            rpy = tuple(
                float(v) for v in (o.get("rpy", "0 0 0").split())
            ) if o is not None else (0.0, 0.0, 0.0)
            axis = np.array(
                [float(v) for v in a.get("xyz", "1 0 0").split()] if a is not None else [1, 0, 0]
            )
            self.joints.append(
                _Joint(
                    name=j.get("name", ""),
                    jtype=j.get("type", "fixed"),
                    parent=j.find("parent").get("link", ""),
                    child=j.find("child").get("link", ""),
                    xyz=xyz,
                    rot=_rpy_to_mat(rpy),  # type: ignore[arg-type]
                    axis=axis / max(float(np.linalg.norm(axis)), 1e-9),
                )
            )
        self.base = f"{side}_base_link"
        self.actuated = [f"{side}_{n}" for n in AERO_JOINT_NAMES]

    def link_positions(self, joints_deg: np.ndarray) -> dict[str, np.ndarray]:
        """16 joint angles (deg) -> {link_name: 3D origin} in base_link frame."""
        q = dict(zip(self.actuated, np.radians(np.asarray(joints_deg, dtype=np.float64))))
        frames = {self.base: (np.zeros(3), np.eye(3))}
        # URDF order is already parent-before-child here; loop until closed.
        for _ in range(4):
            for j in self.joints:
                if j.parent in frames and j.child not in frames:
                    p, R = frames[j.parent]
                    Rj = R @ j.rot
                    if j.jtype == "revolute":
                        th = float(q.get(j.name, 0.0))
                        k = j.axis
                        K = np.array(
                            [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
                        )
                        Rax = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
                        Rj = Rj @ Rax
                    frames[j.child] = (p + R @ j.xyz, Rj)
        return {k: v[0] for k, v in frames.items()}

    def chains(self) -> list[list[str]]:
        """Link chains for drawing: base -> ... -> tip, one per finger."""
        s = self.side
        return [
            [f"{s}_base_link", f"{s}_t_link", f"{s}_thumb_mcp_link", f"{s}_thumb_proximal_link", f"{s}_thumb_distal_link", f"{s}_thumb_tip_link"],
            *[
                [f"{s}_base_link", f"{s}_{f}_proximal_link", f"{s}_{f}_middle_link", f"{s}_{f}_distal_link", f"{s}_{f}_tip_link"]
                for f in ("index", "middle", "ring", "pinky")
            ],
        ]

    def fingertips(self, joints_deg: np.ndarray) -> dict[str, np.ndarray]:
        pos = self.link_positions(joints_deg)
        return {f: pos[f"{self.side}_{f}_tip_link"] for f in ("thumb", "index", "middle", "ring", "pinky")}


def thumb_gap_servo(
    u: np.ndarray,
    aperture_m: np.ndarray,
    fk: "AeroHandFK",
    n_grid: int = 25,
) -> np.ndarray:
    """Aero-v1 thumb correction: replace the joint-angle-copied thumb_curl with
    the value that makes Aero's thumb-index FINGERTIP GAP match the human's
    metric aperture (1-DoF search per frame).

    Why: Aero's thumb curl plane differs from human thumb opposition, so a
    joint-angle copy of the thumb is directionally wrong for pinching (curl can
    OPEN the Aero gap). The four fingers stay semantic; only thumb_curl becomes
    task-space. Frames with non-finite aperture keep the semantic value.

    Returns a copy of u with column 2 (thumb_curl) replaced where possible.
    """
    u = np.array(u, dtype=np.float64, copy=True)
    ap = np.asarray(aperture_m, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n_grid)
    # cache: gap depends on (thumb_abd, thumb_flex, thumb_curl, index_flex);
    # quantize the other dims so repeated poses reuse FK results.
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for t in range(len(u)):
        if not (np.isfinite(u[t]).all() and np.isfinite(ap[t])):
            continue
        key = (int(u[t, 0] * 40), int(u[t, 1] * 40), int(u[t, 3] * 40))
        gaps = cache.get(key)
        if gaps is None:
            gaps = np.empty(n_grid)
            for k, c in enumerate(grid):
                v = u[t].copy()
                v[2] = c
                _, q16, _ = canonical_to_aero(v)
                tips = fk.fingertips(q16)
                gaps[k] = float(np.linalg.norm(tips["thumb"] - tips["index"]))
            cache[key] = gaps
        u[t, 2] = grid[int(np.argmin(np.abs(gaps - ap[t])))]
    return u


def default_urdf(side: str) -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "aero_hand_open" / f"aero_hand_open_{side}.urdf"
