"""Official TetherIA Aero conventions, ported out of ROS (see docs/ego_teleop/AERO_A0_AUDIT.md).

This module is a *transcription*, not a redesign: the palm-local transform, the 21->25 keypoint expansion and the
robot-side joint/actuation orders are exactly what `webcam_mocap.py`, `dex_retargeting_node.py`, `HandMocap.msg`,
`JointControl.msg` and `aero_open_sdk.AeroHandConstants` contain. `tests/teleop/test_aero_official_conventions.py`
pins every constant here against the checked-out upstream source, so an upstream change breaks a test instead of
silently changing what the hand does.

Both hand-pose providers (monocular baseline and RGB-D) emit landmarks in THIS representation, which is what makes the
B0/B1 comparison of spec section 30 an honest one-variable experiment."""
from __future__ import annotations
import numpy as np

# ---- landmark orders ----------------------------------------------------------------------------------------
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP = 0, 4, 5, 8, 9, 12, 13, 16, 17, 20
N_LANDMARKS = 21

# HandMocap.msg keypoint order (25) and the MediaPipe index feeding each slot. MediaPipe has no finger CMC landmark,
# so the official node repeats the wrist there.
MOCAP25_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_cmc", "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_cmc", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_cmc", "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_cmc", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
MEDIAPIPE_TO_MOCAP25 = (0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0, 13, 14, 15, 16, 0, 17, 18, 19, 20)
MOCAP25_TIPS = (4, 9, 14, 19, 24)          # thumb, index, middle, ring, pinky — the DexPilot task links
MP_TIPS = (4, 8, 12, 16, 20)               # the same fingertips in MediaPipe-21 indexing
# MediaPipe HAND_CONNECTIONS, for drawing only.
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))

# ---- robot side (AeroHandConstants / JointControl.msg) ------------------------------------------------------
AERO_JOINT_NAMES = (
    "thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_pip", "index_dip",
    "middle_mcp_flex", "middle_pip", "middle_dip",
    "ring_mcp_flex", "ring_pip", "ring_dip",
    "pinky_mcp_flex", "pinky_pip", "pinky_dip",
)
AERO_JOINT_LOWER_DEG = (0.0,) * 16
AERO_JOINT_UPPER_DEG = (100.0, 55.0, 90.0, 90.0) + (90.0,) * 12


def expand_to_mocap25(l21: np.ndarray) -> np.ndarray:
    """(21,3) MediaPipe landmarks -> (25,3) HandMocap keypoints (wrist duplicated into every finger CMC slot)."""
    J = np.asarray(l21, np.float64).reshape(N_LANDMARKS, 3)
    return J[list(MEDIAPIPE_TO_MOCAP25)]


def palm_frame(l21: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (R, t): palm basis (columns = x,y,z axes) and wrist origin, per `WebcamMocap.process_landmarks`.

    x: index MCP - ring MCP (negated for the left hand), z: middle MCP - wrist, y = z x x, x re-orthogonalised.
    Raises ValueError on a degenerate hand (collinear/zero axes) rather than emitting a silently wrong frame."""
    J = np.asarray(l21, np.float64).reshape(N_LANDMARKS, 3)
    if side not in ("left", "right"): raise ValueError(f"side must be left|right, got {side!r}")

    def _n(v, what):
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-9: raise ValueError(f"degenerate palm frame: {what} has length {n}")
        return v / n

    x = _n(J[INDEX_MCP] - J[RING_MCP], "index-ring MCP axis")
    if side == "left": x = -x
    z = _n(J[MIDDLE_MCP] - J[WRIST], "wrist-middle MCP axis")
    y = _n(np.cross(z, x), "y = z x x")
    x = _n(np.cross(y, z), "re-orthogonalised x")
    return np.array([x, y, z]).T, J[WRIST].copy()


def to_palm_local(l21: np.ndarray, side: str) -> np.ndarray:
    """(21,3) in any frame -> (21,3) wrist-origin, palm-aligned. Units are preserved (metres in on the RGB-D path)."""
    R, t = palm_frame(l21, side)
    return (np.asarray(l21, np.float64).reshape(N_LANDMARKS, 3) - t) @ R


def palm_scale_m(l21: np.ndarray) -> float:
    """Hand size proxy (mean of |index MCP - pinky MCP| and |middle MCP - wrist|), for logging and B0/B1 rescaling."""
    J = np.asarray(l21, np.float64).reshape(N_LANDMARKS, 3)
    return float(0.5 * (np.linalg.norm(J[INDEX_MCP] - J[PINKY_MCP]) + np.linalg.norm(J[MIDDLE_MCP] - J[WRIST])))


def swap_side(side: str) -> str:
    return {"left": "right", "right": "left"}[side]


def handedness_for_image(label: str, *, selfie_mirrored: bool) -> str:
    """MediaPipe labels handedness assuming a SELFIE-MIRRORED image. A head camera looking outward is not mirrored,
    so its labels must be swapped. Never guess this per frame — it is a property of the camera mount."""
    label = label.lower()
    if label not in ("left", "right"): raise ValueError(f"handedness label must be left|right, got {label!r}")
    return label if selfie_mirrored else swap_side(label)
