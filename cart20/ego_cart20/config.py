"""Frozen contract constants for the ego CART20 pipeline (spec v2 + design decisions 2026-10-01).

Every number that defines the meaning of a stored value lives here and is written into metadata.json.
Change nothing here without reporting it first: the reBot v4 fine-tuning data uses the same contract.

Decisions that override spec v2 text (user, 2026-10-01):
  D1  temporal: rows are stored at ~15 Hz, but every state/action time offset is UMI_DT = 3/59.94 s (50.05 ms),
      interpolated on the RAW trajectory -- never the 15 Hz row spacing.  REL16 spans 16 * 50.05 ms = 0.80 s.
  D2  gripper: 0 = fully closed, 1 = fully open (aperture mm / 80 mm), the v4 / R312c RELCART20 / ego gcal1 polarity.
  D3  state.npy = v4 RELCART20 task-anchor state [L pose9 | R pose9 | gL | gR] (grippers at dims 18, 19).
      The spec's previous-relative state is kept as state_prevrel.npy (diagnostic / ablation only, never the default).
"""
from dataclasses import asdict, dataclass

UMI_DT = 3.0 / 59.94                 # 50.05 ms, the v4 contract target / history spacing (UMI76_CONTRACT 2b)
ROW_FPS = 15.0                       # stored row rate
HORIZON = 16                         # REL16: number of future targets (time axis)
ARM_DIM = 10                         # per arm: xyz 3 + rot6d 6 + gripper 1
CART_DIM = 20                        # CART20 = LEFT10 | RIGHT10
AUX_DIM = 12                         # AUX12 = left dq6 | right dq6 (all zero for ego)
MODEL_ACTION_DIM = CART_DIM + AUX_DIM  # 32, X-VLA v4 action width
STATE_DIM = 20
ARMS = ("left", "right")             # LEFT always first

# CART20 / action32 channel map (per arm [xyz, rot6d, g])
ACT_POS = {"left": slice(0, 3), "right": slice(10, 13)}
ACT_ROT = {"left": slice(3, 9), "right": slice(13, 19)}
ACT_GRIP = {"left": 9, "right": 19}
AUX = slice(20, 32)

# state.npy (v4 RELCART20 layout [L pose9 | R pose9 | gL gR])
ST_POSE = {"left": slice(0, 9), "right": slice(9, 18)}
ST_GRIP = {"left": 18, "right": 19}
# state_prevrel.npy (spec v2 layout, per arm [xyz, rot6d, g]) -- same layout as CART20
PR_POSE = {"left": slice(0, 9), "right": slice(10, 19)}
PR_GRIP = {"left": 9, "right": 19}


@dataclass(frozen=True)
class Cart20Config:
    row_fps: float = ROW_FPS
    target_dt_s: float = UMI_DT
    horizon: int = HORIZON
    max_raw_gap_s: float = 0.050        # interpolation only between two VALID raw samples at most this far apart
    image_tol_s: float = 0.020          # a row needs a camera frame within 20 ms (the C8 V3 pairing rule)
    required_cameras: tuple = ("head",)
    gripper_convention: str = "0_closed_1_open"
    rot6d_convention: str = "first_two_rows"   # umi pose_util.mat_to_rot6d = R[:2, :]
    aux_mode: str = "zero"
    stage_weights: tuple = (1.0, 1.0, 1.0, 2.0)
    discontinuity_filter: bool = False    # v2 = False (unchanged); v2b = True (pose_filter.discontinuity_filter, user 2026-10-01)
    jump_max_trans_per_row_m: float = 0.100
    jump_max_rot_per_row_deg: float = 45.0
    jump_window: str = "frame"            # "frame" (rate checked per raw step) | "row" (displacement over one 66.7 ms row)
    jump_isolated_step_m: float = None    # also flag a single raw step > this with both neighbouring steps < 0.3 m/s

    def to_dict(self):
        return asdict(self)


CONTRACT = dict(
    cartesian_action_name="CART20", cartesian_action_dim=CART_DIM, model_action_dim=MODEL_ACTION_DIM,
    action_horizon=HORIZON, state_dim=STATE_DIM, auxiliary_action_dim=AUX_DIM, auxiliary_action_mode="zero",
    row_fps=ROW_FPS, target_dt_s=UMI_DT, horizon_span_s=HORIZON * UMI_DT,
    action_contract="current_anchor_relative: A_k = inv(T(t)) @ T(t + k*UMI_DT), k = 1..16, raw-trajectory interpolation",
    action_layout="[L xyz3 rot6d6 g | R xyz3 rot6d6 g | AUX12 = 0]",
    state_contract="relcart20_task_anchor: per arm inv(T(t_task_start)) @ T(t); layout [L pose9 | R pose9 | gL | gR]",
    state_prevrel_contract="diagnostic only: per arm inv(T(t)) @ T(t - UMI_DT); layout [L pose9 g | R pose9 g]",
    arm_order=list(ARMS), rotation_representation="rot6d, first two ROWS of R (umi pose_util.mat_to_rot6d)",
    gripper_representation="continuous aperture, 0 = closed, 1 = open (caliper mm / 80 mm)",
    pseudo_robot_q=False, sequential_delta=False, tail_padding=False,
)
