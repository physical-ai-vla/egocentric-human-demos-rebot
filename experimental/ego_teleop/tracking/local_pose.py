"""Local control pose vs global map pose (multi-sensor spec section 3, 17).

    T_map_wrist    globally optimised SLAM pose. Jumps when the backend closes a loop or re-optimises its map.
    T_local_wrist  continuous causal pose. NEVER jumps. This is the only pose the robot is allowed to see.

A loop closure is a *correction of the past*, not a motion of the wrist: the operator's hand did not move 8 cm in one
frame because the map decided the room is smaller. Feeding a map pose straight into the relative-SE(3) retargeter would
turn every global correction into a robot command, so the correction is absorbed into an offset instead:

    T_local_wrist(t) = T_local_map · T_map_wrist(t)          T_local_map updated ONLY on a correction, such that
                                                             T_local_wrist is exactly continuous across it.

The offset is permanent (never slewed back): the local frame is an arbitrary, session-local control frame, and only
its increments matter. What the absorber owes the log is the size of what it swallowed -- `correction_m`,
`correction_deg` and the cumulative totals, which are exactly the numbers that say whether the map is behaving.

Policy on an UNFLAGGED discontinuity (the backend did not announce a map update): default is to pass it through, so
`TrackingSupervisor.max_jump_m` still sees it and grades the frame LOST. A tracking failure and a loop closure look
identical in the pose alone; silently absorbing both would hide the failure. Backends that do announce their map
updates (`extra["map_update"]`/`loop_closure`/`global_correction`) get the smooth path."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from ..transforms.se3 import inv_T


@dataclass
class LocalPoseConfig:
    absorb_flagged_only: bool = True          # False: also absorb an unflagged jump that is otherwise healthy
    map_update_keys: tuple = ("map_update", "loop_closure", "global_correction")
    jump_m: float = 0.05                      # what counts as a discontinuity when absorb_flagged_only is False
    jump_deg: float = 10.0
    max_absorb_m: float = 1.0                 # a "correction" larger than this is not a loop closure; refuse it
    max_absorb_deg: float = 90.0


@dataclass
class LocalPoseContinuity:
    """Map pose in, continuous local pose out. One instance per tracked wrist."""
    cfg: LocalPoseConfig = field(default_factory=LocalPoseConfig)
    T_local_map: np.ndarray = field(default_factory=lambda: np.eye(4))
    n_absorbed: int = 0
    n_refused: int = 0
    last_correction_m: float = 0.0
    last_correction_deg: float = 0.0
    cumulative_correction_m: float = 0.0
    cumulative_correction_deg: float = 0.0
    _prev_map: np.ndarray | None = field(default=None, repr=False)
    _prev_local: np.ndarray | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.T_local_map = np.eye(4); self._prev_map = self._prev_local = None
        self.n_absorbed = self.n_refused = 0
        self.last_correction_m = self.last_correction_deg = 0.0
        self.cumulative_correction_m = self.cumulative_correction_deg = 0.0

    @staticmethod
    def flagged(extra: dict | None, keys) -> bool:
        e = extra or {}
        return any(bool(e.get(k)) for k in keys)

    def update(self, T_map: np.ndarray, *, extra: dict | None = None) -> np.ndarray:
        """-> T_local_wrist for this map pose. Absorbs an announced (or, by policy, a detected) map correction."""
        T_map = np.asarray(T_map, np.float64)
        c = self.cfg
        self.last_correction_m = self.last_correction_deg = 0.0
        if self._prev_map is not None and self._prev_local is not None:
            d = inv_T(self._prev_map) @ T_map
            dm = float(np.linalg.norm(d[:3, 3]))
            dd = float(np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude()))
            announced = self.flagged(extra, c.map_update_keys)
            detected = (dm > c.jump_m or dd > c.jump_deg) and not c.absorb_flagged_only
            if announced or detected:
                if dm > c.max_absorb_m or dd > c.max_absorb_deg:
                    self.n_refused += 1            # too big to be a map correction: let it through and be graded
                else:
                    # the corrected map pose must produce EXACTLY the previous local pose: zero commanded motion
                    self.T_local_map = self._prev_local @ inv_T(T_map)
                    self.n_absorbed += 1
                    self.last_correction_m, self.last_correction_deg = dm, dd
                    self.cumulative_correction_m += dm; self.cumulative_correction_deg += dd
        T_local = self.T_local_map @ T_map
        self._prev_map, self._prev_local = T_map.copy(), T_local.copy()
        return T_local

    def stats(self) -> dict:
        return dict(map_absorbed=self.n_absorbed, map_refused=self.n_refused,
                    map_correction_m=self.last_correction_m, map_correction_deg=self.last_correction_deg,
                    map_correction_cum_m=self.cumulative_correction_m, map_correction_cum_deg=self.cumulative_correction_deg)
