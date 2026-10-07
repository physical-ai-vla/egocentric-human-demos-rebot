"""reBot arm branch: Cartesian EEF target -> IK -> joint-limit check -> existing robot_service (/observe, /execute_step).

Units and frames — ONE boundary, here:
    robot_service  /observe      joints_deg (14): arms = FOLLOWER frame, deg; jaws = raw encoder counts (0 closed .. -270 open,
                                 may wrap to ~358 after a power cycle)
    robot_service  /execute_step action (14): arms = LEADER frame, RADIANS (service converts rad->deg);
                                 jaws = COMMAND 0..45 (steady state obs = cmd * -6; negative commands do nothing)
    follower -> leader           sign flip on FLIP_IDX = [0, 1, 5, 7, 8, 12] (both arms), same set infer_core_v4 /
                                 mac_v4_smoke_ui use and robot_service._NEG documents
    IK / FK (handumi rebot_b601 URDF)   RADIANS, FOLLOWER frame (FK of the observed joints = the measured TCP)
    this module's public API (RebotState, command_ee_pose)   RADIANS + metres, FOLLOWER frame
Nothing else in ego_teleop converts angles or flips signs. Sending an /observe vector straight back to /execute_step
drove both arms to their limits on 2026-09-23; `leader_action()` is the only way a vector leaves this module.

14-dim layout (robot_service ARM_IDX / rebot_arm_cfg): [L j1..j6, L gripper, R j1..j6, R gripper]. Teleop drives ONE
side; the other arm is re-sent at its observed pose (converted to the leader frame like everything else) and both jaws
are held at the command that reproduces their observed opening."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np
from ..transforms.se3 import T_to_pose7, pose7_to_T

SIDE_SLICE = {"left": slice(0, 6), "right": slice(7, 13)}
GRIPPER_INDEX = {"left": 6, "right": 13}
N_ACTION = 14
FLIP_IDX = (0, 1, 5, 7, 8, 12)          # follower (observation) frame -> leader (execute_step) frame
GRIP_CMD_MAX = 45.0                     # jaw command range 0..45
GRIP_OBS_PER_CMD = -6.0                 # steady state: obs_raw = cmd * -6 (measured both arms 2026-09-23)
# Follower-frame joint bounds per arm, from the C30 teleop range (c30_ik_execution_validation.py). A command outside
# them is refused, never clamped; widen from data, not by guessing.
DEFAULT_LOWER_RAD = (-2.8, -3.14, -3.14, -1.87, -1.57, -3.14)
DEFAULT_UPPER_RAD = (2.8, 0.0, 0.0, 1.57, 1.57, 3.14)


def unwrap_grip(raw: float) -> float:
    """The jaw encoder wraps (a closed jaw can read 358 instead of -2); same rule as grip_norm_dataset / infer_core_v4."""
    g = float(raw)
    return g - 360.0 if g > 180.0 else g


def grip_hold_command(raw: float) -> float:
    """The jaw command whose steady state is the observed raw opening, clamped to 0..45."""
    return float(np.clip(unwrap_grip(raw) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))


def leader_action(q14_follower_rad) -> list[float]:
    """Follower-frame 14-vector (arms rad, jaws already COMMANDS) -> the /execute_step action (leader frame)."""
    a = [float(v) for v in q14_follower_rad]
    if len(a) != N_ACTION: raise ValueError(f"expected {N_ACTION} values, got {len(a)}")
    for k in FLIP_IDX: a[k] = -a[k]
    for g in GRIPPER_INDEX.values():
        if not 0.0 <= a[g] <= GRIP_CMD_MAX: raise ValueError(f"jaw command {a[g]} outside 0..{GRIP_CMD_MAX}")
    return a


def hold_action(joints_deg_all) -> list[float]:
    """The /execute_step action that holds the robot exactly where /observe says it is (both arms, both jaws)."""
    q = [float(v) for v in joints_deg_all]
    for sl in SIDE_SLICE.values(): q[sl] = [float(np.radians(v)) for v in q[sl]]
    for g in GRIPPER_INDEX.values(): q[g] = grip_hold_command(q[g])
    return leader_action(q)


@dataclass
class RebotState:
    timestamp_ns: int                  # host monotonic at observation
    side: str
    q_rad: np.ndarray                  # (6) teleoperated arm joints, follower frame
    gripper_raw: float
    T_RB_RE: np.ndarray | None         # FK of q_rad (None if no FK available)
    joints_deg_all: list[float] = field(default_factory=list)   # raw 14 from the service (for logging / other arm hold)

    def to_row(self) -> dict:
        r = dict(t_ns=int(self.timestamp_ns), side=self.side, gripper_raw=self.gripper_raw)
        for i in range(6): r[f"q{i+1}_rad"] = float(self.q_rad[i])
        p = T_to_pose7(self.T_RB_RE) if self.T_RB_RE is not None else [np.nan] * 7
        for k, n in zip(p, ("x", "y", "z", "qx", "qy", "qz", "qw")): r[f"ee_{n}"] = float(k)
        return r


@dataclass
class ExecutionReport:
    command_timestamp_ns: int          # coordinator tick time the command belongs to
    execution_timestamp_ns: int        # host monotonic when the robot call returned
    ok: bool
    q_cmd_rad: np.ndarray | None = None
    ik_error_m: float | None = None
    ik_error_deg: float | None = None
    error: str = ""
    action: list[float] | None = None  # the exact /execute_step vector that went out (None = nothing sent)

    @property
    def latency_ms(self) -> float: return (self.execution_timestamp_ns - self.command_timestamp_ns) / 1e6


class IkSolver(Protocol):
    def fk(self, q_rad: np.ndarray) -> np.ndarray: ...
    def ik(self, q_current_rad: np.ndarray, T_RB_RE_target: np.ndarray) -> tuple[np.ndarray, float, float]:
        """-> (q_rad, position error m, rotation error deg) of the achieved pose."""
    # optional: sync(joints_deg_all) — told the full observed robot so the other arm's joints are right in FK/IK


class RebotController(Protocol):
    def observe(self) -> RebotState: ...
    def command_ee_pose(self, T_RB_RE_target: np.ndarray, command_timestamp_ns: int) -> ExecutionReport: ...
    def estop(self) -> None: ...


@dataclass
class JointLimits:
    lower_rad: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_LOWER_RAD))
    upper_rad: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_UPPER_RAD))
    max_step_rad: float = 0.05         # per command vs the previous command; larger = refused (IK flip / singularity guard)

    def check(self, q_prev: np.ndarray, q_new: np.ndarray) -> list[str]:
        r = []
        out = np.flatnonzero((q_new < np.asarray(self.lower_rad) - 1e-9) | (q_new > np.asarray(self.upper_rad) + 1e-9))
        if out.size: r.append("joint_limit:" + "/".join(f"j{i+1}={np.degrees(q_new[i]):.1f}deg" for i in out))
        if np.max(np.abs(q_new - q_prev)) > self.max_step_rad: r.append(f"joint_step:{np.degrees(np.max(np.abs(q_new - q_prev))):.1f}deg")
        return r


class HttpRebotClient:
    """RebotController over robot_service. `http` is injectable (tests, dry runs): http(method, path, json) -> dict.

    IK is seeded with the previous COMMAND (continuity), not the observation: the follower lags its setpoint by
    seconds, so seeding from the observation would re-solve from a stale pose every tick. After `seed_timeout_s`
    without a successful command (hold, clutch, start) the seed falls back to the observation."""

    def __init__(self, base_url: str, side: str, ik: IkSolver, *, limits: JointLimits | None = None, max_ik_pos_err_m: float = 0.01,
                 max_ik_rot_err_deg: float = 5.0, http: Callable[[str, str, dict | None], dict] | None = None, timeout_s: float = 0.2,
                 seed_timeout_s: float = 0.5) -> None:
        if side not in SIDE_SLICE: raise ValueError(side)
        self.base_url = base_url.rstrip("/"); self.side = side; self.ik = ik; self.limits = limits
        self.max_pos_err, self.max_rot_err = max_ik_pos_err_m, max_ik_rot_err_deg
        self._http = http or self._requests_http; self.timeout_s = timeout_s; self.seed_timeout_s = seed_timeout_s
        self._last: RebotState | None = None
        self._q_prev_cmd: np.ndarray | None = None; self._t_prev_cmd_ns = 0

    def _requests_http(self, method: str, path: str, payload: dict | None) -> dict:
        import requests
        r = requests.request(method, self.base_url + path, json=payload, timeout=self.timeout_s); r.raise_for_status(); return r.json()

    def observe(self) -> RebotState:
        d = self._http("GET", "/observe", None)
        joints_deg = [float(v) for v in d["joints_deg"]]
        if len(joints_deg) != N_ACTION: raise RuntimeError(f"/observe returned {len(joints_deg)} joints, expected {N_ACTION}")
        q = np.radians(np.asarray(joints_deg[SIDE_SLICE[self.side]], np.float64))          # <-- the deg->rad boundary
        if self.ik is not None and hasattr(self.ik, "sync"): self.ik.sync(joints_deg)
        T = self.ik.fk(q) if self.ik is not None else None
        self._last = RebotState(time.monotonic_ns(), self.side, q, joints_deg[GRIPPER_INDEX[self.side]], T, joints_deg)
        return self._last

    def reset_seed(self) -> None:
        self._q_prev_cmd = None

    def _seed(self, st: RebotState) -> np.ndarray:
        fresh = self._q_prev_cmd is not None and (time.monotonic_ns() - self._t_prev_cmd_ns) / 1e9 <= self.seed_timeout_s
        return self._q_prev_cmd.copy() if fresh else st.q_rad.copy()

    def build_action(self, q_new: np.ndarray, st: RebotState) -> list[float]:
        """Teleoperated arm at q_new, the other arm held at its observation, both jaws held — in the leader frame."""
        q = [0.0] * N_ACTION
        for side, sl in SIDE_SLICE.items():
            q[sl] = [float(v) for v in (q_new if side == self.side else np.radians(st.joints_deg_all[sl]))]
            q[GRIPPER_INDEX[side]] = grip_hold_command(st.joints_deg_all[GRIPPER_INDEX[side]])
        return leader_action(q)

    def command_ee_pose(self, T_RB_RE_target: np.ndarray, command_timestamp_ns: int) -> ExecutionReport:
        if self._last is None: self.observe()
        st = self._last; seed = self._seed(st)
        q_new, e_pos, e_rot = self.ik.ik(seed, np.asarray(T_RB_RE_target, np.float64))
        q_new = np.asarray(q_new, np.float64)
        if not np.all(np.isfinite(q_new)):
            return ExecutionReport(command_timestamp_ns, time.monotonic_ns(), False, None, e_pos, e_rot, "ik_nan")
        if e_pos > self.max_pos_err or e_rot > self.max_rot_err:
            return ExecutionReport(command_timestamp_ns, time.monotonic_ns(), False, None, e_pos, e_rot, f"ik_unreachable:{e_pos*1000:.1f}mm/{e_rot:.1f}deg")
        if self.limits is not None:
            bad = self.limits.check(seed, q_new)
            if bad: return ExecutionReport(command_timestamp_ns, time.monotonic_ns(), False, q_new, e_pos, e_rot, ",".join(bad))
        action = self.build_action(q_new, st)
        try:
            res = self._http("POST", "/execute_step", {"action": action})
            ok = bool(res.get("ok", False)); err = res.get("error") or ""
        except Exception as exc:                                   # network / service error => report, never retry blindly
            ok, err = False, f"http:{exc}"
        if ok: self._q_prev_cmd = q_new.copy(); self._t_prev_cmd_ns = time.monotonic_ns()
        return ExecutionReport(command_timestamp_ns, time.monotonic_ns(), ok, q_new, e_pos, e_rot, err, action)

    def estop(self) -> None:
        try: self._http("POST", "/estop", {})
        except Exception: pass


class MockRebotController:
    """Ideal kinematic robot for tests/sim: holds an EEF pose, moves to whatever is commanded, optional latency and workspace."""

    def __init__(self, T0: np.ndarray | None = None, side: str = "right", *, fail_every: int = 0) -> None:
        self.T = np.eye(4) if T0 is None else np.asarray(T0, np.float64).copy(); self.side = side
        self.commands: list[tuple[int, np.ndarray]] = []; self.estops = 0; self.fail_every = fail_every

    def observe(self) -> RebotState:
        return RebotState(time.monotonic_ns(), self.side, np.zeros(6), -100.0, self.T.copy(), [0.0] * N_ACTION)

    def command_ee_pose(self, T_RB_RE_target: np.ndarray, command_timestamp_ns: int) -> ExecutionReport:
        n = len(self.commands) + 1
        if self.fail_every and n % self.fail_every == 0: return ExecutionReport(command_timestamp_ns, time.monotonic_ns(), False, error="mock_fail")
        self.T = np.asarray(T_RB_RE_target, np.float64).copy(); self.commands.append((int(command_timestamp_ns), self.T.copy()))
        return ExecutionReport(command_timestamp_ns, time.monotonic_ns(), True, np.zeros(6), 0.0, 0.0)

    def estop(self) -> None: self.estops += 1


class PyrokiIkSolver:
    """IkSolver over the handumi `rebot_b601` embodiment (pyroki/jax, continuity IK: rest cost toward the seed +
    `limit_joint_delta`, same solver as holobrain-mac-model/eef_kin.py). Lives in an interpreter that has handumi
    (~/xvla-mac/bin/python), not in ego_collector/.venv.

    The solver's joint vector is NOT the service's 14-slot layout: `svc_idx[side]` maps service arm slot k -> solver
    index. `sync()` keeps the other arm at its observed joints so FK/IK see the real robot."""

    def __init__(self, solver, side: str, svc_idx: dict[str, list[int]], q_full: np.ndarray) -> None:
        self.solver, self.side, self.svc_idx = solver, side, {k: list(v) for k, v in svc_idx.items()}
        self.q_full = np.asarray(q_full, np.float32).copy()
        self.idx6 = self.svc_idx[side]

    @classmethod
    def for_rebot_b601(cls, side: str, ik_overrides: dict | None = None) -> "PyrokiIkSolver":
        import dataclasses
        from handumi.robots.registry import load_embodiment          # type: ignore
        rt = load_embodiment("rebot_b601")
        w = dataclasses.replace(rt.config.ik_weights, **(ik_overrides or {}))
        solver = rt.solver_cls(config=w); names = list(solver.joint_names); svc_idx = {}
        for s in SIDE_SLICE:
            arm = rt.config.arms[s]; grip = {g.name for g in arm.gripper_joints}
            svc_idx[s] = [names.index(n) for n in arm.joint_names if n not in grip]
            if len(svc_idx[s]) != 6: raise RuntimeError(f"rebot_b601 {s} arm has {len(svc_idx[s])} non-gripper joints, expected 6")
        return cls(solver, side, svc_idx, rt.home_q())

    def sync(self, joints_deg_all) -> None:
        for s, sl in SIDE_SLICE.items(): self.q_full[self.svc_idx[s]] = np.radians(np.asarray(joints_deg_all[sl], np.float64))

    def fk(self, q_rad: np.ndarray) -> np.ndarray:
        q = self.q_full.copy(); q[self.idx6] = q_rad
        L, R = self.solver.fk_pose7(q); return pose7_to_T(np.asarray(L if self.side == "left" else R, np.float64))

    def ik(self, q_current_rad: np.ndarray, T_target: np.ndarray) -> tuple[np.ndarray, float, float]:
        from ..transforms.se3 import rotation_angle_deg
        q = self.q_full.copy(); q[self.idx6] = q_current_rad
        pose = (np.asarray(T_target[:3, 3], np.float32), np.asarray(T_target[:3, :3], np.float32))
        q_new = np.asarray(self.solver.ik(q, left_pose=pose if self.side == "left" else None, right_pose=pose if self.side == "right" else None), np.float64)
        q6 = q_new[self.idx6]
        T_ach = self.fk(q6)
        return q6, float(np.linalg.norm(T_ach[:3, 3] - T_target[:3, 3])), rotation_angle_deg(T_ach[:3, :3], T_target[:3, :3])
