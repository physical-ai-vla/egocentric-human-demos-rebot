"""One clock, one command: the coordinator ticks at the master teleop rate (30 Hz), pulls the latest wrist pose and hand
features, runs the two retargeting branches, the safety layer and the two executors, and emits ONE TeleopCommand per
tick that the recorder logs together with both execution timestamps.

State machine (spec section 19 + VIO lifecycle, 2026-09-10):
    INIT -> SENSOR_READY -> CALIBRATED -> VIO_INITIALIZING -> VIO_STABLE -> ROBOT_READY -> ARMED -> TELEOP <-> CLUTCHED
    TELEOP/CLUTCHED -> PAUSED ; any -> ESTOP -> (clear) ROBOT_READY ; VIO loss while armed/teleop is handled by the
    TrackingSupervisor (hold), a hard tracker reset goes back to VIO_INITIALIZING via 'vio_reset'.
    HOMING / anchoring / recording may only start from VIO_STABLE onwards (tracking_valid + vio_stable readiness).
    Robot commands are sent ONLY in TELEOP / CLUTCHED (clutch: arm frozen)."""
from __future__ import annotations
import enum
import time
from dataclasses import dataclass, field, asdict
import numpy as np
from ..tracking.interfaces import WristPose, HandFeatures, TrackingHealth, HandHealth, WristPoseProvider, HandFeatureProvider
from ..retarget.arm_relative_se3 import RelativeSE3Retargeter, ArmTarget
from ..retarget.aero_retarget import AeroRetargeter, AeroCommand
from ..transforms.se3 import T_to_pose7
from .safety import ArmSafetyPipeline, SafetyVerdict
from .rebot_client import RebotController, RebotState, ExecutionReport
from .aero_client import AeroClient, AeroState


class TeleopState(str, enum.Enum):
    INIT = "INIT"; SENSOR_READY = "SENSOR_READY"; CALIBRATED = "CALIBRATED"; VIO_INITIALIZING = "VIO_INITIALIZING"; VIO_STABLE = "VIO_STABLE"
    ROBOT_READY = "ROBOT_READY"; ARMED = "ARMED"
    TELEOP = "TELEOP"; CLUTCHED = "CLUTCHED"; PAUSED = "PAUSED"; ESTOP = "ESTOP"


@dataclass
class Readiness:
    camera_online: bool = False
    imu_online: bool = False
    tracking_valid: bool = False
    vio_stable: bool = False          # VioStableGate passed (min_stable_s of OK poses while still)
    robot_observed: bool = False
    aero_homed: bool = False
    calibration_loaded: bool = False

    def missing(self) -> list[str]: return [k for k, v in asdict(self).items() if not v]


# event -> (allowed source states, target state, required readiness fields)
_T = {
    "sensors_ready": ({TeleopState.INIT}, TeleopState.SENSOR_READY, ("camera_online", "imu_online")),
    "calibrated":    ({TeleopState.SENSOR_READY}, TeleopState.CALIBRATED, ("calibration_loaded",)),
    "vio_start":     ({TeleopState.CALIBRATED, TeleopState.VIO_STABLE, TeleopState.ROBOT_READY}, TeleopState.VIO_INITIALIZING, ()),
    "vio_reset":     ({TeleopState.VIO_STABLE, TeleopState.ROBOT_READY, TeleopState.ARMED, TeleopState.PAUSED}, TeleopState.VIO_INITIALIZING, ()),
    "vio_stable":    ({TeleopState.VIO_INITIALIZING}, TeleopState.VIO_STABLE, ("tracking_valid", "vio_stable")),
    "robot_ready":   ({TeleopState.VIO_STABLE, TeleopState.ESTOP}, TeleopState.ROBOT_READY, ("robot_observed", "aero_homed", "vio_stable")),
    "arm":           ({TeleopState.ROBOT_READY, TeleopState.PAUSED}, TeleopState.ARMED, ("camera_online", "imu_online", "tracking_valid", "vio_stable", "robot_observed", "aero_homed", "calibration_loaded")),
    "start":         ({TeleopState.ARMED}, TeleopState.TELEOP, ("tracking_valid",)),
    "clutch":        ({TeleopState.TELEOP}, TeleopState.CLUTCHED, ()),
    "release":       ({TeleopState.CLUTCHED}, TeleopState.TELEOP, ("tracking_valid",)),
    "pause":         ({TeleopState.TELEOP, TeleopState.CLUTCHED}, TeleopState.PAUSED, ()),
    "stop":          ({TeleopState.TELEOP, TeleopState.CLUTCHED, TeleopState.PAUSED, TeleopState.ARMED}, TeleopState.ROBOT_READY, ()),
    "estop":         (set(TeleopState), TeleopState.ESTOP, ()),
    "clear_estop":   ({TeleopState.ESTOP}, TeleopState.ROBOT_READY, ("robot_observed",)),
}
COMMANDING_STATES = {TeleopState.TELEOP, TeleopState.CLUTCHED}


class TeleopStateMachine:
    def __init__(self) -> None: self.state = TeleopState.INIT; self.history: list[tuple[int, str, str]] = []

    def can(self, event: str, readiness: Readiness | None = None) -> tuple[bool, str]:
        if event not in _T: return False, f"unknown event {event}"
        srcs, _, req = _T[event]
        if self.state not in srcs: return False, f"{event} not allowed in {self.state.value}"
        if req:
            r = readiness or Readiness(); miss = [k for k in req if not getattr(r, k)]
            if miss: return False, f"{event} blocked, not ready: {miss}"
        return True, ""

    def fire(self, event: str, readiness: Readiness | None = None, *, t_ns: int | None = None) -> TeleopState:
        ok, why = self.can(event, readiness)
        if not ok: raise RuntimeError(why)
        self.history.append((int(t_ns if t_ns is not None else time.monotonic_ns()), self.state.value, event)); self.state = _T[event][1]
        return self.state


@dataclass
class TeleopCommand:
    timestamp_ns: int
    state: TeleopState
    arm_target_xyz: np.ndarray                 # T_RB_RE commanded (after safety) — what was actually sent
    arm_target_quat_xyzw: np.ndarray
    arm_action6: np.ndarray                    # [dxyz, rotvec] inv(T_prev_cmd)·T_cmd  (EEF-local, the a_t arm part)
    hand_target7_deg: np.ndarray               # Aero compact joints, canonical order (the a_t hand part)
    hand_u7: np.ndarray | None                 # normalized human hand state that produced it (None = hold/relaxed)
    wrist_health: TrackingHealth
    hand_health: HandHealth
    arm_sent: bool
    hand_sent: bool
    clutch: bool = False
    recenter: bool = False
    estop: bool = False
    safety_reasons: list[str] = field(default_factory=list)
    arm_exec: ExecutionReport | None = None
    hand_exec_ns: int | None = None
    delta_H6: np.ndarray | None = None         # raw human relative motion (robot-agnostic), for the raw log
    hold_reason: str = ""

    def to_row(self) -> dict:
        p, q = self.arm_target_xyz, self.arm_target_quat_xyzw
        r = dict(t_ns=int(self.timestamp_ns), state=self.state.value, x=p[0], y=p[1], z=p[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3],
                 wrist_health=self.wrist_health.value, hand_health=self.hand_health.value, arm_sent=self.arm_sent, hand_sent=self.hand_sent,
                 clutch=self.clutch, recenter=self.recenter, estop=self.estop, safety=",".join(self.safety_reasons), hold_reason=self.hold_reason,
                 arm_exec_ns=self.arm_exec.execution_timestamp_ns if self.arm_exec else None, arm_exec_ok=self.arm_exec.ok if self.arm_exec else None,
                 arm_exec_error=self.arm_exec.error if self.arm_exec else "", hand_exec_ns=self.hand_exec_ns)
        for i, n in enumerate(("dx", "dy", "dz", "drx", "dry", "drz")): r[f"arm_{n}"] = float(self.arm_action6[i])
        for i, n in enumerate(("dx", "dy", "dz", "drx", "dry", "drz")): r[f"human_{n}"] = float(self.delta_H6[i]) if self.delta_H6 is not None else np.nan
        for i in range(7): r[f"hand_{i}_deg"] = float(self.hand_target7_deg[i]); r[f"u_{i}"] = float(self.hand_u7[i]) if self.hand_u7 is not None else np.nan
        return r


@dataclass
class CoordinatorConfig:
    rate_hz: float = 30.0
    clutch_freezes_hand: bool = False        # default: clutch freezes the arm only (spec section 6)
    hand_enabled: bool = True
    arm_enabled: bool = True
    hand_telemetry_every_n: int = 10


class TeleopCoordinator:
    def __init__(self, *, wrist: WristPoseProvider, hand: HandFeatureProvider | None, arm_retargeter: RelativeSE3Retargeter,
                 aero_retargeter: AeroRetargeter, safety: ArmSafetyPipeline, robot: RebotController, aero: AeroClient | None,
                 cfg: CoordinatorConfig | None = None, readiness: Readiness | None = None) -> None:
        self.wrist, self.hand, self.arm_rt, self.aero_rt, self.safety, self.robot, self.aero = wrist, hand, arm_retargeter, aero_retargeter, safety, robot, aero
        self.cfg = cfg or CoordinatorConfig(); self.sm = TeleopStateMachine(); self.readiness = readiness or Readiness()
        self.dt = 1.0 / self.cfg.rate_hz
        self._prev_cmd_T: np.ndarray | None = None; self._last_robot: RebotState | None = None; self._last_aero: AeroState | None = None
        self._pending_clutch = False; self._pending_recenter = False; self._n = 0
        self.commands: list[TeleopCommand] = []      # the recorder drains this

    # ---- operator events ------------------------------------------------------------------------------------
    def fire(self, event: str, *, t_ns: int | None = None) -> TeleopState:
        self._refresh_readiness(t_ns)
        if event == "estop":
            self.robot.estop(); self.arm_rt.disengage(); return self.sm.fire("estop", self.readiness, t_ns=t_ns)
        st = self.sm.fire(event, self.readiness, t_ns=t_ns)
        if event == "start" or (event == "release"):
            self._anchor(t_ns); self._pending_recenter = event == "release"
        if event == "clutch": self.arm_rt.clutch(); self._pending_clutch = True
        if event in ("stop", "pause"): self.arm_rt.disengage(); self._prev_cmd_T = None
        return st

    def _anchor(self, t_ns: int | None) -> None:
        wp = self.wrist.get_pose(t_ns); rs = self.robot.observe(); self._last_robot = rs
        if wp is None or not wp.valid or rs.T_RB_RE is None: raise RuntimeError("cannot anchor: wrist tracking or robot FK unavailable")
        self.arm_rt.engage(wp, rs.T_RB_RE); self.safety.reset(rs.T_RB_RE); self._prev_cmd_T = rs.T_RB_RE.copy()

    def _refresh_readiness(self, t_ns: int | None) -> None:
        wp = self.wrist.get_pose(t_ns); self.readiness.tracking_valid = bool(wp is not None and wp.valid)
        try: self._last_robot = self.robot.observe(); self.readiness.robot_observed = True
        except Exception: self.readiness.robot_observed = False
        if self.aero is not None: self.readiness.aero_homed = bool(getattr(self.aero, "homed", True))
        else: self.readiness.aero_homed = not self.cfg.hand_enabled

    # ---- the 30 Hz tick --------------------------------------------------------------------------------------
    def tick(self, now_ns: int | None = None) -> TeleopCommand | None:
        t = int(now_ns if now_ns is not None else time.monotonic_ns()); self._n += 1
        if self.sm.state not in COMMANDING_STATES: return None
        wp = self.wrist.get_pose(t)
        hf = self.hand.get_features(t) if (self.hand is not None and self.cfg.hand_enabled) else None
        rs = self.robot.observe(); self._last_robot = rs
        obs_age_ms = (t - rs.timestamp_ns) / 1e6 if rs.timestamp_ns <= t else 0.0
        wrist_health = wp.health if wp is not None else TrackingHealth.LOST
        hand_health = hf.health if hf is not None else HandHealth.LOST

        # ARM branch: relative SE(3) -> safety -> robot
        tgt: ArmTarget = self.arm_rt.update(wp, rs.T_RB_RE)
        verdict: SafetyVerdict = self.safety.check(tgt.T_RB_RE_target, dt_s=self.dt, speed_factor=tgt.speed_factor, robot_obs_age_ms=obs_age_ms)
        arm_exec = None; arm_sent = False
        if self.cfg.arm_enabled and self.sm.state == TeleopState.TELEOP and not tgt.held and verdict.ok:
            arm_exec = self.robot.command_ee_pose(verdict.T_cmd, t); arm_sent = arm_exec.ok
        elif self.cfg.arm_enabled and self.sm.state == TeleopState.TELEOP and tgt.hold_reason == "tracking_lost" and self._prev_cmd_T is not None:
            arm_exec = None  # hold: nothing is sent, the robot keeps its last target
        action6 = ArmTarget(verdict.T_cmd, tgt.state, tgt.held).action6(self._prev_cmd_T)
        self._prev_cmd_T = verdict.T_cmd.copy()

        # HAND branch: u7 -> Aero (independent of the arm; clutch may or may not freeze it)
        if self.sm.state == TeleopState.CLUTCHED and self.cfg.clutch_freezes_hand: acmd: AeroCommand = self.aero_rt.hold(t)
        elif hf is not None and hf.health == HandHealth.OK and hf.u7 is not None: acmd = self.aero_rt.retarget(t, hf.u7)
        elif hf is not None and hf.health == HandHealth.HOLD: acmd = self.aero_rt.hold(t)
        else: acmd = self.aero_rt.relaxed(t)
        hand_sent = False; hand_exec_ns = None
        if self.aero is not None and self.cfg.hand_enabled:
            hand_sent, hand_exec_ns = self.aero.set_compact_deg(acmd.compact_deg)
            if self._n % self.cfg.hand_telemetry_every_n == 0:
                try: self._last_aero = self.aero.observe(telemetry=True)
                except Exception: pass

        p7 = T_to_pose7(verdict.T_cmd)
        dH = None
        if tgt.delta_H is not None:
            from ..transforms.se3 import local_delta
            dH = local_delta(np.eye(4), tgt.delta_H)
        cmd = TeleopCommand(t, self.sm.state, p7[:3], p7[3:], action6, acmd.compact_deg.copy(), acmd.u7, wrist_health, hand_health, arm_sent, hand_sent,
                            clutch=self._pending_clutch or self.sm.state == TeleopState.CLUTCHED, recenter=self._pending_recenter,
                            safety_reasons=list(verdict.reasons), arm_exec=arm_exec, hand_exec_ns=hand_exec_ns, delta_H6=dH, hold_reason=tgt.hold_reason)
        self._pending_clutch = self._pending_recenter = False
        self.commands.append(cmd); return cmd

    @property
    def last_robot_state(self) -> RebotState | None: return self._last_robot
    @property
    def last_aero_state(self) -> AeroState | None: return self._last_aero
