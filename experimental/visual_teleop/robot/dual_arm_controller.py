"""Synchronized bimanual teleop tick (2026-10-14, spec 22-27). ONE loop, ONE timestamp, ONE command vector for both arms.

Per tick (TRANSACTIONAL, spec 27 -- nothing below the line changes before the bridge accepted the command):
    1. per side: candidate target  = relative retarget of the fused human TCP          (pure)
    2.           filter.propose -> limits.propose (NaN, jump, workspace, vel/acc)       (pure)
    3.           IK seeded with the last EXECUTED q -> feasibility (residual, limits, joint step = branch-jump guard)
    4. inter-arm check on the candidate PAIR (held arm = its executed pose)            any violation -> both HOLD
    5. gripper: pinch -> g target -> rate limit + LPF                                   (pure)
    6. bridge.send(one 14-vector) -> accepted?
   ------------------------------------------------------------------------------------------------------------------
    7. ONLY if accepted: commit limits, filter, gripper, q_exec, T_exec for every side that moved.
       otherwise: hold() everywhere -- last_executed stays the last command the robot actually accepted.

Mode precedence: ESTOP > NOT_INITIALISED > DEADMAN_RELEASED > CLUTCH > per-side tracking/IK/safety HOLD > FOLLOW.
Deadman release and clutch both force a re-anchor before the next motion (no jump). E-STOP needs reset + initialise().
`hold_both_on_any` (default true): if one arm must hold, the other holds too (cloth is bimanual; one arm moving alone tears
or drops it). Stage 1-3 use `arms_enabled` / `grippers_enabled` to test one actuator group at a time."""
from __future__ import annotations
import enum
import time
from dataclasses import dataclass, field
import numpy as np
from ego_teleop.transforms.se3 import T_to_pose7
from ..fusion.filters import TransactionalPoseFilter
from ..fusion.wrist_pose_fusion import HandEstimate, HandStatus
from ..gripper.gripper_mapper import GripperMapper, target_from_pinch
from ..gripper.pinch_detector import PinchState
from ..retargeting.relative_se3 import RelativeRetargeter
from ..safety.collision import InterArmCheck
from ..safety.deadman import EStopLatch
from ..safety.limits import ArmLimits
from .bridge import RobotObs
from .ik import ArmIk

SIDES = ("left", "right")


class Mode(str, enum.Enum):
    ESTOP = "ESTOP"; NOT_INITIALISED = "NOT_INITIALISED"; DEADMAN_RELEASED = "DEADMAN_RELEASED"; CLUTCH = "CLUTCH"
    HOLD = "HOLD"; FOLLOW = "FOLLOW"


@dataclass
class ArmState:
    q_exec: np.ndarray
    T_exec: np.ndarray
    limits: ArmLimits
    filt: TransactionalPoseFilter
    grip: GripperMapper
    ik: ArmIk


@dataclass
class TickRecord:
    t_ns: int
    mode: Mode
    reasons: dict                         # side -> [str]; "both" -> [str]
    sent: bool
    accepted: bool
    human_T: dict                         # side -> T_W_H or None
    delta_H: dict
    target_T: dict                        # side -> raw retarget (before filter/limits) or None
    cmd_T: dict                           # side -> candidate command or None
    exec_T: dict                          # side -> last executed TCP after this tick (FK of q_exec)
    q_exec: dict
    g_target: dict
    g_exec: dict
    obs_q: dict
    obs_g: dict
    hand_status: dict
    pinch: dict
    latency_ms: dict                      # stage latencies
    obs_T: dict = field(default_factory=dict)   # side -> FK(OBSERVED q): the robot-executed TCP (training label source)
    action: list | None = None

    def held(self, side: str) -> bool: return bool(self.reasons.get(side)) or bool(self.reasons.get("both"))


@dataclass
class ControllerConfig:
    rate_hz: float = 30.0
    arms_enabled: dict = field(default_factory=lambda: {"left": True, "right": True})
    grippers_enabled: dict = field(default_factory=lambda: {"left": True, "right": True})
    hold_both_on_any: bool = True
    call_bridge_estop: bool = True
    warn_speed_factor: float = 0.5        # hand status WARN => slower (position held / low confidence / IMU-only)
    reanchor_after_s: float = 0.5         # lost longer than this -> re-anchor on reacquisition (no catch-up motion)
    reacquire_frames: int = 2             # consecutive good frames required before the re-anchor


class DualArmController:
    def __init__(self, *, bridge, retarget: RelativeRetargeter, arms: dict, collision: InterArmCheck, estop: EStopLatch,
                 deadman, clutch, cfg: ControllerConfig | None = None) -> None:
        self.bridge, self.rt, self.collision, self.estop, self.deadman, self.clutch = bridge, retarget, collision, estop, deadman, clutch
        self.cfg = cfg or ControllerConfig(); self.dt = 1.0 / self.cfg.rate_hz
        self.arm_factories = arms                 # side -> (limits, filter, gripper, ik) built by the caller
        self.arms: dict[str, ArmState] = {}
        self.initialised = False; self._needs_anchor = set(SIDES); self._last_obs: RobotObs | None = None
        self._lost_since: dict = {}; self._good: dict = {}
        self.records: list[TickRecord] = []; self.counters = dict(ik_reject=0, safety_reject=0, robot_reject=0, collision_hold=0, ticks=0, sent=0)
        if getattr(deadman, "live_allowed", True) is False and getattr(bridge, "live", False):
            raise RuntimeError("live bridge with a deadman that cannot be released (AlwaysPressed) is refused")

    # ---- init / re-anchor ------------------------------------------------------------------------------------------
    def initialise(self, hands: dict, obs: RobotObs | None = None) -> None:
        """Spec 5: robot at its start pose, both hands at the init points (orientation already aligned in fusion).
        q_exec/T_exec := the OBSERVED robot (nothing has been commanded yet); anchors := (human now, T_exec)."""
        if self.estop.latched: raise RuntimeError("E-STOP latched: reset first")
        obs = obs or self.bridge.observe()
        if obs.estop: raise RuntimeError("bridge reports E-STOP / not connected")
        enabled = [s for s in SIDES if self.cfg.arms_enabled.get(s, False)]
        miss = [s for s in enabled if hands.get(s) is None or not hands[s].commandable]
        if miss: raise RuntimeError(f"cannot initialise: hand tracking not commandable on enabled arm(s) {miss}")
        self._needs_anchor = set(SIDES); self.rt.reset_heading()                # INIT = the only place the heading is (re)defined
        for s in SIDES:
            lim, filt, grip, ik = self.arm_factories[s]
            if hasattr(ik.solver, "sync"): ik.solver.sync(obs.joints_deg_all)
            q = obs.q[s].copy(); T = ik.fk(q)
            lim.reset(T); filt.reset(T); grip.reset(obs.g(s))
            self.arms[s] = ArmState(q, T, lim, filt, grip, ik)
            if hands.get(s) is not None and hands[s].commandable:
                self.rt.anchor(s, hands[s].T_W_H, T); self._needs_anchor.discard(s); self._good[s] = self.cfg.reacquire_frames
        self.initialised = True; self._last_obs = obs

    def _reanchor(self, side: str, hand: HandEstimate) -> None:
        a = self.arms[side]; self.rt.anchor(side, hand.T_W_H, a.T_exec); a.filt.reset(a.T_exec); a.limits.rebase_target(); a.limits.hold()

    def trip_estop(self, source: str) -> None:
        self.estop.trip(source)
        if self.cfg.call_bridge_estop: self.bridge.estop()
        self.initialised = False; self._needs_anchor = set(SIDES)

    # ---- the tick ----------------------------------------------------------------------------------------------------
    def tick(self, hands: dict, pinches: dict, now_ns: int | None = None, obs: RobotObs | None = None) -> TickRecord:
        t = int(now_ns if now_ns is not None else time.monotonic_ns()); self.counters["ticks"] += 1
        lat = {}; t_a = time.monotonic_ns()
        try: obs = obs or self.bridge.observe()
        except Exception as e: obs = None; self.trip_estop(f"observe_failed:{type(e).__name__}")
        lat["observe_ms"] = (time.monotonic_ns() - t_a) / 1e6
        if obs is not None and obs.estop and not self.estop.latched: self.trip_estop("bridge_estop")
        elif obs is not None and not obs.estop: self.estop.source_clear("bridge_estop")     # unlatching still needs estop.reset()
        if obs is not None: self._last_obs = obs
        reasons = {s: [] for s in SIDES}; reasons["both"] = []
        rec = lambda mode, **kw: self._record(t, mode, reasons, hands, pinches, obs, lat, **kw)

        if self.estop.latched: reasons["both"].append("estop:" + ",".join(sorted(self.estop.sources)) if self.estop.sources else "estop_latched"); return rec(Mode.ESTOP)
        if not self.initialised: reasons["both"].append("not_initialised"); return rec(Mode.NOT_INITIALISED)
        if not self.deadman.pressed(t):
            self._needs_anchor = set(SIDES); reasons["both"].append("deadman_released"); self._hold_all(); return rec(Mode.DEADMAN_RELEASED)
        if self.clutch.on:
            for s in SIDES:
                if hands.get(s) is not None and hands[s].commandable: self._reanchor(s, hands[s])
            self._needs_anchor = set(SIDES); reasons["both"].append("clutch"); self._hold_all(); return rec(Mode.CLUTCH)
        self.clutch.take_release_edge()
        # 10-15 review: a hand lost for > reanchor_after_s must NOT be followed from the old anchor when it comes back (the robot
        # would "catch up" the whole distance moved while lost). Re-anchor like a clutch release, after reacquire_frames good frames.
        for s in SIDES:
            ok = hands.get(s) is not None and hands[s].commandable
            if not ok:
                self._good[s] = 0
                if self._lost_since.get(s) is None: self._lost_since[s] = t
                elif (t - self._lost_since[s]) / 1e9 > self.cfg.reanchor_after_s and s in self.arms: self._needs_anchor.add(s)
            else:
                self._lost_since[s] = None; self._good[s] = self._good.get(s, 0) + 1
        for s in list(self._needs_anchor):
            if hands.get(s) is not None and hands[s].commandable and self._good.get(s, 0) >= self.cfg.reacquire_frames:
                self._reanchor(s, hands[s]); self._needs_anchor.discard(s)

        # 1-3 per-arm candidates (pure)
        t_b = time.monotonic_ns(); cand_T, cand_q, tgt_T, dH = {}, {}, {}, {}
        for s in SIDES:
            a = self.arms[s]; h = hands.get(s)
            if not self.cfg.arms_enabled.get(s, False): continue          # staged bring-up: a disabled arm is held, not a HOLD event
            if s in self._needs_anchor: reasons[s].append("awaiting_reanchor"); continue
            if h is None or not h.commandable: reasons[s].append("tracking:" + (",".join(h.reasons) if h else "no_estimate")); continue
            try: T_tgt, D = self.rt.target(s, h.T_W_H)
            except Exception as e: reasons[s].append(f"retarget:{e}"); continue
            tgt_T[s], dH[s] = T_tgt, D
            T_f = a.filt.propose(T_tgt)
            sf = self.cfg.warn_speed_factor if h.status == HandStatus.WARN else 1.0
            lv = a.limits.propose(T_f, self.dt, sf)
            if not lv.ok: reasons[s] += lv.reasons; self.counters["safety_reject"] += 1; continue
            fv = a.ik.solve(a.q_exec, lv.T_cmd)
            if not fv.ok: reasons[s] += fv.reasons; self.counters["ik_reject"] += 1; continue
            cand_T[s], cand_q[s] = (lv, fv), fv.q
        lat["ik_ms"] = (time.monotonic_ns() - t_b) / 1e6

        enabled = [s for s in SIDES if self.cfg.arms_enabled.get(s, False)]
        if self.cfg.hold_both_on_any and any(reasons[s] for s in enabled) and cand_T:
            for s in list(cand_T): reasons[s].append("partner_hold")
            cand_T.clear(); cand_q.clear()
        # 4 inter-arm on the would-be pair
        pair = {s: (cand_T[s][0].T_cmd if s in cand_T else self.arms[s].T_exec) for s in SIDES}
        col = self.collision.check(pair["left"], pair["right"])
        if col and cand_T:
            reasons["both"] += col; self.counters["collision_hold"] += 1
            for s in list(cand_T): reasons[s].append("inter_arm")
            cand_T.clear(); cand_q.clear()

        # 5 grippers (pure)
        g_prop, g_tgt = {}, {}
        for s in SIDES:
            p: PinchState | None = pinches.get(s)
            if not self.cfg.grippers_enabled.get(s, False) or p is None or p.pinch is None: continue
            if p.status == "PAUSE": reasons[s].append(f"pinch_lost:{p.dropout_s:.1f}s"); continue
            if p.g_target is None and p.status != "OK": continue            # continuous mode, hand briefly lost: hold the jaw
            g_tgt[s] = p.g_target if p.g_target is not None else target_from_pinch(p.pinch)
            g_prop[s] = self.arms[s].grip.propose(g_tgt[s], self.dt)
        moving_grip = {s for s in g_prop if abs(g_prop[s][1] - self.arms[s].grip.g) > 1e-6}

        if not cand_q and not moving_grip:
            self._hold_all(); return rec(Mode.HOLD if any(reasons[s] for s in SIDES) or reasons["both"] else Mode.FOLLOW, g_target=g_tgt)
        # 6 one vector
        q_send = {s: (cand_q[s] if s in cand_q else self.arms[s].q_exec) for s in SIDES}
        g_send = {s: (g_prop[s][1] if s in g_prop else self.arms[s].grip.g) for s in SIDES}
        res = self.bridge.send(q_send, g_send); self.counters["sent"] += int(res.sent); lat["send_ms"] = res.latency_ms
        if not res.accepted:
            reasons["both"].append(f"robot_reject:{res.error}"); self.counters["robot_reject"] += 1; self._hold_all()
            return rec(Mode.HOLD, sent=res.sent, accepted=False, action=res.action, g_target=g_tgt, cmd={s: cand_T[s][0].T_cmd for s in cand_T})
        # 7 commit
        for s in cand_q:
            a = self.arms[s]; lv, fv = cand_T[s]
            T_ex = a.ik.fk(fv.q)                                 # executed TCP = FK of the accepted joints (not the request)
            a.limits.commit(lv, T_ex); a.filt.commit(T_ex); a.q_exec = fv.q.copy(); a.T_exec = T_ex
        for s in g_prop: self.arms[s].grip.commit(g_prop[s])
        for s in SIDES:
            if s not in cand_q: self.arms[s].limits.hold()
        mode = Mode.HOLD if any(reasons[s] for s in enabled) else Mode.FOLLOW
        return rec(mode, sent=res.sent, accepted=True, action=res.action, g_target=g_tgt, cmd={s: cand_T[s][0].T_cmd for s in cand_T},
                   target=tgt_T, delta=dH)

    def _hold_all(self) -> None:
        for a in self.arms.values(): a.limits.hold()

    def _record(self, t, mode, reasons, hands, pinches, obs, lat, *, sent=False, accepted=False, action=None, g_target=None, cmd=None,
                target=None, delta=None) -> TickRecord:
        A = self.arms
        r = TickRecord(t, mode, {k: list(v) for k, v in reasons.items()}, sent, accepted,
                       {s: (hands[s].T_W_H.copy() if hands.get(s) is not None and hands[s].T_W_H is not None else None) for s in SIDES},
                       {s: (delta or {}).get(s) for s in SIDES}, {s: (target or {}).get(s) for s in SIDES}, {s: (cmd or {}).get(s) for s in SIDES},
                       {s: (A[s].T_exec.copy() if s in A else None) for s in SIDES}, {s: (A[s].q_exec.copy() if s in A else None) for s in SIDES},
                       dict(g_target or {}), {s: (A[s].grip.g if s in A else None) for s in SIDES},
                       {s: (obs.q[s].copy() if obs else None) for s in SIDES}, {s: (obs.g(s) if obs else None) for s in SIDES},
                       {s: (hands[s].status.value if hands.get(s) is not None else "NONE") for s in SIDES},
                       {s: (pinches[s].pinch if pinches.get(s) is not None else None) for s in SIDES},
                       dict(lat, tick_ms=(time.monotonic_ns() - t) / 1e6 if t <= time.monotonic_ns() else 0.0),
                       obs_T={s: (A[s].ik.fk(obs.q[s]) if (obs is not None and s in A) else None) for s in SIDES}, action=action)
        self.records.append(r); return r

    def status(self) -> dict:
        r = self.records[-1] if self.records else None
        return dict(mode=r.mode.value if r else "NONE", reasons=r.reasons if r else {}, counters=dict(self.counters),
                    estop=self.estop.latched, estop_sources=sorted(self.estop.sources), clutch=self.clutch.on,
                    exec_tcp={s: (T_to_pose7(self.arms[s].T_exec).tolist() if s in self.arms else None) for s in SIDES},
                    gripper={s: (self.arms[s].grip.g if s in self.arms else None) for s in SIDES})
