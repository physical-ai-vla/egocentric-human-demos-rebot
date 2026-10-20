"""Spec 37: transactional limiter rollback, IK reject rollback, clutch continuity, deadman HOLD, tracking-loss HOLD, NaN, velocity,
workspace, inter-arm, E-STOP latch, gripper hysteresis/polarity/dropout, stage gating."""
import numpy as np
import pytest
from visual_teleop import config as vcfg
from visual_teleop.gripper.gripper_mapper import GripperMapper, target_from_pinch
from visual_teleop.gripper.pinch_detector import PinchDetector
from visual_teleop.robot.dual_arm_controller import Mode
from visual_teleop.safety.deadman import HeartbeatDeadman, EStopLatch, AlwaysPressed
from visual_teleop.robot.bridge import MockBridge
from visual_teleop.build import build_controller
from visual_teleop.robot.ik import ToyIk, toy_limits
from .conftest import T_of, hand, lost, pinch, DT_NS, Q0
from visual_teleop import config as _vcfg
VMAX = _vcfg.load().limits["max_lin_vel_m_s"]   # limiter speed from the YAML (was hard-coded 0.08)

P_OPEN = {"left": pinch(False), "right": pinch(False)}


def both(T_L, T_R=None, t=0):
    return {"left": hand("left", T_L, t), "right": hand("right", T_L if T_R is None else T_R, t)}


def run(ctl, n, Tfn, t0=0, pinches=None):
    recs = []
    for k in range(n):
        t = t0 + k * DT_NS; T = Tfn(k); recs.append(ctl.tick(both(T, None, t) if not isinstance(T, dict) else T, pinches or P_OPEN, t))
    return recs


def test_follows_and_velocity_limit(rig, cfg):
    ctl, br = rig()
    recs = run(ctl, 30, lambda k: T_of((0.10, 0, 0)))            # a 10 cm step request
    steps = np.diff([r.exec_T["left"][0, 3] for r in recs])
    assert recs[-1].mode == Mode.FOLLOW and all(r.accepted for r in recs)
    assert steps.max() <= cfg.limits["max_lin_vel_m_s"] / 30 + 1e-9                      # never faster than v_max * dt
    np.testing.assert_allclose(br.q["left"][:3], ctl.arms["left"].q_exec[:3])             # executed == what the robot got


def test_ik_reject_rolls_back_and_recovers_without_jump(rig):
    ctl, br = rig(unreachable=lambda T: 0.32 < T[0, 3] < 0.34)
    recs = run(ctl, 60, lambda k: T_of((0.06, 0, 0)))
    x = np.array([r.exec_T["left"][0, 3] for r in recs])
    held = [r for r in recs if r.held("left")]
    assert held and all("ik_unreachable" in ",".join(r.reasons["left"]) for r in held if r.reasons["left"])
    assert x.max() <= 0.32 + VMAX / 30 + 1e-9                                             # never got past the unreachable band
    # the limiter state stayed at the last EXECUTED pose: the executed trajectory has no step bigger than v_max*dt
    assert np.abs(np.diff(x)).max() <= VMAX / 30 + 1e-9


def test_robot_reject_rolls_back_every_state(rig):
    ctl, br = rig()
    run(ctl, 5, lambda k: T_of((0.02, 0, 0)))
    snap = {s: (ctl.arms[s].q_exec.copy(), ctl.arms[s].limits.T_exec.copy(), ctl.arms[s].filt.value.copy(), ctl.arms[s].grip.g) for s in Q0}
    br.reject_next = 3
    recs = run(ctl, 3, lambda k: T_of((0.05, 0, 0)), t0=5 * DT_NS, pinches={"left": pinch(True), "right": pinch(True)})
    assert all(not r.accepted and r.mode == Mode.HOLD for r in recs)
    for s in Q0:
        q, Tl, Tf, g = snap[s]
        np.testing.assert_array_equal(ctl.arms[s].q_exec, q); np.testing.assert_array_equal(ctl.arms[s].limits.T_exec, Tl)
        np.testing.assert_array_equal(ctl.arms[s].filt.value, Tf); assert ctl.arms[s].grip.g == g
    assert np.allclose(ctl.arms["left"].limits.limiter.velocity, 0)


def test_tracking_loss_holds_both_then_reanchors(rig):
    ctl, br = rig()
    run(ctl, 10, lambda k: T_of((0.01 * k, 0, 0)))
    x_before = ctl.arms["left"].T_exec[0, 3]
    r = ctl.tick({"left": lost("left"), "right": hand("right", T_of((0.2, 0, 0)))}, P_OPEN, 10 * DT_NS)
    assert r.mode == Mode.HOLD and r.held("left") and r.held("right") and not r.sent         # hold_both_on_any
    assert ctl.arms["left"].T_exec[0, 3] == x_before


def test_nan_target_rejected(rig):
    ctl, br = rig(); T = T_of((0.01, 0, 0)); T[1, 3] = np.nan
    r = ctl.tick(both(T), P_OPEN, 0)
    assert r.held("left") and not r.sent


def test_workspace_reject_holds_not_clamps(rig):
    ctl, br = rig()
    r = run(ctl, 3, lambda k: T_of((0.0, 0, -0.2)))[-1]                                 # z -> 0.15 - 0.8*0.2 < floor 0.03
    assert any("workspace_reject" in x for x in r.reasons["left"]) and not r.sent


def test_target_jump_glitch_rejected(rig):
    ctl, br = rig(); run(ctl, 3, lambda k: T_of((0.0, 0, 0)))
    r = ctl.tick(both(T_of((0.2, 0, 0))), P_OPEN, 3 * DT_NS)
    assert any("target_jump" in x for x in r.reasons["left"])


def test_clutch_continuity_no_jump(rig):
    ctl, br = rig()
    run(ctl, 20, lambda k: T_of((0.002 * k, 0, 0)))
    x_exec = ctl.arms["left"].T_exec[:3, 3].copy()
    ctl.clutch.set(True)
    recs = run(ctl, 10, lambda k: T_of((0.04 + 0.03 * k, 0.05, 0)), t0=20 * DT_NS)       # human repositions a lot
    assert all(r.mode == Mode.CLUTCH and not r.sent for r in recs)
    np.testing.assert_array_equal(ctl.arms["left"].T_exec[:3, 3], x_exec)
    ctl.clutch.set(False)
    r = ctl.tick(both(T_of((0.04 + 0.03 * 9, 0.05, 0))), P_OPEN, 30 * DT_NS)              # same human pose as last clutch tick
    np.testing.assert_allclose(r.exec_T["left"][:3, 3], x_exec, atol=1e-9)                # zero jump on release
    r = ctl.tick(both(T_of((0.04 + 0.03 * 9 + 0.01, 0.05, 0))), P_OPEN, 31 * DT_NS)       # then relative motion resumes
    assert r.exec_T["left"][0, 3] > x_exec[0]


def test_deadman_release_holds_and_repress_reanchors(rig):
    dm = HeartbeatDeadman(timeout_ms=150); ctl, br = rig(deadman=dm)
    t = 0
    for k in range(10): dm.heartbeat(t); ctl.tick(both(T_of((0.003 * k, 0, 0))), P_OPEN, t); t += DT_NS
    n_sent = len(br.sent); x_exec = ctl.arms["left"].T_exec[:3, 3].copy()
    t += 200_000_000                                                                       # no heartbeat for 200 ms
    r = ctl.tick(both(T_of((0.1, 0, 0))), P_OPEN, t)
    assert r.mode == Mode.DEADMAN_RELEASED and len(br.sent) == n_sent
    dm.heartbeat(t + DT_NS); r = ctl.tick(both(T_of((0.1, 0, 0))), P_OPEN, t + DT_NS)
    np.testing.assert_allclose(r.exec_T["left"][:3, 3], x_exec, atol=1e-9)                # re-press: no jump to the new human pose
    dm.release(); assert not dm.pressed(t + DT_NS)


def test_live_bridge_refuses_unreleasable_deadman(cfg):
    with pytest.raises(RuntimeError):
        build_controller(cfg, bridge=MockBridge(), solvers={s: ToyIk() for s in Q0}, deadman=AlwaysPressed(), joint_limits={s: toy_limits() for s in Q0})


def test_estop_supersedes_and_never_auto_resumes(rig):
    ctl, br = rig(); run(ctl, 3, lambda k: T_of((0.01, 0, 0)))
    br.estop_flag = True
    r = ctl.tick(both(T_of((0.02, 0, 0))), P_OPEN, 3 * DT_NS)
    assert r.mode == Mode.ESTOP and ctl.estop.latched and br.n_estop_calls == 1
    br.estop_flag = False
    r = ctl.tick(both(T_of((0.02, 0, 0))), P_OPEN, 4 * DT_NS); assert r.mode == Mode.ESTOP          # latched
    assert "bridge_estop" not in ctl.estop.sources                                         # bridge reports clear -> source clear, still latched
    ctl.estop.trip("ui"); ok, why = ctl.estop.reset(); assert not ok and "ui" in why
    ctl.estop.source_clear("ui"); ok, _ = ctl.estop.reset(); assert ok
    r = ctl.tick(both(T_of((0.02, 0, 0))), P_OPEN, 5 * DT_NS); assert r.mode == Mode.NOT_INITIALISED   # explicit re-init required


def test_inter_arm_collision_holds_both(rig):
    c = vcfg.load(overrides={"retarget": {"translation_frame": "body"}, "control": {"arms_enabled": {"left": True, "right": True}}, "limits": {"filter": {"alpha_pos": 1.0, "alpha_rot": 1.0}},
                             "workspace": {"left": {"min_xyz": [0.0, -0.5, 0.0]}}})
    ctl, br = rig(c)
    # left hand moves toward -y (into the right arm); right hand still
    recs = run(ctl, 200, lambda k: {"left": hand("left", T_of((0, -0.002 * k, 0))), "right": hand("right", T_of((0, 0, 0)))})
    pl = [r.exec_T["left"][:3, 3] for r in recs]; pr = [r.exec_T["right"][:3, 3] for r in recs]
    d = [np.linalg.norm(a - b) for a, b in zip(pl, pr)]
    assert min(d) >= 0.10 - 1e-9 and any("tcp_separation" in x or "wrist_separation" in x for r in recs for x in r.reasons["both"])


def test_gripper_hysteresis_no_chatter():
    p = PinchDetector(25, 40, 0.5, 2.0, 0.4); t = 0; states = []
    for m in [60, 50, 39, 30, 26, 24, 30, 39, 41, 30, 26, 41]:
        states.append(p.update(m, 0.9, t).pinch); t += DT_NS
    assert states == [False, False, False, False, False, True, True, True, False, False, False, False]
    assert p.n_toggles == 2


def test_gripper_dropout_hold_then_pause():
    p = PinchDetector(25, 40, 0.5, 2.0, 0.4); p.update(10, 0.9, 0)
    s = p.update(None, 0.0, int(0.3e9)); assert s.pinch is True and s.status == "OK"
    s = p.update(80, 0.1, int(1.0e9)); assert s.pinch is True and s.status == "WARN"        # low confidence never toggles
    s = p.update(None, 0.0, int(2.5e9)); assert s.pinch is True and s.status == "PAUSE"
    assert PinchDetector(25, 40).update(None, 0, 0).status == "PAUSE"                        # never measured


def test_gripper_polarity_and_smoothing():
    assert target_from_pinch(True) == 0.0 and target_from_pinch(False) == 1.0              # 0 = closed
    gm = GripperMapper(2.0, 0.3); gm.reset(1.0); gs = []
    for _ in range(40):
        pr = gm.propose(0.0, 1 / 30); gm.commit(pr); gs.append(gm.g)
    assert np.all(np.diff(gs) <= 0) and np.max(-np.diff(gs)) <= 2.0 / 30 + 1e-9 and gs[-1] < 0.05
    assert GripperMapper.jaw_command(0.0) == 0.0 and GripperMapper.jaw_command(1.0) == 45.0


def test_stage1_gripper_only_moves_no_arm(cfg):
    c = vcfg.load(overrides={"retarget": {"translation_frame": "body"}, "control": {"arms_enabled": {"left": False, "right": False}}})
    br = MockBridge(q={s: Q0[s].copy() for s in Q0})
    ctl = build_controller(c, bridge=br, solvers={s: ToyIk() for s in Q0}, deadman=__import__("visual_teleop.safety.deadman", fromlist=["x"]).ScriptedDeadman(),
                           joint_limits={s: toy_limits() for s in Q0})
    ctl.initialise({s: hand(s, np.eye(4)) for s in Q0})
    for k in range(30): ctl.tick(both(T_of((0.01 * k, 0, 0))), {"left": pinch(True), "right": pinch(False)}, k * DT_NS)
    np.testing.assert_allclose(br.q["left"], Q0["left"], atol=1e-12); np.testing.assert_allclose(br.q["right"], Q0["right"], atol=1e-12)
    assert br.g["left"] < 0.1 and br.g["right"] == 1.0


def test_sigmoid_pinch_continuous_and_dropout_holds():
    from visual_teleop.gripper.pinch_detector import SigmoidPinch
    sp = SigmoidPinch(closed=2.0, open=43.0, k=10.0, ema=1.0)
    assert SigmoidPinch(closed=2.0, open=43.0, ema=1.0).update(8.0, 0.9, 0).g_target == 0.0          # a normal squeeze closes FULLY
    g_closed = sp.update(2.0, 0.9, 0).g_target; g_half = sp.update(22.5, 0.9, 1).g_target; g_open = sp.update(43.0, 0.9, 2).g_target
    assert g_closed < 0.01 and abs(g_half - 0.5) < 0.01 and g_open > 0.99                     # 0 = closed, continuous in between
    s = sp.update(None, 0.0, int(0.3e9)); assert s.status == "HOLD" and s.g_target is None       # lost: no new target
    s = sp.update(None, 0.0, int(3e9)); assert s.status == "PAUSE"


def test_sigmoid_lost_hand_never_moves_jaw(cfg):
    from visual_teleop.gripper.pinch_detector import PinchState
    br = MockBridge(q={s: Q0[s].copy() for s in Q0})
    c = vcfg.load(overrides={"retarget": {"translation_frame": "body"}, "control": {"arms_enabled": {"left": False, "right": False}}})
    ctl = build_controller(c, bridge=br, solvers={s: ToyIk() for s in Q0}, deadman=__import__("visual_teleop.safety.deadman", fromlist=["x"]).ScriptedDeadman(),
                           joint_limits={s: toy_limits() for s in Q0})
    ctl.initialise({s: hand(s, np.eye(4)) for s in Q0})
    for k in range(40): ctl.tick(both(np.eye(4)), {"left": None, "right": PinchState(False, "OK", 40.0, 0.0, g_target=0.8)}, k * DT_NS)
    g0 = br.g["right"]
    for k in range(40, 60): ctl.tick(both(np.eye(4)), {"left": None, "right": PinchState(True, "HOLD", None, 0.3, g_target=None)}, k * DT_NS)
    assert abs(br.g["right"] - g0) < 1e-9 and g0 > 0.7


def test_long_loss_reanchors_without_catch_up(rig):
    ctl, br = rig()
    run(ctl, 10, lambda k: T_of((0.002 * k, 0, 0)))
    x_exec = ctl.arms["left"].T_exec[:3, 3].copy(); t = 10 * DT_NS
    for k in range(30): ctl.tick({"left": lost("left"), "right": lost("right")}, P_OPEN, t); t += DT_NS     # 1 s lost
    # the human reappears 15 cm away: frame 1 = not yet (debounce), frame 2 = re-anchor, then NO catch-up toward +15 cm
    for k in range(5): r = ctl.tick(both(T_of((0.17, 0, 0))), P_OPEN, t); t += DT_NS
    np.testing.assert_allclose(r.exec_T["left"][:3, 3], x_exec, atol=1e-9)
    r = ctl.tick(both(T_of((0.18, 0, 0))), P_OPEN, t)                           # then relative motion from the new anchor
    assert 0 < r.exec_T["left"][0, 3] - x_exec[0] <= VMAX / 30 + 1e-9


def test_short_gap_does_not_reanchor(rig):
    ctl, br = rig(); t = 0
    for k in range(5): ctl.tick(both(T_of((0.0, 0, 0))), P_OPEN, t); t += DT_NS
    for k in range(5): ctl.tick({"left": lost("left"), "right": lost("right")}, P_OPEN, t); t += DT_NS   # 0.17 s gap
    a0 = ctl.rt.anchors["left"].T_W_H0.copy()
    ctl.tick(both(T_of((0.01, 0, 0))), P_OPEN, t)
    np.testing.assert_array_equal(ctl.rt.anchors["left"].T_W_H0, a0)          # same anchor: motion during a short gap is kept
