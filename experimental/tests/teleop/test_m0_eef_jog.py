import numpy as np
import pytest
from ego_teleop.robot.rebot_client import HttpRebotClient, JointLimits, hold_action
from ego_teleop.tools.m0_eef_jog import OutboundGuard, GuardViolation, run
from .test_rebot_client_units import ToyIk, FakeHttp, JOINTS


def _rig(live=False, max_step_deg=2.9):
    http = FakeHttp(JOINTS)
    guard = OutboundGuard(http, "right", live=live, max_step_rad=np.radians(max_step_deg))
    return http, guard, HttpRebotClient("http://x", "right", ToyIk(), limits=JointLimits(max_step_rad=np.radians(max_step_deg)), http=guard)


def test_dry_run_sends_nothing_and_every_vector_holds_the_other_arm(tmp_path):
    http, guard, client = _rig()
    s = run(client, guard, axes="x", step=0.02, hz=200.0, max_lin_vel=0.05, dwell_s=0.3, out=tmp_path, show=0)
    assert not http.posted and len(guard.sent) > 10                        # dry run: guarded, logged, never POSTed
    hold = hold_action(JOINTS)
    for a in guard.sent:
        assert a[0:7] == hold[0:7] and a[13] == hold[13]                   # left arm + both jaws exactly held
    assert all(m["failed"] == 0 for m in s) and (tmp_path / "ticks.csv").exists() and (tmp_path / "summary.json").exists()
    # ToyIk puts x into joint 1 of the right arm = service index 7, which is in FLIP_IDX -> leader value is negated:
    # follower q7 swings -8 deg +- 0.02 rad, so the leader element swings +8 deg -+ 0.02 rad
    lead = [a[7] for a in guard.sent]
    assert np.isclose(min(lead), -np.radians(JOINTS[7]) - 0.02, atol=1e-4) and np.isclose(max(lead), -np.radians(JOINTS[7]) + 0.02, atol=1e-4)


def test_guard_refuses_other_arm_change_and_big_steps():
    http, guard, client = _rig(live=True)
    guard.set_observation(JOINTS)
    bad = list(hold_action(JOINTS)); bad[2] += 0.01
    with pytest.raises(GuardViolation): guard("POST", "/execute_step", {"action": bad})
    big = list(hold_action(JOINTS)); big[9] += np.radians(5.0)
    with pytest.raises(GuardViolation): guard("POST", "/execute_step", {"action": big})
    assert not http.posted
    guard("POST", "/execute_step", {"action": hold_action(JOINTS)}); assert len(http.posted) == 1   # live + valid -> sent


def test_stop_rules_each_condition():
    from ego_teleop.tools.m0_eef_jog import StopRules
    r = StopRules(); ok = dict(other_deg=0.0, grip_delta=0.0, along_mm=5.0, dist_mm=10.0, off_mm=1.0, ik_mm=1.0, err="")
    assert r.check(**ok) == ""
    for k, v, word in [("other_deg", 1.0, "inactive"), ("grip_delta", 10.0, "jaw"), ("along_mm", -6.0, "AGAINST"),
                       ("along_mm", 16.0, "overshot"), ("off_mm", 9.0, "off-axis"), ("ik_mm", 20.0, "IK"), ("err", "http:x", "service")]:
        assert word in r.check(**{**ok, k: v})


def test_run_aborts_and_stops_sending_when_the_other_arm_moves():
    from ego_teleop.tools.m0_eef_jog import StopRules

    class Drifting(FakeHttp):                       # after the first command the LEFT arm starts to creep
        def __call__(self, method, path, payload):
            if path == "/observe" and self.posted: self.joints_deg = list(self.joints_deg); self.joints_deg[1] += 0.2
            return super().__call__(method, path, payload)
    http = Drifting(JOINTS)
    guard = OutboundGuard(http, "right", live=True, max_step_rad=np.radians(2.9))
    client = HttpRebotClient("http://x", "right", ToyIk(), limits=JointLimits(max_step_rad=np.radians(2.9)), http=guard)
    s = run(client, guard, axes="x", step=0.01, hz=200.0, max_lin_vel=0.05, dwell_s=0.2, rules=StopRules(), show=0)
    assert s[-1]["move"] == "ABORT" and "inactive arm" in s[-1]["error"]
    assert 1 <= len(http.posted) <= 4                # stopped within a few ticks of the drift crossing 0.5 deg


def test_goto_joints_moves_only_the_active_arm():
    from ego_teleop.tools.m0_eef_jog import goto_joints
    from .test_rebot_client_units import follower_from_leader

    class Follows(FakeHttp):                         # an ideal follower: observation = last command
        def __call__(self, method, path, payload):
            if path == "/execute_step":
                self.joints_deg = [float(v) for v in follower_from_leader(payload["action"])]
                self.joints_deg[6], self.joints_deg[13] = JOINTS[6], JOINTS[13]
            return super().__call__(method, path, payload)
    http = Follows(JOINTS)
    guard = OutboundGuard(http, "right", live=True, max_step_rad=np.radians(2.9))
    client = HttpRebotClient("http://x", "right", ToyIk(), http=guard)
    rep = goto_joints(client, [12.6, -21.8, -8.1, 26.1, 1.8, -5.7], hz=1000.0)
    assert rep["reached"] and np.allclose(rep["arm_deg"], [12.6, -21.8, -8.1, 26.1, 1.8, -5.7], atol=0.01)
    for a in http.posted: assert np.allclose(a[0:7], hold_action(JOINTS)[0:7], atol=1e-12)


class MotorHttp:
    def __init__(self): self.states = {a: {m: 1 for m in ("shoulder_pan", "elbow_flex")} | {"gripper": 0} for a in ("left_arm", "right_arm")}
    def __call__(self, method, path, payload):
        assert path == "/motor_states"
        return {a: {m: {"status": st} for m, st in ms.items()} for a, ms in self.states.items()}


def test_motor_watch_preflight_and_change():
    from ego_teleop.tools.m0_eef_jog import MotorWatch, AbortRun
    h = MotorHttp(); h.states["right_arm"]["elbow_flex"] = 12
    with pytest.raises(AbortRun, match="pre-flight"): MotorWatch(h)          # a faulted arm motor refuses the run
    h = MotorHttp(); w = MotorWatch(h, every_n=1); w.tick()                   # disabled jaws (0) are allowed at baseline
    h.states["left_arm"]["gripper"] = 1
    with pytest.raises(AbortRun, match="left_arm.gripper"): w.tick()          # but any change aborts


def test_joint_step_test_moves_only_the_chosen_joint(tmp_path):
    from ego_teleop.tools.m0_eef_jog import joint_step_test, MOTOR_ORDER, StopRules
    from .test_rebot_client_units import follower_from_leader

    class Fast(FakeHttp):                        # ideal follower behind /joints_fast + /motor_states
        def __call__(self, method, path, payload):
            if path == "/execute_step":
                self.posted.append(payload["action"]); f = follower_from_leader(payload["action"])
                self.joints_deg = [float(v) for v in f]; self.joints_deg[6], self.joints_deg[13] = JOINTS[6], JOINTS[13]
                return {"ok": True}
            if path == "/joints_fast":
                m = {}
                for s, off in (("left_arm", 0), ("right_arm", 7)):
                    m[s] = {n: {"pos": self.joints_deg[off + i], "vel": 0.0, "torq": 0.1, "status": 1, "fresh": True}
                            for i, n in enumerate(MOTOR_ORDER)}
                return {"t0": 0.0, "t1": 0.001, "motors": m}
            if path == "/motor_states":
                return {a: {n: {"status": 1} for n in MOTOR_ORDER} for a in ("left_arm", "right_arm")}
            return super().__call__(method, path, payload)
    http = Fast(JOINTS)
    guard = OutboundGuard(http, "right", live=True, max_step_rad=np.radians(2.9))
    client = HttpRebotClient("http://x", "right", ToyIk(), http=guard)
    s = joint_step_test(client, http, 2, [0.5, -1.0], dwell_s=0.05, hz=500.0, rules=StopRules(), out=tmp_path)
    assert [x["step"] for x in s] == ["j2+0.50", "j2+0.50->home", "j2-1.00", "j2-1.00->home"]
    assert abs(s[0]["commanded_change_deg"] - 0.5) < 1e-6 and abs(s[0]["ss_error_deg"]) < 1e-6
    A = np.array(http.posted); home = hold_action(JOINTS)
    assert np.allclose(A[:, 0:7], home[0:7], atol=1e-12) and np.allclose(A[:, 13], home[13])
    moved = [i for i in range(7, 13) if np.ptp(A[:, i]) > 1e-9]; assert moved == [8]      # j2 = service slot 8 only
