import numpy as np
import pytest
from ego_teleop.robot.rebot_client import (HttpRebotClient, JointLimits, SIDE_SLICE, GRIPPER_INDEX, N_ACTION, FLIP_IDX,
                                           grip_hold_command, hold_action, leader_action, unwrap_grip)
from ego_teleop.transforms.se3 import make_T


class ToyIk:
    """FK: EEF x = sum of joints (rad) in metres; IK: put the whole displacement into joint 1."""
    def fk(self, q): return make_T(None, [float(np.sum(q)), 0, 0])
    def ik(self, q, T):
        q2 = q.copy(); q2[0] += T[0, 3] - float(np.sum(q)); return q2, 0.0, 0.0


class FakeHttp:
    def __init__(self, joints_deg): self.joints_deg = joints_deg; self.posted = []
    def __call__(self, method, path, payload):
        if path == "/observe": return {"joints_deg": list(self.joints_deg), "joints_rad": [], "images": {}}
        if path == "/execute_step": self.posted.append(payload["action"]); return {"ok": True, "sent_deg": []}
        if path == "/estop": return {"ok": True}
        raise AssertionError(path)


def follower_from_leader(a):
    """What the service/plugin does to a leader action: flip back (arms), rad -> deg; jaws are commands."""
    f = list(a)
    for k in FLIP_IDX: f[k] = -f[k]
    return [np.degrees(v) if (i not in GRIPPER_INDEX.values()) else v for i, v in enumerate(f)]


# A realistic pose: every FLIP_IDX joint non-zero so a missing flip cannot hide, jaws half open / wrapped.
JOINTS = [12.0, -35.0, -60.0, 20.0, -10.0, 45.0, -120.0, -8.0, -40.0, -55.0, 15.0, 5.0, -30.0, 358.0]


def test_observe_converts_deg_to_rad_once_and_selects_side():
    joints = [10.0 * i for i in range(N_ACTION)]; joints[6] = -100.0; joints[13] = -200.0
    http = FakeHttp(joints); c = HttpRebotClient("http://x", "right", ToyIk(), http=http)
    st = c.observe()
    assert np.allclose(st.q_rad, np.radians(joints[7:13])) and st.gripper_raw == -200.0 and st.joints_deg_all == joints
    assert np.isclose(st.T_RB_RE[0, 3], np.sum(np.radians(joints[7:13])))


def test_leader_action_flips_exactly_flip_idx():
    q = np.arange(1, 15, dtype=float) * 0.1; q[6] = q[13] = 10.0
    a = leader_action(q)
    for i in range(N_ACTION): assert a[i] == (-q[i] if i in FLIP_IDX else q[i])
    assert set(FLIP_IDX) == {0, 1, 5, 7, 8, 12}                             # verified set (infer_core_v4, robot_service._NEG)


def test_gripper_hold_command_range_and_wrap():                           # TEST 3
    assert grip_hold_command(0.0) == 0.0 and grip_hold_command(-270.0) == 45.0 and grip_hold_command(-120.0) == 20.0
    assert grip_hold_command(-300.0) == 45.0 and grip_hold_command(+5.0) == 0.0          # clamped to 0..45
    assert unwrap_grip(358.0) == pytest.approx(-2.0) and grip_hold_command(358.0) == pytest.approx(2.0 / 6.0)
    with pytest.raises(ValueError): leader_action([0.0] * 6 + [-100.0] + [0.0] * 7)     # a raw count can never go out


@pytest.mark.parametrize("side", ["left", "right"])
def test_hold_command_moves_neither_arm(side):                            # TEST 1 + TEST 2
    """observe -> command the observed TCP -> the service would drive BOTH arms to exactly where they are."""
    http = FakeHttp(JOINTS); c = HttpRebotClient("http://x", side, ToyIk(), http=http)
    st = c.observe(); rep = c.command_ee_pose(st.T_RB_RE.copy(), 1)
    assert rep.ok and rep.action == http.posted[0]
    back = follower_from_leader(http.posted[0])
    for sl in SIDE_SLICE.values(): assert np.allclose(back[sl], JOINTS[sl], atol=1e-9)
    for g in GRIPPER_INDEX.values():
        assert 0.0 <= back[g] <= 45.0 and np.isclose(back[g], grip_hold_command(JOINTS[g]))
    assert http.posted[0] == hold_action(JOINTS)


@pytest.mark.parametrize("side", ["left", "right"])
def test_jog_moves_only_the_intended_arm(side):                           # TEST 4
    http = FakeHttp(JOINTS); c = HttpRebotClient("http://x", side, ToyIk(), http=http)
    st = c.observe(); T = st.T_RB_RE.copy(); T[0, 3] += 0.02
    assert c.command_ee_pose(T, 1).ok
    back = follower_from_leader(http.posted[0]); other = "left" if side == "right" else "right"
    assert np.allclose(back[SIDE_SLICE[other]], JOINTS[SIDE_SLICE[other]], atol=1e-9)
    moved = np.radians(back[SIDE_SLICE[side]]) - np.radians(JOINTS[SIDE_SLICE[side]])
    assert np.isclose(moved[0], 0.02) and np.allclose(moved[1:], 0.0)


def test_ik_seed_is_previous_command_not_stale_observation():
    http = FakeHttp(JOINTS); c = HttpRebotClient("http://x", "right", ToyIk(), http=http)
    st = c.observe(); T = st.T_RB_RE.copy()
    for k in range(1, 4):                      # the follower never moves (observation frozen); commands accumulate
        T[0, 3] += 0.01; assert c.command_ee_pose(T, k).ok
    last = np.radians(follower_from_leader(http.posted[-1])[SIDE_SLICE["right"]])
    assert np.isclose(last[0] - np.radians(JOINTS[7]), 0.03)
    c.reset_seed(); T2 = st.T_RB_RE.copy(); T2[0, 3] += 0.01; assert c.command_ee_pose(T2, 9).ok
    assert np.isclose(np.radians(follower_from_leader(http.posted[-1])[7]) - np.radians(JOINTS[7]), 0.01)


def test_ik_residual_and_joint_step_guards_send_nothing():               # TEST 5
    class BadIk(ToyIk):
        def ik(self, q, T): return q.copy(), 0.05, 0.0
    class NanIk(ToyIk):
        def ik(self, q, T): return np.full(6, np.nan), 0.0, 0.0
    http = FakeHttp([0.0] * N_ACTION); c = HttpRebotClient("http://x", "left", BadIk(), http=http)
    rep = c.command_ee_pose(make_T(None, [0.1, 0, 0]), 1)
    assert not rep.ok and rep.error.startswith("ik_unreachable") and rep.action is None and not http.posted
    rep = HttpRebotClient("http://x", "left", NanIk(), http=http).command_ee_pose(make_T(None, [0.1, 0, 0]), 1)
    assert not rep.ok and rep.error == "ik_nan" and not http.posted
    lim = JointLimits(np.full(6, -3.0), np.full(6, 3.0), max_step_rad=0.01)
    c2 = HttpRebotClient("http://x", "left", ToyIk(), limits=lim, http=http)
    rep = c2.command_ee_pose(make_T(None, [0.5, 0, 0]), 2); assert not rep.ok and "joint_step" in rep.error and not http.posted
    c3 = HttpRebotClient("http://x", "left", ToyIk(), limits=JointLimits(max_step_rad=10.0), http=http)
    rep = c3.command_ee_pose(make_T(None, [3.0, 0, 0]), 3); assert not rep.ok and "joint_limit" in rep.error and not http.posted


def test_service_error_is_reported_and_does_not_advance_the_seed():
    class Down(FakeHttp):
        def __call__(self, method, path, payload):
            if path == "/execute_step": raise ConnectionError("refused")
            return super().__call__(method, path, payload)
    c = HttpRebotClient("http://x", "right", ToyIk(), http=Down(JOINTS)); st = c.observe()
    T = st.T_RB_RE.copy(); T[0, 3] += 0.01
    rep = c.command_ee_pose(T, 1); assert not rep.ok and rep.error.startswith("http:") and c._q_prev_cmd is None


def test_bad_observe_length_raises():
    http = FakeHttp([0.0] * 12)
    with pytest.raises(RuntimeError):
        HttpRebotClient("http://x", "right", ToyIk(), http=http).observe()


def test_default_base_url_is_robot_service_port():
    from ego_teleop.config import RebotCfg
    import yaml, pathlib
    assert RebotCfg().base_url.endswith(":8020")
    y = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2] / "configs/ego_teleop/rebot.yaml").read_text())
    assert y["base_url"].endswith(":8020")
