import time
import numpy as np
from ego_teleop.robot.rebot_client import HttpRebotClient, JointLimits
from ego_teleop.tools import m0_eef_jog as M0
from ego_teleop.tools.hud_robot import RobotPanel
from .test_rebot_client_units import ToyIk, FakeHttp

# inside the C30 bounds, so the joint_bounds gate can pass (ToyIk's FK is a toy; only the plumbing is under test)
SAFE = [0.0, -10.0, -10.0, 10.0, 0.0, 0.0, -5.0, 5.0, -20.0, -8.0, 20.0, 2.0, -5.0, -10.0]


class RobotHttp(FakeHttp):
    def __init__(self, joints, fault=None):
        super().__init__(joints); self.fault = fault; self.estops = 0
    def __call__(self, method, path, payload):
        if path == "/motor_states":
            st = {a: {m: {"status": 1} for m in ("shoulder_pan", "elbow_flex")} | {"gripper": {"status": 0}} for a in ("left_arm", "right_arm")}
            if self.fault: st["right_arm"]["elbow_flex"] = {"status": self.fault}
            return st
        if path == "/status": return {"connected": True, "estop": getattr(self, "estop_on", False)}
        if path == "/estop": self.estops += 1; self.estop_on = True; return {"ok": True}
        if path == "/clear_estop": self.estop_on = False; return {"ok": True}
        return super().__call__(method, path, payload)


def _panel(http, live=False, dwell=0.2, tmp=None):
    guard = M0.OutboundGuard(http, "right", live=live, max_step_rad=np.radians(2.9))
    client = HttpRebotClient("http://x", "right", ToyIk(), limits=JointLimits(max_step_rad=np.radians(2.9)), http=guard)
    return RobotPanel(client=client, guard=guard, base_http=http, axes="x", signs=(+1,), step=0.01, hz=200.0,
                      max_lin_vel=0.05, dwell_s=dwell, out_root=tmp)


def _wait(p, t=10.0):
    t0 = time.time()
    while p.worker is not None and p.worker.is_alive() and time.time() - t0 < t: time.sleep(0.02)


def test_enable_is_refused_while_a_motor_is_faulted(tmp_path):
    p = _panel(RobotHttp(SAFE, fault=12), tmp=tmp_path)
    ok, msg = p.enable()
    assert not ok and "motors" in msg and p.worker is None
    s = p.snapshot(); assert s["gates"]["motors"]["ok"] is False and s["checks"]["motor status"]["level"] == "abort"


def test_dry_run_enable_never_posts_and_reports_moves(tmp_path):
    http = RobotHttp(SAFE); p = _panel(http, tmp=tmp_path)
    ok, _ = p.enable(); assert ok; _wait(p)
    s = p.snapshot()
    assert s["mode"] == "DONE" and not http.posted and len(p.guard.sent) > 0
    assert [m["move"] for m in s["moves"]] == ["+X", "home<+X"] and s["live_metrics"]["other_arm_deg"] == 0.0


def test_hold_stops_a_live_run_and_estop_posts(tmp_path):
    http = RobotHttp(SAFE); p = _panel(http, live=True, dwell=5.0, tmp=tmp_path)
    assert p.enable()[0]; time.sleep(0.3); n = len(http.posted); p.hold(); _wait(p)
    s = p.snapshot(); assert s["mode"] == "HOLD" and "operator HOLD" in s["banner"] and len(http.posted) <= n + 2
    assert p.estop()[0] and http.estops == 1 and p.snapshot()["mode"] == "ESTOP" and p.snapshot()["robot_estop"] is True
    ok, msg = p.enable(); assert not ok and "service" in msg              # nothing starts while the robot is E-STOPped
    assert p.clear_estop()[0] and p.snapshot()["mode"] == "OBSERVE" and p.snapshot()["robot_estop"] is False


def test_estop_failure_is_loud(tmp_path):
    class Dead(RobotHttp):
        def __call__(self, method, path, payload):
            if path == "/estop": raise ConnectionError("refused")
            return super().__call__(method, path, payload)
    p = _panel(Dead(SAFE), tmp=tmp_path)
    ok, msg = p.estop(); assert not ok and "CUT ROBOT POWER" in p.snapshot()["banner"]
