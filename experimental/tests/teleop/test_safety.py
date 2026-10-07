import numpy as np
import pytest
from ego_teleop.robot.safety import ArmSafetyPipeline, SafetyConfig, WorkspaceBox, CartesianLimiter
from .conftest import T_of


def cfg(**kw):
    base = dict(workspace=WorkspaceBox((0, -0.5, 0), (0.6, 0.5, 0.5)), max_lin_vel_m_s=0.2, max_ang_vel_deg_s=60, max_lin_acc_m_s2=100.0, robot_obs_max_age_ms=200)
    base.update(kw); return SafetyConfig(**base)


def test_velocity_limit_linear_and_angular():
    s = ArmSafetyPipeline(cfg()); s.reset(T_of((0.3, 0, 0.2)))
    v = s.check(T_of((0.5, 0, 0.2), (0, 0, 90)), dt_s=1 / 30)
    assert v.ok and np.isclose(np.linalg.norm(v.T_cmd[:3, 3] - [0.3, 0, 0.2]), 0.2 / 30, atol=1e-9)
    assert np.isclose(v.ang_speed_deg_s, 60, atol=1e-6) and np.isclose(v.lin_speed_m_s, 0.2)


def test_speed_factor_zero_does_not_move():
    s = ArmSafetyPipeline(cfg()); s.reset(T_of((0.3, 0, 0.2)))
    v = s.check(T_of((0.5, 0.1, 0.2), (0, 0, 40)), dt_s=1 / 30, speed_factor=0.0)
    assert np.allclose(v.T_cmd, T_of((0.3, 0, 0.2)))


def test_acceleration_limit_ramps():
    s = ArmSafetyPipeline(cfg(max_lin_acc_m_s2=1.0)); s.reset(T_of((0.3, 0, 0.2)))
    v1 = s.check(T_of((0.5, 0, 0.2)), dt_s=1 / 30)
    assert np.isclose(v1.lin_speed_m_s, 1.0 / 30, atol=1e-9)         # first tick: a_max*dt
    v2 = s.check(T_of((0.5, 0, 0.2)), dt_s=1 / 30); assert v2.lin_speed_m_s > v1.lin_speed_m_s


def test_workspace_clamp_and_hold_modes():
    s = ArmSafetyPipeline(cfg(max_lin_vel_m_s=100, max_lin_acc_m_s2=1e6)); s.reset(T_of((0.3, 0, 0.2)))
    v = s.check(T_of((0.9, 0, 0.2)), dt_s=1 / 30)
    assert v.ok and v.clamped and np.isclose(v.T_cmd[0, 3], 0.6) and "workspace_clamped" in v.reasons
    h = ArmSafetyPipeline(cfg(workspace_mode="hold")); h.reset(T_of((0.3, 0, 0.2)))
    v = h.check(T_of((0.9, 0, 0.2)), dt_s=1 / 30)
    assert not v.ok and np.allclose(v.T_cmd, T_of((0.3, 0, 0.2))) and "workspace_violation" in v.reasons


def test_stale_robot_observation_holds():
    s = ArmSafetyPipeline(cfg()); s.reset(T_of((0.3, 0, 0.2)))
    v = s.check(T_of((0.35, 0, 0.2)), dt_s=1 / 30, robot_obs_age_ms=500)
    assert not v.ok and any(r.startswith("robot_obs_stale") for r in v.reasons) and np.allclose(v.T_cmd, T_of((0.3, 0, 0.2)))


def test_check_before_reset_raises():
    with pytest.raises(RuntimeError):
        ArmSafetyPipeline(cfg()).check(T_of(), dt_s=0.03)
