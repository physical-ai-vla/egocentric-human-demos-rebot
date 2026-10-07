"""Aero retargeting backends: the DexPilot port, the semantic-7D baseline, and the command limiter.

The point of these is swappability and safety, not "does the optimiser find a good pose": both backends must accept
the same canonical human hand pose and emit the same 16-joint contract, and nothing may reach the hand that is
outside the joint box, faster than the rate limit, or invented while the hand is not being seen."""
from __future__ import annotations
import numpy as np
import pytest
from ego_teleop.hand3d import aero_mocap as M
from ego_teleop.hand3d.interfaces import HandPoseEstimate, HandPoseHealth
from ego_teleop.retarget.aero_backends import (
    OFFICIAL_SCALE_FACTORS, AeroCommandLimiter, AeroLimiterConfig, AeroTarget,
    DexPilotAeroRetargeter, DexPilotConfig, Semantic7DAeroRetargeter,
)

LOWER, UPPER = np.deg2rad(M.AERO_JOINT_LOWER_DEG), np.deg2rad(M.AERO_JOINT_UPPER_DEG)


def hand_pose(curl: float = 0.3, *, side: str = "right", t_ns: int = 0, metric: bool = True) -> HandPoseEstimate:
    """A real-sized synthetic hand (see tests/teleop/test_hand3d.py) as a canonical HandPoseEstimate."""
    from .test_hand3d import place_in_camera, synthetic_hand
    lm = place_in_camera(synthetic_hand(curl))
    if side == "left": lm = lm * np.array([-1.0, 1.0, 1.0])
    local = M.to_palm_local(lm, side)
    return HandPoseEstimate(t_ns, side, np.zeros((21, 2)), lm, local, np.ones(21, bool), np.full(21, 0.40),
                            np.ones(21), 0.9, HandPoseHealth.OK, "head_rgbd", metric, palm_scale_m=M.palm_scale_m(lm))


@pytest.fixture(scope="module")
def dexpilot():
    return DexPilotAeroRetargeter(DexPilotConfig(side="right"))


def test_dexpilot_emits_16_joints_inside_the_aero_box(dexpilot):
    t = dexpilot.retarget(hand_pose())
    assert t.joints_rad.shape == (16,) and t.method == "dexpilot" and t.pose_source == "head_rgbd"
    assert np.all(t.joints_rad >= LOWER - 1e-12) and np.all(t.joints_rad <= UPPER + 1e-12)
    assert t.raw_rad is not None and t.retarget_done_ns is not None
    assert set(t.to_row()) >= {f"q_{n}_rad" for n in M.AERO_JOINT_NAMES}


def test_dexpilot_reindexes_the_optimiser_output_by_joint_name(dexpilot):
    """dex_retargeting returns its own joint order; mixing it up would move the wrong finger."""
    assert [dexpilot._r.joint_names[i] for i in dexpilot._reindex] == [f"right_{n}" for n in M.AERO_JOINT_NAMES]


def test_scale_factors_match_the_official_arithmetic(dexpilot):
    q = np.full(16, 0.2)
    out = dexpilot.apply_scale_factors(q)
    assert np.isclose(out[0], 0.2 * OFFICIAL_SCALE_FACTORS[0][0] + OFFICIAL_SCALE_FACTORS[0][1])
    assert np.isclose(out[1], 0.2 * 2.0 + np.deg2rad(-60))
    assert np.allclose(out[2:4], 0.2 * 3.5)
    assert np.allclose(out[4:13], 0.2 * 1.15 + np.deg2rad(-10))
    assert np.allclose(out[13:16], 0.2 * 1.2 + np.deg2rad(-10))


def test_official_hacks_are_applied_and_switchable(dexpilot):
    p = hand_pose()
    with_hacks = dexpilot.prepare(p)
    off = DexPilotAeroRetargeter(DexPilotConfig(side="right", apply_hacks=False))
    assert np.allclose(with_hacks - off.prepare(p), np.array([0.0, 0.01, 0.0]), atol=0.021)


def test_retargeting_refuses_incomplete_geometry(dexpilot):
    p = hand_pose(); p.landmarks_local_3d[7] = np.nan
    with pytest.raises(ValueError, match="complete geometry"):
        dexpilot.retarget(p)


def test_retargeting_refuses_the_other_hand(dexpilot):
    with pytest.raises(ValueError, match="left hand pose"):
        dexpilot.retarget(hand_pose(side="left"))


def test_curl_moves_the_fingers_in_the_right_direction(dexpilot):
    """A more curled human hand must not produce a straighter Aero hand."""
    open_q = dexpilot.retarget(hand_pose(curl=0.0)).joints_rad
    closed_q = dexpilot.retarget(hand_pose(curl=1.0)).joints_rad
    assert closed_q[4:16].sum() > open_q[4:16].sum()


def test_position_and_vector_configs_still_build():
    for method in ("position", "vector"):
        r = DexPilotAeroRetargeter(DexPilotConfig(side="right", method=method))
        t = r.retarget(hand_pose())
        assert t.method == method and t.joints_rad.shape == (16,)


def test_semantic7d_backend_consumes_the_same_pose_and_contract():
    t = Semantic7DAeroRetargeter(side="right").retarget(hand_pose())
    assert t.joints_rad.shape == (16,) and t.method == "semantic7d"
    assert np.all(t.joints_rad >= LOWER - 1e-9) and np.all(t.joints_rad <= UPPER + 1e-9)


def test_backends_are_interchangeable(dexpilot):
    p = hand_pose()
    for backend in (dexpilot, Semantic7DAeroRetargeter(side="right")):
        t = backend.retarget(p)
        assert isinstance(t, AeroTarget) and t.joints_rad.shape == (16,)


# ---- limiter ------------------------------------------------------------------------------------------------
def _target(q, t_ns=0):
    return AeroTarget(t_ns, np.asarray(q, np.float64), "dexpilot", "head_rgbd")


def test_limiter_rate_limits_a_jump():
    lim = AeroCommandLimiter(AeroLimiterConfig(max_joint_rate_rad_s=1.0, max_step_rad=1.0))
    lim.step(_target(np.zeros(16)), 0)
    out = lim.step(_target(np.full(16, 1.5)), 100_000_000)          # 100 ms -> 0.1 rad allowed
    assert np.allclose(out.joints_rad, 0.1) and out.source == "rate_limited" and lim.rate_limited == 1


def test_limiter_holds_the_last_command_and_never_opens_the_hand():
    lim = AeroCommandLimiter()
    first = lim.step(_target(np.full(16, 0.4)), 0)
    held = lim.step(None, 100_000_000)
    assert np.allclose(held.joints_rad, first.joints_rad) and held.source == "hold"
    assert not np.allclose(held.joints_rad, 0.0)                    # holding, not relaxing to open
    assert lim.holds == 1


def test_limiter_watchdog_flags_a_stale_hold():
    lim = AeroCommandLimiter(AeroLimiterConfig(command_timeout_ms=50))
    lim.step(_target(np.full(16, 0.2)), 0)
    assert lim.step(None, 500_000_000).diagnostics["watchdog"] is True


def test_limiter_has_nothing_to_hold_before_the_first_command():
    assert AeroCommandLimiter().step(None, 0) is None


def test_limiter_clips_to_the_joint_box():
    lim = AeroCommandLimiter(AeroLimiterConfig(max_joint_rate_rad_s=1e6, max_step_rad=1e6))
    out = lim.step(_target(np.full(16, 10.0)), 0)
    assert np.allclose(out.joints_rad, UPPER)
