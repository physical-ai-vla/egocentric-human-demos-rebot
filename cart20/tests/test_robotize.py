"""[2026-10-16] ego -> reBot robotization math (no IK): modes, frames, orientation policies, scaling, line-search projection."""
import pathlib, sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ego_cart20.robotize import core  # noqa: E402

rng = np.random.default_rng(0)


def traj(n=60, p0=(0.0, 0.0, 0.1), seed=0):
    r = np.random.default_rng(seed)
    p = np.array(p0) + np.cumsum(r.normal(0, 0.004, (n, 3)), 0)
    R = Rot.from_rotvec(np.cumsum(r.normal(0, 0.02, (n, 3)), 0)).as_matrix() @ Rot.from_euler("y", 60, degrees=True).as_matrix()
    return core.make_T(R, p)


def setup(n=60):
    TH = {"left": traj(n, (0.1, 0.25, 0.12), 1), "right": traj(n, (0.1, -0.2, 0.10), 2)}
    T_task_H = core.task_frame(TH["left"][0, :3, 3], TH["right"][0, :3, 3], 0.0)
    T_R0 = {"left": core.make_T(np.eye(3), (0.28, 0.22, 0.15)), "right": core.make_T(np.eye(3), (0.28, -0.22, 0.15))}
    T_task_R = core.task_frame(T_R0["left"][:3, 3], T_R0["right"][:3, 3], -0.054)
    return TH, T_R0, T_task_H, T_task_R


def test_task_frame_axes():
    T = core.task_frame((0, 0.2, 0.3), (0, -0.2, 0.1), 0.0)
    assert np.allclose(T[:3, 3], [0, 0, 0]) and np.allclose(T[:3, 1], [0, 1, 0]) and np.allclose(T[:3, 2], [0, 0, 1]) and np.allclose(T[:3, 0], [1, 0, 0])


def test_mode_b_full_preserves_lr_relative_pose_and_heights():
    TH, T_R0, Th, Tr = setup(); rz = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="B"))
    e = core.rel_pose_err(TH["left"], TH["right"], rz["raw"]["left"], rz["raw"]["right"])
    assert e["pos_p95_mm"] < 1e-6 and e["rot_p95_deg"] < 1e-6
    # heights above the table are kept (human table z = 0, robot table z = -0.054)
    assert np.allclose(rz["raw"]["left"][:, 2, 3] + 0.054, TH["left"][:, 2, 3])


def test_mode_a_reapplies_start_relative_motion_on_robot_start():
    TH, T_R0, Th, Tr = setup(); rz = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="A"))
    for s in TH:
        assert np.allclose(rz["raw"][s][0], T_R0[s])
        lhs = np.einsum("ij,njk->nik", core.inv_T(rz["raw"][s][0]), rz["raw"][s]); rhs = rz["delta"][s]
        assert np.allclose(lhs, rhs, atol=1e-9)                                  # dT robot == dT human (start frame)


def test_r_map_applied_once():
    TH, T_R0, Th, Tr = setup(); M = Rot.from_euler("y", -11, degrees=True).as_matrix()
    a = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="B")); b = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="B", R_map=M))
    assert np.allclose(np.einsum("nij,jk->nik", a["raw"]["left"][:, :3, :3], M), b["raw"]["left"][:, :3, :3])
    A = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="A", R_map=M))
    assert np.allclose(A["raw"]["left"][0], T_R0["left"])                         # A starts exactly at the robot start


def test_task_aligned_drops_tilt_keeps_yaw():
    TH, T_R0, Th, Tr = setup(); rz = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="B", orientation="TASK_ALIGNED"))
    R = rz["raw"]["left"][:, :3, :3]; D = np.einsum("nij,kj->nik", R, R[0])
    assert np.allclose(D[:, :3, 2], [0, 0, 1], atol=1e-9)                      # pure rotation about z
    assert np.max(rz["clamp"]["left"]) > 1.0


def test_yaw_dominant_clamps_tilt():
    TH, T_R0, Th, Tr = setup(); rz = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="B", orientation="YAW_DOMINANT", max_tilt_deg=5))
    R = rz["raw"]["right"][:, :3, :3]; D = np.einsum("nij,kj->nik", R, R[0])
    tilt = [core.rot_angle_deg(core.swing_twist_z(d)[1]) for d in D]
    assert max(tilt) <= 5.0 + 1e-6


def test_scale_acts_on_task_axes():
    n = 20; TH = {s: np.repeat(core.make_T(np.eye(3), p)[None], n, 0) for s, p in (("left", (0, 0.2, 0.1)), ("right", (0, -0.2, 0.1)))}
    for s in TH: TH[s][:, 0, 3] += np.linspace(0, 0.1, n)                        # both hands move 10 cm forward (task x)
    Th = core.task_frame(TH["left"][0, :3, 3], TH["right"][0, :3, 3]); _, T_R0, _, Tr = setup()
    rz = core.robotize(TH, T_R0, Th, Tr, core.RobotizeCfg(mode="B", scale_xyz=(0.9, 1, 1)))
    d = rz["raw"]["left"][-1, :3, 3] - rz["raw"]["left"][0, :3, 3]
    assert np.allclose(d, [0.09, 0, 0], atol=1e-9)


def test_project_line_search_keeps_feasible_and_shared_alpha():
    n = 30; raw = {s: np.repeat(core.make_T(np.eye(3), (0.3, y, 0.1))[None], n, 0) for s, y in (("left", 0.2), ("right", -0.2))}
    raw["left"][10:20, 2, 3] = -0.2                                              # dips below the table
    chk = lambda TL, TR, st: (min(TL[2, 3], TR[2, 3]) >= 0.0, {"reason": None if min(TL[2, 3], TR[2, 3]) >= 0 else "table"}, (st or 0) + 1)
    pr = core.project(raw, chk, core.RobotizeCfg(mode="C"))
    assert np.all(pr["feasible"]["left"][:, 2, 3] >= 0.0) and pr["alpha"][12] < 1.0 and pr["alpha"][25] == 1.0
    assert pr["ok"].all()


def test_safety_numbers_match_deployment():
    sys.path.insert(0, str(pathlib.Path.home() / "laundry_system"))
    try:
        from deployment.run_rebot import rebot_safety
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"deployment not importable here: {ex}")
    from ego_cart20.robotize.rebot import SAFETY
    c = rebot_safety(0.05)
    assert c.table_z_m == SAFETY["table_z_m"] and c.table_margin_m == SAFETY["table_margin_m"] and c.min_bimanual_dist_m == SAFETY["min_bimanual_dist_m"]
    for k, kk in (("left", "L"), ("right", "R")):
        assert np.allclose(c.workspace_min[kk], SAFETY["workspace_min"][k]) and np.allclose(c.workspace_max[kk], SAFETY["workspace_max"][k])
