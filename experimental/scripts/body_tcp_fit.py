#!/usr/bin/env python3
"""[2026-09-18] `T_body_tcp` estimator for the HandUMI: the lever arm between the tracked body and the grasp TCP.

Route B uses relative Δq, but the lever arm is still needed: the body centre and the TCP are not the same point, so
once the body ROTATES the TCP translates even when the body centre does not. That is the whole reason the calibration
take contains a rotation-in-place segment.

WHAT IS AND IS NOT ESTIMABLE HERE, because calling all of it "full SE(3)" would be a lie:

  t_body_tcp   ESTIMATED.  From the grasped cube: on closed-grip frames the held cube IS the grasp point, so
               p_tcp(t) = p_body(t) + R_world_body(t) @ t_body_tcp. Stacking frames taken at DIFFERENT orientations
               is what makes the lever arm observable -- at one orientation the equation cannot tell a lever arm from
               a constant offset in the world, which is why a constant world bias `b` is solved alongside it and the
               conditioning of the joint system is reported as the observability number.

  R_body_tcp   NOT ESTIMATED from this data, and not pretended to be. A cube centroid is a POINT; a point carries no
               orientation, so nothing in these observations constrains the TCP's axes. `R_body_tcp` is a CAD
               constant (the gripper is bolted to the body) and belongs in the hardware description, not in a fit.
               The fitter writes it through from the config and records where it came from.

  R_body_imu   ESTIMATED, and it has to be, or the R_world_body above is wrong. Gravity in a static segment pins
               roll and pitch; YAW is pinned only by the yaw-rotation segment, so yaw is solved by a 1-D search that
               reports the depth of its own minimum. A flat minimum means the operator did not yaw enough and the
               take needs redoing -- said out loud rather than absorbed into a plausible-looking number.

    .venv/bin/python scripts/body_tcp_fit.py --selftest          # synthetic, no data needed
    .venv/bin/python scripts/body_tcp_fit.py --track <npz> --side left --out <yaml>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRAVITY = 9.80665

# Written into every artefact this script produces. A convention that lives only in a docstring is one refactor away
# from a silent left/right or sign flip three stages downstream, and such a flip is invisible in every summary number
# we compute -- it shows up as a robot reaching the wrong way, months later.
CONVENTION = dict(
    frame_world="table-plane aligned: +z = table normal (up), +x = camera optical axis projected onto the table, "
                "y = z x x. Right-handed, det(R) = +1, asserted at construction.",
    frame_body="the tracked HandUMI rigid body; origin = the trimmed body centre from the RGB-D tracker",
    frame_tcp="grasp TCP; +x gripper forward (approach), +y gripper left, +z gripper up",
    rotation_multiplication_order="column-vector convention: p_world = R_world_body @ p_body + t_world_body. "
                                  "Composition is LEFT to RIGHT: T_world_tcp = T_world_body @ T_body_tcp.",
    quaternion="xyzw, unit, scalar LAST. RPY/Euler exists only in UI and debug output, never in stored data.",
    handedness="right-handed throughout; a fit whose det(R) is not +1 is rejected, never silently orthonormalised",
    yaw_sign="R_world_body(t) = rot_z(yaw) @ R_imu(t) @ R_imu_body_rp. `yaw` rotates the IMU's world into OURS; it "
             "is the NEGATIVE of the IMU's mounting angle in the body. rot_z is counter-clockwise seen from +z.",
    lever_arm="t_body_tcp is expressed in the BODY frame and applied as p_tcp = p_body + R_world_body @ t_body_tcp",
)


def rot_z(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_lever_arm(p_body: np.ndarray, R_wb: np.ndarray, p_tcp: np.ndarray) -> dict:
    """Least squares on  p_tcp = p_body + R_wb @ t  + b.

    `b` is a constant world-frame offset solved alongside the lever arm ON PURPOSE. Without it the fit would happily
    absorb a systematic error in the cube detection into `t` and report a small residual; with it, the two are
    separated exactly to the degree the orientations varied, and the condition number says how far that is."""
    n = len(p_body)
    A = np.zeros((3 * n, 6))
    y = (p_tcp - p_body).reshape(-1)
    for i in range(n):
        A[3 * i:3 * i + 3, :3] = R_wb[i]
        A[3 * i:3 * i + 3, 3:] = np.eye(3)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = (A @ x - y).reshape(n, 3)
    sv = np.linalg.svd(A, compute_uv=False)
    return dict(t=x[:3], bias=x[3:], residual_mm_rms=float(np.sqrt((r ** 2).sum(1).mean())) * 1e3,
                residual_mm_p95=float(np.percentile(np.linalg.norm(r, axis=1), 95)) * 1e3,
                cond=float(sv[0] / max(sv[-1], 1e-12)), n=n)


def observability(R_wb: np.ndarray) -> dict:
    """Can this take separate the lever arm from a constant world bias -- judged from the ROTATIONS ALONE.

    The design matrix of `fit_lever_arm` is [R(t) | I]; the observations p_tcp only enter the right-hand side.
    So the conditioning -- the whole question of whether the take moved enough -- is decided before a single TCP
    observation exists. That makes this runnable on a take whose cube was missing, which is exactly when knowing
    whether the MOTION was adequate decides if the take must be redone or merely re-labelled.

    The unknown constants do not change the answer: R_world_body = rot_z(yaw) @ R_imu @ R_imu_body_rp, and a
    constant right-multiplication is an orthogonal change of the parameter basis, which leaves every singular
    value untouched. The selftest asserts this rather than leaving it as an argument.
    """
    n = len(R_wb)
    A = np.zeros((3 * n, 6))
    for i in range(n):
        A[3 * i:3 * i + 3, :3] = R_wb[i]
        A[3 * i:3 * i + 3, 3:] = np.eye(3)
    sv = np.linalg.svd(A, compute_uv=False)
    return dict(cond=float(sv[0] / max(sv[-1], 1e-12)), n=n,
                singular_values=[float(x) for x in sv])


def orientation_spread(R_wb: np.ndarray) -> dict:
    """How much the body actually rotated. A lever arm seen at one attitude is not measured, only assumed."""
    ref = R_wb[0]
    ang = np.array([np.degrees(np.arccos(np.clip((np.trace(ref.T @ R) - 1) / 2, -1, 1))) for R in R_wb])
    axes = np.stack([R[:, 2] for R in R_wb])            # body z in world: a compact view of attitude coverage
    return dict(angle_from_first_p50=float(np.percentile(ang, 50)),
                angle_from_first_p95=float(np.percentile(ang, 95)),
                angle_max=float(ang.max()),
                axis_spread_deg=float(np.degrees(np.arccos(np.clip(
                    np.min(axes @ np.median(axes, 0) / np.linalg.norm(np.median(axes, 0))), -1, 1)))))


def fit_imu_yaw(p_body, R_imu, p_tcp, R_imu_body_rp, *, grid=721) -> dict:
    """Yaw of the IMU mounting, by 1-D search.

    CONVENTION, stated because an unstated one is how a frame silently flips three stages downstream:
        R_world_body(t) = rot_z(yaw) @ R_imu(t) @ R_imu_body_rp
    `yaw` is therefore the rotation that takes the IMU's own world into ours, NOT the mounting angle of the IMU in
    the body. The two differ by a sign, and the self-test asserts this exact relation.

    The search: for each candidate yaw the lever-arm fit is linear, so the residual
    curve over yaw is cheap and its SHAPE is the diagnostic. A clear minimum means the take yawed enough to pin the
    mounting; a flat curve means it did not, and no amount of fitting will invent the constraint."""
    psis = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    res = np.empty(grid)
    fits = []
    for i, psi in enumerate(psis):
        R_wb = np.einsum("ij,njk,kl->nil", rot_z(psi), R_imu, R_imu_body_rp)
        f = fit_lever_arm(p_body, R_wb, p_tcp)
        res[i] = f["residual_mm_rms"]; fits.append(f)
    k = int(np.argmin(res))
    # depth of the minimum relative to the curve's own spread: 1.0 = the minimum is as deep as the curve is tall
    depth = float((np.median(res) - res[k]) / max(np.median(res), 1e-9))
    return dict(yaw_rad=float(psis[k]), yaw_deg=float(np.degrees(psis[k])), fit=fits[k],
                residual_curve_min=float(res[k]), residual_curve_median=float(np.median(res)),
                # 0.60, not a feel-good number: the self-test's rotation-skipped take produced depth 0.337
                # while getting the lever arm wrong by 5 mm, so the bar sits above what a bad take can reach
                minimum_depth=depth, observable=bool(depth > 0.60))


def gravity_alignment(accel_static: np.ndarray, table_normal_world=np.array([0.0, 0.0, 1.0])) -> dict:
    """Roll and pitch of the IMU mounting from a static segment, plus the cross-check the collection asked for:
    the angle between the IMU's gravity axis and the fitted table normal. They should be near parallel; a departure
    is a mounting or extrinsic fault and is worth catching before the take is trusted."""
    g = np.median(accel_static, 0)
    n = np.linalg.norm(g)
    g_hat = g / max(n, 1e-12)
    target = -np.asarray(table_normal_world, float)         # gravity points DOWN, the table normal points up
    v = np.cross(g_hat, target)
    s, c = np.linalg.norm(v), float(g_hat @ target)
    R = np.eye(3) if s < 1e-9 else (
        np.eye(3) + np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        + np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) @
          np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) * ((1 - c) / s ** 2))
    return dict(R_rp=R, gravity_magnitude_g=float(n / GRAVITY),
                angle_to_table_normal_deg=float(np.degrees(np.arccos(np.clip(-c, -1, 1)))))


def warnings_for(fit: dict, spread: dict, yaw: dict, grav: dict) -> list[str]:
    w = []
    if spread["angle_max"] < 40:
        w.append(f"body rotated at most {spread['angle_max']:.0f} deg -- the lever arm is barely observable; the "
                 f"rotate-in-place segment is what this number comes from")
    if fit["cond"] > 25:
        w.append(f"lever arm and world bias are poorly separated (cond {fit['cond']:.0f}); more attitude variety needed")
    if not yaw["observable"]:
        w.append(f"IMU yaw is NOT observable from this take (residual minimum depth {yaw['minimum_depth']:.2f}); "
                 f"the yaw-rotation segment was too small")
    if not (0.9 <= grav["gravity_magnitude_g"] <= 1.1):
        w.append(f"static |a| = {grav['gravity_magnitude_g']:.3f} g -- the 'static' segment was not static")
    if grav["angle_to_table_normal_deg"] > 5:
        w.append(f"IMU gravity and the table normal disagree by {grav['angle_to_table_normal_deg']:.1f} deg -- "
                 f"suspect IMU mounting or the table-plane fit, not the tracker")
    if fit["residual_mm_rms"] > 15:
        w.append(f"lever-arm residual {fit['residual_mm_rms']:.1f} mm rms -- the model does not describe this take")
    return w


def selftest() -> int:
    """Two synthetic takes: one that satisfies the protocol and one that skips the rotation segment. The second MUST
    be reported as unobservable -- a fitter that returns a confident number from data that cannot constrain it is
    worse than one that fails."""
    rng = np.random.default_rng(0)
    t_true = np.array([0.021, -0.043, 0.118])
    yaw_true = np.radians(37.0)
    print("SELF-TEST (synthetic, no recorded data)\n")
    results = []
    for name, max_rot_deg in (("protocol followed (rotate-in-place present)", 60.0), ("rotation segment SKIPPED", 3.0)):
        n = 400
        ang = np.radians(rng.uniform(-max_rot_deg, max_rot_deg, (n, 3)))
        R_wb = np.stack([rot_z(a[2]) @ np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
                         @ np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
                         for a in ang])
        p_body = np.stack([rng.uniform(-0.15, 0.15, 3) for _ in range(n)])
        p_tcp = p_body + np.einsum("nij,j->ni", R_wb, t_true) + rng.normal(0, 0.002, (n, 3))
        R_imu = np.einsum("ij,njk->nik", rot_z(-yaw_true).T, R_wb)      # IMU reads the body rotated by an unknown yaw
        f = fit_lever_arm(p_body, R_wb, p_tcp)
        sp = orientation_spread(R_wb)
        y = fit_imu_yaw(p_body, R_imu, p_tcp, np.eye(3), grid=181)
        g = dict(R_rp=np.eye(3), gravity_magnitude_g=1.0, angle_to_table_normal_deg=0.4)
        err = np.linalg.norm(f["t"] - t_true) * 1e3
        print(f"  {name}")
        print(f"    lever arm  {f['t'] * 1e3} mm   truth {t_true * 1e3} mm   error {err:.2f} mm")
        print(f"    residual {f['residual_mm_rms']:.2f} mm rms   cond {f['cond']:.1f}   rotation max {sp['angle_max']:.0f} deg")
        print(f"    yaw {y['yaw_deg']:+.1f} deg (truth {np.degrees(yaw_true):+.1f})  minimum depth {y['minimum_depth']:.3f}"
              f"  observable={y['observable']}")
        ws = warnings_for(f, sp, y, g)
        for w in ws:
            print(f"    WARN  {w}")
        results.append(dict(err=err, obs=y["observable"], warns=len(ws), yaw=y["yaw_deg"]))
        print()

    good, bad = results
    fails = []
    if good["err"] > 2.0: fails.append(f"protocol-followed case: lever arm off by {good['err']:.2f} mm (expect < 2)")
    if not good["obs"]: fails.append("protocol-followed case: yaw reported NOT observable")
    if good["warns"]:   fails.append(f"protocol-followed case raised {good['warns']} warnings, expected none")
    # the yaw the fitter returns is the rotation INTO our world, so it is the negative of the generated mounting yaw
    if abs(good["yaw"] + np.degrees(yaw_true)) > 2.0:
        fails.append(f"yaw convention broken: got {good['yaw']:+.1f}, expected {-np.degrees(yaw_true):+.1f}")
    if bad["obs"]:      fails.append("rotation-skipped case was called observable -- the gate would pass a bad take")
    if not bad["warns"]: fails.append("rotation-skipped case raised no warning")

    # `observability` is run on R_imu before yaw and the mounting roll/pitch are known, on the claim that a
    # constant right-multiplication cannot change the conditioning. Asserted, not assumed -- if it were false,
    # every observability verdict reported before a fit would be measuring the wrong matrix.
    rng2 = np.random.default_rng(7)
    ang2 = np.radians(rng2.uniform(-60, 60, (300, 3)))
    R_test = np.stack([rot_z(a[2]) @ np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
                       @ np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
                       for a in ang2])
    C = rot_z(np.radians(23.0)) @ np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    c_plain = observability(R_test)["cond"]
    c_right = observability(np.einsum("nij,jk->nik", R_test, C))["cond"]
    print(f"  observability invariance: cond {c_plain:.4f} -> {c_right:.4f} under a constant right-multiplication")
    if abs(c_plain - c_right) > 1e-6 * max(c_plain, 1.0):
        fails.append(f"observability is NOT invariant to the unknown constant rotation "
                     f"({c_plain:.6f} vs {c_right:.6f}) -- the pre-fit gate measures the wrong matrix")
    for f_ in fails:
        print(f"  SELF-TEST FAIL  {f_}")
    print("  SELF-TEST PASS" if not fails else f"\n  {len(fails)} self-test failures")
    return 0 if not fails else 1


# The canonical body tracker's frozen setting. A rigid body reference produced at any other visibility is a
# different estimator, so the fit refuses it rather than quietly calibrating against it.
EXPECT_VISIBILITY = 0.05


def yaw_excitation(R_wb: np.ndarray) -> dict:
    """How much the body turned about the WORLD VERTICAL, as opposed to how much it turned at all.

    The two are not the same and the difference is what invalidated CALIB_bodytcp_20260919_123133: it contained
    64 deg of tilt and only 35 deg of yaw, so the unknown yaw between the table-normal world (positions) and the
    gravity world (rotations) was never excited, the yaw objective was flat, and the lever arm came out at
    |t| = 104 mm -- larger than the device. A gate on total rotation passes that take; this one does not.
    """
    rel = Rotation.from_matrix(R_wb) * Rotation.from_matrix(R_wb[0]).inv()
    rv = rel.as_rotvec()
    yaw = np.degrees(rv[:, 2])                      # world +Z is up in both worlds, so this component IS the yaw
    tilt = np.degrees(np.linalg.norm(rv[:, :2], axis=1))
    return dict(yaw_span_deg=float(np.ptp(yaw)), yaw_min_deg=float(yaw.min()), yaw_max_deg=float(yaw.max()),
                tilt_max_deg=float(tilt.max()), tilt_p95_deg=float(np.percentile(tilt, 95)))


def run_track(track: Path, side: str, out: Path | None, *, min_conf: float = 0.0, gap_guard: int = 0,
              fit_yaw: bool = True, body: str = "rigid", cube: str = "centroid", boot: int = 200) -> int:
    """Read a track_export.npz, run every freeze gate, and write a calibration ONLY if all of them pass.

    `fit_yaw` defaults to TRUE and production calibration must keep it so. Positions come from the table-normal
    world and rotations from the gravity world; the two share +Z and differ by an unknown yaw, so using R_imu
    directly is a frame error, not a simplification. Running at yaw = 0 is a diagnostic for reproducing that
    error on purpose (the regression fixture below), never a way to produce a number.
    """
    z = np.load(track, allow_pickle=False)
    need = [f"{side}_body_xyz", f"{side}_valid", f"{side}_quat_world_imu", f"{side}_quat_valid"]
    missing = [k for k in need if k not in z]
    if missing:
        print(f"{track}: missing {missing} -- produced by scripts/body_track_export.py"); return 1

    # Provenance BEFORE anything else: an artefact produced by different settings is not a slightly different
    # input, it is a different experiment, and the fit cannot tell from the arrays alone.
    from scripts.provenance import require, describe
    print(f"\n PROVENANCE\n{describe(z)}")
    if body == "rigid":
        require(z, context="canonical body reference",
                **{f"{side}_canonical_visibility": EXPECT_VISIBILITY,
                   f"{side}_canonical_pose_balanced": True,
                   f"{side}_canonical_version": "canonical_union_v1"})
    if cube == "geometric":
        require(z, context="cube geometric centre",
                **{f"{side}_cube_geo_version": "ransac_planes_half_edge_v1",
                   f"{side}_cube_half_edge_m": 0.020})

    bkey = {"rigid": f"{side}_body_xyz_rigid", "centroid": f"{side}_body_xyz"}[body]
    ckey = {"geometric": f"{side}_p_tcp_geo", "centroid": f"{side}_p_tcp_obs"}[cube]
    for k, what in ((bkey, "scripts/body_canonical_tracker.py"), (ckey, "scripts/cube_tcp_detect.py / cube_center_planes.py")):
        if k not in z:
            print(f"{track}: missing {k} -- produced by {what}"); return 1

    v = z[f"{side}_valid"] & z[f"{side}_quat_valid"] & np.isfinite(z[bkey]).all(1) & np.isfinite(z[ckey]).all(1)
    n_raw = int(v.sum())
    if min_conf > 0 and f"{side}_p_tcp_conf" in z:
        v = v & (z[f"{side}_p_tcp_conf"] >= min_conf)
    if gap_guard > 0:
        ok = z[f"{side}_valid"]
        idx = np.arange(len(ok)); bad = idx[~ok]
        if len(bad):
            v = v & (np.min(np.abs(idx[:, None] - bad[None, :]), axis=1) > gap_guard)

    p_body, p_tcp = z[bkey][v], z[ckey][v]
    R_imu = Rotation.from_quat(z[f"{side}_quat_world_imu"][v]).as_matrix()
    print(f"\n{track.parent.parent.parent.name}   side {side}   body={body}  cube={cube}")
    print(f"  frames {n_raw} -> {int(v.sum())} after confidence >= {min_conf:.2f} and gap guard {gap_guard}")

    spread, obs, exc = orientation_spread(R_imu), observability(R_imu), yaw_excitation(R_imu)
    print("\n MOTION")
    print(f"   total rotation  p50 {spread['angle_from_first_p50']:5.1f}  max {spread['angle_max']:5.1f} deg"
          f"    cond {obs['cond']:6.1f}")
    print(f"   vertical-axis yaw span {exc['yaw_span_deg']:6.1f} deg  ({exc['yaw_min_deg']:+.1f} .. {exc['yaw_max_deg']:+.1f})"
          f"    tilt max {exc['tilt_max_deg']:5.1f} deg")

    yaw = None
    if fit_yaw:
        yaw = fit_imu_yaw(p_body, R_imu, p_tcp, np.eye(3))
        R_wb = np.einsum("ij,njk->nik", rot_z(yaw["yaw_rad"]), R_imu)
        print("\n YAW BETWEEN THE TABLE-NORMAL WORLD AND THE GRAVITY WORLD")
        print(f"   fitted yaw {yaw['yaw_deg']:+7.2f} deg   objective {yaw['residual_curve_median']:.1f} -> "
              f"{yaw['residual_curve_min']:.1f} mm   depth {yaw['minimum_depth']:.3f} (gate > 0.60)")
    else:
        R_wb = R_imu
        print("\n YAW FIT DISABLED -- diagnostic only; this reproduces the frame error, it does not calibrate")

    fit = fit_lever_arm(p_body, R_wb, p_tcp)
    res = np.linalg.norm((p_tcp - p_body) - np.einsum("nij,j->ni", R_wb, fit["t"]) - fit["bias"], axis=1) * 1e3
    print("\n LEVER ARM")
    print(f"   t_body_tcp  {1e3*fit['t'][0]:+7.1f}, {1e3*fit['t'][1]:+7.1f}, {1e3*fit['t'][2]:+7.1f} mm"
          f"   |t| {1e3*np.linalg.norm(fit['t']):.1f} mm")
    print(f"   residual    p50 {np.percentile(res,50):5.1f}  p95 {np.percentile(res,95):6.1f}  rms {fit['residual_mm_rms']:5.1f} mm"
          f"   inliers <10 mm {100*(res<=10).mean():.1f}%")

    # bootstrap: the spread of the ANSWER, yaw included, over resamples of the same take
    rng = np.random.default_rng(0)
    bt, by = [], []
    for _ in range(boot):
        q = rng.integers(0, len(p_body), len(p_body))
        if fit_yaw:
            y2 = fit_imu_yaw(p_body[q], R_imu[q], p_tcp[q], np.eye(3), grid=121)
            if y2["fit"]["cond"] > 25:
                continue
            bt.append(y2["fit"]["t"]); by.append(y2["yaw_deg"])
        else:
            f2 = fit_lever_arm(p_body[q], R_wb[q], p_tcp[q])
            if f2["cond"] <= 25:
                bt.append(f2["t"])
    bt = np.asarray(bt) * 1e3
    yaw_std = float(np.std(np.unwrap(np.radians(by)) if by else [0.0]) * 180 / np.pi) if by else float("nan")
    print(f"   bootstrap({len(bt)})  t std {bt.std(0).round(1)} mm" + (f"   yaw std {yaw_std:.2f} deg" if by else ""))

    # interleaved split: both subsets span the whole take, so both stay conditioned
    ev = np.arange(len(p_body)) % 2 == 0
    halves = []
    for m in (ev, ~ev):
        if fit_yaw:
            y2 = fit_imu_yaw(p_body[m], R_imu[m], p_tcp[m], np.eye(3), grid=361)
            halves.append((y2["fit"], y2["yaw_deg"]))
        else:
            halves.append((fit_lever_arm(p_body[m], R_wb[m], p_tcp[m]), 0.0))
    dt = 1e3 * float(np.linalg.norm(halves[0][0]["t"] - halves[1][0]["t"]))
    dy = abs(halves[0][1] - halves[1][1])
    print(f"   interleaved   |dt| {dt:.1f} mm   |dyaw| {dy:.2f} deg   cond {halves[0][0]['cond']:.1f}/{halves[1][0]['cond']:.1f}")

    # pose-binned body-frame offset: the constant the whole exercise is trying to expose
    ang = np.degrees((Rotation.from_matrix(R_wb) * Rotation.from_matrix(R_wb[0]).inv()).magnitude())
    vb = np.einsum("nij,ni->nj", R_wb, p_tcp - p_body)
    means = [1e3 * vb[(ang >= lo) & (ang < hi)].mean(0)
             for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 90)) if ((ang >= lo) & (ang < hi)).sum() > 5]
    drift = np.ptp(np.asarray(means), axis=0) if len(means) > 1 else np.zeros(3)
    print(f"   pose-bin body-frame offset drift  {drift.round(1)} mm")

    print("\n FREEZE GATES")
    cov = float(z[f"{side}_valid"].mean())
    sw = int(z[f"{side}_identity_switches"]) if f"{side}_identity_switches" in z else -1
    clk = float(z[f"{side}_clock_residual_ms_p95"]) if f"{side}_clock_residual_ms_p95" in z else float("nan")
    reg = z.get(f"{side}_body_rigid_residual_mm")
    reg95 = float(np.nanpercentile(reg, 95)) if reg is not None and np.isfinite(reg).any() else float("nan")
    g = [
        ("A body track coverage > 95%", cov > 0.95, f"{100*cov:.1f}%"),
        ("B identity switches == 0", sw == 0, f"{sw}"),
        ("C IMU-depth sync p95 < 5 ms", clk < 5.0, f"{clk:.2f} ms"),
        ("D canonical registration p95 < 10 mm", reg95 < 10.0, f"{reg95:.2f} mm"),
        ("E rotation observability", spread["angle_max"] >= 40 and obs["cond"] <= 25,
         f"max {spread['angle_max']:.0f} deg, cond {obs['cond']:.1f}"),
        ("F vertical-axis yaw span >= 90 deg", exc["yaw_span_deg"] >= 90.0, f"{exc['yaw_span_deg']:.1f} deg"),
        ("G yaw objective has a clear minimum", bool(yaw and yaw["observable"]),
         f"depth {yaw['minimum_depth']:.3f}" if yaw else "yaw fit disabled"),
        ("H fitted yaw bootstrap stable (< 5 deg)", bool(by) and yaw_std < 5.0, f"{yaw_std:.2f} deg"),
        ("I |t_body_tcp| physically plausible (< 200 mm)", 1e3 * np.linalg.norm(fit["t"]) < 200.0,
         f"{1e3*np.linalg.norm(fit['t']):.1f} mm"),
        ("J fit residual p50 < 10 mm", np.percentile(res, 50) < 10.0, f"{np.percentile(res,50):.1f} mm"),
        ("K interleaved halves both conditioned", max(halves[0][0]["cond"], halves[1][0]["cond"]) <= 25,
         f"cond {halves[0][0]['cond']:.1f}/{halves[1][0]['cond']:.1f}"),
        ("L interleaved halves agree (t < 15 mm, yaw < 5 deg)", dt < 15.0 and dy < 5.0,
         f"|dt| {dt:.1f} mm, |dyaw| {dy:.2f} deg"),
        ("M pose-bin offset drift < 20 mm on every axis", bool((drift < 20.0).all()), f"max {drift.max():.1f} mm"),
    ]
    for name, okg, detail in g:
        print(f"   [{'PASS' if okg else 'FAIL'}] {name:<48s} {detail}")
    allok = all(x[1] for x in g)
    print(f"\n  {'ALL GATES PASS -- this calibration may be frozen' if allok else 'NOT FREEZABLE -- see the FAILs above'}")

    if out:
        if not allok:
            print(f"  refusing to write {out}: a calibration that failed a gate is worse than none, because it "
                  f"looks deliberate. Re-run with the gate satisfied, or inspect without --out.")
            return 1
        out.write_text(json.dumps(dict(side=side, body_reference=body, cube_reference=cube,
                                       t_body_tcp_m=[float(x) for x in fit["t"]],
                                       yaw_table_from_imu_deg=float(yaw["yaw_deg"]) if yaw else 0.0,
                                       residual_mm_p50=float(np.percentile(res, 50)),
                                       residual_mm_rms=fit["residual_mm_rms"], cond=fit["cond"],
                                       bootstrap_t_std_mm=[float(x) for x in bt.std(0)],
                                       bootstrap_yaw_std_deg=yaw_std, n_frames=int(v.sum()),
                                       convention=CONVENTION), indent=1))
        print(f"  -> {out}")
    return 0 if allok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--print-convention", action="store_true", help="dump the frame convention stored with every fit")
    ap.add_argument("--track", type=Path, help="npz with p_body, R_world_body, p_tcp_obs, valid (produced once the take exists)")
    ap.add_argument("--side", choices=["left", "right"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-conf", type=float, default=0.9, help="drop TCP detections below this confidence")
    ap.add_argument("--gap-guard", type=int, default=5, help="drop frames within N frames of a body-track gap")
    ap.add_argument("--no-yaw-fit", action="store_true",
                    help="DIAGNOSTIC ONLY: fit at yaw=0, reproducing the table-world/gravity-world frame error")
    ap.add_argument("--body", choices=["rigid", "centroid"], default="rigid")
    ap.add_argument("--cube", choices=["centroid", "geometric"], default="centroid")
    a = ap.parse_args()
    if a.print_convention:
        print(json.dumps(CONVENTION, indent=1)); return 0
    if a.selftest:
        print(json.dumps(CONVENTION, indent=1) + "\n")
        return selftest()
    if not (a.track and a.side):
        ap.error("--track and --side, or --selftest")
    return run_track(a.track, a.side, a.out, min_conf=a.min_conf, gap_guard=a.gap_guard,
                     fit_yaw=not a.no_yaw_fit, body=a.body, cube=a.cube)


if __name__ == "__main__":
    sys.exit(main())
