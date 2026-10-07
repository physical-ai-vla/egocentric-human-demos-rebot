"""P4-P7 of the multi-sensor ladder: chest RGB-D + wrist Arducam + wrist IMU -> fused wrist pose -> reBot.

    --stage compare   P4  both branches side by side. No fusion output is used for anything, no robot.
                          Reads out: RGB-D <-> VI discrepancy, relative drift, jitter, dropouts, the alignment.
    --stage fused     P5  the fused pose drives a VIRTUAL TCP through the real mapper/retargeter/safety. No robot.
    --stage virtual   P6  the full coordinator state machine + MockRebotController: engage, clutch, dropouts,
                          re-engage, map corrections. Still no physical robot.
    --stage real      P7  the same, with the HTTP reBot client. Refuses to start without --i-am-at-the-robot.

    --ablation        run the take three times -- rgbd_only (A), vi_only (B), fused (C) -- and print the comparison
                      table of spec section 21. Fusion is NOT assumed to win; the table is how you find out.

Inputs (one of):
    --episode DIR --vi-poses PARQUET    a recorded take: DIR/head/{color,depth} for the chest camera, and the wrist
                                        VI trajectory the offline pipeline already produced
                                        (derived/pose_<backend>/<side>_camera_pose.parquet, i.e. `m1_vio` output).
    --synthetic                         a generated take with a known answer: plumbing / CI, never evidence.
    --live                              the three real sensors (ego_teleop.tracking.fused_live).

    .venv/bin/python -m ego_teleop.tools.f4_fusion --stage compare --episode datasets/fusion/ep001 \
        --vi-poses datasets/fusion/ep001/derived/pose_openvins/right_camera_pose.parquet --protocol --ablation

Every stage writes ONE report json and (with --record) the fused stream into an episode. The three headline numbers,
printed first: stationary XYZ jitter (p95, mm) | return-to-start error (mm) | tracking-loss rate (%)."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from handumi_collector.pose.estimator import PoseEstimate, TrackingState
from ..config import FusedWristCfg, TeleopCfg, load_teleop_cfg
from ..hand3d.head_camera import RecordedRgbdSource
from ..recorder.episode_logger import TeleopEpisodeLogger
from ..retarget.aero_retarget import AeroRetargeter
from ..retarget.arm_relative_se3 import RelativeSE3Retargeter
from ..robot.coordinator import CoordinatorConfig, Readiness, TeleopCoordinator
from ..robot.rebot_client import HttpRebotClient, JointLimits, MockRebotController, PyrokiIkSolver
from ..robot.safety import ArmSafetyPipeline, SafetyConfig
from ..tracking.fused_wrist import FusedWristPoseProvider, FusionMode
from ..tracking.interfaces import TrackingHealth, WristPose
from ..tracking.vio_stable import VioStableGate
from ..tracking.wrist_pose_provider import EstimatorWristPoseProvider, TrackingSupervisor
from ..transforms.frames import HumanRobotFrameMapper
from ..transforms.se3 import make_T, pose7_to_T
from .a1_hand3d import _dist
from .p1_rgbd_arm import PROTOCOL_60S, _gate, _seconds, _window, still_windows

ABLATIONS = ("rgbd_only", "vi_only", "fused")       # A, B, C of spec section 21
JUMP_MM = 30.0
MOCK_TCP = (0.35, 0.0, 0.25)                         # a plausible reBot TCP for the virtual stages                                       # a fused-position step above this is "catastrophic", not motion


# ---- the wrist VI branch, replayed from a trajectory the offline pipeline already produced ----------------------
class _ReplayEstimator:
    """Placeholder so `EstimatorWristPoseProvider` can be reused for grading a replayed trajectory: the supervisor,
    the T_H_C transform and the health policy must be the SAME code the live path runs."""
    class info: name = "replay"
    def get_quality(self) -> dict: return dict(backend="replay")


def vi_provider_from_parquet(path: str | Path, *, T_H_C: np.ndarray, supervisor_cfg) -> tuple[EstimatorWristPoseProvider, pd.DataFrame]:
    df = pd.read_parquet(path)
    need = {"t_ns", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if not need <= set(df.columns): raise SystemExit(f"{path}: expected columns {sorted(need)}, got {sorted(df.columns)}")
    return EstimatorWristPoseProvider(_ReplayEstimator(), T_H_C=T_H_C, supervisor=TrackingSupervisor(supervisor_cfg),
                                      source="vi_replay"), df.sort_values("t_ns").reset_index(drop=True)


def vi_estimates(df: pd.DataFrame):
    """-> (t_ns, PoseEstimate) for every row, keeping whatever quality columns the backend actually filled."""
    has = lambda c: c in df.columns
    for i, r in enumerate(df.itertuples(index=False)):
        d = r._asdict()
        valid = bool(d.get("valid", True)) and np.isfinite([d["x"], d["y"], d["z"]]).all()
        T = pose7_to_T(np.array([d["x"], d["y"], d["z"], d["qx"], d["qy"], d["qz"], d["qw"]], float)) if valid else None
        st = TrackingState(d["tracking_state"]) if has("tracking_state") and isinstance(d.get("tracking_state"), str) \
            else (TrackingState.TRACKING if valid else TrackingState.LOST)
        yield int(d["t_ns"]), PoseEstimate(int(d["t_ns"]), int(d.get("frame_index", i)), T, st,
                                           confidence=d.get("confidence"), num_features=d.get("num_features"),
                                           reprojection_error=d.get("reprojection_error"))


# ---- a synthetic take (plumbing only) ---------------------------------------------------------------------------
def synthetic_take(n: int = 1800, *, hz: float = 30.0, drift_m_s: float = 0.002, lever=(0.02, -0.03, 0.06),
                   drift_from_s: float = 30.0, seed: int = 0):
    """A take with a known answer, following the 60 s protocol of section 20 window for window:

        0-10 still | 10-20 translate | 20-30 ROTATE ONLY | 30-40 both | 40-50 fast | 50-60 return + still

    VI drifts linearly from `drift_from_s` (after the alignment windows, as in a real session); the chest camera does not. The rotation-only window is what makes the
    lever arm identifiable, and the still windows are what make the jitter and return-to-start numbers mean anything
    — which is the whole reason the protocol has that shape. NEVER evidence about the hardware: it exists so the
    pipeline (pairing, alignment, correction, retargeting, safety, report) can be exercised end to end in CI."""
    rng = np.random.default_rng(seed)
    T_D_W = make_T(Rotation.from_euler("xyz", [22.0, -8.0, 155.0], degrees=True).as_matrix(), [0.28, -0.10, 0.50])
    inv = np.linalg.inv(T_D_W)
    p0 = np.array([0.05, 0.02, -0.03])
    AWAY = np.array([0.10, -0.05, 0.04])          # where the fast window leaves the hand, and what the return undoes

    def pose(s: float) -> tuple[np.ndarray, np.ndarray]:
        """True wrist pose at time s, per the protocol window it falls in."""
        rv = np.zeros(3); p = p0.copy()
        if s < 10:                                    # stationary
            pass
        elif s < 20:                                  # slow translation, orientation held
            u = (s - 10) / 10
            p = p0 + np.array([0.14 * np.sin(2 * np.pi * u), 0.10 * np.sin(4 * np.pi * u), 0.06 * np.sin(2 * np.pi * u)])
        elif s < 30:                                  # ROTATION ONLY -- this is what identifies the lever arm
            u = (s - 20) / 10
            rv = np.radians(45.0) * np.array([np.sin(2 * np.pi * u), 0.7 * np.sin(4 * np.pi * u), 0.5 * np.sin(6 * np.pi * u)])
        elif s < 40:                                  # translation + rotation
            u = (s - 30) / 10
            p = p0 + np.array([0.12 * np.sin(2 * np.pi * u), 0.08 * np.sin(4 * np.pi * u), 0.05 * np.sin(6 * np.pi * u)])
            rv = np.radians(30.0) * np.array([np.sin(6 * np.pi * u), 0.6 * np.sin(4 * np.pi * u), 0.4 * np.sin(2 * np.pi * u)])
        elif s < 50:                                  # fast natural reach, ending away from home
            u = (s - 40) / 10
            p = p0 + np.array([0.20 * np.sin(6 * np.pi * u), 0.12 * np.sin(4 * np.pi * u), 0.10 * np.sin(8 * np.pi * u)]) \
                + u * AWAY
            rv = np.radians(40.0) * np.array([np.sin(8 * np.pi * u), 0.5 * np.sin(6 * np.pi * u), 0.6 * np.sin(4 * np.pi * u)])
        else:                                         # return to start, then hold still
            p = p0 + max(1.0 - (s - 50) / 3.0, 0.0) * AWAY
        # every window starts and ends at (p0 + its own end offset) with zero rotation, so the take is continuous:
        # a teleporting hand would show up as a catastrophic jump and the metric would be measuring the fixture
        return p, Rotation.from_rotvec(rv).as_matrix()

    vi_rows, rgbd_rows, truth = [], [], []
    dhat = np.array([1.0, -0.4, 0.6]); dhat = dhat / np.linalg.norm(dhat)
    for i in range(n):
        t = int(i / hz * 1e9); s = i / hz
        p, R = pose(s)
        drift = dhat * drift_m_s * max(s - drift_from_s, 0.0)
        q = Rotation.from_matrix(R).as_quat()
        truth.append(np.concatenate([[t], p]))
        vi_rows.append(dict(t_ns=t, frame_index=i, valid=True, tracking_state="tracking",
                            x=p[0] + drift[0], y=p[1] + drift[1], z=p[2] + drift[2],
                            qx=q[0], qy=q[1], qz=q[2], qw=q[3], confidence=0.9, num_features=200))
        palm_W = p + R @ np.asarray(lever, float)
        rgbd_rows.append((t, inv[:3, :3] @ palm_W + inv[:3, 3] + rng.normal(0, 0.002, 3), 0.9))
    return pd.DataFrame(vi_rows), rgbd_rows, np.asarray(truth), T_D_W


# ---- the run ----------------------------------------------------------------------------------------------------
def _readiness() -> Readiness:
    """The coordinator's flags are named for the production wrist-VIO source; here `camera_online`/`imu_online` mean
    'all three sensors are streaming' and calibration is the fused branch's own. They are set explicitly, never
    defaulted true, so a missing sensor still blocks ARM."""
    return Readiness(camera_online=True, imu_online=True, calibration_loaded=True, robot_observed=True, aero_homed=True)


def run(*, events, cfg: FusedWristCfg, mode: str, stage: str, rgbd_provider=None, vi_provider,
        robot=None, logger: TeleopEpisodeLogger | None = None, engage_at_s: float = 3.0,
        clutch_at_s: float | None = None, release_at_s: float | None = None, tick_hook=None) -> pd.DataFrame:
    """Replay one take (an ordered stream of sensor events) through the fusion stack at `cfg.rate_hz`.

    `events` yields (t_ns, "rgbd"|"vi", payload): an RgbdFrame (or a ready (p_xyz, confidence) pair) and a
    PoseEstimate. One time-ordered stream is the point — the fusion never sees the future of either sensor.

    `tick_hook(row, prov, tick_state)` is called after every fusion tick, for a live observer (`f5_teleop_hud`). It
    is the ONLY thing a HUD may do to this loop: it reads what the tick produced and it may sleep to pace a replay
    to wall clock. It must not touch the provider, and a replay with no hook runs exactly as it did before."""
    import dataclasses as dc
    prov = FusedWristPoseProvider(dc.replace(cfg.provider_config(), mode=mode))
    mapper = HumanRobotFrameMapper(cfg.frames)
    stable = VioStableGate(cfg.stable)
    co = None
    if stage in ("virtual", "real"):
        # a mock TCP inside the configured workspace: the default identity pose sits outside it, and the anchor box
        # would then refuse (correctly) before anything else could be measured
        robot = robot or MockRebotController(T0=make_T(np.eye(3), MOCK_TCP), side=cfg.side)
        T_anchor = robot.observe().T_RB_RE
        safety_cfg = SafetyConfig(**{**asdict(cfg.safety), "workspace": cfg.anchor_workspace(T_anchor)}) if cfg.workspace_from_anchor else cfg.safety
        co = TeleopCoordinator(wrist=prov, hand=None, arm_retargeter=RelativeSE3Retargeter(mapper),
                               aero_retargeter=AeroRetargeter(), safety=ArmSafetyPipeline(safety_cfg),
                               robot=robot, aero=None,
                               cfg=CoordinatorConfig(rate_hz=cfg.rate_hz, hand_enabled=False, arm_enabled=(stage == "real")),
                               readiness=_readiness())
        for ev in ("sensors_ready", "calibrated", "vio_start"): co.fire(ev, t_ns=0)

    dt_ns = int(1e9 / cfg.rate_hz)
    rows: list[dict] = []
    tick_state: dict = {}
    t0 = next_tick = None
    engaged = clutched = False
    for t_ns, kind, payload in events:
        if t0 is None: t0, next_tick = t_ns, t_ns
        if kind == "vi":
            prov.push_vi(vi_provider.ingest_estimate(payload))
        elif kind == "rgbd":
            if tick_hook is not None: tick_state["last_rgbd"] = payload   # the HUD's chest-camera view, when there is one
            if rgbd_provider is not None and not isinstance(payload, tuple):
                prov.push_rgbd(rgbd_provider.push_image(t_ns, getattr(payload, "frame_index", 0), payload))
            else:
                p, conf = payload
                ok = np.all(np.isfinite(p))
                prov.push_rgbd(WristPose.from_T(t_ns, make_T(np.eye(3), p), health=TrackingHealth.OK if ok else TrackingHealth.LOST,
                                                confidence=conf, source="rgbd_hand_palm",
                                                tracking_state=TrackingState.TRACKING if ok else TrackingState.LOST))
        while next_tick <= t_ns:                      # the fusion clock, independent of both sensor clocks
            rows.append(_tick(prov, co, stable, cfg, mapper, next_tick, t0, logger, state=tick_state,
                              engage=(not engaged and (next_tick - t0) / 1e9 >= engage_at_s),
                              clutch=(engaged and not clutched and clutch_at_s is not None and (next_tick - t0) / 1e9 >= clutch_at_s),
                              release=(clutched and release_at_s is not None and (next_tick - t0) / 1e9 >= release_at_s)))
            engaged = engaged or bool(rows[-1].get("engaged"))
            clutched = bool(rows[-1].get("clutch_state", clutched))
            if tick_hook is not None: tick_hook(rows[-1], prov, tick_state)
            next_tick += dt_ns
    df = pd.DataFrame(rows)
    df.attrs["stats"] = prov.stats()
    df.attrs["mode"] = mode
    return df


def _tick(prov, co, stable, cfg, mapper, t_ns, t0, logger, *, engage: bool, clutch: bool, release: bool,
          state: dict | None = None) -> dict:
    t_call = time.monotonic_ns()
    wp = prov.get_pose(t_ns)
    row = dict(t_ns=int(t_ns), s=(t_ns - t0) / 1e9, engaged=False, clutch_state=False,
               fusion_ms=(time.monotonic_ns() - t_call) / 1e6)
    if wp is None:
        row.update(tracking_health=TrackingHealth.LOST.value, fusion_state="INITIALIZING")
        return row
    stable.update(wp)
    p, q = wp.position_xyz_m, wp.quaternion_xyzw
    e = wp.extra
    row.update(tracking_health=wp.health.value, x=float(p[0]), y=float(p[1]), z=float(p[2]),
               qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]),
               vx=float(wp.linear_velocity_xyz[0]), vy=float(wp.linear_velocity_xyz[1]), vz=float(wp.linear_velocity_xyz[2]),
               pose_age_ms=(t_ns - wp.timestamp_ns) / 1e6, stable=bool(stable.stable))
    for k in ("fusion_state", "fusion_reason", "vi_health", "rgbd_health", "anchored", "align_rms_m", "align_pairs",
              "lever_arm_m", "lever_arm_observable", "residual_m", "residual_corrected_m", "correction_m", "correction_pending_m",
              "c_x", "c_y", "c_z", "vi_x", "vi_y", "vi_z", "vi_map_x", "vi_map_y", "vi_map_z",
              "rgbd_x", "rgbd_y", "rgbd_z", "rgbd_w_x", "rgbd_w_y", "rgbd_w_z",
              "map_correction_m", "map_absorbed", "held",
              # the first-hardware-run gate (section 3): the pre-optimisation visual pose and what the backend
              # announced, beside the local pose the robot actually followed
              "visual_raw_x", "visual_raw_y", "visual_raw_z", "d_visual_raw_m", "vi_map_update", "vi_map_update_m",
              "vi_keyframe", "vi_server_ms", "vi_net_ms", "vi_cap_to_send_ms", "vi_link_ms"):
        row[k] = e.get(k)
    if logger is not None: logger.add_fused_wrist_pose(cfg.side, wp)
    if co is None:
        # stage `fused` (P5): no coordinator, no robot — but the human->robot mapping still runs, so the take shows
        # which robot axis each hand motion lands on. This is the table the axis map is read off before P6.
        if state is not None and wp.health != TrackingHealth.LOST:
            if state.get("T_H0") is None and stable.stable: state["T_H0"] = wp.T()
            if state.get("T_H0") is not None:
                from ..transforms.se3 import inv_T
                dH = inv_T(state["T_H0"]) @ wp.T()
                dR = mapper.map_translation(dH[:3, 3])
                for i, a in enumerate("xyz"):
                    row[f"human_d{a}_mm"] = float(dH[i, 3] * 1000.0); row[f"robot_d{a}_mm"] = float(dR[i] * 1000.0)
        return row

    if engage and co.readiness.tracking_valid is not None:
        co.readiness.tracking_valid = bool(wp.valid); co.readiness.vio_stable = bool(stable.stable)
        if wp.valid and stable.stable:
            for ev in ("vio_stable", "robot_ready", "arm", "start"): co.fire(ev, t_ns=t_ns)
            row["engaged"] = True
            if logger is not None: logger.add_event("engage", t_ns, stage="fusion")
    if clutch:
        co.fire("clutch", t_ns=t_ns); prov.set_clutched(True)
        if logger is not None: logger.add_event("clutch", t_ns, stage="fusion")
    if release:
        co.fire("release", t_ns=t_ns); prov.set_clutched(False)
        if logger is not None: logger.add_event("release", t_ns, stage="fusion")
    row["clutch_state"] = prov._clutched
    cmd = co.tick(t_ns)
    if cmd is not None:
        row.update(state=cmd.state.value, hold_reason=cmd.hold_reason, clutch=bool(cmd.clutch),
                   safety=",".join(cmd.safety_reasons), clamped="workspace_clamped" in cmd.safety_reasons,
                   tcp_x=float(cmd.arm_target_xyz[0]), tcp_y=float(cmd.arm_target_xyz[1]), tcp_z=float(cmd.arm_target_xyz[2]),
                   step_mm=float(np.linalg.norm(cmd.arm_action6[:3]) * 1000.0),
                   arm_sent=bool(cmd.arm_sent),
                   exec_latency_ms=cmd.arm_exec.latency_ms() if cmd.arm_exec else np.nan)
        if logger is not None:
            logger.add_command(cmd); logger.add_fused_relative_pose(cfg.side, row)
    return row


# ---- P7 / live ---------------------------------------------------------------------------------------------------
def build_real_robot(teleop: TeleopCfg):
    """The physical arm (stage `real`). Needs the handumi-sw kinematics (pyroki/jax), which lives in ITS interpreter:
    run this tool from that venv, or inject a controller. Nothing is faked — a missing solver raises."""
    r = teleop.rebot
    try:
        ik = PyrokiIkSolver.for_rebot_b601(r.side)
    except ImportError as exc:
        raise SystemExit("--stage real needs the handumi rebot_b601 kinematics (pyroki/jax) for IK/FK: "
                         f"{exc}. Run this tool from ~/xvla-mac/bin/python, or finish at --stage virtual.") from None
    lim = r.joint_limits_rad or {}
    limits = JointLimits(**{k: np.asarray(v, np.float64) for k, v in lim.items()}, max_step_rad=r.joint_max_step_rad)
    return HttpRebotClient(r.base_url, r.side, ik, limits=limits,
                           max_ik_pos_err_m=r.max_ik_pos_err_m, max_ik_rot_err_deg=r.max_ik_rot_err_deg)


def run_live(args, teleop: TeleopCfg, segments: dict, *, tick_hook=None, rig=None) -> int:
    """The three real sensors, at `rate_hz`, through the same tick/report code the replay stages use.

    NOT YET RUN ON THE RIG (the wrist IMU unit is the missing hardware). P0 (`tools/f0_sensors.py`) proves the
    sensors first, through the collector's own device path, which is deliberately not this one."""
    import dataclasses as dc
    from ..tracking.fused_live import FusedLiveRig
    cfg = teleop.fused_wrist
    mode = args.mode or cfg.mode
    rig = rig or FusedLiveRig(cfg=dc.replace(cfg, mode=mode), teleop=teleop, hardware=args.hardware, downscale=args.downscale)
    logger = TeleopEpisodeLogger(args.record, metadata=dict(tool="f4_fusion", stage=args.stage, live=True)) if args.record else None
    robot = build_real_robot(teleop) if args.stage == "real" else None
    rows: list[dict] = []
    with rig as r:
        prov = r.provider
        mapper = HumanRobotFrameMapper(cfg.frames)
        stable = VioStableGate(cfg.stable)
        co = None
        if args.stage in ("virtual", "real"):
            robot = robot or MockRebotController(T0=make_T(np.eye(3), MOCK_TCP), side=cfg.side)
            T_anchor = robot.observe().T_RB_RE
            safety_cfg = SafetyConfig(**{**asdict(cfg.safety), "workspace": cfg.anchor_workspace(T_anchor)}) if cfg.workspace_from_anchor else cfg.safety
            co = TeleopCoordinator(wrist=prov, hand=None, arm_retargeter=RelativeSE3Retargeter(mapper),
                                   aero_retargeter=AeroRetargeter(), safety=ArmSafetyPipeline(safety_cfg),
                                   robot=robot, aero=None,
                                   cfg=CoordinatorConfig(rate_hz=cfg.rate_hz, hand_enabled=False, arm_enabled=(args.stage == "real")),
                                   readiness=_readiness())
            for ev in ("sensors_ready", "calibrated", "vio_start"): co.fire(ev, t_ns=0)
        dt = 1.0 / cfg.rate_hz
        t_start = time.monotonic_ns(); t0 = t_start
        tick_state: dict = {}
        engaged = clutched = False
        try:
            while (time.monotonic_ns() - t_start) / 1e9 < args.live_seconds:
                tick_t = time.monotonic_ns(); s = (tick_t - t0) / 1e9
                row = _tick(prov, co, stable, cfg, mapper, tick_t, t0, logger, state=tick_state,
                            engage=(not engaged and s >= args.engage_at),
                            clutch=(engaged and not clutched and args.clutch_at is not None and s >= args.clutch_at),
                            release=(clutched and args.release_at is not None and s >= args.release_at))
                engaged = engaged or bool(row.get("engaged")); clutched = bool(row.get("clutch_state", clutched))
                rows.append(row)
                if tick_hook is not None:
                    tick_state["last_rgbd"] = r.last_rgbd_frame
                    tick_state["last_wrist"] = r.last_wrist_frame
                    tick_hook(row, prov, tick_state)
                slack = dt - (time.monotonic_ns() - tick_t) / 1e9
                if slack > 0: time.sleep(slack)
                else: row["overrun_ms"] = -slack * 1000.0
        except KeyboardInterrupt:
            print("\ninterrupted — stopping")
        finally:
            if co is not None and args.stage == "real": co.fire("estop", t_ns=time.monotonic_ns())
        health = r.health()
    df = pd.DataFrame(rows); df.attrs["stats"] = health.get("fusion", {})
    rep = report(df, cfg, mode, args.stage, segments); rep["live"] = health
    print_report(rep)
    out = args.out or ((Path(args.record) / "fusion_report.json") if args.record else Path("fusion_report.json"))
    out.write_text(json.dumps(dict(reports={mode: rep}), indent=1, default=str))
    if args.record: df.to_parquet(Path(args.record) / f"fusion_ticks_{mode}.parquet", index=False)
    if logger is not None: logger.close()
    print(f"\nreport -> {out}")
    return 0


# ---- metrics (spec section 20) -----------------------------------------------------------------------------------
def _xyz(df: pd.DataFrame) -> np.ndarray:
    if df.empty or "x" not in df: return np.zeros((0, 3))
    P = df[["x", "y", "z"]].to_numpy(np.float64)
    return P[np.isfinite(P).all(1)]


def stationary(df: pd.DataFrame) -> dict:
    P = _xyz(df)
    out = dict(frames=int(len(df)),
               jitter_mm=_dist(np.linalg.norm(P - P.mean(0), axis=1) * 1000.0) if len(P) > 2 else _dist([]),
               rms_mm=float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()) * 1000.0) if len(P) > 2 else float("nan"))
    if "qx" in df and len(df) > 2:
        Q = df[["qx", "qy", "qz", "qw"]].to_numpy(np.float64); Q = Q[np.isfinite(Q).all(1)]
        if len(Q) > 2:
            R = Rotation.from_quat(Q)
            d = np.degrees((R.mean().inv() * R).magnitude())
            out["rotation_jitter_deg"] = _dist(d); out["rotation_rms_deg"] = float(np.sqrt((d ** 2).mean()))
    return out


def dropouts(df: pd.DataFrame) -> dict:
    if df.empty: return dict(loss_rate=float("nan"), events=0)
    s = _seconds(df); lost = (df["tracking_health"].to_numpy() == TrackingHealth.LOST.value)
    runs, start = [], None
    for i, f in enumerate(lost):
        if f and start is None: start = i
        if (not f or i == len(lost) - 1) and start is not None:
            runs.append(float(s[i] - s[start])); start = None
    counts = df["fusion_state"].value_counts(dropna=False) if "fusion_state" in df else {}
    return dict(loss_rate=float(lost.mean()), events=len(runs), duration_s=_dist(runs),
                longest_s=float(max(runs)) if runs else 0.0,
                valid_duty=float((~lost).mean()),
                ok_fraction=float((df["fusion_state"] == "TRACKING_OK").mean()) if "fusion_state" in df else float("nan"),
                fusion_states={str(k): int(v) for k, v in dict(counts).items()})


def jumps(df: pd.DataFrame, *, jump_mm: float = JUMP_MM) -> dict:
    P = _xyz(df[df["tracking_health"] != TrackingHealth.LOST.value]) if "tracking_health" in df else _xyz(df)
    if len(P) < 3: return dict(n=0, max_step_mm=float("nan"))
    step = np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000.0
    return dict(n=int((step > jump_mm).sum()), threshold_mm=jump_mm, step_mm=_dist(step), max_step_mm=float(step.max()))


def rotation_leak(df: pd.DataFrame, window) -> dict:
    """Translation seen while the operator only ROTATED. Non-zero here is an uncompensated lever arm (section 8)."""
    if window is None: return dict(note="no rotation_only window in this take")
    d = _window(df, window); P = _xyz(d)
    if len(P) < 6: return dict(note="too few tracked frames")
    k = max(len(P) // 5, 2)
    return dict(displacement_mm=float(np.linalg.norm(P[-k:].mean(0) - P[:k].mean(0)) * 1000.0),
                span_mm=float(np.linalg.norm(P.max(0) - P.min(0)) * 1000.0),
                path_mm=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1000.0))


def branch_discrepancy(df: pd.DataFrame) -> dict:
    """|RGB-D anchor − VI prediction| once aligned: the number P4 exists to produce (section 21 A vs B)."""
    out = dict(residual_mm=_dist(df["residual_m"].to_numpy(np.float64) * 1000.0) if "residual_m" in df else _dist([]),
               residual_corrected_mm=_dist(df["residual_corrected_m"].to_numpy(np.float64) * 1000.0)
               if "residual_corrected_m" in df else _dist([]))
    if {"rgbd_w_x", "vi_x"} <= set(df.columns):
        A = df[["rgbd_w_x", "rgbd_w_y", "rgbd_w_z"]].to_numpy(np.float64)
        B = df[["vi_x", "vi_y", "vi_z"]].to_numpy(np.float64)
        m = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if m.sum() > 2:
            d = np.linalg.norm(A[m] - B[m], axis=1) * 1000.0
            out["anchor_minus_vi_mm"] = _dist(d)
            out["anchor_minus_vi_drift_mm"] = float(d[-max(len(d) // 10, 1):].mean() - d[:max(len(d) // 10, 1)].mean())
    if "correction_m" in df: out["correction_mm"] = _dist(df["correction_m"].to_numpy(np.float64) * 1000.0)
    return out


def return_to_start(df: pd.DataFrame, start_w, end_w) -> dict:
    if start_w is None or end_w is None or start_w == end_w:
        return dict(position_error_mm=float("nan"), note="needs a still window at the start AND at the end")
    a, b = _xyz(_window(df, start_w)), _xyz(_window(df, end_w))
    if len(a) < 3 or len(b) < 3: return dict(position_error_mm=float("nan"), note="too few tracked frames")
    return dict(position_error_mm=float(np.linalg.norm(b.mean(0) - a.mean(0)) * 1000.0))


def _segment_direction(d: pd.DataFrame) -> dict:
    """Displacement of one window, in the human wrist frame and mapped into robot axes. Reported, never applied —
    `HumanRobotFrameMapper` stays the only place human and robot axes are related."""
    cols = [f"human_d{a}_mm" for a in "xyz"] + [f"robot_d{a}_mm" for a in "xyz"]
    if d.empty or not set(cols) <= set(d.columns): return dict(frames=int(len(d)), note="no mapped delta in this window")
    V = d[cols].to_numpy(np.float64); V = V[np.isfinite(V).all(1)]
    if len(V) < 6: return dict(frames=int(len(d)), note="too few tracked frames")
    k = max(len(V) // 5, 2); disp = V[-k:].mean(0) - V[:k].mean(0)
    return dict(frames=int(len(d)), human_mm=dict(zip("xyz", disp[:3].round(2))), robot_mm=dict(zip("xyz", disp[3:].round(2))))


def report(df: pd.DataFrame, cfg: FusedWristCfg, mode: str, stage: str, segments: dict) -> dict:
    auto = still_windows(df) if {"vx", "vy", "vz"} <= set(df.columns) else []
    starts = [n for n in segments if n.startswith(("stationary", "static", "still"))]
    ends = [n for n in segments if n.startswith("return")]
    rot = [n for n in segments if "rotation" in n and "translation" not in n]
    start_w = segments[starts[0]] if starts else (auto[0] if auto else None)
    end_w = segments[ends[-1]] if ends else (auto[-1] if len(auto) > 1 else None)
    s = _seconds(df)
    engaged = bool(df["engaged"].any()) if "engaged" in df else False
    rep = dict(stage=stage, mode=mode, side=cfg.side, ticks=int(len(df)),
               engaged=engaged, stable_fraction=float(df["stable"].mean()) if "stable" in df else float("nan"),
               duration_s=float(s[-1] - s[0]) if len(s) > 1 else 0.0,
               rate_hz=float((len(df) - 1) / max(s[-1] - s[0], 1e-6)) if len(s) > 1 else float("nan"),
               fusion_ms=_dist(df["fusion_ms"]) if "fusion_ms" in df else _dist([]),
               stationary=stationary(_window(df, start_w)) if start_w else None,
               stationary_window_s=list(start_w) if start_w else None,
               return_to_start=return_to_start(df, start_w, end_w),
               rotation_only_leak=rotation_leak(df, segments[rot[0]] if rot else None),
               dropout=dropouts(df), jumps=jumps(df), branches=branch_discrepancy(df),
               alignment=df.attrs.get("stats", {}).get("alignment", {}),
               local_pose=df.attrs.get("stats", {}).get("local_pose", {}),
               correction=df.attrs.get("stats", {}).get("correction", {}),
               pairs=df.attrs.get("stats", {}).get("pairs"))
    if "robot_dx_mm" in df and segments:
        rep["segment_directions"] = {n: _segment_direction(_window(df, w)) for n, w in segments.items()}
    if "step_mm" in df:
        rep["command"] = dict(step_mm=_dist(df["step_mm"]), clamped_fraction=float(df.get("clamped", pd.Series(dtype=bool)).mean())
                              if "clamped" in df else float("nan"),
                              exec_latency_ms=_dist(df["exec_latency_ms"]) if "exec_latency_ms" in df else _dist([]))
    if stage in ("virtual", "real") and not engaged:
        rep["engage_note"] = ("never reached VIO_STABLE, so nothing was ever commanded: every command metric below is "
                              "missing rather than good. The stability gate wants min_stable_s of TRACKING_OK poses "
                              "under max_lin_vel_m_s — an unfiltered 30 Hz position source whose own noise exceeds "
                              "that threshold can never pass it, which is itself a result worth reporting.")
    g = cfg.gates
    jitter = (rep["stationary"] or {}).get("jitter_mm", {}).get("p95", float("nan"))
    rot_j = (rep["stationary"] or {}).get("rotation_jitter_deg", {}).get("p95", float("nan"))
    ret = rep["return_to_start"].get("position_error_mm", float("nan"))
    rep["headline"] = dict(stationary_xyz_jitter_p95_mm=jitter, return_to_start_error_mm=ret,
                           tracking_loss_rate=rep["dropout"]["loss_rate"])
    checks = {
        "fusion rate >= min (Hz)": (rep["rate_hz"], g.min_fusion_rate_hz, "min"),
        "TRACKING_OK fraction >= min": (rep["dropout"]["ok_fraction"], g.min_tracking_ok_fraction, "min"),
        "stationary XYZ jitter p95 <= max (mm)": (jitter, g.max_stationary_jitter_mm_p95, "max"),
        "stationary rotation jitter p95 <= max (deg)": (rot_j, g.max_stationary_rotation_jitter_deg_p95, "max"),
        "return-to-start <= max (mm)": (ret, g.max_return_to_start_mm, "max"),
        "catastrophic jumps <= max": (float(rep["jumps"]["n"]), float(g.max_catastrophic_jumps), "max"),
    }
    leak = rep["rotation_only_leak"].get("displacement_mm")
    if leak is not None: checks["rotation-only translation leak <= max (mm)"] = (leak, g.max_rotation_translation_leak_mm, "max")
    disc = (rep["branches"].get("residual_corrected_mm") or {}).get("p95")
    # Gated on the residual AFTER the correction (|e − C|): how far apart the two sensors still are about the pose
    # the robot actually got. `residual_mm` (|e|) is the drift the anchor sees and stays large by definition while
    # the anchor is working, and `anchor_minus_vi_mm` additionally contains the lever arm — neither is a gate.
    if disc is not None and np.isfinite(disc):
        checks["RGB-D <-> fused residual p95 <= max (mm)"] = (disc, g.max_rgbd_vi_discrepancy_mm_p95, "max")
    if "command" in rep:
        checks["command step p95 <= max (mm)"] = (rep["command"]["step_mm"]["p95"], g.max_command_discontinuity_mm, "max")
    rep["gate"] = _gate(checks)
    rep["gate_note"] = ("Thresholds are pre-measurement starting points (configs/ego_teleop/fused_wrist.yaml). "
                        "Re-fit them from the first real A/B/C table before a PASS is read as evidence.")
    return rep


def print_report(rep: dict) -> None:
    h = rep["headline"]
    print(f"\n=== {rep['stage']} / {rep['mode']} — {rep['ticks']} ticks, {rep['duration_s']:.1f} s @ {rep['rate_hz']:.1f} Hz")
    print(f"    stationary XYZ jitter p95 : {h['stationary_xyz_jitter_p95_mm']:.1f} mm")
    print(f"    return-to-start error     : {h['return_to_start_error_mm']:.1f} mm")
    print(f"    tracking loss rate        : {h['tracking_loss_rate']*100:.2f} %")
    if rep.get("engage_note"): print(f"    NOT ENGAGED: {rep['engage_note']}")
    a = rep.get("alignment") or {}
    if a:
        print(f"    alignment  aligned={a.get('aligned')} rms={a.get('rms_m', float('nan'))*1000:.1f} mm "
              f"lever_arm={np.round(a.get('lever_arm_m', [0, 0, 0]), 4).tolist()} m "
              f"(sigma {a.get('lever_arm_sigma_m', float('nan'))*1000:.1f} mm, observable={a.get('lever_arm_observable')})")
    leak = rep["rotation_only_leak"].get("displacement_mm")
    if leak is not None: print(f"    rotation-only translation leak: {leak:.1f} mm")
    for k, v in rep["gate"].items():
        print(f"    [{'PASS' if v['passed'] else 'FAIL'}] {k}: {v['value']:.4g} (thr {v['threshold']:.4g})")


def ablation_table(reports: dict[str, dict]) -> str:
    rows = [("metric", *reports.keys())]
    def get(r, path, scale=1.0):
        v = r
        for k in path:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        return float("nan") if v is None else float(v) * scale
    metrics = [("valid duty %", ("dropout", "valid_duty"), 100.0),
               ("stationary jitter p95 mm", ("stationary", "jitter_mm", "p95"), 1.0),
               ("stationary rot jitter p95 deg", ("stationary", "rotation_jitter_deg", "p95"), 1.0),
               ("return-to-start mm", ("return_to_start", "position_error_mm"), 1.0),
               ("rotation-only leak mm", ("rotation_only_leak", "displacement_mm"), 1.0),
               ("anchor-fused residual p95 mm", ("branches", "residual_corrected_mm", "p95"), 1.0),
               ("drift seen by anchor p95 mm", ("branches", "residual_mm", "p95"), 1.0),
               ("anchor-VI raw gap p95 mm", ("branches", "anchor_minus_vi_mm", "p95"), 1.0),
               ("catastrophic jumps", ("jumps", "n"), 1.0),
               ("longest loss s", ("dropout", "longest_s"), 1.0),
               ("command step p95 mm", ("command", "step_mm", "p95"), 1.0)]
    for name, path, sc in metrics:
        rows.append((name, *[f"{get(r, path, sc):.2f}" for r in reports.values()]))
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in rows)


# ---- CLI ----------------------------------------------------------------------------------------------------------
def parse_segments(items) -> dict:
    out = {}
    for it in items or []:
        n, a, b = it.split(":"); out[n] = (float(a), float(b))
    return out


def build_events(args, cfg: FusedWristCfg, teleop: TeleopCfg):
    """-> (events, rgbd_provider, vi_provider, note). One time-ordered stream; neither branch ever sees the future."""
    T_H_C = np.eye(4)
    note = ""
    if args.synthetic:
        vi_df, rgbd_rows, _, _ = synthetic_take(n=getattr(args, 'synthetic_frames', None) or 1800)
        vip, vi_df = vi_provider_from_parquet_df(vi_df, T_H_C=T_H_C, supervisor_cfg=cfg.vi_tracking)
        ev = sorted([(t, "rgbd", (p, c)) for t, p, c in rgbd_rows] + [(t, "vi", e) for t, e in vi_estimates(vi_df)],
                    key=lambda x: (x[0], x[1] != "vi"))
        return ev, None, vip, "SYNTHETIC take: plumbing only, never evidence about the hardware"
    if not (args.episode and args.vi_poses):
        raise SystemExit("need --synthetic, or --episode DIR together with --vi-poses PARQUET (see --help)")
    from handumi_collector.pose.calibration import load_versioned
    _, d = load_versioned(f"wrist_camera_{cfg.side}")
    if d and "T_wrist_camera" in d:
        T_H_C = np.asarray(d["T_wrist_camera"], float).reshape(4, 4)
    elif args.allow_identity_mount:
        note = ("T_H_C is IDENTITY: the Arducam is treated as sitting exactly at the wrist rotation centre. Every "
                "wrist rotation then fabricates translation (spec section 8). Calibrate wrist_camera_%s before "
                "any number here is used." % cfg.side)
        print("WARNING: " + note)
    else:
        raise SystemExit(f"no wrist_camera_{cfg.side} calibration (T_H_C). Pass --allow-identity-mount only for a "
                         f"plumbing check — it fabricates translation from every wrist rotation.")
    vip, vi_df = vi_provider_from_parquet(args.vi_poses, T_H_C=T_H_C, supervisor_cfg=cfg.vi_tracking)
    src = RecordedRgbdSource(args.episode)
    rgbd_provider = cfg.build_rgbd_provider(head_rgbd=teleop.head_rgbd)
    ev = sorted([(int(f.timestamp_ns), "rgbd", f) for f in src] + [(t, "vi", e) for t, e in vi_estimates(vi_df)],
                key=lambda x: (x[0], x[1] != "vi"))
    return ev, rgbd_provider, vip, note


def vi_provider_from_parquet_df(df: pd.DataFrame, *, T_H_C, supervisor_cfg):
    return EstimatorWristPoseProvider(_ReplayEstimator(), T_H_C=T_H_C, supervisor=TrackingSupervisor(supervisor_cfg),
                                      source="vi_replay"), df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("compare", "fused", "virtual", "real"), default="compare")
    ap.add_argument("--mode", choices=ABLATIONS, default=None, help="one ablation arm (default: the config's mode)")
    ap.add_argument("--ablation", action="store_true", help="run all three arms of spec section 21 and compare")
    ap.add_argument("--episode", type=Path, help="recorded take with head/ RGB-D frames")
    ap.add_argument("--vi-poses", type=Path, help="wrist VI trajectory parquet (m1_vio output)")
    ap.add_argument("--synthetic", action="store_true", help="generated take with a known answer (plumbing / CI)")
    ap.add_argument("--live", action="store_true", help="the three real sensors (ego_teleop.tracking.fused_live)")
    ap.add_argument("--live-seconds", type=float, default=60.0)
    ap.add_argument("--hardware", default="handumi_v1", help="--live: collector profile for the WRIST camera + IMU "
                    "(must NOT list the Orbbec: the fusion opens the chest camera itself)")
    ap.add_argument("--downscale", type=int, default=1, help="--live: wrist image downscale fed to the VI backend")
    ap.add_argument("--protocol", action="store_true", help="label the standard 60 s trial windows (section 20)")
    ap.add_argument("--segment", action="append", default=[], metavar="name:t0:t1")
    ap.add_argument("--engage-at", type=float, default=3.0)
    ap.add_argument("--clutch-at", type=float, default=None)
    ap.add_argument("--release-at", type=float, default=None)
    ap.add_argument("--synthetic-frames", type=int, default=None, help="length of the synthetic take, in sensor "
                    "frames at 30 Hz (default 1800 = the full 60 s protocol)")
    ap.add_argument("--record", type=Path, help="write the fused stream into this episode directory")
    ap.add_argument("--out", type=Path, help="report json (default: <record>/fusion_report.json or ./fusion_report.json)")
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--allow-identity-mount", action="store_true")
    ap.add_argument("--i-am-at-the-robot", action="store_true", help="required for --stage real")
    args = ap.parse_args(argv)

    teleop = load_teleop_cfg(args.config_dir) if args.config_dir else load_teleop_cfg()
    cfg = teleop.fused_wrist
    if args.stage == "real" and not args.i_am_at_the_robot:
        raise SystemExit("--stage real drives the physical arm. Re-run with --i-am-at-the-robot, at low scale, with "
                         "the e-stop in hand, and only after the virtual stage's numbers are clean (ladder P6 -> P7).")
    segments = parse_segments(args.segment)
    if args.protocol: segments = {n: (a, b) for n, a, b, _ in PROTOCOL_60S} | segments

    if args.live:
        return run_live(args, teleop, segments)

    modes = list(ABLATIONS) if args.ablation else [args.mode or cfg.mode]
    reports: dict[str, dict] = {}
    logger = TeleopEpisodeLogger(args.record, metadata=dict(tool="f4_fusion", stage=args.stage)) if args.record else None
    for mode in modes:
        events, rgbd_provider, vi_provider, note = build_events(args, cfg, teleop)
        df = run(events=events, cfg=cfg, mode=mode, stage=args.stage, rgbd_provider=rgbd_provider,
                 vi_provider=vi_provider, logger=logger if mode == (args.mode or cfg.mode) else None,
                 engage_at_s=args.engage_at, clutch_at_s=args.clutch_at, release_at_s=args.release_at)
        rep = report(df, cfg, mode, args.stage, segments)
        if note: rep["note"] = note
        reports[mode] = rep
        print_report(rep)
        if args.record: df.to_parquet(Path(args.record) / f"fusion_ticks_{mode}.parquet", index=False)
    if len(reports) > 1:
        print("\n=== ablation (spec section 21): does each sensor earn its place?\n")
        print(ablation_table(reports))
        print("\nFusion is not assumed to win. If C is not better than the better of A and B, the anchor is not "
              "helping and the reason is in `branches` / `alignment`.")
    out = args.out or ((Path(args.record) / "fusion_report.json") if args.record else Path("fusion_report.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(config=dict(mode=cfg.mode, side=cfg.side, rate_hz=cfg.rate_hz), reports=reports),
                              indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print(f"\nreport -> {out}")
    if logger is not None: logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
