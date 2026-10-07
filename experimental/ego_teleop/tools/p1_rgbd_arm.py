"""V0 / V1 of the vision-only RGB-D arm POC — a fixed RGB-D camera as a temporary arm pose source.

    V0  raw palm pose only, no robot, no mapping     -> is the pose good enough to drive anything?
    V1  3-DoF virtual reBot TCP, no robot command    -> does the EXISTING arm stack behave on it?

Both stages run the production downstream unchanged (`HumanRobotFrameMapper`, `RelativeSE3Retargeter`,
`ArmSafetyPipeline`, `TeleopCoordinator`, the reBot client) — the only new thing is the pose source. V2 (3-DoF on the
real arm) needs no new code: it is `arm_enabled: true` plus an `HttpRebotClient`, and it is deliberately not wired
here until V1's numbers have been read.

    .venv/bin/python -m ego_teleop.tools.p1_rgbd_arm --stage v0 --episode datasets/HumanRGBD_v1/episode_000001 --viz
    .venv/bin/python -m ego_teleop.tools.p1_rgbd_arm --stage v1 --live --protocol --record datasets/rgbd_arm_poc/ep001

The three numbers this exists to produce, printed first in every report:

    stationary XYZ jitter (mm, p95)   return-to-start error (mm)   tracking-loss rate (%)

Everything else is there to explain those three. Windows are labelled with `--segment name:t0:t1` (or `--protocol`
for the standard 60 s trial); when nothing is labelled the still windows are found automatically, so the headline
numbers exist even on an unlabelled take."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from ..config import RgbdArmPocCfg, TeleopCfg, load_teleop_cfg
from ..hand3d.head_camera import OrbbecHeadCamera, RecordedRgbdSource
from ..hand3d.palm_pose import POSE_SOURCE, ArmPoseHealth
from ..recorder.episode_logger import HeadRgbdWriter, TeleopEpisodeLogger
from ..retarget.aero_retarget import AeroRetargeter
from ..retarget.arm_relative_se3 import RelativeSE3Retargeter
from ..robot.coordinator import CoordinatorConfig, Readiness, TeleopCoordinator
from ..robot.rebot_client import MockRebotController
from ..robot.safety import ArmSafetyPipeline, SafetyConfig
from ..tracking.interfaces import TrackingHealth
from ..tracking.vio_stable import VioStableGate
from ..transforms.frames import HumanRobotFrameMapper
from ..transforms.se3 import T_to_pose7
from .a1_hand3d import _dist, draw_overlay

# The standard trial of the plan (section 15). One take, robot-free, so every metric below has a window to live in.
# CANONICAL: these names travel from the recorder's voice cue, through `--segment`, into the report's window lookup.
# `~/orbbec_recorder.py` duplicates the table (standalone script, its own venv) and
# tests/teleop/test_rgbd_arm_poc.py pins the two against each other — a renamed segment silently NaNs a headline
# number, because the report finds the return window by name.
# The hint is ASCII on purpose: it is drawn with OpenCV's Hershey fonts, which have no Korean glyphs and would
# render the instruction as "????????" on the one screen the operator is actually looking at. The spoken cue is
# Korean and lives in the recorder (`POC_SAY`), which is the only place that speaks.
PROTOCOL_60S = (("stationary", 0, 10, "hold still, hand open"),
                ("slow_xyz", 10, 20, "slow: left/right, up/down, near/far"),
                ("palm_rotation_only", 20, 30, "rotate wrist only, do NOT translate"),
                ("translation_rotation", 30, 40, "translate and rotate together"),
                ("fast_natural", 40, 50, "fast natural motion"),
                ("return_stationary", 50, 60, "return to start, then hold still"))
SEGMENT_ARGS = " ".join(f"--segment {n}:{a}:{b}" for n, a, b, _ in PROTOCOL_60S)
STILL_SPEED_M_S = 0.02        # auto still-window detection
MIN_STILL_S = 1.5


# ---- stage V0: the pose source alone ---------------------------------------------------------------------------
def run_v0(source, poc: RgbdArmPocCfg, head_rgbd, *, max_frames: int | None = None, out_dir: Path | None = None,
           viz: bool = False, live_window: bool = False, logger: TeleopEpisodeLogger | None = None,
           head_writer: HeadRgbdWriter | None = None, live: bool = False) -> pd.DataFrame:
    """Push RGB-D frames through the POC pose source and record every pose it produced, graded or not."""
    provider = poc.build_pose_provider(head_rgbd=head_rgbd)
    est = provider.est                       # RgbdPalmPoseEstimator — for the raw palm read-out and the hand estimate
    stable = VioStableGate(poc.stable)
    rows: list[dict] = []
    writer = None
    t_wall = time.monotonic()
    n = 0
    for frame in source:
        if frame is None: continue
        t_call = time.monotonic_ns()
        wp = provider.push_image(frame.timestamp_ns, getattr(frame, "frame_index", n), frame)
        proc_ms = (time.monotonic_ns() - t_call) / 1e6
        palm, hand = est.last_palm, est.last_hand
        hand_est = hand.estimate if hand is not None else None
        stable.update(wp)
        p = wp.position_xyz_m
        row = dict(frame_index=getattr(frame, "frame_index", n), t_ns=int(frame.timestamp_ns),
                   tracking_health=wp.health.value, arm_pose_health=palm.health.value, arm_pose_reason=palm.reason,
                   hand_health=hand.health.value if hand else "", hand_reason=hand.reason if hand else "",
                   x=float(p[0]), y=float(p[1]), z=float(p[2]),
                   vx=float(wp.linear_velocity_xyz[0]), vy=float(wp.linear_velocity_xyz[1]), vz=float(wp.linear_velocity_xyz[2]),
                   n_palm_landmarks=palm.n_palm_landmarks, palm_depth_spread_m=palm.palm_depth_spread_m,
                   wrist_offset_m=palm.wrist_offset_m, palm_scale_m=palm.palm_scale_m,
                   span_index_pinky_m=palm.span_index_pinky_m, span_wrist_middle_m=palm.span_wrist_middle_m,
                   palm_scale_ratio=palm.palm_scale_ratio, stable=bool(stable.stable),
                   n_valid=hand_est.n_valid if hand_est else 0, n_filled=hand_est.n_filled if hand_est else 0,
                   proc_ms=proc_ms,
                   # capture->pose latency needs a real capture clock; a replayed episode has a synthetic 1/fps one,
                   # so it is reported ONLY for a live camera rather than as a fabricated number
                   latency_ms=(hand_est.latency_ms() if hand_est else float("nan")) if live else float("nan"))
        # raw palm orientation, recorded in V0 even though the arm runs 3-DoF: V3/V4 are then decided on measurements
        if palm.R_palm is not None:
            q = Rotation.from_matrix(palm.R_palm).as_quat()
            for k, v in zip(("raw_qx", "raw_qy", "raw_qz", "raw_qw"), q): row[k] = float(v)
        rows.append(row)
        if logger is not None:
            logger.add_rgbd_hand_pose(poc.side, wp)
            if hand_est is not None:
                logger.add_hand_pose(hand_est, supervised_health=hand.health.value, reason=hand.reason)
        if head_writer is not None: head_writer.add(frame)
        if viz or live_window:
            img = _overlay(frame, hand_est, palm, wp)
            if viz and out_dir is not None:
                import cv2
                if writer is None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out_dir / "v0_palm_pose.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             frame.calib.fps, (img.shape[1], img.shape[0]))
                writer.write(img)
            if live_window:
                import cv2
                cv2.imshow("V0 fixed RGB-D palm pose", img)
                if cv2.waitKey(1) & 0xFF == 27: break
        n += 1
        if max_frames and n >= max_frames: break
    if writer is not None: writer.release()
    df = pd.DataFrame(rows)
    df.attrs["wall_rate_hz"] = n / max(time.monotonic() - t_wall, 1e-6)
    df.attrs["quality"] = provider.stats()["quality"]
    return df


def _overlay(frame, hand_est, palm, wp):
    """A1's hand overlay plus the arm-root read-out: palm origin, its health, and the camera-frame XYZ."""
    import cv2
    img = draw_overlay(frame, hand_est if hand_est is not None and hand_est.n_valid else None)
    col = {"ARM_POSE_OK": (0, 220, 0), "ARM_POSE_DEGRADED": (0, 200, 255), "ARM_POSE_LOST": (0, 0, 255)}[palm.health.value]
    p = wp.position_xyz_m
    txt = f"{palm.health.value} {wp.health.value}  palm=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})m  {palm.reason}"
    cv2.putText(img, txt, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if palm.position_m is not None and hand_est is not None:
        uv = frame.calib.color.project(palm.position_m[None, :])[0]
        if np.isfinite(uv).all(): cv2.circle(img, tuple(uv.astype(int)), 7, col, 2, cv2.LINE_AA)
    return img


# ---- windows ---------------------------------------------------------------------------------------------------
def _seconds(df: pd.DataFrame) -> np.ndarray:
    return (df["t_ns"].to_numpy(np.float64) - float(df["t_ns"].iloc[0])) / 1e9 if len(df) else np.zeros(0)


def _window(df: pd.DataFrame, w: tuple[float, float]) -> pd.DataFrame:
    s = _seconds(df)
    return df[(s >= w[0]) & (s <= w[1])]


def still_windows(df: pd.DataFrame, *, speed_m_s: float = STILL_SPEED_M_S, min_s: float = MIN_STILL_S) -> list[tuple[float, float]]:
    """Maximal windows where the palm is tracked and barely moving. Lets the headline numbers exist without labels."""
    if df.empty: return []
    s = _seconds(df)
    v = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(np.float64), axis=1)
    ok = (df["tracking_health"].to_numpy() == TrackingHealth.OK.value) & np.isfinite(v) & (v <= speed_m_s)
    out, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None: start = i
        if (not flag or i == len(ok) - 1) and start is not None:
            end = i if not flag else i + 1
            if s[end - 1] - s[start] >= min_s: out.append((float(s[start]), float(s[end - 1])))
            start = None
    return out


def _xyz(df: pd.DataFrame) -> np.ndarray:
    P = df[["x", "y", "z"]].to_numpy(np.float64)
    return P[np.isfinite(P).all(1)]


def _quat(df: pd.DataFrame) -> np.ndarray | None:
    if "raw_qx" not in df: return None
    Q = df[["raw_qx", "raw_qy", "raw_qz", "raw_qw"]].to_numpy(np.float64)
    Q = Q[np.isfinite(Q).all(1)]
    return Q if len(Q) else None


def stationary_metrics(df: pd.DataFrame) -> dict:
    """Position scatter about the window mean, and orientation scatter about the window's mean rotation.

    Orientation is measured in every V0 run even though V0/V1 command translation only: whether 6-DoF is worth
    building is a question about this number, and guessing it later from a 3-DoF log is impossible."""
    P = _xyz(df)
    out = dict(frames=int(len(df)),
               jitter_mm=_dist(np.linalg.norm(P - P.mean(0), axis=1) * 1000.0) if len(P) > 2 else _dist([]),
               per_axis_std_mm={a: float(P[:, i].std() * 1000.0) if len(P) > 2 else float("nan") for i, a in enumerate("xyz")},
               depth_std_mm=float(P[:, 2].std() * 1000.0) if len(P) > 2 else float("nan"))
    Q = _quat(df)
    if Q is not None and len(Q) > 2:
        R = Rotation.from_quat(Q)
        out["orientation_jitter_deg"] = _dist(np.degrees((R.mean().inv() * R).magnitude()))
    else:
        out["orientation_jitter_deg"] = _dist([])
    return out


def dropout_metrics(df: pd.DataFrame) -> dict:
    """Tracking-loss rate and the shape of the losses: a 2 % loss made of one 3 s blackout is a different robot
    experience from a 2 % loss made of 60 single dropped frames, and only the second is survivable."""
    if df.empty: return dict(loss_rate=float("nan"), events=0)
    s = _seconds(df)
    lost = df["tracking_health"].to_numpy() == TrackingHealth.LOST.value
    durations, start = [], None
    for i, f in enumerate(lost):
        if f and start is None: start = i
        if (not f or i == len(lost) - 1) and start is not None:
            durations.append(float(s[i] - s[start])); start = None
    arm = df["arm_pose_health"].value_counts()
    return dict(loss_rate=float(lost.mean()), events=len(durations), duration_s=_dist(durations),
                longest_s=float(max(durations)) if durations else 0.0,
                arm_pose_counts={k: int(v) for k, v in arm.items()},
                arm_pose_ok_fraction=float((df["arm_pose_health"] == ArmPoseHealth.OK.value).mean()),
                reasons={k: int(v) for k, v in df.loc[df["arm_pose_reason"].astype(bool), "arm_pose_reason"].value_counts().head(10).items()})


def segment_directions(df: pd.DataFrame, segments: dict[str, tuple[float, float]], mapper: HumanRobotFrameMapper) -> dict:
    """Per-segment displacement, in camera axes and mapped into robot EEF axes.

    This is the table you read the axis map off: move the hand right and see which robot axis the mapped column puts
    it on. It is reported, never applied — `HumanRobotFrameMapper` stays the only place axes are related."""
    out = {}
    for name, w in segments.items():
        d = _window(df, w)
        P = _xyz(d[d["tracking_health"] != TrackingHealth.LOST.value])
        if len(P) < 6:
            out[name] = dict(frames=int(len(d)), note="too few tracked frames"); continue
        k = max(len(P) // 5, 2)
        disp = P[-k:].mean(0) - P[:k].mean(0)
        path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        out[name] = dict(frames=int(len(d)),
                         camera_displacement_mm={a: float(disp[i] * 1000) for i, a in enumerate("xyz")},
                         robot_displacement_mm={a: float(v * 1000) for a, v in zip("xyz", mapper.map_translation(disp))},
                         path_length_mm=path * 1000.0, span_mm=float(np.linalg.norm(P.max(0) - P.min(0)) * 1000.0),
                         speed_mm_s=_dist(np.linalg.norm(d[["vx", "vy", "vz"]].to_numpy(np.float64), axis=1) * 1000.0))
    return out


def v0_report(df: pd.DataFrame, poc: RgbdArmPocCfg, segments: dict[str, tuple[float, float]]) -> dict:
    mapper = HumanRobotFrameMapper(poc.frames)
    auto = still_windows(df)
    # windows are found by name PREFIX, so `return_stationary` (canonical), `return` or `return_to_start` all work
    starts = [n for n in segments if n.startswith(("stationary", "static", "still"))]
    ends = [n for n in segments if n.startswith("return")]
    start_w = segments[starts[0]] if starts else (auto[0] if auto else None)
    end_w = segments[ends[-1]] if ends else (auto[-1] if len(auto) > 1 else None)

    rep: dict = dict(stage="v0", pose_source=POSE_SOURCE, camera_mode=poc.camera_mode, imu_used=False, causal=True,
                     frames=int(len(df)), rate_hz=float(df.attrs.get("wall_rate_hz", float("nan"))),
                     frames_with_hand=int((df["n_valid"] > 0).sum()),
                     processing_ms=_dist(df["proc_ms"]),
                     tracking_counts={k: int(v) for k, v in df["tracking_health"].value_counts().items()},
                     capture_to_pose_latency_ms=_dist(df["latency_ms"]),   # live sources only; NaN on a replay
                     still_windows_s=auto,
                     palm_scale_mm=_dist(df["palm_scale_m"] * 1000.0), palm_scale_ratio=_dist(df["palm_scale_ratio"]),
                     palm_depth_spread_mm=_dist(df["palm_depth_spread_m"] * 1000.0),
                     wrist_to_centroid_mm=_dist(df["wrist_offset_m"] * 1000.0),
                     n_palm_landmarks=_dist(df["n_palm_landmarks"]))
    rep["dropout"] = dropout_metrics(df)
    # span jumps: the evidence for the plan's geometry-sanity thresholds, which ship as null (measure-only)
    for col in ("span_index_pinky_m", "span_wrist_middle_m"):
        v = df[col].to_numpy(np.float64)
        rep[f"{col}_mm"] = _dist(v * 1000.0)
        rep[f"{col}_frame_jump_mm"] = _dist(np.abs(np.diff(v)) * 1000.0)
    rep["stationary"] = stationary_metrics(_window(df, start_w)) if start_w else None
    rep["stationary_window_s"] = list(start_w) if start_w else None
    rep["return_to_start"] = _return_to_start(df, start_w, end_w)
    rep["segments"] = segment_directions(df, segments, mapper) if segments else {}
    rep["frame_mapping_under_test"] = mapper.describe()
    rep["quality"] = df.attrs.get("quality", {})

    g = poc.gates
    jitter = (rep["stationary"] or {}).get("jitter_mm", {}).get("p95", float("nan"))
    ori = (rep["stationary"] or {}).get("orientation_jitter_deg", {}).get("p95", float("nan"))
    ret = rep["return_to_start"].get("position_error_mm", float("nan"))
    loss = rep["dropout"]["loss_rate"]
    rep["headline"] = dict(stationary_xyz_jitter_p95_mm=jitter, return_to_start_error_mm=ret, tracking_loss_rate=loss)
    rep["gate"] = _gate({
        "rate_hz >= min": (rep["rate_hz"], g.min_rate_hz, "min"),
        "arm_pose OK fraction >= min": (rep["dropout"]["arm_pose_ok_fraction"], g.min_arm_pose_ok_fraction, "min"),
        "stationary XYZ jitter p95 <= max (mm)": (jitter, g.max_stationary_jitter_mm_p95, "max"),
        "return-to-start error <= max (mm)": (ret, g.max_return_to_origin_mm, "max"),
        "stationary orientation jitter p95 <= max (deg)": (ori, g.max_stationary_orientation_jitter_deg_p95, "max"),
    })
    rep["gate_note"] = ("Thresholds are pre-measurement starting points. The orientation check is informational for "
                        "V0/V1 (3-DoF commands translation only) and is what decides whether V3/V4 are worth it.")
    return rep


def _return_to_start(df: pd.DataFrame, start_w, end_w) -> dict:
    """How far the reported palm position has drifted after the hand leaves and comes back to the same place.

    This is the number that says whether relative motion accumulates error: the human returns to the start, so a
    non-zero result is the pose source's, not the operator's."""
    if start_w is None or end_w is None or start_w == end_w:
        return dict(position_error_mm=float("nan"), note="needs a still window at the start AND at the end "
                                                         "(label one `return`, or hold still for 1.5 s at both ends)")
    A, B = _xyz(_window(df, start_w)), _xyz(_window(df, end_w))
    if len(A) < 3 or len(B) < 3: return dict(position_error_mm=float("nan"), note="too few tracked frames in a window")
    d = B.mean(0) - A.mean(0)
    out = dict(position_error_mm=float(np.linalg.norm(d) * 1000.0),
               per_axis_mm={a: float(d[i] * 1000.0) for i, a in enumerate("xyz")},
               start_window_s=list(start_w), end_window_s=list(end_w))
    QA, QB = _quat(_window(df, start_w)), _quat(_window(df, end_w))
    if QA is not None and QB is not None and len(QA) > 2 and len(QB) > 2:
        out["orientation_error_deg"] = float(np.degrees((Rotation.from_quat(QA).mean().inv() * Rotation.from_quat(QB).mean()).magnitude()))
    return out


def _gate(checks: dict[str, tuple[float, float, str]]) -> dict:
    out = {}
    for k, (val, thr, sense) in checks.items():
        ok = (val >= thr) if sense == "min" else (val <= thr)
        out[k] = dict(passed=bool(ok) if np.isfinite(val) else False, value=val, threshold=thr,
                      **({"note": "not measured"} if not np.isfinite(val) else {}))
    return out


# ---- stage V1: 3-DoF virtual TCP through the existing arm stack --------------------------------------------------
def _poc_readiness() -> Readiness:
    """The coordinator's readiness flags are named for the PRODUCTION pose source (wrist fisheye + IMU + OpenVINS).

    This POC has no IMU and no VIO, so two of them are satisfied by their RGB-D equivalents: `imu_online` by the depth
    stream being up, `vio_stable` by the reused `VioStableGate` proving the palm pose held TRACKING_OK and still for
    `stable.min_stable_s`. `tracking_valid` is NOT substituted — it is the real health of the RGB-D pose. Recorded in
    every episode as `readiness_note`, so no log can be read as if an IMU had been present."""
    return Readiness(camera_online=True, imu_online=True, tracking_valid=False, vio_stable=False,
                     robot_observed=True, aero_homed=True, calibration_loaded=True)


READINESS_NOTE = ("POC: no IMU and no VIO. coordinator.readiness.imu_online = RGB-D depth stream up; "
                  "vio_stable = VioStableGate on the RGB-D palm pose. tracking_valid is the real RGB-D health.")


def run_v1(source, poc: RgbdArmPocCfg, head_rgbd, *, engage_at_s: float = 2.0, clutch_at_s: float | None = None,
           release_at_s: float | None = None, max_frames: int | None = None,
           logger: TeleopEpisodeLogger | None = None, head_writer: HeadRgbdWriter | None = None) -> tuple[pd.DataFrame, dict]:
    """Drive the REAL arm stack from the RGB-D pose source, with the robot command disabled.

    `arm_enabled=False` is the only difference from V2 (3-DoF on the real arm): the retargeter, the workspace clamp,
    the velocity/acceleration limiter and the coordinator state machine all run exactly as they would with a robot
    attached, and the TCP target they produce is recorded instead of sent."""
    provider = poc.build_pose_provider(head_rgbd=head_rgbd)
    est = provider.est
    robot = MockRebotController(side=poc.side)
    T_anchor = robot.observe().T_RB_RE
    safety_cfg = SafetyConfig(**{**asdict(poc.safety), "workspace": poc.anchor_workspace(T_anchor)}) if poc.workspace_from_anchor else poc.safety
    co = TeleopCoordinator(wrist=provider, hand=None, arm_retargeter=RelativeSE3Retargeter(HumanRobotFrameMapper(poc.frames)),
                           aero_retargeter=AeroRetargeter(), safety=ArmSafetyPipeline(safety_cfg), robot=robot, aero=None,
                           cfg=CoordinatorConfig(rate_hz=poc.rate_hz, hand_enabled=False, arm_enabled=False),
                           readiness=_poc_readiness())
    stable = VioStableGate(poc.stable)
    fired = ["sensors_ready", "calibrated", "vio_start"]
    for ev in fired: co.fire(ev, t_ns=0)

    rows: list[dict] = []
    t0 = None
    engaged = clutched = False
    n = 0
    for frame in source:
        if frame is None: continue
        t = int(frame.timestamp_ns)
        if t0 is None: t0 = t
        s = (t - t0) / 1e9
        wp = provider.push_image(t, getattr(frame, "frame_index", n), frame)
        palm = est.last_palm
        co.readiness.tracking_valid = bool(wp.valid)
        co.readiness.vio_stable = stable.update(wp)
        if logger is not None:
            logger.add_rgbd_hand_pose(poc.side, wp)
            if est.last_hand is not None and est.last_hand.estimate is not None:
                logger.add_hand_pose(est.last_hand.estimate, supervised_health=est.last_hand.health.value)
        if head_writer is not None: head_writer.add(frame)

        # ---- operator events, scripted by the CLI so an offline replay exercises the same path a human would
        if not engaged and co.readiness.vio_stable and s >= engage_at_s:
            for ev in ("vio_stable", "robot_ready", "arm", "start"):
                co.fire(ev, t_ns=t)
                if logger is not None: logger.add_event(ev, t, stage="v1", note=READINESS_NOTE if ev == "arm" else "")
            engaged = True
            if not np.allclose(robot.observe().T_RB_RE[:3, 3], T_anchor[:3, 3], atol=0.01):
                raise RuntimeError("robot moved between workspace-box construction and ENGAGE; rebuild the box")
        if engaged and clutch_at_s is not None and not clutched and s >= clutch_at_s:
            co.fire("clutch", t_ns=t); clutched = True
            if logger is not None: logger.add_event("clutch", t, stage="v1")
        if clutched and release_at_s is not None and s >= release_at_s:
            co.fire("release", t_ns=t); clutched = False
            if logger is not None: logger.add_event("release", t, stage="v1")

        cmd = co.tick(t)
        if cmd is not None:
            tgt = co.arm_rt.last_target
            row = dict(frame_index=getattr(frame, "frame_index", n), t_ns=t, s=s, state=cmd.state.value,
                       tracking_health=cmd.wrist_health.value, arm_pose_health=palm.health.value,
                       hold_reason=cmd.hold_reason, clutch=bool(cmd.clutch),
                       safety=",".join(cmd.safety_reasons), clamped="workspace_clamped" in cmd.safety_reasons,
                       tcp_x=float(cmd.arm_target_xyz[0]), tcp_y=float(cmd.arm_target_xyz[1]), tcp_z=float(cmd.arm_target_xyz[2]),
                       step_mm=float(np.linalg.norm(cmd.arm_action6[:3]) * 1000.0),
                       speed_mm_s=float(np.linalg.norm(cmd.arm_action6[:3]) * poc.rate_hz * 1000.0),
                       unclamped_x=float(tgt[0, 3]) if tgt is not None else np.nan,
                       unclamped_y=float(tgt[1, 3]) if tgt is not None else np.nan,
                       unclamped_z=float(tgt[2, 3]) if tgt is not None else np.nan)
            for i, a in enumerate("xyz"):
                row[f"human_d{a}_mm"] = float(cmd.delta_H6[i] * 1000.0) if cmd.delta_H6 is not None else np.nan
            rows.append(row)
            if logger is not None:
                logger.add_command(cmd); logger.add_rebot_state(co.last_robot_state)
                logger.add_rgbd_relative_pose(poc.side, row)
        n += 1
        if max_frames and n >= max_frames: break
    df = pd.DataFrame(rows)
    df.attrs["anchor_workspace"] = asdict(safety_cfg.workspace)
    df.attrs["anchor_tcp"] = T_to_pose7(T_anchor).tolist()
    df.attrs["quality"] = provider.stats()["quality"]
    return df, dict(engaged=engaged, ticks=len(rows))


def v1_report(df: pd.DataFrame, poc: RgbdArmPocCfg, meta: dict) -> dict:
    g = poc.gates
    if df.empty:
        return dict(stage="v1", pose_source=POSE_SOURCE, ticks=0, gate={}, gate_passed=False,
                    note="never engaged: the pose source did not hold TRACKING_OK and still long enough to anchor")
    held = df[df["hold_reason"].astype(bool)]
    P = df[["tcp_x", "tcp_y", "tcp_z"]].to_numpy(np.float64)
    U = df[["unclamped_x", "unclamped_y", "unclamped_z"]].to_numpy(np.float64)
    clamp_err = np.linalg.norm(np.nan_to_num(U - P), axis=1)
    rep = dict(stage="v1", pose_source=POSE_SOURCE, camera_mode=poc.camera_mode, imu_used=False,
               dof=3, orientation="fixed at ENGAGE (palm orientation measured but not commanded)",
               ticks=int(len(df)), engaged=bool(meta.get("engaged")),
               states={k: int(v) for k, v in df["state"].value_counts().items()},
               hold_reasons={k: int(v) for k, v in held["hold_reason"].value_counts().items()},
               hold_fraction=float(len(held) / len(df)),
               tcp_path_mm=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1000.0),
               tcp_span_mm={a: float((P[:, i].max() - P[:, i].min()) * 1000.0) for i, a in enumerate("xyz")},
               commanded_speed_mm_s=_dist(df["speed_mm_s"]),
               speed_limit_mm_s=poc.safety.max_lin_vel_m_s * 1000.0,
               speed_limit_hit_fraction=float((df["speed_mm_s"] >= poc.safety.max_lin_vel_m_s * 1000.0 * 0.98).mean()),
               workspace_clamp_fraction=float(df["clamped"].mean()),
               workspace_clamp_depth_mm=_dist(clamp_err * 1000.0),
               anchor_workspace=df.attrs.get("anchor_workspace"), anchor_tcp=df.attrs.get("anchor_tcp"),
               human_travel_mm={a: float(np.nanmax(df[f"human_d{a}_mm"]) - np.nanmin(df[f"human_d{a}_mm"])) for a in "xyz"},
               translation_scale=list(poc.frames.translation_scale),
               frame_mapping=HumanRobotFrameMapper(poc.frames).describe(),
               quality=df.attrs.get("quality", {}), readiness_note=READINESS_NOTE,
               ik="not checked (virtual stage: MockRebotController has no IK; V2 runs the pyroki solver)")
    # Clutch freezes the RETARGETER target. The commanded TCP may still be travelling toward it under the velocity
    # limiter, so the number that must be ~0 is the unclamped target, not the command.
    clutch = df[df["hold_reason"] == "clutch"]
    if len(clutch) > 1:
        C = clutch[["unclamped_x", "unclamped_y", "unclamped_z"]].to_numpy(np.float64)
        C = C[np.isfinite(C).all(1)]
        if len(C) > 1: rep["clutch_frozen_mm"] = float(np.linalg.norm(C - C[0], axis=1).max() * 1000.0)
    rep["gate"] = _gate({
        "hold fraction <= max": (rep["hold_fraction"], 0.25, "max"),
        "workspace clamp fraction <= max": (rep["workspace_clamp_fraction"], g.max_workspace_clamp_fraction, "max"),
        "engaged": (1.0 if rep["engaged"] else 0.0, 1.0, "min"),
    })
    if "clutch_frozen_mm" in rep:
        rep["gate"]["clutch freezes the target (mm)"] = dict(passed=rep["clutch_frozen_mm"] < 1e-6,
                                                             value=rep["clutch_frozen_mm"], threshold=0.0)
    rep["note"] = ("`tcp_*` is the pose that WOULD have been sent (post workspace clamp and velocity/acceleration "
                   "limit); `unclamped_*` is the retargeter target before safety. The gap between them is the "
                   "safety layer doing its job, and is reported as workspace_clamp_depth_mm / speed_limit_hit.")
    return rep


# ---- CLI ---------------------------------------------------------------------------------------------------------
def _segments(args) -> dict[str, tuple[float, float]]:
    segs = {n: (float(a), float(b)) for n, a, b, _ in PROTOCOL_60S} if args.protocol else {}
    for spec in args.segment or []:
        name, a, b = spec.split(":"); segs[name] = (float(a), float(b))
    return segs


def _source(args, cfg: TeleopCfg):
    if args.live:
        cam = OrbbecHeadCamera(fps=int(cfg.rgbd_arm_poc.rate_hz)).open()
        def gen():
            while True: yield cam.read()
        return gen()
    return RecordedRgbdSource(args.episode)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RGB-D-only arm POC: V0 (palm pose) / V1 (3-DoF virtual reBot TCP)")
    ap.add_argument("--stage", choices=("v0", "v1"), default="v0")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episode", help="recorded RGB-D episode dir (color/*.jpg + aligned depth/*.png)")
    src.add_argument("--live", action="store_true", help="live fixed Orbbec camera (macOS: needs sudo)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", default="outputs/rgbd_arm_poc")
    ap.add_argument("--viz", action="store_true", help="V0: write an overlay mp4")
    ap.add_argument("--window", action="store_true", help="V0: live preview window")
    ap.add_argument("--protocol", action="store_true",
                    help="label the standard 60 s trial: " + " ".join(f"{n}:{a}-{b}s" for n, a, b, _ in PROTOCOL_60S))
    ap.add_argument("--segment", action="append", metavar="NAME:T0:T1", help="label a window, repeatable")
    ap.add_argument("--engage-at", type=float, default=2.0, help="V1: seconds before ENGAGE (after the stability gate)")
    ap.add_argument("--clutch-at", type=float, default=None)
    ap.add_argument("--release-at", type=float, default=None)
    ap.add_argument("--record", default=None, help="write a teleop episode (raw/rgbd_hand_pose_live_*, human_hand, events)")
    a = ap.parse_args(argv)

    cfg = load_teleop_cfg()
    poc = cfg.rgbd_arm_poc
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logger = head_writer = None
    if a.record:
        logger = TeleopEpisodeLogger(a.record, metadata=dict(
            stage=a.stage, pose_source=POSE_SOURCE, camera_mode=poc.camera_mode, imu_used=False, causal=True,
            source_episode=a.episode, readiness_note=READINESS_NOTE, config=dict(palm=asdict(poc.palm),
            tracking=asdict(poc.tracking), frames=poc.frames.to_dict(), safety=asdict(poc.safety))))
        if a.live: head_writer = HeadRgbdWriter(a.record)

    if a.stage == "v0":
        df = run_v0(_source(a, cfg), poc, cfg.head_rgbd, max_frames=a.frames, out_dir=out, viz=a.viz,
                    live_window=a.window, logger=logger, head_writer=head_writer, live=a.live)
        rep = v0_report(df, poc, _segments(a))
        df.to_parquet(out / "v0_palm_pose.parquet", index=False)
    else:
        df, meta = run_v1(_source(a, cfg), poc, cfg.head_rgbd, engage_at_s=a.engage_at, clutch_at_s=a.clutch_at,
                          release_at_s=a.release_at, max_frames=a.frames, logger=logger, head_writer=head_writer)
        rep = v1_report(df, poc, meta)
        df.to_parquet(out / "v1_virtual_tcp.parquet", index=False)
    rep["episode"] = a.episode
    rep["gate_passed"] = all(c["passed"] for c in rep.get("gate", {}).values()) if rep.get("gate") else False
    (out / f"{a.stage}_report.json").write_text(json.dumps(rep, indent=1, default=float))
    if logger is not None:
        logger.add_calibration("rgbd_arm_poc", json.loads(json.dumps(rep, default=float)))
        logger.close(status="KEEP", notes=f"RGB-D arm POC {a.stage}")
    if head_writer is not None: head_writer.close()

    print(json.dumps(rep, indent=1, default=float))
    if a.stage == "v0":
        h = rep["headline"]
        print(f"\nV0 HEADLINE   stationary XYZ jitter p95: {h['stationary_xyz_jitter_p95_mm']:.1f} mm"
              f"   return-to-start: {h['return_to_start_error_mm']:.1f} mm"
              f"   tracking-loss rate: {h['tracking_loss_rate']*100:.1f} %")
    print(f"\n{a.stage.upper()} GATE: {'PASS' if rep['gate_passed'] else 'FAIL'}")
    for k, v in rep.get("gate", {}).items():
        print(f"  [{'ok' if v['passed'] else 'XX'}] {k}: {v['value']} (threshold {v['threshold']}){' - ' + v['note'] if v.get('note') else ''}")
    return 0 if rep["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
