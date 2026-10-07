"""Robot-agnostic teleop episode log (spec sections 14-15). Raw video + IMU + camera timestamps are written by the
existing handumi_collector recorder (MCAP/MP4); this logger adds the teleop-specific streams next to them and never
touches or overwrites a raw file that already exists.

episode_xxxxxx/
├── metadata.json
├── head.mp4  left_wrist.mp4  right_wrist.mp4  sensors.mcap (/left/imu, /right/imu, frame_meta)   <- handumi_collector recorder (raw)
├── raw/      wrist_pose_live_left.parquet  wrist_pose_live_right.parquet  finger_state.parquet  events.parquet
│          human_hand.parquet   (head RGB-D: 2D + camera-metric + palm-local landmarks, validity, depth confidence,
│                                handedness, tracking state — the CANONICAL raw hand data, spec section 23)
├── head/     color/frame_NNNN.jpg  depth/frame_NNNN.png (16-bit, aligned)  frames.parquet  (HeadRgbdWriter)
├── robot/    rebot_state.parquet rebot_command.parquet aero_state.parquet aero_command.parquet aero_target.parquet
├── calibration/  fisheye_intrinsics.json camera_imu_extrinsic.json wrist_camera_extrinsic.json human_robot_frame.json aero_retarget.json
└── processed/    (written later by processing; never by the live logger)

Every row keeps the source/host timestamps it was produced with; nothing is resampled here (invariant E).
Head cameras are observation streams only: neither the C922 nor the RGB-D unit has a pose row anywhere in this
layout, and neither feeds the arm. The RGB-D head camera additionally reconstructs the operator's HAND — that is
`raw/human_hand`, a human-side stream, and it drives the Aero hand alone.

Human pose is saved BEFORE retargeting and is never replaced by the Aero target (spec section 23): if Aero is later
swapped for another dexterous hand, the episodes stay usable.

LIVE vs REFINED (storage contract frozen 2026-09-11, before bulk recording). `live` is what the causal/online
estimator produced while recording; `refined` is recomputed offline and non-causally from the same raw camera + IMU.
They are different data:

    INVARIANT: a refined trajectory NEVER overwrites or deletes the live one.

Pose streams therefore carry an explicit `_live` suffix from the start, so adding `wrist_pose_refined_*` /
`delta_tcp_refined_*` later (ladder step 11) needs no rename and leaves no unmarked stream. Training configs then pick
`action_pose_source: live | refined` explicitly, which makes that an ablation for free. Provenance per stream lives in
metadata.json `pose_streams`. `delta_tcp_*` does not exist yet — when it does it follows the same rule
(`delta_tcp_live_*`, later `delta_tcp_refined_*`). Pose is strictly a derived signal, but the live one is kept inside
the raw episode bundle because it is recording-time state; the invariant above matters more than the folder."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

EPISODE_LAYOUT = {
    "raw": ("wrist_pose_live_left", "wrist_pose_live_right",
            "rgbd_hand_pose_live_left", "rgbd_hand_pose_live_right",
            "rgbd_relative_pose_live_left", "rgbd_relative_pose_live_right",
            "fused_wrist_pose_live_left", "fused_wrist_pose_live_right",
            "fused_relative_pose_live_left", "fused_relative_pose_live_right",
            "finger_state", "human_hand", "events"),
    "robot": ("rebot_state", "rebot_command", "aero_state", "aero_command", "aero_target"),
}


class TeleopEpisodeLogger:
    def __init__(self, episode_dir: str | Path, *, metadata: dict | None = None) -> None:
        self.dir = Path(episode_dir)
        for sub in ("raw", "robot", "calibration", "processed"): (self.dir / sub).mkdir(parents=True, exist_ok=True)
        clash = [str(self.dir / grp / f"{n}.parquet") for grp, names in EPISODE_LAYOUT.items() for n in names if (self.dir / grp / f"{n}.parquet").exists()]
        if clash: raise FileExistsError(f"refusing to overwrite raw episode files: {clash}")
        self.meta = dict(metadata or {}); self.meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.meta.setdefault("start_monotonic_ns", time.monotonic_ns()); self.meta.setdefault("start_wall_ns", time.time_ns())
        self._rows: dict[str, list[dict]] = {n: [] for names in EPISODE_LAYOUT.values() for n in names}
        (self.dir / ".incomplete").write_text("")
        self.closed = False

    # ---- streams -------------------------------------------------------------------------------------------
    def add_wrist_pose(self, side: str, wp) -> None:
        """Each wrist unit is an independent pose sensor with its own VIO world; streams are never merged here.

        This is the **live** stream: whatever the causal/online estimator produced at recording time. An offline
        refined trajectory (ladder step 11) is a DIFFERENT stream and never overwrites this one."""
        if side not in ("left", "right"): raise ValueError(side)
        name = f"wrist_pose_live_{side}"
        if name not in self.meta.setdefault("pose_streams", {}):
            self.declare_pose_stream(name, estimator=getattr(wp, "source", "") or "unknown",
                                     source_camera=f"{side}_wrist", source_imu=f"{side}_wrist")
        self._rows[name].append(dict(wp.to_row(), side=side))

    def add_rgbd_hand_pose(self, side: str, wp, *, palm=None) -> None:
        """The RGB-D-only POC arm pose source (`fixed_rgbd_hand`) — a DIFFERENT sensor from the wrist VIO.

        It gets its own stream family on purpose: `wrist_pose_live_*` means "wrist fisheye + IMU + OpenVINS" and must
        keep meaning that, so an episode recorded with the POC source can never be read as a wrist-VIO episode. The
        raw palm read-out (`rgbd_palm_position_raw` / `rgbd_palm_orientation_raw` of the plan) travels as columns of
        THIS table rather than as two extra files: they are the same per-frame samples, before grading."""
        if side not in ("left", "right"): raise ValueError(side)
        name = f"rgbd_hand_pose_live_{side}"
        if name not in self.meta.setdefault("pose_streams", {}):
            self.declare_pose_stream(name, estimator=getattr(wp, "source", "") or "unknown",
                                     source_camera="head_rgbd", source_imu="", pose_source="fixed_rgbd_hand",
                                     camera_mode="fixed", imu_used=False)
        row = dict(wp.to_row(), side=side)
        e = dict(getattr(wp, "extra", {}) or {})
        for k in ("arm_pose_health", "arm_pose_reason", "hand_health", "hand_reason", "n_palm_landmarks",
                  "palm_depth_spread_m", "wrist_offset_m", "palm_scale_m", "span_index_pinky_m", "span_wrist_middle_m",
                  "palm_scale_ratio", "origin", "orientation", "age_ms"):
            row[k] = e.get(k)
        R = e.get("R_palm")                     # raw palm orientation, recorded even while the arm runs 3-DoF
        if R is not None:
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(np.asarray(R, float)).as_quat()
            for k, v in zip(("raw_qx", "raw_qy", "raw_qz", "raw_qw"), q): row[k] = float(v)
        self._rows[name].append(row)

    def add_fused_wrist_pose(self, side: str, wp) -> None:
        """The multi-sensor fused arm pose source (`fused_wrist`): chest RGB-D + wrist Arducam + wrist IMU.

        A third stream family, for the same reason the POC got its own (multi-sensor spec section 18): `wrist_pose_live_*`
        means "wrist fisheye + IMU alone" and must keep meaning that. THIS table is the exact causal trajectory that was
        available to robot control, and it carries, per row, the evidence behind it — the RGB-D anchor (all three axes,
        raw in the chest camera frame and mapped into the VI world), the VI pose both as the backend's map pose and as
        the continuous local pose, the applied correction, the map correction that was absorbed, both branch healths and
        the unified fusion state. A later offline refinement is a DIFFERENT stream and never overwrites this one."""
        if side not in ("left", "right"): raise ValueError(side)
        name = f"fused_wrist_pose_live_{side}"
        e = dict(getattr(wp, "extra", {}) or {})
        if name not in self.meta.setdefault("pose_streams", {}):
            self.declare_pose_stream(name, estimator=getattr(wp, "source", "") or "unknown",
                                     source_camera=f"head_rgbd+{side}_wrist", source_imu=f"{side}_wrist",
                                     pose_source="fused_wrist", fusion_mode=e.get("fusion_mode", ""),
                                     camera_mode="chest_rgbd+wrist", imu_used=True)
        row = dict(wp.to_row(), side=side)
        for k, v in e.items():
            if k in row or isinstance(v, (list, tuple, dict)): continue      # never shadow the pose columns
            row[k] = v
        self._rows[name].append(row)

    def add_fused_relative_pose(self, side: str, row: dict) -> None:
        """Per-tick relative motion of the fused source: dT_human from ENGAGE, the mapped robot delta, the TCP target
        actually commanded, and the latencies. Sampled on the teleop clock, not on any sensor clock."""
        if side not in ("left", "right"): raise ValueError(side)
        self._rows[f"fused_relative_pose_live_{side}"].append(dict(row, side=side))

    def add_rgbd_relative_pose(self, side: str, row: dict) -> None:
        """Per-tick relative motion of the POC source: dT_human from ENGAGE, the mapped robot delta, and the resulting
        TCP target. Sampled on the teleop clock, not the camera clock, which is why it is its own table."""
        if side not in ("left", "right"): raise ValueError(side)
        self._rows[f"rgbd_relative_pose_live_{side}"].append(dict(row, side=side))

    def declare_pose_stream(self, name: str, *, pose_type: str = "live", estimator: str = "", causal: bool = True,
                            source_camera: str = "", source_imu: str = "", **extra) -> None:
        """Provenance for one pose stream, recorded in metadata.json.

        `live` = produced by a causal/online estimator while recording. `refined` = recomputed offline, non-causal,
        from the episode's full raw camera + IMU. A future refined stream adds source_episode / refinement_backend /
        refinement_version here; those fields are deliberately not invented now."""
        self.meta.setdefault("pose_streams", {})[name] = dict(
            pose_type=pose_type, estimator=estimator, causal=bool(causal),
            source_camera=source_camera, source_imu=source_imu, **extra)
    def add_finger_state(self, hf) -> None: self._rows["finger_state"].append(hf.to_row())
    add_hand = add_finger_state
    def add_hand_pose(self, est, *, supervised_health: str | None = None, reason: str = "") -> None:
        """One `HandPoseEstimate` (head RGB-D or monocular). The supervisor's verdict is stored next to the provider's
        own so a later analysis can tell "the provider saw a hand" from "the policy accepted it"."""
        row = est.to_row()
        row["supervised_health"] = supervised_health or est.health.value
        row["health_reason"] = reason or est.extra.get("health_reason", "")
        self._rows["human_hand"].append(row)

    def add_aero_target(self, target) -> None:
        """Retargeter output (16 joints, rad) with its optimiser diagnostics, before the SDK's own conversion."""
        self._rows["aero_target"].append(target.to_row())

    def add_rebot_state(self, rs) -> None: self._rows["rebot_state"].append(rs.to_row())
    def add_aero_state(self, a) -> None: self._rows["aero_state"].append(a.to_row())
    def add_aero_command(self, c) -> None: self._rows["aero_command"].append(c.to_row())
    def add_command(self, cmd) -> None: self._rows["rebot_command"].append(cmd.to_row())

    def add_event(self, name: str, t_ns: int, **payload) -> None:
        self._rows["events"].append(dict(t_ns=int(t_ns), event=name, payload=json.dumps(payload, default=str)))

    def add_calibration(self, name: str, obj: dict) -> None:
        (self.dir / "calibration" / f"{name}.json").write_text(json.dumps(obj, indent=1, default=_json_default))

    # ---- finalize ------------------------------------------------------------------------------------------
    def close(self, *, status: str = "KEEP", notes: str = "") -> dict:
        if self.closed: return self.meta
        counts = {}
        for grp, names in EPISODE_LAYOUT.items():
            for n in names:
                rows = self._rows[n]; counts[n] = len(rows)
                pd.DataFrame(rows).to_parquet(self.dir / grp / f"{n}.parquet", index=False)
        self.meta.update(status=status, notes=notes, end_monotonic_ns=time.monotonic_ns(), end_wall_ns=time.time_ns(), row_counts=counts,
                         duration_s=round((time.monotonic_ns() - self.meta["start_monotonic_ns"]) / 1e9, 3))
        (self.dir / "metadata.json").write_text(json.dumps(self.meta, indent=1, default=_json_default))
        (self.dir / ".incomplete").unlink(missing_ok=True); self.closed = True
        return self.meta


class HeadRgbdWriter:
    """Head RGB-D raw frames, in the same on-disk shape the Orbbec recorder already uses (colour jpg + 16-bit aligned
    depth png + a timestamp table). Writing is a per-frame call so the capture thread never blocks on encoding more
    than one frame, and the calibration the frames were taken with is copied in beside them."""

    def __init__(self, episode_dir: str | Path, calib=None, *, jpeg_quality: int = 95) -> None:
        import cv2
        self._cv2 = cv2
        self.dir = Path(episode_dir) / "head"
        (self.dir / "color").mkdir(parents=True, exist_ok=True)
        (self.dir / "depth").mkdir(parents=True, exist_ok=True)
        self.q = int(jpeg_quality); self.rows: list[dict] = []
        if calib is not None:
            (self.dir / "calibration.json").write_text(json.dumps(calib.to_dict(), indent=1, default=_json_default))
            self.depth_scale_m = calib.depth_scale_m
        else:
            self.depth_scale_m = 0.001

    def add(self, frame) -> int:
        i = len(self.rows)
        self._cv2.imwrite(str(self.dir / "color" / f"frame_{i:06d}.jpg"), frame.color_bgr, [self._cv2.IMWRITE_JPEG_QUALITY, self.q])
        raw = np.rint(np.nan_to_num(frame.depth_m / self.depth_scale_m, nan=0.0)).astype(np.uint16)  # rint, not truncate: the store must be lossless
        self._cv2.imwrite(str(self.dir / "depth" / f"frame_{i:06d}.png"), raw)
        self.rows.append(dict(frame_index=i, t_ns=int(frame.timestamp_ns), source_t_ns=frame.source_timestamp_ns))
        return i

    def close(self) -> None:
        pd.DataFrame(self.rows).to_parquet(self.dir / "frames.parquet", index=False)


LEGACY_STREAM_NAMES = {"wrist_pose_live_left": "wrist_pose_left", "wrist_pose_live_right": "wrist_pose_right"}


def stream_path(episode_dir: str | Path, name: str, group: str = "raw") -> Path | None:
    """Resolve one stream file, preferring the canonical name and falling back to the pre-2026-09-11 filename.

    A legacy `wrist_pose_<side>.parquet` IS a live stream — it was written by the online estimator at recording time —
    so it is returned as such. Old episodes stay readable; new recordings only ever write canonical names."""
    d = Path(episode_dir)
    p = d / group / f"{name}.parquet"
    if p.exists(): return p
    legacy = LEGACY_STREAM_NAMES.get(name)
    if legacy:
        lp = d / group / f"{legacy}.parquet"
        if lp.exists(): return lp
    return None


def _json_default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return str(o)
