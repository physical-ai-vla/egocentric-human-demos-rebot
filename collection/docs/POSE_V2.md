# HandUMI Pose Tracking Architecture v2 — as implemented (M2, 2026-09-10)

Fiducial-free: **Arducam fisheye + ICM42688P → VIO/VO backend → T_world_camera → T_camera_tcp → episode-local TCP → head-timeline
canonical table → HOME-return / static / jump / IMU-consistency QA.** No AprilTag / ArUco / ChArUco / AprilGrid anywhere in
`handumi_collector` (guarded by `tests/handumi/test_pose_pipeline.py::test_no_fiducial_dependency_in_handumi_collector`). Legacy
fiducial code stays in `ego_collector/` (V0) and is never imported by the production package.

Acquisition and pose processing are separate: the recorder never imports `handumi_collector.pose`; a pose failure produces a
REJECT verdict in `derived/`, never a lost raw episode (`test_pipeline_survives_backend_crash_and_unavailable_backend`,
`test_live_runner_failure_never_breaks_raw_recording`).

## Code map (`handumi_collector/pose/`)
| file | role |
|---|---|
| `estimator.py` | `PoseEstimator` (initialize / push_imu / push_image / get_pose / get_quality / reset / finish), `PoseEstimate` (metrics `None` when a backend lacks them), `TrackingState` UNINITIALIZED→INITIALIZING→TRACKING→DEGRADED→LOST, `BackendInfo(uses_imu, metric_scale, online_capable)`, `BackendUnavailable` |
| `backends/` | registry (`pose.yaml backend:`): `mock` (GT replay, tests only, refused by tools without `--allow-mock`), `opencv_vo` (in-repo KLT+essential+PnP monocular VO, **non-metric scale**), `orbslam3` (visual-inertial, external process: EuRoC export + settings yaml → `mono_inertial_euroc` → TUM import; also `trajectory_file=` import), `dpvo` (needs `dpvo` package + weights; rectifies fisheye→pinhole) |
| `calibration.py` | versioned `configs/calibration/{fisheye,camera_imu,camera_tcp}_<side>_vNNN.yaml`; missing = `None`, recorded as `uncalibrated` flags |
| `timing.py` | Teensy device→host clock fit (lower-envelope), `imu_host_times_ns` (+ configured offset), `estimate_camera_imu_offset` (per-frame rotation magnitude vs gyro integral; ±2 ms on synthetic) |
| `imu.py` | gyro integration, still-window detection (HOME start/end), bias from the still start |
| `run.py` | `process_episode` → `derived/pose_<backend>/{left,right}_camera_pose, *_tcp_pose, canonical.parquet, pose_qa.json` |
| `sync.py`, `relative.py` | head-timeline resampling (no long-gap interpolation), `relative_tcp` = inv(T_t)T_{t+lead} clamped (== robot `relative_ee_state`), `c_state_rows` (14-D) |
| `qa.py` | thresholds from `pose.yaml`: HOME return drift PASS/WARN/REJECT, static jitter REVIEW, jumps, valid ratio, lost duration, IMU/VIO rotation residual |
| `live.py` | `LivePoseRunner` (continuous_session mode, online backends; isolated thread, latest-frame tap) |

Tools: `tools.pose_process`, `tools.pose_inspector`, `tools.pose_benchmark`, `tools.camera_imu_offset`, `tools.synth_episode`.
UI/session: `CollectorSession.mark()` / **H** key writes `home_leave` / `home_return` events (optional; QA auto-detects still windows).

## Design points to remember
- Per-hand relative motion only (`ΔT_L`, `ΔT_R`); **no `T_L^-1 T_R`** is produced (V1). The inspector's 3-D view says so.
- Episode-local frame: anchor = mean pose over the HOME-start still window (else first valid pose). `canonical_anchor` is recorded.
- Monocular visual-only backends cannot initialise while still → no poses in the HOME-start window → HOME-return drift is
  reported as unavailable (not faked). A visual-inertial backend is required for metric drift QA.
- Offsets: `pose.yaml offsets.measured=false` ⇒ QA flag until `tools.camera_imu_offset` has been run on real Hpilot data.
- Synthetic end-to-end: `tools.synth_episode` writes a complete raw episode (rendered 3-D point scene, trajectory-consistent IMU
  with bias, one-sided USB latency, configurable camera–IMU offset) used by the tests; `opencv_vo` on it: median rotation error
  < 2°, 0 lost, IMU residual p95 < 5°.

## Runbook
```bash
.venv/bin/python -m handumi_collector.tools.pose_process <session_or_episode> [--backend opencv_vo|orbslam3|dpvo] [--cal-dir …]
.venv/bin/python -m handumi_collector.tools.pose_inspector --episode <ep> [--backend …] [--save out.png --no-show]
.venv/bin/python -m handumi_collector.tools.camera_imu_offset <session> --backend <b>     # → paste offsets: block into pose.yaml
.venv/bin/python -m handumi_collector.tools.pose_benchmark <session> --backends opencv_vo orbslam3   # benchmark.md / .json
```
Remaining M2 hardware steps (§30 of the design): fisheye intrinsics + T_camera_imu (imported as versioned files), camera↔IMU
offset measurement, T_camera_tcp (pivot calibration), Hpilot 10–20, backend comparison, tracker freeze, open-loop replay.
