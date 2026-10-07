# Versioned calibrations (latest vNNN wins; never overwrite; sessions/episodes record the version name)
- `gripper_vNNN.yaml` — jaw encoder ticks_closed / ticks_open per side (UI SET CLOSED / SET OPEN / SAVE).
- `fisheye_<side>_vNNN.yaml` — Arducam intrinsics: `model: kannala_brandt` (cv2.fisheye) or `pinhole`, `K` 3x3, `D` [k1..k4], `image_size` [w,h]. Written by `tools.calibrate_fisheye` (plain checkerboard) or imported from Kalibr.
- `camera_imu_<side>_vNNN.yaml` — `T_camera_imu` 4x4 (IMU frame in camera frame), optional `time_offset_ms`, `imu_noise`.
- `camera_tcp_<side>_vNNN.yaml` — `T_camera_tcp` 4x4 (pinch-centre TCP in camera frame), `method`.
- `head_mount_vNNN.yaml` — the fixed head-camera holder: hand-measured mount `geometry` (height, table-edge distance, pitch/yaw/roll) so the mount can be physically restored, `T_table_camera` 4x4 (camera expressed in the table frame, so it carries camera points into table coordinates), the `board` it was solved from, `rms_reproj_px`, an optional depth-plane `depth_check`, and `repeatability` across remounts. Written by `tools.calibrate_table_frame`.

Missing file ⇒ consumers record `uncalibrated` explicitly (identity is never assumed silently).

**On fiducials.** The rule is no AprilTag / ArUco / ChArUco anywhere in the runtime pose pipeline, and it is enforced by
`tests/handumi/test_pose_pipeline.py::test_no_fiducial_dependency_in_handumi_collector`. It has never meant "no
calibration targets": `fisheye_<side>` is solved from a plain checkerboard, and `head_mount` is too. A checkerboard is a
one-off calibration-time object that leaves nothing to detect at recording time, which is the property the rule is
about — as distinct from a tag-based world map the pipeline would depend on frame by frame.


## head_mount provenance (2026-09-14)

`latest` is **v005** and it is the one to use. The earlier four are kept rather than deleted, because why a calibration
was rejected is part of its record, and all five are the same rig on the same afternoon.

| | why not |
|---|---|
| v001 | The table frame's Z pointed into the table, so the camera read as height **−48.3 cm** at pitch −62.5° — under the table, looking up. |
| v002 | Z fixed; roll reported as the image x axis tipping out of the table plane, which is a projection that shrinks with pitch (7.1° for a 10° roll at 45° of tilt). |
| v003 | Roll correct at last, but the capture behind v001–v003 turned the board 180° partway through "for diversity". Averaging across that defines two table frames and returns neither. |
| v004 | Board still moved between views, and the tooling did not yet record `view_spread`, so there is no way to check it after the fact. |
| **v005** | Board held still: `view_spread` **1.99 mm / 0.49°**, against a 5 mm / 1.0° limit. Reprojection 0.414 px, depth plane 1.14 mm RMS agreeing with the board to 0.34°, valid depth 100 %. |

Holder geometry repeated independently across v003–v005 at **48.2 cm / 62° pitch / −9° roll** (spread 2 mm and 1°),
which is the part that should agree — the mount did not move between captures. `board_distance` and `yaw` differ because
the board was put down somewhere different each time, and that is precisely what it means for the table frame to be
defined by where the board is.

Two lessons worth keeping. Diversity is what **lens intrinsics** need and the opposite of what an **extrinsic** needs:
`fisheye_<side>` wants the board everywhere in frame, `head_mount` wants it nailed down. And nothing in the per-view
reprojection error reveals a board that moved — every view still solves at 0.4 px — which is why `view_spread` is
written into the file and the tools refuse past 5 mm / 1°.
