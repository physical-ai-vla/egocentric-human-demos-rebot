# Hardware bring-up baseline — frozen before the first three-sensor run

Frozen 2026-09-22 12:11 +0900 so a regression found during hardware work can be told apart from a failure that was
already there. Re-run the same command after hardware changes and diff the list.

```
commit      12db478b33c7a5c78868dcf13955542eaa0fc168
branch      orbbec-rgbd
subject     upstream v2 (implementation only, E50 runs untouched): MASt3R-only mode, spike-filtered re
tracked mods 199 | untracked 682
```

## Test baseline

```
12 failed, 526 passed, 1 warning in 111.95s (0:01:51)
```

These 12 fail **before any hardware work** and none of them import `ego_teleop` (checked). They fail on
physical collector state — the gripper jaw reading OPEN at rest, the AVFoundation listing position, a real
recording session. A 13th failure, or any `tests/teleop` failure, is a regression from hardware work.

```
FAILED tests/handumi/test_autoloop.py::test_real_session_records_two_episodes_by_itself
FAILED tests/handumi/test_autoloop.py::test_toggle_off_mid_episode_and_while_waiting
FAILED tests/handumi/test_camera_identity.py::test_a_listing_position_is_not_an_opencv_index
FAILED tests/handumi/test_m15_production.py::test_record_without_imu_and_optional_aux_missing
FAILED tests/handumi/test_m15_production.py::test_error_review_on_required_device_failure
FAILED tests/handumi/test_m15_production.py::test_disk_gate_and_preflight_gate
FAILED tests/handumi/test_m15_production.py::test_hardware_check_records_and_validates
FAILED tests/handumi/test_m15_production.py::test_integrity_validator_detects_problems
FAILED tests/handumi/test_m15_production.py::test_ui_previews_every_configured_camera_and_renders_depth
FAILED tests/handumi/test_m15_production.py::test_ui_without_a_c922_still_previews_the_depth_camera
FAILED tests/handumi/test_m1_gripper_calibration.py::test_set_closed_open_via_session
FAILED tests/handumi/test_pose_pipeline.py::test_session_home_marks_written_to_events
```

## Hardware attached at freeze time

```
XIAO nRF52840 Sense   /dev/cu.usbmodem31401   D2FB7365FE48ED0F   PRESENT  (P0-1 PASS)
Teensy IMU left       /dev/cu.usbmodem205401601  20540160        present (not used by the fusion branch)
Teensy IMU right      /dev/cu.usbmodem207780701  20778070        present (not used by the fusion branch)
3x C922                                                            present (reBot rig, not this branch)
wrist Arducam         'Arducam 1080P Low Light'                    ABSENT   <- blocks P0-2 and P0-4
chest Orbbec RGB-D    pyorbbecsdk query_devices() == 0             ABSENT   <- blocks P0-3 and P0-4
```
