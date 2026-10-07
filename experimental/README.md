# Experimental

This folder holds research code that is **not** part of the production HandUMI → reBot path. It is kept for its
results and because parts of it are reused elsewhere. Source: `~/ego_collector` (branch `orbbec-rgbd`, HEAD `12db478`,
plus the uncommitted working tree as of 2026-10-07). See `MANIFEST.md`.

| path | what | status |
|---|---|---|
| `ego_collector/` | **V0 egocentric collector** (2026-08-28 to 08-31). A head C922 plus AprilTag gloves (wrist 10/20, finger 11/12/21/22) and 4 world tags (100–103) feed an offline pseudo-action pipeline: `record → process-tags → process-hands[3d] → generate-actions → qa`. It also has WiLoR/HaWoR 3D-hand and Aero-hand adapters. | **Superseded.** Real-hardware tests showed the head C922 misses the printed tags too often. The work pivoted to raw ego video, then to HandUMI. **Still a live dependency:** `ego_collector.camera.*` (UVC/Orbbec capture, ChArUco helpers) is imported by `../collector`, so do not delete this package. |
| `ego_teleop/` | **Wearable teleop / VIO** (2026-09-10 onward). The wristband is the fisheye plus IMU of the HandUMI unit. Tag-free VIO gives a relative SE(3) that drives the reBot arm (clutch, LOST-hold, safety). Head RGB-D metric hand pose → DexPilot → Aero Hand. It includes Kalibr/Allan calibration tooling (`calibration/bundle.py`, `tools/kalibr_*`, `tools/imu_allan.py`), OpenVINS and MASt3R-live tracking backends, a fused wrist provider (chest RGB-D + wrist cam + IMU), an RGB-D arm proof of concept, HUDs, and `tools/g0_mount_qual.py` (gripper-first wrist mount qualification, 2026-10-02). | **Research, not deployed.** The software for milestones A0–A2/M1 is built and tested. Live wrist VIO is the blocker: OpenVINS drifts and diverges when a hand waits, and the user retired OpenVINS on 2026-09-17. MASt3R-SLAM gives a good RGB-only visual pose on a GPU, but its scale must come from elsewhere (IMU-VI per episode, or cube PnP). The finger source is still gate-pending. The head cameras never drive the arm. |
| `scripts/` (RGB-D body tracking) | `handumi_body_track.py`, `body_*`, `cube_tcp_detect.py`, `cube_center_planes.py`, `imu_orientation.py`, `rgbd_qa10_report.py`, `rgbd_tcp_qa.py`, `hand_detector_audit.py`, `cube_pnp_anchor.py`, `imu_bridge_test.py`, `camera_to_grasp_offset.py` (2026-09-18/19). These track the HandUMI **body** in the head Orbbec RGB-D (table-plane RANSAC, then a depth band, then dark-blob segmentation), as a metric teacher label. | **Parked.** The RGB-D sensor passed QA 8/8. The palm route failed: MediaPipe finds 0 hands with the gloved HandUMI. Body-centroid tracking works when still (0.04–1.9 mm), but forearm merging causes most of the jumps above 50 mm. A visible-surface centroid is not rigid, and the `T_body_tcp` fit was blocked by an unmodelled world-frame yaw. A canonical-model tracker was built. The Orbbec cannot run alongside the wrist UVC cameras, so this was never a production path. |
| `scripts/mast3r_live_server.py` | Live MASt3R socket front end for `ego_teleop` | Research |
| `scripts/generate_apriltags.py` | V0 tag print sheets | V0 only |

Other status notes:

- **Wrist VIO calibration is closed**, so do not reopen it. Right: `camera_imu_right_v002` / `wrist_bundle_right_v005`
  (Kalibr, offset 32.77 ms). Left: `camera_imu_left_v001` / `wrist_bundle_left_v005` (offset 30.17 ms frozen by decision).
  The left camera latency looked session-dependent (30–36 ms). On 2026-10-06 the IMU extrinsic `T_b_c1` used in the
  exports was found to be about 18° off about camera x; the correction is `camera_imu_right_rotfix_20261006.npy` in
  `../collector/configs/calibration/`.
- The segmented relative VIO (`../collector/handumi_collector/pose/backends/segmented.py`) helped the right wrist
  (usable fraction 0.63 → 0.91) but did not fix left-wrist divergence.

## Layout and install

- `configs/` holds the configs for these packages: `ego_teleop/` plus the V0 yamls (`world_tag_map`,
  `wrist_extrinsics`, `qa`, `aperture_calibration`, `collection_plan`, `charuco*`, `camera/`).
- `configs/calibration` and `configs/handumi` are **symlinks** to `../collector/configs/...`. `ego_teleop` resolves
  config paths from its own repo root, and the calibration files must stay single-sourced.
- `assets/aero_hand_open/` holds the Aero Hand Open URDFs (Apache-2.0, TetherIA, see `NOTICE`) for the FK tests and
  the Aero branch.
- `tests/teleop` holds the `ego_teleop` tests. `tests/v0` holds the `ego_collector` V0 tests and their `synth.py`.

```bash
cd experimental
uv venv && uv pip install -e . -e ../collector     # ego_teleop imports handumi_collector
uv run pytest tests -q                             # 356 passed on this copy (same as the source tree)
```

`pyorbbecsdk` (head RGB-D), `dex-retargeting` (`--extra aero`), OpenVINS (`ov_bridge`) and MASt3R are external and
optional; their tests skip when they are absent. `scripts/rgbd_tcp_qa.py` shells out to `scripts/ego_ik_retarget.py`,
which belongs to the ego pipeline (`../pipeline`) and is not in this folder.

Runbooks are in `docs/ego_teleop/`:

- `PLAN.md`
- `M1_RUNBOOK.md`
- `MULTISENSOR_FUSION.md`
- `MAST3R_LIVE_PROTOCOL.md`
- `RGBD_ARM_POC.md`
- `AERO_RGBD_RUNBOOK.md`
- `P0_SENSORS.md`
- `HW_BASELINE.md`
- `TELEOP_IMPROVEMENT_PLAN.md`
- `AERO_A0_AUDIT.md`

The full RGB-D teacher QA record is `../collector/docs/RGBD_TEACHER_QA.md`.
