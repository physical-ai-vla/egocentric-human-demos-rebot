# umi_laundry: UMIalign pose + visual-gripper pipeline (2026-10-12 → 10-16)

Turns a servo-less HandUMI UMIalign session (T-shirt flattening) into per-frame wrist TCP poses and binary gripper labels.
Overview and numbers: [`docs/UPDATE_2026-10-20.md`](../../docs/UPDATE_2026-10-20.md), Sec. 2–4.

Order per session:

| step | code | output |
|---|---|---|
| raw QA, wrist-flip check, end trim | `umi_fusion_scripts/raw_qa.py`, `detect_wrist_flip.py`, `space_trim_points.py` | `raw_qa.json`, `exclude_from_training`, `trim.json` |
| visual jaw opening | `umi_fusion_scripts/jaw_track.py` (uses `jaw_redx.red_mask`) | `derived/jaw/jaw_{side}.npz` |
| MASt3R-SLAM (restart_v1) | `umi_fusion_scripts/m3_runner/run_perframe_restart.py`; drivers `drivers/*m3*.sh`, `../../collection/scripts/robotlike_session_check_4090.sh` | `runs_restart/<tag>/ss1.csv` |
| Orbbec body tracking | `orbbec_tracking/handumi_body_track.py --profile laundry_v1`, `handumi_canonical_track.py` | `derived/body_track/canonical.npz` |
| AprilTag anchor gate | `umi_fusion_scripts/orbbec_tag_gate.py` | `derived/body_track/tag_gate.npz` |
| smoother | `../../cart20/ego_cart20/umi_fusion/run_smoother.py --rcb data_v003 --rc-prior tips_v001 --sigma-rc 0.0001 --tag-gate` (drivers `drivers/smooth_*.sh`) | `derived/umi_fusion_rcb3gate/smoothed_{left,right}.npz` |
| absolute acceptance | `umi_fusion_scripts/orbbec_tag_check.py` (TCP vs jaw-tape tips in the Orbbec) | per-episode table |
| camera↔IMU check | `umi_fusion_scripts/imu_cam_extrinsic_check.py` | csv |
| metrics | `drivers/umialign_metrics.py --session <dir>` | `metrics_*.md/csv` |
| export | `../../cart20/ego_cart20/sources/hfold_export.py` (+ `jaw_visual.py`, `trim.py`) → LeRobot converter `--fusion-dir umi_fusion_rcb3gate` | dataset |

See `MANIFEST.md` for sources.
