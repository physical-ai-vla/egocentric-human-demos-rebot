# pipeline/ MANIFEST

This folder holds the human-data pipeline: HandUMI export, MASt3R pose and scale, gripper extraction, census, Gen-1 pseudo-joint retargeting,
the C-old dataset builders, retarget v2 with its frozen contracts, and the Gen-2 CART20 package. Assembled 2026-10-07. Sources were not modified.

## Choice rule

`~/egocentric-human-demos-rebot` (the cleaned public-style copy) was taken as the base. Each file was then compared with its working-dir origin,
using `docs/SOURCES_v2.md` and a basename search over `~/c8`, `~/umi_bridge`, `~/ego_collector`, `~/humanik_retarget`, `~/holobrain-mac-model` and `~/ego_cart20`.

- If the working copy was newer and differed in code, the working copy won.
- If the repo copy differed only by path sanitization (`${HOME}`, `${REMOTE_HOME}`, `${SHARED_ROOT}`, ...), the working original was restored.
  Those placeholders sit inside Python strings and bash literals, so the sanitized copies do not run. This is a private company repo, so the internal paths are kept.

## Source -> dest

| dest | source | choice |
|---|---|---|
| `export/`, `grip/`, `census/`, `pseudo_joint/`, `dataset/`, `dataset_v2/`, `retarget_v2/` (incl. `contracts/`, `chains/`), `mast3r_pose/` | `~/egocentric-human-demos-rebot/pipeline/*` | repo copy (identical code to origins except the items below) |
| `calibration/` (fisheye/camera-IMU/head C922 yamls, kalibr results, gripper_v0xx, camera_tcp_v2, tcp_convention) | `~/egocentric-human-demos-rebot/calibration/` | repo copy |
| `pseudo_joint/eef_kin.py` | `~/holobrain-mac-model/eef_kin.py` (09-29) | **newer working copy**: `max_joint_delta=None` support (repo copy crashes on None) |
| `mast3r_pose/c8_rest490_resident.sh`, `mast3r_pose/c8_run_5080.sh` | `~/c8/` | original restored (repo copy = path-sanitized only) |
| `mast3r_pose/cp6_scale.py` | `~/umi_bridge/trackA_mast3r_pose_v1/cp6_scale.py` | original restored (sanitized only) |
| `retarget_v2/contracts/R30_TR_EPISODES.md`, `r150_nested_subset_v1.json` | `~/umi_bridge/track_c/v2k/` | original restored (sanitized only) |
| `retarget_v2/chains/hrl80_chain.sh` | `~/c8/robotlike/hrl80_chain.sh` | original restored (sanitized only) |
| `cart20/ego_cart20/**` (package) | `~/egocentric-human-demos-rebot/cart20/ego_cart20` **plus** files that exist only in `~/ego_cart20/ego_cart20` | see below |
| `cart20/tests/` | repo `cart20/tests` | repo copy (differs from `~/ego_cart20/tests` only by "user" wording in a docstring) |
| `cart20/configs/ego_pretrain_relonly_d6.yaml` | repo `cart20/ego_pretrain_relonly_d6.yaml` | repo copy, with `${REMOTE_HOME}` replaced by `/home/bh-aiteam` (the original value) |
| `cart20/README.md`, `cart20/docs/SPEC_STATUS_2026-10-01.md` | `~/ego_cart20/README.md`, `SPEC_STATUS_2026-10-01.md` | working dir (not in the repo copy) |
| `cart20/docs/*__*.md` | `~/ego_cart20/reports/{ego_vs_r312c,domain_slot_audit}/*.md` | analysis write-ups only (INTERPRETATION, STATE_JUMP_ROOT_CAUSE, HANDOFF_joint5_lock_deploy, summary, D6_D20_PROVENANCE) |

### CART20 package: repo copy vs `~/ego_cart20`

The 7 files present in both differ **only in comment wording** ("user decision" -> "design decision"). The repo copy (10-04) is newer and was kept:
`config.py`, `labels/build_state20.py`, `preprocessing/{gripper,pose_filter,resample}.py`, `right_only.py` and `scripts/export_lerobot_robotized.py`.

Files that exist only in `~/ego_cart20` were added, because they are newer:

- `sources/handumi_export.py`: the raw HandUMI loader that `right_only.py` imports. The repo copy lacked it, so the repo's `right_only` could not import.
- `scripts/export_lerobot_right_only_robotcam.py`: the robotcam v2 exporter, measured C922 plus hand-eye (HRA v2 / A100).
- `scripts/viz_hra_mast3r.py`
- `validation/parity_gcal1.py`
- Soft-fold experiment (10-07): `softfold_h32.py`, `sources/softfold_export.py`, `scripts/{convert_dataset_softfold,export_lerobot_softfold,export_softfold_abs,validate_dataset_softfold}.py`.

`cube_pnp.py` (v2 with the single-face IPPE model and jaw-occlusion reject) and `right_only.py` (sanity gate) are the same in both, apart from comments.

## Excluded

- `.DS_Store`, `__pycache__`.
- From `~/ego_cart20`: `launch/` (moved to `training/launchers/ego_cart20/`, logs dropped), `launch/*.log`, `launch/train_relcart20_v4_relonly_d6.sh.orig`
  (the node copy in `training/launchers/node4090/` is used instead), all of `reports/` except the `.md` write-ups
  (csv/json outputs, `preserved_diag_20261004/ray_logs/*.log`), and `.git` (an empty repo with no commits).
- No pyproject.toml / setup.py exists in either source. The package is used via `cd ~/ego_cart20 && python -m ego_cart20...` or `PYTHONPATH`.
  The task brief mentioned one; it does not exist.
- The `.npy` files in `retarget_v2/contracts/` are small frozen seed banks and are kept. No dataset, parquet, video or checkpoint files were copied.

## Notes

- `calibration/README.md` is the repo copy. `~/ego_cart20/README.md` was flagged as "newer", but only because the basename matches; it is a different document and is copied to `cart20/README.md`.
- Scripts reference the working roots `~/c8`, `~/umi_bridge`, `~/ego_collector`, `~/ego_cart20`, `/srv/data/johann` (5090 NFS) and `/home/bh-aiteam` (4090).
