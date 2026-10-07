# Upstream: HandUMI (robonet-ai)

Our HandUMI → reBot work builds on two open-source repositories. Both are licensed Apache-2.0.

| repo | URL (our `origin`) | base commit we build on | commit date |
|---|---|---|---|
| handumi-sw | https://github.com/robonet-ai/handumi-sw | `33cc437229e43ed4bb92b25d4465248f0edbbcd9` ("feat(visualize): replace Rerun preview with lightweight OpenCV wrist view") | 2026-08-26 |
| handumi-hw | https://github.com/robonet-ai/handumi-hw | `e58de33b9a88bd208e1a40e52ea575eaf68963db` ("docs link updated") | 2026-08-20 |

Note that the upstream READMEs link to `github.com/murobotics-ai/...`. That looks like the same project under another
org name. The hashes above are the source of truth.

## handumi-sw → local branch `rebot-ego`

The local clone (`~/handumi-sw`) is on branch `rebot-ego`, which points at the same commit as upstream `main`
(`33cc437`). **All of our changes are uncommitted working-tree changes**, so they are exported here as files:

| path | content |
|---|---|
| `handumi-sw-rebot-ego/tracked.patch` | `git -C ~/handumi-sw diff -- . ':(exclude)uv.lock'`, which changes 8 tracked files (+111/−8, see `tracked.diffstat.txt`). `uv.lock` is left out: it is regenerated, and the diff was the re-resolve after the vosk marker change plus marker/ordering churn. |
| `handumi-sw-rebot-ego/new_files/` | The 33 untracked files (`git ls-files --others --exclude-standard`, listed in `new_files.list`) plus the local `configs/rig.yaml`, which upstream gitignores. Paths are preserved. None is larger than 5 MB, so nothing was skipped. |

The fork adds **ego-bimanual capture for the reBot B601** (runbook: `new_files/docs/rebot/RUNBOOK.md`):

- AprilTag EEF tracking:
  - `tracking/apriltag.py`: a head fisheye sees world tags 100–103, which gives `T_cam_world`.
  - L/R UMI tag bundles 10/11 and 20/21 give the TCP.
  - New recording device `--device apriltag`.
  - `scripts/apriltag_tools.py` (print, preview, bundle-calib).
  - `scripts/setup/calibrate_head_camera.py` (fisheye intrinsics, world map).
- Feetech jaw telemetry: `feetech/telemetry.py` and `FeetechBus.read_status_block` (position, speed, load, V, T,
  current in one transaction). It is recorded as extra columns.
- Derived and robot-independent data:
  - `dataset/derived.py`, `dataset/eef_actions.py`.
  - `handumi dataset derive|mask-tags|label`.
  - `handumi convert-eef` (14-D TCP state/action).
  - `processing/tag_mask.py` (`head_clean` video).
- Tasks and eval:
  - `tasks/cube_stack.py` and `handumi task cube` (3-cube order plan).
  - `eval/tracker_eval.py` and `handumi tracking eval` (jitter, return, L/R agreement, table-z gates).
- Robot profile:
  - `configs/robots/rebot_b601.yaml`, `assets/rebot_b601/rebot_b601_bimanual.urdf`.
  - `configs/calibration/table/rebot_b601.yaml`, `configs/quality_mvp.yaml`.
  - Note: `ik()` takes quaternions in xyzw order.
- Small upstream edits:
  - `pyproject.toml`: vosk is excluded on darwin.
  - `control_tcp.py`, `raw.py`, `record.py`, `teleop_record.py`: support for the telemetry and `head` camera names.
- `configs/rig.yaml` is a **local rig config**. It has OpenCV indices, Feetech port names, and the Quest 3S LAN
  IP/ports (`192.168.50.200:65432`). These are site-specific, but nothing in it is a secret.

### Reconstructing the fork

```bash
git clone https://github.com/robonet-ai/handumi-sw && cd handumi-sw
git checkout -b rebot-ego 33cc437229e43ed4bb92b25d4465248f0edbbcd9
git apply ../handumi-rebot/upstream/handumi-sw-rebot-ego/tracked.patch
cp -R ../handumi-rebot/upstream/handumi-sw-rebot-ego/new_files/. .
uv sync --extra sim        # regenerates uv.lock
```

The patch and files were verified on 2026-10-07. A clean `git archive 33cc437` with the patch applied and the new files
copied is byte-identical to the working tree, excluding `.venv`, caches and `uv.lock`.

**Status**: this Quest/AprilTag tracker path (V1, 2026-08/09) is not the current production capture path. Production
uses `../collector`, which is fiducial-free, with fisheye + IMU on the HandUMI wrist units. The fork is kept for the
reBot URDF/profile, the jaw telemetry, and the EEF conversion code.

## handumi-hw

`~/handumi-hw` has **no local modifications**: it is clean at `e58de33`. The only untracked files are `.DS_Store` and a
Bambu slicer `result.json`, and neither was exported. Our print package (`../hardware/print`) is derived from its
`hardware/STL` meshes. They are only translated to positive coordinates, and the geometry is unchanged.

## Licenses

- `LICENSE-handumi-apache-2.0.txt` is a verbatim copy of `handumi-sw/LICENSE`. It is Apache-2.0, with the HandUMI
  copyright (BrikHMP18 and HandUMI contributors), third-party notices for yubi-sw (Apache-2.0), and MIT notices for the
  vendored axol/piper assets.
- `LICENSE-handumi-hw-apache-2.0.txt` is a verbatim copy of `handumi-hw/LICENSE` (Apache-2.0, Copyright 2026
  BrikHMP18). The two files differ only in their appendix and notices.
- `NOTICE.md` is our statement of modifications, as required by Apache-2.0 §4(b).
