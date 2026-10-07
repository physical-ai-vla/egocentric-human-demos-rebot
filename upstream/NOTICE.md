# NOTICE: modifications to HandUMI

This repository contains material derived from:

- **HandUMI software**: https://github.com/robonet-ai/handumi-sw, Copyright 2026 BrikHMP18 and HandUMI contributors,
  licensed under the Apache License, Version 2.0 (`LICENSE-handumi-apache-2.0.txt`). It includes third-party
  components under their own notices, reproduced in that file.
- **HandUMI hardware**: https://github.com/robonet-ai/handumi-hw, Copyright 2026 BrikHMP18, licensed under the
  Apache License, Version 2.0 (`LICENSE-handumi-hw-apache-2.0.txt`).

As required by Apache-2.0 §4(b), modified files carry this prominent notice: **the files under
`upstream/handumi-sw-rebot-ego/` were modified or added by Beyond Honeycomb (2026-08/10)** relative to handumi-sw
commit `33cc437229e43ed4bb92b25d4465248f0edbbcd9`.

- `tracked.patch` modifies these upstream files:
  - `pyproject.toml`
  - `src/handumi/calibration/control_tcp.py`
  - `src/handumi/dataset/raw.py`
  - `src/handumi/feetech/bus.py`
  - `src/handumi/feetech/gripper.py`
  - `src/handumi/scripts/cli.py`
  - `src/handumi/scripts/record.py`
  - `src/handumi/scripts/teleop_record.py`
- Every file under `new_files/` is new: AprilTag EEF tracking, Feetech telemetry, derived/EEF datasets, cube task,
  tracker eval, the reBot B601 robot profile and URDF, the local rig config, and the runbook.

Hardware: the STL meshes in `hardware/print/{left,right,tips}/stl/` are the upstream handumi-hw meshes, **translated**
so all coordinates are positive. The geometry is unchanged. The 3MF plate projects were generated from them by
Beyond Honeycomb (2026-08-26).

Code under `collector/` and `experimental/` (the `handumi_collector`, `ego_teleop` and `ego_collector` packages) is original work and is not derived from HandUMI source. `experimental/assets/aero_hand_open/` carries its own NOTICE (TetherIA aero-hand-open, Apache-2.0).
