# pipeline/umi_laundry MANIFEST

Assembled 2026-10-20. Sources were only read.

| source | dest |
|---|---|
| `~/ego_collector/scripts/umi_fusion/` (incl. `m3_runner/run_perframe_restart.py`) | `umi_fusion_scripts/` |
| `~/ego_collector/scripts/handumi_body_track.py`, `handumi_canonical_track.py`, `handumi_canonical_eval.py` | `orbbec_tracking/` |
| `~/c8/robotlike/*.sh`, `umialign_metrics.py` | `drivers/` (one-off session drivers, kept as a record of how each session was run) |

Related code elsewhere in this repo: `cart20/ego_cart20/{umi_fusion,robotize}/`, `sources/{hfold_export,jaw_visual,trim}.py`.

## Excluded

| item | reason |
|---|---|
| Vast.ai / RunPod bundle scripts and the prebuilt MASt3R image (5090 NFS `runpod_bundle/`, `m3image/`) | host addresses, account specifics, 9–13 GB |
| recordings, `derived/` outputs, MASt3R runs, metric tables | data |

## Notes

- Scripts insert `~/ego_cart20` and their own folder into `sys.path` and drivers use `~/c8/robotlike` and node paths
  (Ray at the tailnet head, 4090 pinned via `node:100.64.0.2`). Adjust before running elsewhere.
