# R30 — real teleop reference set for ego TR (Trajectory-guided Retargeting)

Written 2026-09-28 by the ego-pipeline agent. **Frozen; do not change.** Source of truth = `~/umi_bridge/track_c/v2k/r30_episodes.json`
(sha256 `532bddc90308722905d6d3cb4310efab124805e01d4227b6abfe8ef1986f2764`).

## What it is
- The 30 real reBot teleop episodes that **every real-derived quantity** of the ego C-old v2 retarget comes from (R30 contract):
  - the TR window library: `build_windows` over these 30 episodes, stride 5
  - the Kr seed bank: `seedbank_R30_q12.npy`, 1195 seeds (sha256 `755041b41bd4c69d026e50fba1e5df74d06ea355ddb58ca75a94947d82f052f0`)
  - workspace gate and dynamics references
- Evaluation of the retarget (posture NN / cluster / W1) is against the **held-out R120** (R150 minus these 30), never R30.
- Used by: hybrid master (old259 + HRL80), C-old v2 TR-only dataset, REL16 ego dataset.

## Rule (predates all results; no episode chosen by performance)
R_n = collection `set_id < n/6` in `~/c8/r150_nested_subset_v1.json` (sha256 `8996a20fc88a8e4bb32dad4b1260556552c911159a06c128eb5eeaa4b52b5744`).
Global set_id = chronological rank within order. R30 = set_id 0–4 → 5 sets × 6 stacking orders = 30.

- **Source dataset:** `rebot/rebot_3stack_R150_headview`
  - 4090: `${REMOTE_HOME}/holobrain-data/lerobot/rebot_3stack_R150_headview`
  - Mac: `~/c8/r150_ds`
- **Session:** all 30 from `local_rebot_R120_headview_20260907_144206`.
- **Total frames:** 37512 (30 fps).
- **R30 ⊂ R60 ⊂ R90 ⊂ R120 ⊂ R150:** nested.

## Episodes (R150 `episode_index`)
`[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]`

| episode_index | set_id | order | session ep | frames | recorded | in probe val-10 |
|---|---|---|---|---|---|---|
| 90 | 0 | RBP | ep000 | 1368 | 2026-09-07T14:43:02 |  |
| 91 | 0 | RPB | ep001 | 1213 | 2026-09-07T14:43:57 | yes |
| 92 | 0 | BRP | ep002 | 1216 | 2026-09-07T14:44:46 |  |
| 93 | 0 | BPR | ep003 | 1302 | 2026-09-07T14:45:40 |  |
| 94 | 0 | PRB | ep004 | 1133 | 2026-09-07T14:46:26 |  |
| 95 | 0 | PBR | ep005 | 1197 | 2026-09-07T14:47:14 |  |
| 96 | 1 | RBP | ep006 | 1218 | 2026-09-07T14:48:03 |  |
| 97 | 1 | RPB | ep007 | 1356 | 2026-09-07T14:48:57 |  |
| 98 | 1 | BRP | ep008 | 1086 | 2026-09-07T14:49:42 | yes |
| 99 | 1 | BPR | ep009 | 1433 | 2026-09-07T14:50:41 |  |
| 100 | 1 | PRB | ep010 | 1459 | 2026-09-07T14:51:37 |  |
| 101 | 1 | PBR | ep011 | 1437 | 2026-09-07T14:52:35 |  |
| 102 | 2 | RBP | ep012 | 1386 | 2026-09-07T14:53:41 |  |
| 103 | 2 | RPB | ep013 | 1224 | 2026-09-07T14:54:30 | yes |
| 104 | 2 | BRP | ep014 | 1393 | 2026-09-07T14:55:27 |  |
| 105 | 2 | BPR | ep015 | 1161 | 2026-09-07T14:56:16 |  |
| 106 | 2 | PRB | ep016 | 1119 | 2026-09-07T14:57:02 |  |
| 107 | 2 | PBR | ep017 | 1161 | 2026-09-07T14:57:49 | yes |
| 108 | 3 | RBP | ep018 | 1096 | 2026-09-07T14:58:33 |  |
| 109 | 3 | RPB | ep019 | 1105 | 2026-09-07T14:59:17 |  |
| 110 | 3 | BRP | ep020 | 1230 | 2026-09-07T15:00:09 |  |
| 111 | 3 | BPR | ep021 | 1239 | 2026-09-07T15:00:59 |  |
| 112 | 3 | PRB | ep022 | 1305 | 2026-09-07T15:01:53 |  |
| 113 | 3 | PBR | ep023 | 1123 | 2026-09-07T15:02:38 |  |
| 114 | 4 | RBP | ep024 | 1151 | 2026-09-07T15:03:48 |  |
| 115 | 4 | RPB | ep025 | 1656 | 2026-09-07T15:04:53 |  |
| 116 | 4 | BRP | ep026 | 1247 | 2026-09-07T15:05:43 |  |
| 117 | 4 | BPR | ep027 | 1206 | 2026-09-07T15:06:33 |  |
| 118 | 4 | PRB | ep028 | 1156 | 2026-09-07T15:07:20 |  |
| 119 | 4 | PBR | ep029 | 1136 | 2026-09-07T15:08:06 |  |

Orders: `RBP` = red bottom / blue middle / purple top, and so on (task string in the dataset).

## Caveats for other agents
- **Probe val-10 overlap:** 4 R30 episodes (91, 98, 103, 107) are in the probe val-10 set (`val10_in_R120`). This was already true of R150 / R120 FT data. When a real FT evaluation uses val-10, those 4 episodes are also retarget reference episodes.
- **Held-out means R120:** "held-out" in C-old v2 metrics means R150 minus R30 = 120 episodes. Do not report R30-based numbers as held-out.
- **Do not reselect:** do not grow, shrink or reselect R30. Changing it changes the TR windows, the Kr bank and the gates, and invalidates the frozen master.

## Downstream real FT budget (user 2026-09-28)
Any R60 / R90 / R120 fine-tuning after the TR ego pretrain must use the NESTED ladder R30 ⊂ R60 ⊂ R90 ⊂ R120 ⊂ R150
(`r150_nested_subset_v1.json`, R_n = set_id < n/6), i.e. R60 CONTAINS these 30 episodes. Then the real calibration core of the
pretrain and the downstream real-data budget overlap, and the count of unique real demonstrations stays R_n.
