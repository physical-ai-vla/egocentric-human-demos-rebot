# Wrist-pose benchmark (MASt3R-SLAM + per-episode IMU-VI scale)

14 separate ArUco-marker episodes (28 hand tracks, sessions 2026-09-22/23), not part of the 349-episode dataset.
`valB` (9 episodes / 18 tracks) was used to choose the IMU-VI window (W = 15, no bias); `valC` (5 / 10) is held out.
All numbers are from the RTX 5080 run that produced the dataset poses (`ss1_5080`).

- `cp6_ss1_5080.log`: scale error and UMI-style k-step relative position error per method; held-out table under `EVAL (valC)`.
  IMU-VI all: scale err p50/p90 4.6 / 17.0 %, k16 p90 11.3 mm, k32 p90 16.1 mm; global constant 18.5 / 23.8 %.
- `cp6_scale_ss1_5080.json`: per-track rows behind that table.
- `cp7_eval.json`: rotation error vs the ArUco reference (`eval_raw`: p50 0.837 deg, p90 2.13 deg).
- `qa_all.json`: per-track coverage / tracking QA for ORB-SLAM3 (`ORB_*`) and MASt3R-SLAM (`M3_*`).
