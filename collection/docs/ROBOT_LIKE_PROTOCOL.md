# Robot-like ego collection — protocol `robot_like_v1` (2026-09-28)

Goal: add ~80 ego demonstrations whose motion looks like the reBot's (R150), on top of the 259 C8 episodes, and compare
`old ego only` vs `old + robot-like ego`. Quantity is secondary; embodiment match is the point.

## Record

```bash
cd ~/ego_collector
.venv/bin/python -m handumi_collector.collect --hardware handumi_v1 --dataset HRL80 --protocol robot_like_v1
```

- `--dataset HRL80` → sessions in `datasets/human_handumi_raw/HRL80/`, 6 orders × 14 = 84. Never mixes with `Hpilot/`.
- `--protocol robot_like_v1` (`configs/handumi/robot_like_v1.yaml`):
  - AUTO stack time **30 s** (legacy 20 s)
  - every episode_meta gets `protocol: robot_like_v1`
  - the RECORD tab shows the **ROBOT-LIKE PROTOCOL** panel
- Without `--protocol`, the collector behaves exactly as before.

### Operator rules (also shown in the panel)

- Stay inside the robot workspace box on the HEAD preview.
- Keep the wrists level. Rotate only as much as the robot would.
- Move one stage at a time: approach → grasp → transport → place. Pause about 0.3 s at each grasp and release.
- No sudden direction changes. Move at robot speed, about half of natural speed.
- Keep the two hands apart, and never cross over the other arm.

### Live panel (IMU + gripper only)

| lamp | signal | GREEN if ≤ | source |
|---|---|---|---|
| rot | wrist \|ω\|, gyro averaged ±1/30 s on a 15 Hz grid | R150 p95 0.996 rad/s | `probe_v2_motion.py` |
| rot-acc | \|Δω\|·15 | R150 p95 4.413 rad/s² | same |
| lin-acc proxy (○) | \|a − g_lowpass\| | 1.157 m/s², **advisory only** | not the same quantity as R150 lin_a |
| stage | jaw hysteresis 0.35 / 0.65 | — | grasp / release counted per arm |

Lamps: AMBER up to 1.5 × p95, RED above that.

Episode verdict (`derived/robot_like/live_summary.json`, `episode_meta.robot_like_live`) is FAIL when:
- more than 10 % of ticks on either wrist are above p95 (R150 itself is above its own p95 5 % of the time), or
- there are fewer than 2 grasps.

A FAIL makes the preliminary QA `REVIEW`. AUTO `on_warn: keep` still keeps the episode, with the note attached.

The live panel cannot show mapped robot TCP, IK or joint margin, because there is no reliable live metric pose on this
Mac:
- OpenVINS diverges on a waiting hand.
- mast3r_live has never run on a GPU.
- The Orbbec cannot share the host with the wrist cameras.

For the same reason, inter-arm distance has no live source.

## Offline robot check (per session)

```bash
scripts/robotlike_session_check.sh datasets/human_handumi_raw/HRL80/HRL80_<ts>
```

1. Export, using the same exporter and flags as C8.
2. Upload to the 5090 NFS.
3. Run MASt3R-SLAM on the RTX 5080 via Ray (`c8_run_5080.sh`, pinned `node:<gpu-node-ip>`). This takes about 11 GPU-minutes per episode.
4. Pull the results back.
5. Run `scripts/robotlike_offline_check.py` with `~/xvla-mac/bin/python`, which reuses the frozen C8 Phase-3 chain from `~/c8/c8_phase3.py`:
   - metric scale (imu_vi + degenerate guard)
   - camera_tcp_v2
   - |dt| < 20 ms sync
   - 65-frame segments
   - r150_active_median re-anchor
   - R150 workspace
   - continuity IK

   It adds:
   - per-joint margin to the URDF limits (j2/j3 upper = 0 is reported but not gated)
   - `c8_collision_v1` clearance, with hard fail below 0.075 m and warn below 0.194 m
   - robot-space TCP dynamics vs R150 p95

Outputs:
- `<episode>/derived/robot_like/offline_check.json`
- `<session>/robot_like_offline_report.json`
- the QA tab button "Load robot-like offline check".

Episode verdict: PASS if at least 50 % of its segments pass workspace + IK + collision.

## Validation (2026-09-28)

- `tests/handumi/test_robotlike.py`: 8 tests covering monitor math (angular accel = the probe definition), gap handling, grasp hysteresis, session tag + summary, legacy session unchanged, UI panel + QA loader.
- Offline checker parity on 3 C8 episodes vs `phase3_census.json`: segment starts, scale, IK status and first-fail all identical. ws_in is equal to 1e-4.
- The video→CORI offset used without the grip census (N_video − n_CORI) equals the census on 8/8 sampled sides.
