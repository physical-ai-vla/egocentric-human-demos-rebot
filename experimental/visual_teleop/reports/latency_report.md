# Latency report — 2026-10-14 (per stage, p50 / p90 / p99, ms)

Measured on this Mac with `visual_teleop/tools/benchmark.py`. Live fingertip timings come from the attached right-wrist
Arducam. Robot, RGB-D and IMU stages could not be measured live today.

| stage | p50 | p90 | p99 | max | source |
|---|---|---|---|---|---|
| fingertip student, 1280×720, offline (no undistort) | 4.66 | 4.75 | 5.02 | 25.3 | `rec_01.mp4`, 300 frames (~214 fps) |
| fingertip node, live 1920×1080 + undistort + detect | 15.1 | 18.9 | 20.4 | — | right Arducam, 366 frames, 28.7 Hz |
| fingertip capture → control process receive | 15.2 | — | 20.6 | — | same run, over UDP |
| AHRS update (per IMU sample) | 0.016 | 0.018 | 0.026 | 0.22 | capacity ~38 kHz vs 200–416 Hz needed |
| pyroki IK, per arm | 0.52 | 1.4–2.4 | 3.2 | 3.3 | 300 targets around home |
| controller tick, both arms, pyroki, mock bridge | 4.56 | 7.18 | 7.61 | 27.8 | 300 ticks |
| control loop, Stage 1, mock robot | — | — | 3.5 | — | `live_teleop --mock-robot` |
| RGB-D MediaPipe + depth, per frame | **not measured** | | | | The Orbbec needs `sudo`, which cannot run from this session. |
| `/observe` + `/status` round trip (MIT bridge) | **not measured** | | | | Bridge was up but had no robot connected. Earlier measurement: 30–45 Hz, p95 tick 33–63 ms, because `/observe` returns base64 images. |

## Budget, human → robot command

- Wrist cam pinch path: about 15–20 ms detection + up to 33 ms tick wait + bridge.
- RGB-D wrist path: one camera frame (33 ms) + MediaPipe (≈15–30 ms, unmeasured here) + fusion publish (≤10 ms) + tick
  wait (≤33 ms) + the `/observe` round trip.

**`/observe` is the likely bottleneck.** It ships camera images on every call. If it measures above ~30 ms, add a
joints-only endpoint to the bridge; do not guess before measuring.

The < 100 ms end-to-end target is plausible but **unverified**. On live runs the dashboard shows `observe_ms`, `ik_ms`,
`send_ms`, `tick_ms` and `loop_p99_ms`, and `ticks.npz` stores `lat_*` per tick.
