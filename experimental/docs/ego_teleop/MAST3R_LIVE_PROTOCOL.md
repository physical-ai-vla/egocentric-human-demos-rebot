# MASt3R live wrist tracking — wire protocol v1

The live visual frontend of the multi-sensor wrist branch (`docs/ego_teleop/MULTISENSOR_FUSION.md`, spec §2 and
§10). MASt3R-Fusion is CUDA + torch + gtsam and lives at `~/vio_bakeoff/MASt3R-Fusion` on the GPU box; the
teleoperation loop runs on the Mac that holds the cameras. Nothing of MASt3R-Fusion is copied into this repo.

```
   Mac (cameras, fusion, reBot)                         GPU box (~/vio_bakeoff/MASt3R-Fusion)
   ─────────────────────────────                        ──────────────────────────────────────
   wrist Arducam ─┐
                  ├─► Mast3rLiveBackend ──JPEG+IMU────► mast3r_live_server.py ─► FrameTracker
   wrist IMU  ────┘   (a PoseEstimator)  ◄───pose─────                          + FactorGraph
```

| side | file | notes |
|---|---|---|
| client | `ego_teleop/tracking/backends/mast3r_live.py` | registered as `mast3r_live`; `--selftest` runs it against an in-process echo server |
| server | `scripts/mast3r_live_server.py` | **deployed into the MASt3R-Fusion checkout**, imports `mast3r_fusion.*` |
| doc | this file | the two halves assert the same `MAGIC`/version/struct in `tests/teleop/test_mast3r_live.py` |

## Running it

```bash
scp scripts/mast3r_live_server.py gpu-5090:~/vio_bakeoff/MASt3R-Fusion/
ssh gpu-5090 'cd ~/vio_bakeoff/MASt3R-Fusion && python mast3r_live_server.py --config config/base_euroc.yaml --port 5577'
```

```yaml
# configs/ego_teleop/fused_wrist.yaml
vi_backend: mast3r_live
vi_backend_options: {host: gpu-5090, port: 5577, jpeg_quality: 82}
```

No server, a refused handshake or a version mismatch ⇒ `BackendUnavailable` at construction. A pose source is never
substituted, and the client never invents a pose.

## Framing

Every message, both directions:

```
offset  size  field
0       4     magic "M3RF"
4       4     u32 header_len     little-endian
8       4     u32 payload_len
12      H     header, UTF-8 JSON
12+H    P     payload (only `frame` has one: a JPEG)
```

## Messages

### client → server `hello` (first)
```json
{"type":"hello","version":1,"encoding":"jpeg",
 "intrinsics":{"model":"pinhole","width":960,"height":540,"fx":327.8,"fy":328.5,"cx":469.9,"cy":254.4,
               "distortion":[0,0,0,0],"rectified_from":"kannala_brandt","balance":0.0},
 "image_size":[960,540],
 "T_camera_imu":[16 floats, row-major],
 "imu_noise":{"accelerometer_noise_density":...,"gyroscope_noise_density":...,
              "accelerometer_random_walk":...,"gyroscope_random_walk":...}}
```

**Intrinsics are always `pinhole`.** The wrist Arducam is Kannala-Brandt; the CLIENT rectifies it, with the same
`cv2.fisheye` call and the same `balance` that `ego_teleop/tools/export_euroc.py` uses offline
(`ego_teleop/transforms/fisheye.py` is the one implementation both share). That is deliberate: MASt3R-Fusion's
`Intrinsics.from_calib` knows pinhole+radtan and omnidir/mei only, and if the live path and the offline bake-off
rectified differently their numbers could not be compared. The server rejects a non-pinhole `hello`.

### server → client `ready`
```json
{"type":"ready","version":1,"backend":"mast3r_fusion","img_size":512,"grid":[H,W],"device":"cuda:0","config":"config/base_euroc.yaml"}
```
Anything else (`{"type":"error","error":"..."}`) is a refused session.

### client → server `frame`
```json
{"type":"frame","seq":123,"t_ns":...,"frame_index":...,
 "imu":[[t_ns, gx, gy, gz, ax, ay, az], ...]}
```
payload = JPEG of the rectified frame.

* `t_ns` is host-monotonic nanoseconds. The client has already applied the camera↔IMU time offset, so **camera and
  IMU are on one clock** and the server only re-bases both to seconds against the first frame of the session.
* `gyro` is **rad/s**, `accel` is **m/s²** — the canonical host units of `firmware/teensy_imu/PROTOCOL.md`. The
  server converts gyro to deg/s internally because that is what `IMUPool` hands the factor graph; getting that
  wrong is a silent 57× rotation error.
* IMU is batched onto the next frame. 416 Hz of single-sample writes would spend more time in the kernel than in
  the estimator, and the factor graph consumes IMU per frame interval anyway.
* **At most one frame is in flight.** The client will not send another until this one is answered.

### server → client `pose`
```json
{"type":"pose","seq":123,"frame_index":...,"t_ns":...,
 "T_world_camera":[16 floats, row-major] | null,
 "state":"init|tracking|reloc|degraded|lost",
 "keyframe":false,"server_ms":31.4,"dropped":0,
 "confidence":null,"num_features":null,
 "map_update":true,"map_update_m":0.031}
```

`map_update` is the load-bearing field. MASt3R-Fusion runs loop closure and global BA, so a pose it emits can move
the whole past trajectory. The server re-reads the pose of the last emitted keyframe after every optimisation step;
if the optimiser moved it by more than `--map-update-eps-m` (default 2 mm) the next pose carries the flag and the
size of the shift. `LocalPoseContinuity` (`provider.local.map_update_keys` in `fused_wrist.yaml`) then absorbs the
step into an offset so the map is corrected and **the arm is not**.

Without the flag a loop closure and a tracking failure are the same event in the pose alone, and the only safe
reading of that pair — grade it LOST — would throw away every loop closure. An *unflagged* jump therefore stays a
jump and `TrackingSupervisor.max_jump_m` grades it LOST, which is the intended behaviour, not a gap.

### either → `bye`, server → `error`
```json
{"type":"bye"}          {"type":"error","error":"..."}
```
`error` does not end the session; the client marks the link failed and every subsequent estimate is LOST until it
reconnects.

## Latency, not FPS (spec §25)

Nothing in this path is allowed to queue frames.

* the client sends at most one frame at a time and **drops** any frame offered while one is in flight
  (`frames_dropped_busy`);
* the server's `SocketStream` holds exactly one pending frame and drops the older one (`dropped`, echoed on every
  pose);
* `push_image` returns the newest pose the reader thread has, **carrying that pose's own timestamp** — so
  `TrackingSupervisor` sees the real age and a stalled server becomes DEGRADED and then LOST on schedule instead of
  handing the arm fresh-looking poses of where the hand used to be.

`get_quality()` reports `frames_sent`, `frames_dropped_busy`, `poses`, `map_updates`, `server_ms_last`,
`link_ms_last` and `imu_pending`; `f5_teleop_hud` shows the latency strip built from them.

## Testing it without the GPU box

```bash
.venv/bin/python -m ego_teleop.tracking.backends.mast3r_live --selftest   # echo server, same wire format
.venv/bin/python -m pytest tests/teleop/test_mast3r_live.py -q
.venv/bin/python scripts/mast3r_live_server.py --dry-run                  # imports no CUDA
```

`EchoPoseServer` is a protocol fixture, never a tracker: it must never be used to produce a number about the
hardware.

## Status

The client, the protocol and the framing are tested. **The server has not been run against a GPU yet** — it is
`main.py`'s loop with the dataset replaced, and the three adaptations (`SocketStream`, `AppendableImuPool`,
map-update detection) are the parts to watch on the first real run. Nothing about the tracking quality of this path
is known yet; that is what the §20 protocol run is for.
