# Putting the cameras and the wrist IMUs on one timeline

Measured 2026-09-15 on `datasets/human_handumi_calib/TSYNC/TSYNC_20260914_180606`, a 30 s take with all three cameras
and both IMUs.

## There is exactly one constant to calibrate, and it is not the one you would reach for first

    head frame at capture_ns
      -> nearest wrist frame, looked up per frame          no calibration; record the delta each time
      -> + the wrist camera <-> wrist IMU offset           ONE constant, estimated from gyro correlation
      -> IMU sample, interpolated to that instant

The tempting model — measure `head <-> wrist camera`, measure `wrist camera <-> wrist IMU`, add them, store the sum —
is wrong in its first term. **Do not write a fixed `head <-> wrist camera` offset anywhere.**

## Why: the cameras free-run at genuinely different rates

    stream        mean frame period      implied rate
    head_depth       33.3045 ms          30.0260 Hz
    left_wrist       33.3298 ms          30.0032 Hz
    right_wrist      33.3281 ms          30.0047 Hz

The head camera runs 0.0228 Hz faster than the left wrist. Nothing triggers them together, so their relative phase
winds continuously: at that difference it advances about 23 ms over a 30 s take, most of one 33.3 ms frame period. The
offset measured over five-second windows does not drift in one direction, it circulates:

    window      left_wrist    right_wrist
     0- 5 s       -2.99 ms      +2.08 ms
     5-10 s       +1.63 ms      -3.09 ms
    10-15 s       +2.80 ms      -0.26 ms
    15-20 s       -2.91 ms      +2.86 ms
    20-25 s       -1.89 ms      -1.94 ms
    25-30 s       +2.08 ms      -2.34 ms

This also explains a discrepancy that looked like a bug in the skew report: `left_wrist~head_depth` showed a start
offset of +25.26 ms and a median nearest-neighbour distance of 8.64 ms. Those disagree for a constant offset and agree
for a rotating phase — the first frames happened to land 25 ms apart, and the typical distance over the whole take is
what a wrapped phase gives.

Sampled once, an offset like this looks like a calibration. Applied later, it is wrong by up to a frame.

## What follows

**Look up, do not correct.** For each head frame, take the wrist frame whose `capture_ns` is nearest, and record the
delta that lookup accepted. `inspect_episode --replay` prints exactly this per stream; the rest of the pipeline should
carry it the same way. A viewer or a dataset that silently substitutes the nearest frame claims a synchronisation the
hardware does not provide.

**Estimate one constant.** `wrist camera <-> wrist IMU` is a real, physical, constant latency between two devices in
the same unit, and `pose.timing.estimate_camera_imu_offset` — visual angular speed against gyro magnitude — is the
right tool for it. That estimator needs the camera and the IMU to move together, which holds here and does not hold for
a head camera bolted to a table: it never rotates, so there is nothing to correlate. This is the whole reason the head
is reached by lookup rather than by correlation.

**Interpolate the IMU.** At 200 Hz the nearest sample is within 2.5 ms, which is already small, but gyro and accel are
smooth and linear interpolation to the frame instant costs nothing and removes a quantisation nobody has to carry.

## The error bound this leaves

Nearest-frame lookup between two free-running 30 fps streams is worst-case **±16.7 ms**, half a frame period, and
uniformly distributed across the take because the phase wraps. That is structural: no calibration improves it, only
hardware-triggered cameras or a faster frame rate would.

For cube stacking — reach, grasp, place, at hand speeds — this is very likely fine. It is worth writing down anyway,
because it is the kind of limit that stops being fine quietly when a later task involves fast motion, and because the
number is a property of the rig rather than of any recording.
