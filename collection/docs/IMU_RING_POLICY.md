# TODO / risk: the tx ring drops the newest packet, not the oldest

Status: **recorded, not changed.** The drain workaround in `imu_logger` removes the symptom at session start; the
latency behaviour under a mid-session stall is still the one described here. Decide before the recording pipeline is
frozen. Raised 2026-09-14 while bringing up the right wrist.

## What it does today

`firmware/teensy_imu/teensy_imu.ino`:

```c
static bool ring_push(uint8_t type, const uint8_t* payload, uint8_t len) {
  uint16_t next = (g_head + 1) % TX_RING_PACKETS;
  if (next == g_tail) { g_dropped++; return false; }   // host too slow: drop, count, keep sampling
  ...
}
```

When the ring is full the **incoming** sample is discarded and the 256 already queued are kept. Sampling never stalls
and the loss is counted, which is the part that is right. What is wrong is *which* samples survive: the queue holds the
oldest data, so a host that pauses and resumes is served a backlog before it sees the present.

Observed on the bench: the board free-ran unread for ~32 s, and the first thing a connecting host received was 233
samples timestamped 69.7–70.9 s followed by a single 6357-sample sequence jump. Read naively that is *34 % loss at
129 Hz* on a board that had lost nothing; the same recording after the jump ran at 197.91 Hz with 0.0000 % loss.

## Why it matters beyond the drain

The drain only covers *session start*, where we know a backlog exists and can throw it away. A stall **during** a
recording behaves differently, and worse:

- **Latency ratchets.** After a stall of duration T the queue holds up to 1.28 s of old samples (256 at 200 Hz) and the
  host stays that far behind until the ring empties. For a file-based recording that is harmless — the device clock is
  what times the data, and `device_timestamp_us` is correct regardless. For anything consuming the stream live
  (teleop, a UI freshness indicator, an online filter) the freshest state is precisely what is withheld.
- **The gap lands in the wrong place.** With newest-drop the sequence gap appears where the host *resumes*, so the
  samples nearest the stall are the ones lost, and what survives is the stale run before it. Oldest-drop puts the gap
  at the stall and delivers the present immediately after.
- It gets more likely as the rate rises. At 500 Hz the same 256-slot ring is 0.51 s deep and fills 2.5x faster, so the
  500 Hz ablation is the natural moment to settle this.

## Oldest-drop, if we change it

```c
if (next == g_tail) { g_tail = (g_tail + 1) % TX_RING_PACKETS; g_dropped++; }   // evict the oldest, then push
```

**For:** the host always receives the most recent sample the device has; latency is bounded by USB, not by the backlog;
the session-start drain becomes a convenience rather than a correctness requirement; loss stays fully accounted for,
because the sequence numbers still show exactly what went missing.

**Against:** it changes which data is lost, so a stall now costs the *oldest* samples in the queue — for a recording,
losing 1.28 s of continuous history is arguably worse than losing 1.28 s of staleness. Recording and live consumption
want opposite policies, which is the real tension.

**A third option:** keep newest-drop but shrink `TX_RING_PACKETS` so the worst-case backlog is small (32 slots = 160 ms
at 200 Hz). Bounds the latency without choosing a side. Costs recording continuity across even short stalls.

## Before changing anything

Nothing here is worth doing on the strength of an argument; it needs the measurement we do not have yet.

1. **Does a mid-session stall actually happen?** Instrumented 2026-09-14: the status packet's spare `uint16` now
   carries the deepest the ring has been since the previous packet, and is reset each period so the host's max over a
   recording is that recording's high-water (a maximum cannot be differenced like a counter, so a since-boot figure
   would carry in the pre-session backlog — `imu_dual_smoke` resets it after the drain for the same reason).

   **Measured 2026-09-14, both wrists, 60 s: 1-2 of 256 packets at 200 Hz, and still 2 of 256 at 500 Hz** — where the
   same ring is only 0.51 s deep and fills 2.5x faster. While a host is reading, the ring does not queue. That is the
   single-digit figure this document was written to wait for, so **the question is closed on present evidence**: there
   is no latency to ratchet and no case for changing the policy. Re-open it only if a real recording ever reports a
   high-water mark in the tens, which is now a number every session carries.

   One trap in reading it: the device reports once a second, so the first status packet after a session's drain still
   describes the second before it. Baselining against that packet reported high-water 255/256 and 16270 device drops
   next to a host-side loss of 0.0000 % — at 500 Hz, where the ring fills in half a second, the entire pre-session
   backlog leaked into the session's figures. Tools must wait for a genuinely fresh status packet first.
2. If stalls are absent in practice, do nothing — the drain covers session start and the question is academic.
3. If they occur, decide by consumer: the recorder wants continuity, live teleop wants freshness. A per-build flag
   (`#define RING_EVICT_OLDEST`) is cheap and lets the two coexist rather than forcing one answer.
4. Whatever is chosen, `tests/handumi/test_imu_session.py` needs a case that pins it: fill the ring, stall, resume, and
   assert which samples survive and that `seq` still accounts for every loss.

Related: `docs/handumi_collector/IMU_BRINGUP.md` (drain, host-stamp quantisation), `firmware/teensy_imu/PROTOCOL.md`.
