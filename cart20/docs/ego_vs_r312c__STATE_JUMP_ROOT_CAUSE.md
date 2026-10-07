# Ego state-jump root cause (2026-10-01, read-only; no data changed)

## Conclusion
1. **About 57% of the reported ">30 mm per adjacent row" jumps are not jumps.** They are gaps in time.
   - Ego LeRobot frames contain training rows only: rows with all 16 valid targets and a head frame within 20 ms.
   - So adjacent frames in the same episode can be far apart in time. 2,344 of 51,314 pairs are not adjacent rows; among them the gap is p50 1.2 s, max 15.7 s.
   - Of the jumps over 30 mm, 1,482 out of 2,598 (57%) are on such pairs. Of those over 100 mm, 811 out of 829 are.
   - In R312c every frame is adjacent in time. Any analysis that treats neighbouring frames as consecutive in time therefore misreads the ego data. Ego frames carry `aux.row` and `aux.time_s`, so this can be checked directly.
2. **The real jumps are in the raw MASt3R-SLAM camera trajectory. The pipeline stages only pass them through.**
   - 55 raw steps exceed 3 m/s, which is not physically possible for a hand. They occur in 24 train episodes and no val episode.
   - Measured per stage, the scaled camera translation jump divided by the tool-pose jump has median 0.99. So scale × camera jump produces the whole jump, and the tool transform adds nothing.
   - Rotation changes are small (median 2.7°). Most events are normal frame intervals (dt ≈ 33 ms; 3 have a 1–3 ms timestamp gap).
   - There is no tracking-lost flag before any of them (0/55). SLAM jumps the translation without declaring tracking lost.
   - 50 events are one-way jumps (no return) and 5 are spikes (jump then return). They often come in bursts of 3–5 consecutive frames.
   - One garbage pose exists: 100511_000056 left jumps 1,886 m.
3. **What was ruled out:**
   - State and action use different sources: no. Both sample the same `PoseTrack` object (same raw track, same tool transform, same interpolation).
   - The anchor is reset partway: no. It is computed once per episode, at row 0.
   - Segment stitching: no. Whole episodes are used with no stitching.
   - The X order differs between state and action: no. A single `T @ X` is applied once in the exporter.
4. **Why "the state jumps but the action is smooth":**
   - The state is relative to the task anchor, `inv(T_start) T(t)`. A one-way jump therefore stays in every later state of the episode.
   - The action is relative to the current pose, `inv(T_t) T(t+k dt)`. It is only affected in windows (≤ 0.8 s) that cross the jump.

## Impact on ego_cart20_v2_train (raw step > 3 m/s criterion)
| | Rows | Share of train (51,612) |
|---|---:|---:|
| Action window crosses a jump | 145 | 0.28% |
| State after the first jump (offset from the anchor) | 1,771 | 3.4% |
| All rows of the 24 affected episodes | 4,153 | 8.0% |

## Reference: robot vs ego, rows adjacent in time
| | > 30 mm / row | > 100 mm / row | max |
|---|---:|---:|---:|
| R312c | 1.2% (2,105 / 173,700) | 1 | 126 mm |
| Ego | 2.3% (1,116 / 48,970) | 18 | 423 mm |

30 mm per 66.7 ms is 0.45 m/s, which a real hand can reach. Over 100 mm per row (≥ 1.5 m/s) is very likely a SLAM glitch.

## Files
- `raw_jump_events.json`: the 55 raw events.
- `raw_jump_stage_dump.json`: each event's camera jump (raw / scaled), rotation, tool-pose jump, dt, scale, and lost frames in the previous 15 samples.

## Fix options (decision pending; not applied)
- **A. Truncate (recommended).**
  - Mark raw steps over v_max as invalid.
  - Return spikes: invalidate only that sample.
  - One-way jumps: drop the rest of the episode from the first jump on.
  - This keeps the task-anchor meaning. Cost: about 3.4% of rows.
- **B. Drop the whole episode.** Removes 24 episodes (8%).
- **C. Stitch.** Apply a rigid correction for the jump to the trajectory after it. This keeps the data, but SLAM drift may not be a pure offset, so it is risky.
- **Threshold proposal:** v_max = 1.5 m/s (100 mm / 66.7 ms) or 45° per row as a hard limit. Flag-only (no removal) at 30 mm or 15° per row.
