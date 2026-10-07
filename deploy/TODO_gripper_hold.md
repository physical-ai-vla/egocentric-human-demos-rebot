# TODO (2026-10-03): generic V4_GRIPPER=hold sends the RAW jaw reading as a COMMAND

`infer_core_v4.py` `_post_act` and `solve_waypoint`, GRIPPER_MODE `hold`: `cmd[gi] = float(q_now[gi])`.
`q_now[gi]` is the /observe jaw RAW (0 closed .. -270 open, obs = cmd * -6); `/execute_step` wants a COMMAND 0..45.
A held open jaw (raw -100) is sent as -100 -> clipped to 0 -> CLOSES. The HRA one-arm path (V4_ARM_ONLY) already uses
`jaw_hold_cmd(raw) = clip(unwrap_grip(raw) / -6, 0, 45)`.

User decision: do NOT patch the shared path while the current FT / stacking sessions run. Fix afterwards as a separate patch,
with a regression test:  raw 0 -> 0,  raw -90 -> 15,  raw -180 -> 30,  raw -270 -> 45,  raw 358 (wrapped closed) -> 0,
and a real left / right jaw polarity check on the robot (hold must not move either jaw).
