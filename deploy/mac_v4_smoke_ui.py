"""[2026-09-22] Real-robot SMOKE panel for the r150-umi-v4 X-VLA checkpoint (body-frame SE(3) deltas).

This is not a rollout UI. It exists to answer one question with the robot powered: does a predicted body-frame
delta arrive at the arm as the same motion, in direction and magnitude?

    predicted dT  ->  clamped dT  ->  commanded joints  ->  measured TCP dT

so all four are logged side by side for every press. There is no continuous loop on purpose: one press moves
the arm by at most V4_CLAMP_MM per step, for V4_N_ACTION steps, and then stops. The gripper is held.

Sending uses the same path the other UIs use -- observe -> leader frame (FLIP_IDX) -> slew-limited
/execute_step sub-steps -- because posting a target straight to /execute_step has already driven this arm into
its limits once.

Env: V4_* (see infer_core_v4), V4_UI_PORT (8016), V4_ROBOT (http://localhost:8020).
"""
import os, math, time, threading, base64, itertools
import numpy as np, requests, torch
from flask import Flask, jsonify, request
from scipy.spatial.transform import Rotation as Rot

from infer_core_v4 import V4Inferencer, CAM_MAP, GRIP_IDX, FLIP_IDX, ARM_IDX, LIMITS, set_limits
from infer_core_v4 import CART20_W_OPEN as INF_W_OPEN
CYCLE_LOG = os.path.expanduser(os.environ.get("V4_CYCLE_LOG", "~/v4_cycles.jsonl"))


def IC_EXEC_K():
    import infer_core_v4 as _IC
    return _IC.EXEC_K

ROBOT = os.environ.get("V4_ROBOT", "http://localhost:8020")
UI_PORT = int(os.environ.get("V4_UI_PORT", "8016"))
OBS_GAP_S = 1.0 / float(os.environ.get("V4_EXEC_HZ", "15"))
SLEW_MAX_STEP_RAD = float(os.environ.get("V4_SLEW_RAD", "0.02"))     # half the v3 cap: this is a first test
SLEW_HZ = float(os.environ.get("V4_SLEW_HZ", "20"))
# The follower tracks slowly: one send of a +3 deg target realises +0.11 deg, and 40 identical sends over ~2 s
# reach +1.27 deg. A waypoint therefore has to be HELD, not fired once, or the arm executes a fraction of the
# commanded motion (and a small command looks like no motion at all). Hold until the joint error stops
# shrinking or the budget runs out.
# A waypoint is reached when BOTH the joint error and the Cartesian error are small -- joints alone can read
# "close" while the TCP is still millimetres out, and TCP alone hides a wrist that is still turning.
CMD_HZ = float(os.environ.get("V4_CMD_HZ", "20"))
JOINT_TOL_DEG = float(os.environ.get("V4_JOINT_TOL_DEG", "1.2"))
TCP_TOL_MM = float(os.environ.get("V4_TCP_TOL_MM", "6"))
DWELL_TIMEOUT_S = float(os.environ.get("V4_DWELL_TIMEOUT_S", "0.6"))
# [2026-10-02 user] pass-through: with several waypoints per cycle (V4_N_ACTION > 1), the intermediate ones are NOT dwelt on until
# "reached" -- the next target is sent as soon as the TCP is within V4_PASS_TCP_MM (or V4_PASS_DWELL_S runs out). Only the LAST
# waypoint of the cycle keeps the full reach test (JOINT_TOL_DEG + TCP_TOL_MM, V4_DWELL_TIMEOUT_S), so the end target is unchanged.
PASS_THROUGH = os.environ.get("V4_PASS_THROUGH", "0") == "1"
PASS_TCP_MM = float(os.environ.get("V4_PASS_TCP_MM", "12"))
PASS_DWELL_S = float(os.environ.get("V4_PASS_DWELL_S", "0.25"))
# [2026-10-02 user] prefetch (pipelining): while the LAST waypoint of a cycle is being driven, a worker observes and runs the next
# inference, so the next cycle starts without the inference stop. The prefetched chunk is anchored on the pose observed during that
# last motion (absolute base-frame targets, as always); the next cycle skips the head waypoints the arm has already passed
# (nearest target to the measured TCP, then one beyond it if within V4_PASS_TCP_MM) and executes V4_N_ACTION waypoints from there.
# A prefetch older than V4_PREFETCH_MAX_AGE_S, or one that failed, is discarded and the cycle observes + infers as before.
PREFETCH = os.environ.get("V4_PREFETCH", "0") == "1"
CYCLE_SETTLE_S = float(os.environ.get("V4_CYCLE_SETTLE_S", "0.2"))   # [2026-10-06] post-cycle settle before the travel measurement
_REMOTE_URL0 = os.environ.get("V4_REMOTE_INFER", "")      # [2026-10-04] remote server URL at startup (5080b / 5090), survives remote=0 -> 1
PREFETCH_MAX_AGE_S = float(os.environ.get("V4_PREFETCH_MAX_AGE_S", "2.5"))
# [2026-10-04 user] prefetch lead: start the prefetch V4_PREFETCH_LEAD waypoints BEFORE the last one (0 = at the last, the old
# behaviour), so the inference overlaps more of the motion; the next cycle's nearest-target shift skips what was passed meanwhile.
PREFETCH_LEAD = max(-1, int(os.environ.get("V4_PREFETCH_LEAD", "0")))
# -1 = AUTO: lead = ceil((inference EMA + 0.2 s) / waypoint-period EMA), so the prefetch finishes before the cycle ends
_PFT = {"inf": None, "wp": None, "lead": None}
def _ema(key, v, a=0.3):
    _PFT[key] = v if _PFT[key] is None else (1 - a) * _PFT[key] + a * v
# [2026-10-04 user "움직임 연결 부드럽게"] pass-last: with prefetch running, the LAST waypoint of a cycle is passed through too (no
# full reach dwell) -- the next, already-inferred chunk takes over while the arm is still moving. 0 = old behaviour.
PASS_LAST = os.environ.get("V4_PASS_LAST", "0") == "1"
_PF = {"thread": None, "res": None}


def _prefetch_worker():
    try:
        t_req = time.perf_counter()
        h0 = _observe()
        time.sleep(OBS_GAP_S)
        h1 = _observe()
        if h0 is None or h1 is None:
            _PF["res"] = {"error": "observe failed"}; return
        obs_ms = round((time.perf_counter() - t_req) * 1000.0, 1)
        t0 = time.time(); t_i = time.perf_counter()
        INF._rtc_delay_hint = _PF.get("delay")      # [2026-10-06] RTC delay = waypoints the arm runs while this chunk is inferred
        try:
            cmd = INF.infer([h0, h1])
        finally:
            INF._rtc_delay_hint = None
        _PF["res"] = dict(h0=h0, h1=h1, cmd=cmd, L=INF.last, t0=t0, t_obs=time.time(), observe_ms=obs_ms,
                          infer_ms=round((time.perf_counter() - t_i) * 1000.0, 1), exec_k_now=getattr(INF, "exec_k_now", None))
        _ema("inf", _PF["res"]["infer_ms"] / 1000.0)
    except Exception as e:  # noqa: BLE001
        _PF["res"] = {"error": f"prefetch inference failed: {e}"}


def _prefetch_start():
    _PF["res"] = None
    th = threading.Thread(target=_prefetch_worker, daemon=True); _PF["thread"] = th; th.start()


def _prefetch_take():
    """the finished prefetch (waits for a running one), or None"""
    th = _PF["thread"]
    if th is None:
        return None
    th.join(timeout=10.0); _PF["thread"] = None
    r = _PF["res"]; _PF["res"] = None
    if not r or r.get("error"):
        if r: print(f"[v4-smoke] prefetch dropped: {r.get('error')}", flush=True)
        return None
    if time.time() - r["t_obs"] > PREFETCH_MAX_AGE_S:
        print(f"[v4-smoke] prefetch dropped: {time.time() - r['t_obs']:.1f}s old", flush=True); return None
    return r
# [2026-09-30 user] pose gotos (TRAIN START, probes: no TCP target) need an accurate start pose, policy steps need a fast loop:
# separate dwell for gotos. Policy steps keep V4_DWELL_TIMEOUT_S.
GOTO_DWELL_S = float(os.environ.get("V4_GOTO_DWELL_S", "2.0"))   # [2026-09-30] 1.5 -> 2.0: left pan follower lag left 9.6 mm at 1.5 s
# [2026-09-30 user] goto settle: TCP err < tol AND max joint err < V4_GOTO_JOINT_TOL_DEG for 2 consecutive samples. Stationary motion
# between the last two samples is logged (still_mm / still_warn), not a gate. V4_GOTO_STILL_S is unused. Policy steps keep the old rule.
GOTO_JOINT_TOL_DEG = float(os.environ.get("V4_GOTO_JOINT_TOL_DEG", "1.0"))
GOTO_STILL_S = float(os.environ.get("V4_GOTO_STILL_S", "0.25"))
JOINT_NAMES = ["pan", "lift", "elbow", "wflex", "wyaw", "wroll"]
# R150 150 episodes' frame-0 median pose (arms deg, gripper raw) -- the same value the v3/EEF UIs use. Start a
# rollout here or the very first observation is a start pose the policy never saw.
# The six order instructions exactly as meta/tasks.parquet stores them -- a paraphrase is a different input to
# the language encoder, so these are copied verbatim rather than retyped.
TASKS = [
    ("RBP", "Stack the red cube on the bottom, blue cube in the middle, and purple cube on the top, on the plate."),
    ("RPB", "Stack the red cube on the bottom, purple cube in the middle, and blue cube on the top, on the plate."),
    ("BRP", "Stack the blue cube on the bottom, red cube in the middle, and purple cube on the top, on the plate."),
    ("BPR", "Stack the blue cube on the bottom, purple cube in the middle, and red cube on the top, on the plate."),
    ("PRB", "Stack the purple cube on the bottom, red cube in the middle, and blue cube on the top, on the plate."),
    ("PBR", "Stack the purple cube on the bottom, blue cube in the middle, and red cube on the top, on the plate."),
]

TRAIN_START_DEG = [float(x) for x in os.environ.get(
    "V4_TRAIN_START_DEG",
    "-4.5,-2.0,-8.3,30.4,-6.9,-5.2,0.0, 4.2,-14.8,-1.1,6.0,3.2,-3.2,0.0").split(",")]

app = Flask(__name__)
INF_CKPT = os.environ.get("V4_CKPT", "")
REPLAY_FILE = os.environ.get("V4_REPLAY", "~/holobrain-mac-model/chunks_ep0.npz")
# 0 = no guard, as asked: the replay stops on E-STOP or `replay stop`, nothing else
REPLAY_MAX_STEP_MM = float(os.environ.get("V4_REPLAY_MAX_STEP_MM", "0"))
import collections as _collections
STATE = {"busy": False, "presses": 0, "last": {}, "err": None, "arm": ("right" if os.environ.get("V4_ARM_ONLY", "").strip().lower() == "right" else "both"),
         "running": False, "cycles": 0, "travel_mm": [0.0, 0.0], "stop_reason": None, "task": "RBP",
         "zlog": _collections.deque(maxlen=400)}
# A continuous loop needs its own stops, because the per-step clamp only bounds ONE step: 8 steps x 25 mm is
# 200 mm of travel. These are budgets on the whole run, and any of them ends it.
# 0 = no budget. The loop then ends only on E-STOP, STOP, an error, or IK failing to reach the first
# predicted pose -- the per-step clamp is still what bounds any single move.
MAX_CYCLES = int(os.environ.get("V4_MAX_CYCLES", "0"))
MAX_TRAVEL_MM = float(os.environ.get("V4_MAX_TRAVEL_MM", "0"))
INF = None
_lock = threading.Lock()


DT_STEP_MS = 1000.0 / 15.0                 # the policy's action step; latency is reported in these units


_SESS = requests.Session()          # keep-alive: /motor_states answers in 0.4 ms, /observe in 33 ms


def _motor_pos():
    """Light pose read for motion-onset detection: /observe ships three JPEGs, /motor_states does not."""
    try:
        d = _SESS.get(ROBOT + "/motor_states", timeout=5).json()
    except Exception:
        return None
    if not isinstance(d, dict) or "left_arm" in d and not isinstance(d.get("left_arm"), dict):
        return None
    try:
        arm, grip = [], []
        for side in ("left_arm", "right_arm"):
            for name, m in sorted(d[side].items()):
                (grip if name == "gripper" else arm).append(float(m["pos_deg"]))
        return np.asarray(arm), np.asarray(grip)
    except Exception:
        return None


def _wait_motion(base, arm_tol=0.03, grip_tol=0.3, timeout_s=0.6):
    # 0.15 deg was above the first command's realised motion (a 3 deg target moves 0.11 deg on the first
    # send), so every onset timed out. /motor_states polls at 0.4 ms, so the resolution is the threshold.
    """Time from now until the arms (and the jaws) actually start moving. Returns ms, or None on timeout."""
    t0 = time.perf_counter()
    arm_t = grip_t = None
    while time.perf_counter() - t0 < timeout_s:
        cur = _motor_pos()
        if cur is None:
            break
        if arm_t is None and float(np.abs(cur[0] - base[0]).max()) > arm_tol:
            arm_t = (time.perf_counter() - t0) * 1000.0
        if grip_t is None and float(np.abs(cur[1] - base[1]).max()) > grip_tol:
            grip_t = (time.perf_counter() - t0) * 1000.0
        if arm_t is not None and grip_t is not None:
            break
    return arm_t, grip_t


def _robot(path, method="get", **kw):
    try:
        return getattr(requests, method)(ROBOT + path, timeout=30, **kw).json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"robot_service: {e}"}


OBS_RETRIES = int(os.environ.get("V4_OBS_RETRIES", "5"))   # [2026-09-30] see _observe
_OBS_FAILS = {"retried": 0, "gave_up": 0}


def _observe():
    """[2026-09-30 user: "추론 중간에 멈춰"] robot_service /observe returns 500 whenever ONE camera read times out (right wrist,
    ~9 % of reads since the 15:51 USB re-enumeration), and a single failure ended the run loop ("observe failed"). A failed read
    is now retried up to V4_OBS_RETRIES times (50 ms apart) before it counts as a failure; retries are counted in /status."""
    for i in range(OBS_RETRIES + 1):
        o = _robot("/observe")
        if "error" not in o:
            break
        if i < OBS_RETRIES:
            _OBS_FAILS["retried"] += 1; time.sleep(0.05)
    else:
        _OBS_FAILS["gave_up"] += 1
        return None
    # Stamp the observation HERE. umi76 interpolates its history at a physical 50.05 ms, so it needs the
    # time the robot was read, not the time inference happened to start -- the two observations of a cycle
    # are OBS_GAP_S apart when taken and microseconds apart when replayed into the buffer.
    return {"images": o["images"], "joints14": o["joints_rad"], "t": time.time(),
            "cam_age_ms": o.get("cam_age_ms"), "t_wall": o.get("t_wall")}   # [2026-09-30] robot_service frame capture ages


def _leader_now():
    o = _observe()
    if o is None:
        return None
    a = list(o["joints14"])
    for k in FLIP_IDX:
        a[k] = -a[k]
    # the jaw elements are raw encoder counts and the encoder wraps (a closed jaw can read 358, not -2).
    # Anything that ramps from or holds the current pose has to start from the unwrapped value, or it
    # commands a 358-count journey to reach what is already the current position.
    import infer_core_v4 as _IC
    for g in GRIP_IDX:
        a[g] = _IC.unwrap_grip(a[g])
    return a


# [2026-10-03 user] HRA right-only safety (only when the core runs with V4_ARM_ONLY=right):
#   * at /run the LEFT arm (leader frame) and both jaws (as 0..45 commands) are latched; during a rollout every /execute_step
#     must carry exactly those values -- a deviation > HRA_LEFT_TOL_RAD / HRA_JAW_TOL means a leak of an unsupervised output
#     and the step is REFUSED and the run stopped (never silently corrected)
#   * approach-only stop: run time > HRA_MAX_RUN_S, or the predicted right TCP displacement at the chunk end < HRA_STILL_MM for
#     HRA_STILL_N inferences in a row (the approach has converged; the task has no grasp, so it must not keep pushing)
HRA_MAX_RUN_S = float(os.environ.get("V4_HRA_MAX_RUN_S", "0"))    # 0 = off (user 10-04: "최대시간같은거 없애줘"; 6 s stopped after 1-4 step-mode cycles)
HRA_STILL_MM = float(os.environ.get("V4_HRA_STILL_MM", "5"))
HRA_STILL_N = int(os.environ.get("V4_HRA_STILL_N", "3"))
HRA_LEFT_TOL_RAD = float(os.environ.get("V4_HRA_LEFT_TOL_RAD", "0.02"))
HRA_JAW_TOL = float(os.environ.get("V4_HRA_JAW_TOL", "2.0"))
HRA_GATE = os.environ.get("V4_HRA_GATE", "1") != "0"        # 0 = no refusal gate (user 10-04 "다 빼줘"); the core still holds left + jaws
HRA_MAX_CYCLE_MM = float(os.environ.get("V4_HRA_MAX_CYCLE_MM", "80"))     # measured right TCP move between two inferences
HRA_MAX_TRAVEL_MM = float(os.environ.get("V4_HRA_MAX_TRAVEL_MM", "450"))  # measured right TCP net move since /run


def _arm_only():
    import infer_core_v4 as _IC
    return _IC.ARM_ONLY


def _hra_latch():
    """latch the left arm + jaw commands at /run (leader frame, jaws as commands)"""
    import infer_core_v4 as _IC
    a = _leader_now()
    if a is None:
        return "observe failed"
    o = _observe()
    m0, _ = INF.tcp_now(o["joints14"])
    CUBE_STOP["hits"] = 0
    STATE["hra"] = dict(left=[float(a[i]) for i in range(6)], jaw=[_IC.jaw_hold_cmd(o["joints14"][g]) for g in GRIP_IDX],
                        t0=time.time(), still=0, last_call=getattr(INF, "calls", 0), gate_refusals=0, max_left_dev=0.0,
                        p0=m0[1][:3, 3].copy(), p_prev=m0[1][:3, 3].copy(), travel_mm=0.0, max_cycle_mm=0.0)
    return None


def _hra_gate(a):
    """None = send; else the refusal text. Only during a rollout of a one-arm model."""
    h = STATE.get("hra")
    if not (HRA_GATE and _arm_only() and STATE.get("running") and h):
        return None
    dev = max(abs(float(a[i]) - h["left"][i]) for i in range(6)); h["max_left_dev"] = max(h["max_left_dev"], dev)
    jd = max(abs(float(a[g]) - h["jaw"][j]) for j, g in enumerate(GRIP_IDX))
    if dev > HRA_LEFT_TOL_RAD or jd > HRA_JAW_TOL:
        h["gate_refusals"] += 1
        return f"HRA gate: left arm command off by {dev:.4f} rad (tol {HRA_LEFT_TOL_RAD}) / jaw off by {jd:.2f} (tol {HRA_JAW_TOL})"
    return None


# [2026-10-07 user "일정크기 되면 그냥 멈추게"] vision stop: the red cube in the RAW right-wrist C922 frame (640x480, before any zoom)
# measured as sqrt(area) of the largest red blob; stop when it is >= CUBE_STOP["px"] on CUBE_STOP["n"] consecutive checks (0 = off).
CUBE_STOP = {"px": float(os.environ.get("V4_CUBE_STOP_PX", "0")), "n": 2, "every_s": 0.1, "last_t": 0.0, "hits": 0, "px_now": None,
             # [2026-10-07 user "큐브가 다시 그 사이즈보다 작으면 추론 계속"] follow: after a cube-size stop, watch the cube and /run again
             # (same fixed anchor) once it is VISIBLE (>= min_px) and smaller than resume_frac * px on resume_n checks; user STOP ends it
             "follow": os.environ.get("V4_CUBE_FOLLOW", "1") != "0", "waiting": False, "resume_frac": 0.85, "resume_n": 3, "min_px": 6.0, "resumes": 0}


def cube_px(bgr):
    import cv2
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (((h[..., 0] < 10) | (h[..., 0] > 170)) & (h[..., 1] > 110) & (h[..., 2] > 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return 0.0 if n < 2 else float(np.sqrt(st[1:, 4].max()))


def _cube_check():
    c = CUBE_STOP
    if time.time() - c["last_t"] < c["every_s"]:
        return None
    c["last_t"] = time.time()
    try:
        import cv2
        r = requests.get(ROBOT + "/frame/right", timeout=1.0)
        px = cube_px(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR))
    except Exception:
        return None
    c["px_now"] = round(px, 1)
    if c["px"] <= 0:
        return None
    c["hits"] = c["hits"] + 1 if px >= c["px"] else 0
    if c["hits"] >= c["n"]:
        c["waiting"] = bool(c["follow"])
        return f"HRA: cube size {px:.0f} px >= stop size {c['px']:.0f} px (right wrist C922)" + (" -- following (resumes when smaller)" if c["follow"] else "")
    return None


def _cube_follow_watch():
    """[2026-10-07] background: while a cube-size stop is waiting, re-/run when the cube looks smaller again (cube moved away)."""
    import cv2
    ok_n = 0
    while True:
        time.sleep(0.1)
        c = CUBE_STOP
        if not (c["waiting"] and c["follow"] and c["px"] > 0) or STATE["running"]:
            ok_n = 0; continue
        try:
            r = requests.get(ROBOT + "/frame/right", timeout=1.0)
            px = cube_px(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR))
        except Exception:
            continue
        c["px_now"] = round(px, 1)
        ok_n = ok_n + 1 if (c["min_px"] <= px < c["resume_frac"] * c["px"]) else 0
        if ok_n >= c["resume_n"]:
            ok_n = 0; c["waiting"] = False; c["hits"] = 0
            with app.test_request_context("/run?reuse_anchor=1", method="POST"):
                rr = run().get_json()
            c["resumes"] += 1
            print(f"[cube-follow] cube {px:.0f} px < {c['resume_frac'] * c['px']:.0f} px -> resumed ({rr})", flush=True)


threading.Thread(target=_cube_follow_watch, daemon=True).start()


def _hra_stop():
    """approach-only stop conditions; None = keep going"""
    h = STATE.get("hra")
    if not (_arm_only() and h):
        return None
    cs = _cube_check()
    if cs:
        return cs
    if HRA_MAX_RUN_S > 0 and time.time() - h["t0"] > HRA_MAX_RUN_S:      # 0 = no time limit (user 10-04)
        return f"HRA: max run time {HRA_MAX_RUN_S:.0f} s"
    c = getattr(INF, "calls", 0)
    if c != h["last_call"] and INF.last.get("tgt_pos") is not None:
        h["last_call"] = c
        p = INF.last["cur_mat"][1][:3, 3]                      # MEASURED right TCP at this inference
        dc = float(np.linalg.norm(p - h["p_prev"])) * 1000.0; h["p_prev"] = p.copy(); h["max_cycle_mm"] = max(h["max_cycle_mm"], dc)
        h["travel_mm"] = float(np.linalg.norm(p - h["p0"])) * 1000.0
        if HRA_MAX_CYCLE_MM > 0 and dc > HRA_MAX_CYCLE_MM:
            return f"HRA: right TCP moved {dc:.0f} mm between two inferences (> {HRA_MAX_CYCLE_MM:.0f} mm)"
        if HRA_MAX_TRAVEL_MM > 0 and h["travel_mm"] > HRA_MAX_TRAVEL_MM:
            return f"HRA: right TCP net travel {h['travel_mm']:.0f} mm since /run (> {HRA_MAX_TRAVEL_MM:.0f} mm)"
        d = float(np.linalg.norm(INF.last["tgt_pos"][-1][1] - INF.last["cur_mat"][1][:3, 3])) * 1000.0
        h["last_pred_mm"] = round(d, 1); h["still"] = h["still"] + 1 if d < HRA_STILL_MM else 0
        if HRA_STILL_N > 0 and h["still"] >= HRA_STILL_N:
            return f"HRA: approach converged (predicted right TCP move < {HRA_STILL_MM:.0f} mm for {HRA_STILL_N} chunks)"
    return None


def _exec_step(a, substep=1, of=1):
    """Post one /execute_step and remember exactly what went out.

    Every path that moves the robot goes through here -- the rollout loop, the jaw probes, the pose moves --
    so the panel on the page is the actual wire and not a reconstruction of it. Units are execute_step's:
    arm joints in RADIANS in the LEADER frame (sign-flipped on FLIP_IDX against /observe), jaw as 0..45.
    """
    g = _hra_gate(a)
    if g:                                                   # refused: nothing goes on the wire, the rollout stops
        STATE["stop_reason"] = g; STATE["running"] = False
        print(f"[hra] {g}", flush=True)
        return {"error": g}
    r = _robot("/execute_step", "post", json={"action": [float(x) for x in a]})
    STATE["sent"] = {"substep": substep, "of": of, "leader_rad": [round(float(x), 5) for x in a],
                     "arm_deg": [round(math.degrees(a[i]), 2) for i in ARM_IDX],
                     "grip_cmd": [round(float(a[g]), 2) for g in GRIP_IDX],
                     "err": (r or {}).get("error") if isinstance(r, dict) else None,
                     "t": round(time.time() % 10000, 1)}
    return r


def _goto(target14, tcp_target=None, ik_fk_err=0.0, motion_base=None, pass_through=False):
    """Drive to ONE waypoint and hold it until reached or the dwell times out.

    The Damiao follower does not take a setpoint and go: a single send of a +3 deg target realises +0.11 deg,
    and 40 identical sends over ~2 s reach +1.27 deg. So each waypoint is a short closed loop -- keep sending
    the same target at CMD_HZ, watch the error, and stop on reach or timeout. Firing once per waypoint (the
    old behaviour) meant the target moved on before the arm had gone anywhere, which is why a small commanded
    motion looked like no motion at all.
    """
    cur = _leader_now()
    if cur is None:
        return {"error": "observe failed"}
    n = max(1, int(math.ceil(max(abs(target14[i] - cur[i]) for i in ARM_IDX) / SLEW_MAX_STEP_RAD)))
    arm_lat = grip_lat = None
    for k in range(1, n + 1):
        if STATE["stop_reason"] == "estop":
            return {"stopped": True, "at": k}
        # Ramp the ARM joints only. `cur` comes from /observe, whose jaw element is a RAW count (0..-270),
        # while the target's jaw is a COMMAND (0..45) -- interpolating between the two units sends values
        # like -84.9 on the intermediate sub-steps and only lands on the intended command at the last one.
        # The jaw has its own controller and does not need slewing, so it gets the target directly.
        a = [cur[i] + (target14[i] - cur[i]) * k / n for i in range(14)]
        for _g in GRIP_IDX:
            a[_g] = target14[_g]
        r = _exec_step(a, k, n)
        if isinstance(r, dict) and r.get("error"):
            return r
        time.sleep(1.0 / CMD_HZ)

    # The joint target came out of IK, so its own FK residual is a floor the robot cannot beat: if
    # ||FK(q_target) - desired_TCP|| is already 7 mm, no amount of dwell brings the measured TCP error under a
    # 6 mm tolerance. Raise the bar by the residual so a timeout means "the follower did not get there",
    # not "IK never offered a pose that close".
    tcp_tol = max(TCP_TOL_MM, ik_fk_err + 2.0)
    dwell_s = DWELL_TIMEOUT_S if tcp_target is not None else GOTO_DWELL_S
    if pass_through and tcp_target is not None:
        dwell_s = PASS_DWELL_S
    # [2026-09-30] no TCP target (pose goto): measure against FK of the TARGET joint pose instead of logging 0 -- the old
    # tcp_err_mm=0 hid a real 16 mm left-arm gap at TRAIN START.
    if tcp_target is None:
        _tf = list(target14)
        for f in FLIP_IDX:
            _tf[f] = -_tf[f]
        _mt, _ = INF.tcp_now(_tf)
        tgt_pos = [np.asarray(_mt[r_][:3, 3], float) for r_ in (0, 1)]
    else:
        tgt_pos = [np.asarray(tcp_target[0][r_], float) for r_ in (0, 1)]
    t0 = time.time(); reps = 0; reached = False
    joint_err = tcp_err = float("nan"); detail = None; settle = 0; still_mm = None; prev_tcp = None
    while time.time() - t0 < dwell_s:
        if STATE["stop_reason"] == "estop":
            break
        r = _exec_step(target14)        # the dwell repeats the same target; show it too
        if isinstance(r, dict) and r.get("error"):
            return r
        reps += 1
        time.sleep(1.0 / CMD_HZ)
        o = _observe()
        if o is None:
            break
        now = list(o["joints14"])
        for f in FLIP_IDX:
            now[f] = -now[f]
        joint_err = math.degrees(max(abs(target14[i] - now[i]) for i in ARM_IDX))
        m, _ = INF.tcp_now(o["joints14"])
        exyz = [(np.asarray(m[r_][:3, 3], float) - tgt_pos[r_]) * 1000 for r_ in (0, 1)]
        tcp_err = max(float(np.linalg.norm(e)) for e in exyz)
        jerr = [math.degrees(now[i] - target14[i]) for i in ARM_IDX]
        detail = {arm: {"target_tcp_mm": [round(float(x) * 1000, 1) for x in tgt_pos[r_]],
                        "measured_tcp_mm": [round(float(x) * 1000, 1) for x in m[r_][:3, 3]],
                        "tcp_err_xyz_mm": [round(float(x), 1) for x in exyz[r_]],
                        "tcp_err_norm_mm": round(float(np.linalg.norm(exyz[r_])), 1),
                        "target_q_deg": [round(math.degrees(target14[i]), 2) for i in ARM_IDX[r_ * 6:(r_ + 1) * 6]],
                        "measured_q_deg": [round(math.degrees(now[i]), 2) for i in ARM_IDX[r_ * 6:(r_ + 1) * 6]],
                        "joint_err_deg": dict(zip(JOINT_NAMES, [round(x, 2) for x in jerr[r_ * 6:(r_ + 1) * 6]])),
                        "joint_err_max_deg": round(max(abs(x) for x in jerr[r_ * 6:(r_ + 1) * 6]), 2),
                        "worst_joint": JOINT_NAMES[int(np.argmax([abs(x) for x in jerr[r_ * 6:(r_ + 1) * 6]]))]}
                  for arm, r_ in (("L", 0), ("R", 1))}
        if tcp_target is not None:
            if pass_through and tcp_err < max(PASS_TCP_MM, tcp_tol):
                reached = True
                break
            if joint_err < JOINT_TOL_DEG and tcp_err < tcp_tol:
                reached = True
                break
        else:                                            # pose goto: 2 consecutive passes (stationary = log/warning only)
            # [2026-09-30 user] the hard stationary gate + reset made false timeouts (final L/R 2.9/3.7 mm, joint 0.24/0.98 deg,
            # reached=false, still_mm=null). Accept on TCP < tol AND max joint < GOTO_JOINT_TOL_DEG for 2 samples in a row.
            ok_now = joint_err < GOTO_JOINT_TOL_DEG and tcp_err < tcp_tol
            settle = settle + 1 if ok_now else 0
            if prev_tcp is not None:
                still_mm = max(float(np.linalg.norm(np.asarray(m[r_][:3, 3], float) - prev_tcp[r_])) * 1000 for r_ in (0, 1))
            prev_tcp = [np.asarray(m[r_][:3, 3], float).copy() for r_ in (0, 1)]
            if settle >= 2:
                reached = True
                break
    cause = None; still_warn = None
    if not reached:
        # [2026-09-30] cause from the SAME last sample as tcp_err/joint_err/per_arm, with the thresholds this move used
        jt = JOINT_TOL_DEG if tcp_target is not None else GOTO_JOINT_TOL_DEG
        if ik_fk_err > TCP_TOL_MM:
            cause = "IK_RESIDUAL"
        elif joint_err >= jt:
            cause = "FOLLOWER_LAG"
        elif tcp_err >= tcp_tol:
            cause = "CARTESIAN_GAP"
        else:
            cause = "IN_TOL_NOT_CONSECUTIVE"
    if tcp_target is None and still_mm is not None and still_mm >= 1.0:
        still_warn = f"still moving {still_mm:.1f} mm between the last two samples"
    return {"ok": True, "substeps": n, "repeats": reps, "dwell_ms": int((time.time() - t0) * 1000),
            "joint_err_deg": round(joint_err, 2), "tcp_err_mm": round(tcp_err, 2),
            "ik_fk_err_mm": round(float(ik_fk_err), 2), "tcp_tol_mm": round(tcp_tol, 2),
            "reached": reached, "timeout": not reached, "cause": cause, "dwell_limit_s": dwell_s, "per_arm": detail, "still_mm": None if still_mm is None else round(still_mm, 2), "still_warn": still_warn,
            "arm_latency_ms": None if arm_lat is None else round(arm_lat, 1),
            "gripper_latency_ms": None if grip_lat is None else round(grip_lat, 1)}


def _mask_arm(cmd, arm, q_now):
    """Freeze one arm at its current pose so the first tests can move a single arm."""
    if arm == "both":
        return cmd
    hold = list(range(6, 14)) if arm == "left" else list(range(0, 7))     # indices to leave untouched
    lead_now = list(q_now)
    for k in FLIP_IDX:
        lead_now[k] = -lead_now[k]
    out = cmd.copy()
    for i in hold:
        out[:, i] = lead_now[i]
    return out


def _one_press():
    T = {}
    pf = _prefetch_take() if PREFETCH else None
    if pf is not None:
        # [2026-10-02] pipelined cycle: observation + inference already ran during the previous cycle's last waypoint
        h0, h1, cmd, t0 = pf["h0"], pf["h1"], pf["cmd"], pf["t0"]
        INF.last = pf["L"]
        if pf.get("exec_k_now") is not None:
            INF.exec_k_now = pf["exec_k_now"]
        T["observe_ms"] = pf["observe_ms"]; T["infer_ms"] = pf["infer_ms"]; T["prefetched"] = True
        T["prefetch_age_ms"] = round((time.time() - pf["t_obs"]) * 1000.0, 1)
        infer_s = T["infer_ms"] / 1000.0
    else:
        t_req = time.perf_counter()
        h0 = _observe()
        if h0 is None:
            return {"error": "observe failed (robot connected?)"}
        time.sleep(OBS_GAP_S)
        h1 = _observe()
        if h1 is None:
            return {"error": "observe failed"}

        T["observe_ms"] = round((time.perf_counter() - t_req) * 1000.0, 1)   # upper bound on observation age:
        # /observe returns no capture timestamp, so the true age of the pixels is at most this and cannot be
        # separated from the round trip until robot_service stamps its frames.
        t0 = time.time(); t_i = time.perf_counter()
        try:
            cmd = INF.infer([h0, h1])
        except Exception as e:  # noqa: BLE001
            return {"error": f"inference failed: {e}"}
        infer_s = time.time() - t0
        T["infer_ms"] = round((time.perf_counter() - t_i) * 1000.0, 1)
        T["prefetched"] = False
    # [2026-09-30 user] frame age: capture wall time of each camera frame fed to the model (h1 = the current observation),
    # from robot_service cam_age_ms + t_wall; ages at inference start and at each waypoint command go into the cycle log.
    _cap = {c: h1["t_wall"] - a / 1000.0 for c, a in (h1.get("cam_age_ms") or {}).items() if a is not None and h1.get("t_wall")}
    _age = lambda t_: {c: round((t_ - v) * 1000.0, 1) for c, v in _cap.items()}
    age_infer = _age(t0)
    L = INF.last
    # [2026-09-29] keep the exact inference input of every cycle (both observations: images b64, joints, t) so a real frame
    # can be re-run offline and compared with the training images (grasp-phase OOD check). -> V4_FRAME_DIR/<t>.json
    try:
        _fd = os.path.expanduser(os.environ.get("V4_FRAME_DIR", "~/v4_cycle_frames")); os.makedirs(_fd, exist_ok=True)
        with open(os.path.join(_fd, f"{h1['t']:.3f}.json"), "w") as f_:
            __import__("json").dump(dict(cycle=STATE["cycles"], ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")),
                                         exec_k=int(os.environ.get("V4_EXEC_K", "0") or 0), task=INF.task,
                                         obs=[{"t": h["t"], "joints14": list(map(float, h["joints14"])), "images": h["images"]} for h in (h0, h1)],
                                         state=[float(x) for x in L["state"]]), f_)
    except Exception as e_:  # noqa: BLE001
        print("[frame-log] skipped:", e_, flush=True)
    cmd = _mask_arm(cmd, STATE["arm"], h1["joints14"])

    m_before, _ = INF.tcp_now(h1["joints14"])
    # Axis-wise trace, in the BASE frame, so "it goes up when it should go down" can be attributed instead of
    # argued: the model's own body-frame delta rotated into base (pred), the commanded step after clamping
    # (cmd), and what the TCP actually did (actual). A sign that flips between pred and cmd is a compose bug;
    # between cmd and actual is execution; agreement all the way through is the policy.
    # Three-way vertical decomposition, one row per cycle, kept across cycles. With n_action=1 the loop
    # re-anchors on a fresh T_now every time, so a small per-step bias does not average out -- it is re-applied
    # to the new pose and the arm walks. Separating the three z's says which layer it comes from:
    #   a1_body_dz  the model's own output, before any frame or IK work touches it
    #   tgt_dz      the same step after T_now @ A[1] and the frame adapter, i.e. what was asked of the arm
    #   act_dz      what the TCP did
    # pred up and target up -> the policy. pred flat and target up -> decode/anchor. target flat and actual
    # up -> IK or the follower.
    for r in (0, 1):
        a1 = L["act"][0, r * 10:r * 10 + 3]
        tgt_dz = float(L["tgt_pos"][0][r][2] - m_before[r][2, 3]) * 1000.0
        STATE["zlog"].append({"t": round(time.time() % 10000, 1), "cycle": STATE["cycles"],
                              "arm": "LR"[r], "a1_body_dz": round(float(a1[2]) * 1000, 2),
                              "a1_body_norm": round(float(np.linalg.norm(a1)) * 1000, 2),
                              "tgt_dz": round(tgt_dz, 2), "tcp_z": round(float(m_before[r][2, 3]) * 1000, 1),
                              "act_dz": None})
    axes = {0: [], 1: []}
    prev_tgt = {r: m_before[r][:3, 3].copy() for r in (0, 1)}
    prev_act = {r: m_before[r][:3, 3].copy() for r in (0, 1)}
    sent = 0; track = []
    # execute from EXEC_K, not from the first waypoint; each is anchored on the same measured pose, so this
    # skips the low-signal head of the chunk rather than skipping motion
    _k0 = max(0, int(getattr(INF, "exec_k_now", None) or IC_EXEC_K()) - 1)     # [2026-10-01] adaptive k: what infer() chose
    _shift = 0
    if T.get("prefetched"):
        # the arm kept moving after the prefetch observation: skip the waypoints it has already passed
        _o = _observe()
        if _o is not None:
            _m = INF.tcp_now(_o["joints14"])[0]
            _d = [max(float(np.linalg.norm(np.asarray(L["tgt_pos"][i][r_], float) - _m[r_][:3, 3])) for r_ in (0, 1)) * 1000.0
                  for i in range(min(len(L["tgt_pos"]), _k0 + max(1, len(cmd))))]
            _j = int(np.argmin(_d)); _shift = _j + (1 if _d[_j] < PASS_TCP_MM else 0) - _k0
            _shift = max(0, min(_shift, max(0, len(cmd) - 1)))
        T["prefetch_shift"] = _shift
    wp_indices = [i for i in range(_k0 + _shift, _k0 + _shift + max(1, len(cmd))) if i < L["valid_step"]]
    _lead = PREFETCH_LEAD
    if _lead < 0:                                       # [2026-10-04] AUTO prefetch lead (see PREFETCH_LEAD)
        _lead = int(math.ceil(((_PFT["inf"] if _PFT["inf"] is not None else float(T.get("infer_ms") or 1500) / 1000.0) + 0.2)
                              / max(_PFT["wp"] if _PFT["wp"] is not None else 0.2, 0.05)))
        _PFT["lead"] = _lead
    _t_wp = None
    for k in wp_indices:
        # re-solve this waypoint from where the arm actually is; the Cartesian target itself is untouched
        _now = time.perf_counter()
        if _t_wp is not None: _ema("wp", min(_now - _t_wp, 1.0))
        _t_wp = _now
        o = _observe()
        if o is None:
            return {"error": "observe failed mid-chunk", "sent": sent}
        t_k = time.perf_counter()
        a, ikerr, ok, clipped = INF.solve_waypoint(o["joints14"], L["tgt_pos"][k], L["tgt_quat"][k],
                                                   L["widths"][k] if L["gripper_mode"] in ("predict", "binary", "continuous", "relative") else None)
        ik_ms = (time.perf_counter() - t_k) * 1000.0
        t_cmd_k = time.time()   # [2026-09-30] waypoint command send time (frame age at command)
        if getattr(INF, "_lik_stop", None):        # [2026-09-29] pure learned-IK A/B: stop, do not execute, record why
            print(f"[v4-smoke] LEARNED_IK_STOP: {INF._lik_stop}", flush=True)
            return {"error": f"LEARNED_IK_STOP: {INF._lik_stop}", "sent": sent}
        base = _motor_pos() if k == 0 else None
        a = _mask_arm(np.asarray(a)[None, :], STATE["arm"], o["joints14"])[0]
        _pt = PASS_THROUGH and (k != wp_indices[-1] or (PASS_LAST and PREFETCH and STATE["running"]))       # [2026-10-02] intermediate waypoint: pass through, last one: full reach test
        if PREFETCH and k == wp_indices[max(0, len(wp_indices) - 1 - _lead)] and STATE["running"] and _PF["thread"] is None:
            _PF["delay"] = len(wp_indices) - wp_indices.index(k)      # this waypoint + the rest of the cycle run during the inference
            _prefetch_start()
        if base is not None:
            r = _goto(list(a), (L["tgt_pos"][k], L["tgt_quat"][k]), ikerr, motion_base=base, pass_through=_pt)
        else:
            r = _goto(list(a), (L["tgt_pos"][k], L["tgt_quat"][k]), ikerr, pass_through=_pt)
        r["pass_through"] = _pt
        r["ik_ok"] = ok
        r["dq_clipped"] = clipped
        r["ik_ms"] = round(ik_ms, 1)
        if k == 0:
            T["ik_ms"] = round(ik_ms, 1)
            T["arm_latency_ms"] = r.get("arm_latency_ms")
            T["gripper_latency_ms"] = r.get("gripper_latency_ms")
        if isinstance(r, dict) and r.get("error"):
            return {"error": r["error"], "sent": sent}
        if r.get("stopped"):
            break
        sent += 1
        o2 = _observe()
        m_now = INF.tcp_now(o2["joints14"])[0] if o2 else None
        # [2026-09-29] per-cycle grasp diagnostic (arm vs gripper horizon): full predicted g[1:16], the selected g[k], what was
        # sent, TCP now / IK target / after, IK residual, q now / q command (follower frame), limit excess. -> V4_CYCLE_LOG
        try:
            q_cmd_f = np.asarray(a, np.float64).copy(); q_cmd_f[FLIP_IDX] *= -1.0          # leader -> follower frame
            wp = getattr(INF, "last_wp", {}) or {}
            rec_c = dict(t=round(time.time(), 3), cycle=STATE["cycles"], ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")),
                         state_mode=os.environ.get("V4_STATE_MODE"), exec_k=int(k) + 1,
                         frame_age_at_infer_ms=age_infer, frame_age_at_cmd_ms=_age(t_cmd_k), infer_ms=T.get("infer_ms"),
                         prefetched=T.get("prefetched"), prefetch_shift=T.get("prefetch_shift"), prefetch_age_ms=T.get("prefetch_age_ms"), pass_through=r.get("pass_through"),
                         g_pred=[[round(float(x), 3) for x in L["act"][:, r_ * 10 + 9]] for r_ in (0, 1)],
                         g_sel=[round(float(L["act"][k, r_ * 10 + 9]), 3) for r_ in (0, 1)],
                         grip_cmd_sent=[round(float(a[g_]), 1) for g_ in GRIP_IDX],
                         tcp_now_mm=[[round(float(x) * 1000, 1) for x in m_before[r_][:3, 3]] for r_ in (0, 1)],
                         tcp_target_mm=[[round(float(x) * 1000, 1) for x in L["tgt_pos"][k, r_]] for r_ in (0, 1)],
                         tcp_after_mm=None if m_now is None else [[round(float(x) * 1000, 1) for x in m_now[r_][:3, 3]] for r_ in (0, 1)],
                         ik_residual_mm=round(float(ikerr), 2), ik_ok=bool(ok), reached=r.get("reached"), tcp_err_mm=r.get("tcp_err_mm"),
                         q_now=[round(float(x), 4) for x in np.asarray(o["joints14"], np.float64)[ARM_IDX]],
                         q_cmd=[round(float(x), 4) for x in q_cmd_f[ARM_IDX]],
                         ik_backend_executed=getattr(INF, "_ik_used", None), lik_stop_reason=getattr(INF, "_lik_stop", None), shadow_numerical=getattr(INF, "_shadow", None), rel_input_diff_mm=getattr(INF, "_rel_diff_mm", None),
                         grip_relative=getattr(INF, "_grip_rel_log", None),
                         temporal_ensemble=getattr(INF, "te_info", None), deadband=getattr(INF, "db_info", None), adaptive_k=getattr(INF, "ak_info", None),
                         anchor_mm=None if getattr(INF, "_anchor", None) is None else [[round(float(x) * 1000, 1) for x in INF._anchor[r_][:3, 3]] for r_ in (0, 1)],
                         limit_excess=wp.get("limit_excess"), grip_width_now_mm=[round(float(x) * 1000, 1) for x in (L["state"][[36 + 1, 74 + 1]] if len(L["state"]) == 76 else L["state"][18:20] * INF_W_OPEN)])
            with open(CYCLE_LOG, "a") as f_:
                f_.write(__import__("json").dumps(rec_c) + "\n")
        except Exception as e_:  # noqa: BLE001 -- diagnostics must never break a waypoint
            print("[cycle-log] skipped:", e_, flush=True)
        for r_ in (0, 1):
            raw_body = L["act"][k, r_ * 10:r_ * 10 + 3]                       # model output, body frame, m
            # UMI current-anchor puts EVERY step of the chunk on the same measured T_now, so the body->world
            # rotation is m_before for all k, not only the first. This used to be computed at k == 0 only,
            # which left pred_world null for the whole run whenever exec_k > 1 (execution starts at wp
            # exec_k-1) -- and pred_world is the only field comparable with cmd_world, so the one check the
            # comment above calls for ("a sign that flips between pred and cmd is a compose bug") could not
            # be made. pred_body is in the body frame and must never be compared with cmd_world directly.
            R_cur = m_before[r_][:3, :3]
            pred_world = (R_cur @ raw_body) * 1000.0
            cmd_world = (L["tgt_pos"][k, r_] - prev_tgt[r_]) * 1000.0
            act_world = ((m_now[r_][:3, 3] - prev_act[r_]) * 1000.0) if m_now is not None else None
            axes[r_].append({
                "wp": k,
                "tcp_xyz_mm": [round(float(x) * 1000, 1) for x in prev_act[r_]],
                "pred_body_dxyz_mm": [round(float(x) * 1000, 2) for x in raw_body],
                "pred_world_dxyz_mm": None if pred_world is None else [round(float(x), 2) for x in pred_world],
                "cmd_world_dxyz_mm": [round(float(x), 2) for x in cmd_world],
                "actual_world_dxyz_mm": None if act_world is None else [round(float(x), 2) for x in act_world]})
            prev_tgt[r_] = L["tgt_pos"][k, r_].copy()
            if m_now is not None:
                prev_act[r_] = m_now[r_][:3, 3].copy()
        track.append({k_: r.get(k_) for k_ in ("repeats", "dwell_ms", "joint_err_deg", "tcp_err_mm",
                                               "ik_fk_err_mm", "tcp_tol_mm", "reached", "timeout", "cause",
                                               "ik_ok", "dq_clipped")})
    time.sleep(CYCLE_SETTLE_S)     # [2026-10-06] was a fixed 0.2 s (only for the m_after travel telemetry); HRA loader sets 0
    h2 = _observe()
    m_after, _ = (INF.tcp_now(h2["joints14"]) if h2 else (None, None))

    rows = []
    for r, nm in ((0, "left"), (1, "right")):
        k = sent - 1 if sent else 0
        want = np.linalg.inv(m_before[r]) @ np.array(
            np.vstack([np.hstack([Rot.from_quat(L["tgt_quat"][k, r]).as_matrix(), L["tgt_pos"][k, r][:, None]]),
                       [0, 0, 0, 1]]))
        got = np.linalg.inv(m_before[r]) @ m_after[r] if m_after is not None else np.eye(4)
        cos = float(np.dot(want[:3, 3], got[:3, 3]) /
                    max(np.linalg.norm(want[:3, 3]) * np.linalg.norm(got[:3, 3]), 1e-12))
        rows.append({
            "arm": nm,
            "pred_mm": round(float(L["pred_mm"][:sent or 1, r].sum()), 3),
            "pred_deg": round(float(L["pred_deg"][:sent or 1, r].sum()), 3),
            "cmd_mm": round(float(np.linalg.norm(want[:3, 3]) * 1000), 3),
            "cmd_deg": round(float(np.degrees(np.linalg.norm(Rot.from_matrix(want[:3, :3]).as_rotvec()))), 3),
            "actual_mm": round(float(np.linalg.norm(got[:3, 3]) * 1000), 3),
            "actual_deg": round(float(np.degrees(np.linalg.norm(Rot.from_matrix(got[:3, :3]).as_rotvec()))), 3),
            "dir_cos": round(cos, 3),
            "fkerr_mm": round(float(L["fkerr_mm"][:sent or 1, r].max()), 2),
        })

    STATE["presses"] += 1
    for i, row in enumerate(rows):
        STATE["travel_mm"][i] += row["actual_mm"]
    tel = {"infer_s": round(infer_s, 2), "sent": sent, "kept": int(L["kept"]), "valid": int(L["valid_step"]),
           "clamped_steps": int(L["clamped_steps"]), "dq_clipped": int(L["clipped"]),
           "arm": STATE["arm"], "presses": STATE["presses"], "rows": rows,
           "state20": [round(float(x), 4) for x in L["state"]],
           "gripper": L["gripper_mode"],
           "latency": {**T,
                       "total_ms": round(sum(v for v in (T.get("observe_ms"), T.get("infer_ms"),
                                                         T.get("ik_ms"), T.get("arm_latency_ms"))
                                             if v is not None), 1),
                       "step_ms": round(DT_STEP_MS, 1)},
           "skip_steps_would_be": round(sum(v for v in (T.get("observe_ms"), T.get("infer_ms"),
                                                        T.get("ik_ms"), T.get("arm_latency_ms"))
                                            if v is not None) / DT_STEP_MS, 2),
           "tracking": track,
           "axes": {"left": axes[0], "right": axes[1]},
           "reached": sum(1 for t in track if t.get("reached")),
           "timeouts": sum(1 for t in track if t.get("timeout")),
           "grip_width_mm": [round(float(L["widths"][:sent or 1, r].mean()) * 1000, 1) for r in (0, 1)],
           "grip_cmd": [round(float(L["grip_cmd"][:sent or 1, r].mean()), 1) for r in (0, 1)],
           "clamp": f"{LIMITS['mm']} mm / {LIMITS['deg']} deg per step"}
    STATE["last"] = tel
    return tel


def _loop():
    """Continuous rollout: same one-press cycle, with run-level budgets on top of the per-step clamp."""
    STATE["cycles"] = 0; STATE["travel_mm"] = [0.0, 0.0]; STATE["stop_reason"] = None
    _jaw_torque(True)          # a run can start straight after a rest, which left the jaws unpowered
    while STATE["running"]:
        with _lock:
            tel = _one_press()
        STATE["cycles"] += 1
        if isinstance(tel, dict) and tel.get("error"):
            STATE["stop_reason"] = f"error: {tel['error']}"; break
        if MAX_CYCLES and STATE["cycles"] >= MAX_CYCLES:
            STATE["stop_reason"] = f"cycle budget {MAX_CYCLES} reached"; break
        if MAX_TRAVEL_MM and max(STATE["travel_mm"]) >= MAX_TRAVEL_MM:
            STATE["stop_reason"] = f"travel budget {MAX_TRAVEL_MM} mm reached"; break
        if tel.get("valid", 0) < 1:
            STATE["stop_reason"] = "IK could not reach the first predicted pose"; break
        hs = _hra_stop()
        if hs:
            STATE["stop_reason"] = hs; break
    STATE["running"] = False
    if STATE["stop_reason"] is None:
        STATE["stop_reason"] = "stopped by user"
    print(f"[v4-smoke] loop ended: {STATE['stop_reason']} after {STATE['cycles']} cycles, "
          f"travel {[round(x,1) for x in STATE['travel_mm']]} mm", flush=True)


# [2026-09-30 user] gripper zero gate: the left zero re-latches after every left-arm power drop (read +18/+22/357/362 today).
# A slipped zero reads outside the valid raw range, the state clips it to "closed" and commands land in the wrong frame,
# so RUN / MOVE 1 CHUNK / PROMPT A/B are refused until both jaws read inside [V4_GRIP_GATE_LO, V4_GRIP_GATE_HI] (-280, +5).
GRIP_GATE = (float(os.environ.get("V4_GRIP_GATE_LO", "-280")), float(os.environ.get("V4_GRIP_GATE_HI", "5")))


def _grip_gate():
    """None if both jaws read a valid raw, else an error dict (shown on the page)."""
    ob = _observe()
    if not ob or "joints14" not in ob:
        return {"error": "gripper zero gate: observe failed"}
    raw = [float(ob["joints14"][g]) for g in GRIP_IDX]
    bad = [a for a, r in zip(("left", "right"), raw) if not (GRIP_GATE[0] <= r <= GRIP_GATE[1])]
    if bad:
        return {"error": f"gripper zero slipped on {'/'.join(bad)}: raw L/R {raw[0]:.1f}/{raw[1]:.1f} outside "
                         f"[{GRIP_GATE[0]:.0f}, {GRIP_GATE[1]:.0f}] -- re-zero (1 release -> close by hand -> 2 set zero)",
                "grip_raw_LR": [round(r, 2) for r in raw]}
    return None


# ---------------------------------------------------------------------------------------------------------------------------
# [2026-10-01 user: "추론과 실행을 겹치는 파이프라이닝(동료 방식) 이걸 가져와서 구현해줘"] STREAMING rollout, after the colleague's
# bh_indy7_LeRobot chunk_lookahead.py + action_timeline.merge_chunk: the next forward runs WHILE the current chunk executes.
#   inference thread: observe (h0, h1) -> INF.infer -> the chunk's 16 absolute base-frame targets get a time each,
#                     tau_j = t_obs + (j+1) * DT_EXEC; rows whose time has passed are dropped, rows that overlap a pending one are
#                     blended (V4_STREAM_BLEND new weight, 0.7 = lerobot weighted_average), the rest are added. Back-to-back.
#   executor (this run's loop): every DT_EXEC pops the row due now, re-solves IK from the measured joints (/motor_states, fast,
#                     no images) and posts /execute_step WITHOUT waiting for arrival; the robot side (MIT bridge profile) joins
#                     the targets. A dry timeline sends nothing (the arm holds its last target) and is counted (starvation).
# DT_EXEC = UMI_HISTORY_DT (50.05 ms, the chunk's own step) / V4_STREAM_SPEED. The chunk must cover two forwards
# (colleague's note): 16 * DT_EXEC >= 2 * latency -> fp32 (~0.95 s) needs SPEED <= ~0.4, fp16 (~0.45 s) <= ~0.9.
import json, collections
from infer_core_v4 import UMI_HISTORY_DT as UMI_DT       # 50.05 ms: the chunk's own step
STREAM = os.environ.get("V4_STREAM", "0") == "1"
STREAM_SPEED = float(os.environ.get("V4_STREAM_SPEED", "0.4"))
STREAM_BLEND = float(os.environ.get("V4_STREAM_BLEND", "0.7"))
STREAM_LOG = os.path.expanduser(os.environ.get("V4_STREAM_LOG", "~/v4_stream.jsonl"))
# [2026-10-01 user: "맞는때에 추론을 넣게"] ChunkLookahead schedule: the next observe+forward starts only when the pending timeline
# covers no more than the expected latency (EMA of the measured one) + V4_STREAM_MARGIN_S, so the new chunk lands just before
# the current one runs out (fresh observation, no wasted rows). V4_STREAM_SCHED=0 = back-to-back forwards (previous behaviour).
STREAM_SCHED = os.environ.get("V4_STREAM_SCHED", "1") == "1"
STREAM_MARGIN_S = float(os.environ.get("V4_STREAM_MARGIN_S", "0.2"))
# [2026-10-01 user: "사이에 약간의 텀을 줘"] pause at every chunk boundary (the executor switches to rows of a newer chunk): the arm
# holds for V4_STREAM_PAUSE_S and the whole pending timeline is shifted by the same amount (nothing is dropped as late).
STREAM_PAUSE_S = float(os.environ.get("V4_STREAM_PAUSE_S", "0.3"))
STREAM_WP_PAUSE_S = float(os.environ.get("V4_STREAM_WP_PAUSE_S", "0.0"))
# [2026-10-01 user: "사이가 멈춰도 돼 ... 갑자기 확 튀던가 예전 동작으로 돌아가"] SYNC streaming: a pipelined chunk is predicted from
# an observation ~1 s old while the arm kept moving, so at the boundary the new plan jumps or points BACK along the path. With
# V4_STREAM_SYNC=1 the next observe+forward starts only when the timeline is empty and the arm has settled
# (V4_STREAM_SETTLE_S), and the chunk's first V4_STREAM_ROWS rows are timed from NOW (tau_j = now + j*dt, nothing dropped or
# blended): stream within a chunk, stop between chunks for the forward.
STREAM_SYNC = os.environ.get("V4_STREAM_SYNC", "0") == "1"
STREAM_ROWS = int(os.environ.get("V4_STREAM_ROWS", "8"))
# [2026-10-02 user "step k는 8"] non-sync streaming: only the first V4_STREAM_K rows (k1..kK) of every chunk enter the timeline
# (0 = all valid rows, the previous behaviour); the next chunk is scheduled/merged as before, so each plan is executed up to kK.
STREAM_K = int(os.environ.get("V4_STREAM_K", "0"))
STREAM_SETTLE_S = float(os.environ.get("V4_STREAM_SETTLE_S", "0.15"))   # [2026-10-01 user] extra hold after every waypoint (timeline shifted too)
_MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]


def _joints14_fast():
    """/motor_states (no images) -> the joints14 of /observe: arm joints in RAD (raw frame), jaws raw degrees."""
    try:
        d = _SESS.get(ROBOT + "/motor_states", timeout=2).json()
        out = []
        for side in ("left_arm", "right_arm"):
            for n in _MOTOR_ORDER:
                v = float(d[side][n]["pos_deg"]); out.append(v if n == "gripper" else math.radians(v))
        return out
    except Exception:
        return None


def _stream_merge(tl, rows, now):
    """merge_chunk: rows = [(tau, pos(2,3), quat(2,4), width(2), chunk_id, j)] into the pending timeline tl (list, sorted)."""
    half = 0.5 * (UMI_DT / STREAM_SPEED)
    kept = [r for r in tl if r[0] >= now - half]
    dropped_old = len(tl) - len(kept); dropped_new = 0; blended = 0
    for (tau, P, Q, W, cid, j) in rows:
        if tau < now - half:
            dropped_new += 1; continue
        hit = next((i for i, r in enumerate(kept) if abs(r[0] - tau) <= half), None)
        if hit is None:
            kept.append((tau, P, Q, W, cid, j))
        else:
            o = kept[hit]; w = STREAM_BLEND
            q = np.stack([Rot.from_quat(np.stack([o[2][r], Q[r]])).mean(weights=[1 - w, w]).as_quat() for r in (0, 1)])
            kept[hit] = (o[0], (1 - w) * o[1] + w * P, q, (1 - w) * o[3] + w * W, cid, j); blended += 1
    kept.sort(key=lambda r: r[0])
    return kept, dict(dropped_old=dropped_old, dropped_new=dropped_new, blended=blended)


def _stream_infer_worker(TL, stats):
    cid = 0; lat_ema = None
    while STATE["running"]:
        if STREAM_SYNC and cid > 0:
            while STATE["running"]:
                with TL["lock"]:
                    empty = not TL["rows"]
                if empty and time.time() - TL.get("last_sent_t", 0.0) >= STREAM_SETTLE_S + UMI_DT / STREAM_SPEED:
                    break
                time.sleep(0.01)
            if not STATE["running"]:
                break
        elif STREAM_SCHED and cid > 0:
            # wait until the timeline left to play is about one forward long
            while STATE["running"]:
                with TL["lock"]:
                    left = (TL["rows"][-1][0] - time.time()) if TL["rows"] else 0.0
                if left <= (lat_ema or 1.0) + STREAM_MARGIN_S:
                    break
                time.sleep(0.02)
            if not STATE["running"]:
                break
        t0 = time.time()
        h0 = _observe()
        if h0 is None:
            stats["obs_fail"] += 1; time.sleep(0.05); continue
        time.sleep(OBS_GAP_S)
        h1 = _observe()
        if h1 is None:
            stats["obs_fail"] += 1; time.sleep(0.05); continue
        t_obs = time.time()
        try:
            INF.infer([h0, h1])
        except Exception as e:  # noqa: BLE001
            STATE["stop_reason"] = f"inference failed: {e}"; STATE["running"] = False; break
        L_ = INF.last; T = min(int(L_["valid_step"]), len(L_["tgt_pos"])); dt = UMI_DT / STREAM_SPEED
        if STREAM_SYNC:
            t_start = time.time()
            rows = [(t_start + j * dt, L_["tgt_pos"][j].copy(), L_["tgt_quat"][j].copy(), L_["widths"][j].copy(), cid, j) for j in range(min(T, STREAM_ROWS))]
        else:
            rows = [(t_obs + (j + 1) * dt, L_["tgt_pos"][j].copy(), L_["tgt_quat"][j].copy(), L_["widths"][j].copy(), cid, j) for j in range(min(T, STREAM_K) if STREAM_K > 0 else T)]
        now = time.time()
        with TL["lock"]:
            TL["rows"], m = _stream_merge(TL["rows"], rows, now)
        cid += 1; stats["chunks"] = cid
        lat = now - t0; stats["lat_s"].append(lat); lat_ema = lat if lat_ema is None else 0.7 * lat_ema + 0.3 * lat
        try:
            with open(STREAM_LOG, "a") as f:
                f.write(json.dumps(dict(t=round(now, 3), kind="chunk", chunk=cid, latency_s=round(lat, 3), infer_ms=round((now - t_obs) * 1000, 1),
                                        speed=STREAM_SPEED, pending=len(TL["rows"]), lat_ema_s=round(lat_ema, 3), sched=STREAM_SCHED,
                                        # [2026-10-02 user: gripper never closes] the chunk's full predicted gripper g, k1..k16, per arm
                                        g16=[[round(float(x), 3) for x in np.asarray(L_["widths"])[:T, r_]] for r_ in (0, 1)], stream_k=STREAM_K, **m)) + "\n")
        except Exception:
            pass


# ==== [2026-10-06 user: "추론에서_모터까지" port] PLAN mode ==================================================================
# One forward is always in flight. Rows of the plan table carry a global row number; the bridge (/plan/*) plays them at 100 Hz as
# knots (segment position + slope as velocity feed-forward). A forward is requested with anchor = the row the playback heads to
# at the observation; the rows that will run while it computes are RESERVED (anchor .. F) and never replaced: the result replaces
# only the rows after max(C handed, F). Rows are handed in ~V4_PLAN_WINDOW_S portions; while a forward runs nothing beyond F is
# handed, and PACE slows the playback so the reserved rows last until the result (+ a brake lead). If the rows still run out the
# arm settles on the last knot. RTC (if on) uses the plan rows as the prefix: hard for the reserved d = F - anchor rows.
# Model rows = UMI_DT (50.05 ms, the training spacing); playback pace <= V4_STREAM_SPEED (the UI speed %).
PLAN_MODE = os.environ.get("V4_PLAN", "0") == "1"
PLAN_WINDOW_S = float(os.environ.get("V4_PLAN_WINDOW_S", "0.12"))
PLAN_BRAKE_LEAD_S = float(os.environ.get("V4_PLAN_BRAKE_LEAD_S", "0.15"))
PLAN_TICK_S = float(os.environ.get("V4_PLAN_TICK_S", "0.02"))
PLAN_LOG = os.path.expanduser(os.environ.get("V4_PLAN_LOG", "~/v4_plan.jsonl"))
# [2026-10-06 user: "지그재그"] seam smoothing (the doc's §07 EMA + an overlap crossfade) and a minimum execution horizon:
#   PLAN_EMA     q = prev + a (q_raw - prev) on the arm joints, prev = the boundary row (row F, already fixed); 0 or >= 1 = off
#   PLAN_XFADE   the first N new rows are blended with the previous plan's rows at the same row numbers, weight (i+1)/(N+1)
#   PLAN_MIN_EXEC the next forward is requested only after this many rows of the installed chunk have started (or earlier if the
#                 plan would run dry first): fewer, longer-lived chunks = fewer direction changes
PLAN_EMA = float(os.environ.get("V4_PLAN_EMA", "0.35"))
PLAN_XFADE = int(os.environ.get("V4_PLAN_XFADE", "4"))
PLAN_MIN_EXEC = int(os.environ.get("V4_PLAN_MIN_EXEC", "6"))
PLAN_TRACE = os.environ.get("V4_PLAN_TRACE", "1") == "1"
# [2026-10-06 user: "평활화"] within-chunk smoothing of the model's 16 targets before IK (the jitter is IN the chunk: 2nd difference
# ~69 % of the step). Positions: least-squares polynomial of degree PLAN_SMOOTH_DEG in the step index, the current TCP (k = 0) pinned
# with a large weight so the chunk still starts where the arm is; rotations: the same fit on the rotation vectors relative to the
# current TCP rotation (degree min(2, deg)). 0 = off. No lag across chunks (one chunk at a time).
PLAN_SMOOTH_DEG = int(os.environ.get("V4_PLAN_SMOOTH_DEG", "3"))


def _smooth_chunk(P, Q, cur):
    """P (T,2,3), Q (T,2,4 xyzw), cur (2,4,4) -> smoothed copies (both arms)."""
    T = len(P); deg = min(PLAN_SMOOTH_DEG, T - 1)
    if deg < 1 or T < 3:
        return P, Q
    P = np.asarray(P, float).copy(); Q = np.asarray(Q, float).copy()
    k = np.arange(0, T + 1, dtype=float); w = np.ones(T + 1); w[0] = 100.0      # k = 0: the current pose, pinned
    for r in range(2):
        Y = np.vstack([cur[r][:3, 3][None], P[:, r]])
        for ax in range(3):
            c = np.polyfit(k, Y[:, ax], deg, w=w); P[:, r, ax] = np.polyval(c, k[1:])
        R0 = cur[r][:3, :3]; rv = np.vstack([np.zeros(3)[None], Rot.from_matrix(np.einsum("ji,tjk->tik", R0, Rot.from_quat(Q[:, r]).as_matrix())).as_rotvec()])
        dr = min(2, deg)
        for ax in range(3):
            c = np.polyfit(k, rv[:, ax], dr, w=w); rv[1:, ax] = np.polyval(c, k[1:])
        Q[:, r] = Rot.from_matrix(np.einsum("ij,tjk->tik", R0, Rot.from_rotvec(rv[1:]).as_matrix())).as_quat()
    return P, Q


def _jag(P):
    v = np.diff(P, axis=0); a = np.diff(v, axis=0)
    return round(float(np.linalg.norm(a, axis=1).mean() / max(np.linalg.norm(v, axis=1).mean(), 1e-9)), 3) if len(P) >= 3 else None      # [10-06] per-tick measured right TCP + per-chunk targets/seams in PLAN_LOG


def _plan_leader(q14):
    a = list(map(float, q14))
    for i in FLIP_IDX:
        a[i] = -a[i]
    import infer_core_v4 as _IC
    for g in GRIP_IDX:
        a[g] = float(_IC.jaw_hold_cmd(q14[g])) if hasattr(_IC, "jaw_hold_cmd") else float(np.clip(_IC.unwrap_grip(q14[g]) / -6.0, 0, 45))
    return a


def _plan_forward_job(job, out):
    try:
        t_req = time.time()
        h0 = _observe(); time.sleep(OBS_GAP_S); h1 = _observe()
        if h0 is None or h1 is None:
            out["res"] = {"error": "observe failed"}; return
        t_obs = time.time()
        import infer_core_v4 as _IC
        INF._rtc_plan_prefix = job["prefix"] if _IC.RTC_ON else None; INF._rtc_delay_hint = job["d"]
        try:
            INF.infer([h0, h1])
        finally:
            INF._rtc_plan_prefix = None; INF._rtc_delay_hint = None
        L_ = INF.last; T = min(int(L_["valid_step"]), len(L_["tgt_pos"]))
        seed = list(map(float, h1["joints14"])); rows = {}; anchor = job["anchor"]; ik_bad = None
        P_all = np.asarray(L_["tgt_pos"][:T], float); Q_all = np.asarray(L_["tgt_quat"][:T], float); jag_raw = _jag(P_all[:, 1]); jag_sm = None
        if PLAN_SMOOTH_DEG > 0 and L_.get("cur_mat") is not None:
            P_all, Q_all = _smooth_chunk(P_all, Q_all, L_["cur_mat"]); jag_sm = _jag(P_all[:, 1])
        # [2026-10-06 seam fix] compile from the BOUNDARY row (row F, the command the new rows continue from), not from the measured
        # joints of the observation: rows <= F are discarded anyway, and an IK chain seeded at the observation can pick a different
        # wrist posture for the same TCP -> a joint jump at the seam. seam_jump_deg = first new row vs the boundary row.
        if job.get("boundary") is not None:
            seed = list(job["boundary"])
            for i in FLIP_IDX:
                seed[i] = -seed[i]
            seed[GRIP_IDX[0]] = float(h1["joints14"][GRIP_IDX[0]]); seed[GRIP_IDX[1]] = float(h1["joints14"][GRIP_IDX[1]])
        seam_jump = None
        for j in range(T):
            if anchor + j + 1 <= job["F"] and job.get("boundary") is not None:
                continue
            P = P_all[j].copy(); Q = Q_all[j].copy(); W = np.asarray(L_["widths"][j]).copy()
            a, ikerr, ok, clipped = INF.solve_waypoint(seed, P, Q, None)
            if not ok or getattr(INF, "_lik_stop", None):
                ik_bad = f"IK failed at step {j + 1}"; break
            a = list(map(float, a))
            if job.get("left_hold") is not None:
                a[0:7] = job["left_hold"]
            r_no = anchor + j + 1
            if 0.0 < PLAN_EMA < 1.0 and r_no > job["F"] and job.get("boundary") is not None:
                prev = job["boundary"]
                for i_ in ARM_IDX:
                    a[i_] = prev[i_] + PLAN_EMA * (a[i_] - prev[i_])
            if r_no > job["F"]:
                if seam_jump is None and job.get("boundary") is not None:
                    seam_jump = round(float(np.degrees(np.abs(np.asarray(a)[ARM_IDX] - np.asarray(job["boundary"])[ARM_IDX]).max())), 2)
                job["boundary"] = list(a)
            rows[r_no] = dict(action=a, P=P, Q=Q, W=W, ikerr=float(ikerr), clipped=int(clipped))
            nxt = list(rows[r_no]["action"]) if False else list(a)
            for i in FLIP_IDX:
                nxt[i] = -nxt[i]
            nxt[GRIP_IDX[0]] = seed[GRIP_IDX[0]]; nxt[GRIP_IDX[1]] = seed[GRIP_IDX[1]]; seed = nxt
        out["res"] = dict(anchor=anchor, rows=rows, t_req=t_req, t_obs=t_obs, t_done=time.time(), ik_bad=ik_bad,
                          rtc=(INF.last or {}).get("rtc"), infer_ms=round((time.time() - t_obs) * 1000, 1), jag_raw=jag_raw, jag_sm=jag_sm,
                          seam_jump_deg=seam_jump)
    except Exception as e:  # noqa: BLE001
        out["res"] = {"error": f"forward failed: {e}"}


def _plan_loop():
    STATE["cycles"] = 0; STATE["travel_mm"] = [0.0, 0.0]; STATE["stop_reason"] = None
    _jaw_torque(True)
    stats = {"chunks": 0, "handed": 0, "dry_ticks": 0, "lat_s": collections.deque(maxlen=50), "pace": None, "last": None}
    STATE["plan"] = stats
    o = _observe()
    if o is None:
        STATE["stop_reason"] = "observe failed"; STATE["running"] = False; return
    import infer_core_v4 as _IC
    left_hold = _plan_leader(o["joints14"])[0:7] if _IC.ARM_ONLY == "right" else None
    sid = int(time.time() * 1000) % 1000000000
    r = _robot("/plan/open", "post", json={"session": sid})
    if r.get("error"):
        STATE["stop_reason"] = f"plan open: {r['error']}"; STATE["running"] = False; return
    plan = {}; plan_end = 0; F = None; fw = {"th": None, "res": None, "job": None, "t0": None}; lat_ema = None
    base = max(0.05, STREAM_SPEED); pace_sent = 1.0; rows_prev = None; last_cut = 0

    def contiguous_end(c):
        e = c
        while (e + 1) in plan:
            e += 1
        return e

    def log(**kw):
        try:
            with open(PLAN_LOG, "a") as f_:
                f_.write(json.dumps(dict(t=round(time.time(), 3), session=sid, **kw)) + "\n")
        except Exception:
            pass
    try:
        while STATE["running"]:
            tick = time.time()
            hs = _hra_stop()
            if hs:
                STATE["stop_reason"] = hs; break
            prog = _robot("/plan/progress")
            if prog.get("error") or prog.get("session") != sid:
                STATE["stop_reason"] = f"plan session gone: {prog.get('error') or prog.get('glitch') or 'closed'}"; break
            C = int(prog["last_row"])
            if PLAN_TRACE:
                q14_ = _joints14_fast()
                if q14_ is not None:
                    try:
                        m_, _ = INF.tcp_now(q14_)
                        log(kind="tick", exec_row=int(prog["exec_row"]), tcp_mm=[round(float(x) * 1000, 1) for x in m_[1][:3, 3]], pace=round(pace_sent, 3))
                    except Exception:
                        pass
            # 1. install a finished forward: replace only the rows after max(C, F)
            if fw["th"] is not None and not fw["th"].is_alive():
                res = fw["res"]; job = fw["job"]; fw["th"] = None
                if not res or res.get("error"):
                    STATE["stop_reason"] = (res or {}).get("error", "forward lost"); break
                cut = max(C, job["F"])
                old = {r_: plan[r_] for r_ in plan if r_ > cut}
                seam_mm = [round(float(np.linalg.norm(np.asarray(res["rows"][r_]["P"])[1] - np.asarray(old[r_]["P"])[1]) * 1000), 1)
                           for r_ in sorted(old)[:6] if r_ in res["rows"]]
                tgt_mm = {int(r_): [round(float(x) * 1000, 1) for x in np.asarray(v["P"])[1]] for r_, v in res["rows"].items()}
                for r_ in old:
                    del plan[r_]
                for r_, v in res["rows"].items():
                    if r_ > cut:
                        i_ = r_ - cut - 1
                        if PLAN_XFADE > 0 and i_ < PLAN_XFADE and r_ in old:
                            w_ = (i_ + 1) / (PLAN_XFADE + 1); a_ = list(v["action"]); o_ = old[r_]["action"]
                            for k_ in ARM_IDX:
                                a_[k_] = (1 - w_) * o_[k_] + w_ * a_[k_]
                            v = dict(v, action=a_)
                        plan[r_] = v
                last_cut = cut
                plan_end = contiguous_end(C)
                lat = time.time() - res["t_req"]; lat_ema = lat if lat_ema is None else 0.7 * lat_ema + 0.3 * lat
                stats["chunks"] += 1; stats["lat_s"].append(round(lat, 3)); STATE["cycles"] = stats["chunks"]
                stats["last"] = dict(anchor=job["anchor"], F=job["F"], C=C, cut=cut, new_rows=sum(1 for r_ in res["rows"] if r_ > cut),
                                     jag_raw=res.get("jag_raw"), jag_sm=res.get("jag_sm"), seam_jump_deg=res.get("seam_jump_deg"),
                                     plan_end=plan_end, lat_s=round(lat, 3), infer_ms=res.get("infer_ms"), ik_bad=res.get("ik_bad"))
                log(kind="chunk", **stats["last"], seam_mm=seam_mm, tgt_mm=tgt_mm, rtc=res.get("rtc") and {k: res["rtc"].get(k) for k in ("delay", "horizon", "leftover", "source")},
                    pace=round(pace_sent, 3), lat_ema=round(lat_ema, 3))
                F = None
                if plan_end <= C and not res["rows"]:
                    STATE["stop_reason"] = res.get("ik_bad") or "empty chunk"; break
            # 2. hand rows: keep ~PLAN_WINDOW_S queued on the bridge; never past F while a forward runs
            limit = F if (fw["th"] is not None and F is not None) else plan_end
            rem = float(prog["remaining_s"]); batch = []; nxt = C + 1
            while rem < PLAN_WINDOW_S and nxt <= limit and nxt in plan:
                batch.append(dict(row=nxt, action=plan[nxt]["action"], dur_s=UMI_DT)); rem += UMI_DT / max(pace_sent, 1e-3); nxt += 1
            if batch:
                rr = _robot("/plan/rows", "post", json={"session": sid, "rows": batch})
                if rr.get("error"):
                    STATE["stop_reason"] = f"plan rows: {rr['error']}"; break
                stats["handed"] += len(batch); C = nxt - 1
            for r_ in [r_ for r_ in plan if r_ < int(prog["exec_row"]) - 1]:
                del plan[r_]
            # 3. always one forward: anchor = the row the playback heads to now; reserve the rows that run meanwhile
            ex_ = int(prog["exec_row"])
            low = (plan_end - ex_) * UMI_DT / max(pace_sent, 1e-3) < ((lat_ema or 1.0) + PLAN_BRAKE_LEAD_S) * 1.3
            if fw["th"] is None and (stats["chunks"] == 0 or ex_ - last_cut >= PLAN_MIN_EXEC or low):
                anchor = int(prog["exec_row"]); L = lat_ema or 1.0
                n_res = int(math.ceil((L + PLAN_BRAKE_LEAD_S) * max(pace_sent, 0.05) / UMI_DT)) + 1
                F = max(C, min(plan_end, anchor + n_res)) if plan_end > anchor else C
                pre = [(plan[r_]["P"], plan[r_]["Q"], plan[r_]["W"]) for r_ in range(anchor + 1, plan_end + 1) if r_ in plan][:16]
                job = dict(anchor=anchor, F=F, d=max(0, F - anchor), prefix=pre or None, left_hold=left_hold,
                           boundary=list(plan[F]["action"]) if F in plan else None)
                fw.update(res=None, job=job, t0=time.time())
                fw["th"] = threading.Thread(target=_plan_forward_job, args=(job, fw), daemon=True); fw["th"].start()
            # 4. pace: the reserved rows must last until the result arrives (+ brake lead)
            p = base
            if fw["th"] is not None and F is not None and lat_ema is not None:
                avail_1x = float(prog["remaining_1x_s"]) + max(0, F - C) * UMI_DT
                l_rem = max(0.05, lat_ema - (time.time() - fw["t0"]))
                p = float(np.clip(avail_1x / (l_rem + PLAN_BRAKE_LEAD_S), 0.1, base))
            if abs(p - pace_sent) / max(pace_sent, 1e-3) > 0.05:
                _robot("/plan/pace", "post", json={"session": sid, "pace": p}); pace_sent = p
            stats["pace"] = round(pace_sent, 3)
            if prog.get("dry") and stats["chunks"] > 0:
                stats["dry_ticks"] += 1
            sl = PLAN_TICK_S - (time.time() - tick)
            if sl > 0:
                time.sleep(sl)
    except Exception as e:  # noqa: BLE001
        STATE["stop_reason"] = f"plan loop: {e}"
    finally:
        _robot("/plan/close", "post", json={"session": sid})
        STATE["running"] = False
        if STATE["stop_reason"] is None:
            STATE["stop_reason"] = "stopped by user"
        log(kind="end", reason=STATE["stop_reason"], chunks=stats["chunks"], handed=stats["handed"], dry_ticks=stats["dry_ticks"],
            lat_p50=float(np.median(stats["lat_s"])) if stats["lat_s"] else None)
        print(f"[v4-smoke] PLAN loop ended: {STATE['stop_reason']} after {stats['chunks']} chunks, {stats['handed']} rows, "
              f"dry ticks {stats['dry_ticks']}", flush=True)


def _stream_loop():
    """Executor: one row per DT_EXEC from the merged timeline, IK from the measured joints, no arrival wait."""
    STATE["cycles"] = 0; STATE["travel_mm"] = [0.0, 0.0]; STATE["stop_reason"] = None
    _jaw_torque(True)
    TL = {"rows": [], "lock": threading.Lock()}
    stats = {"chunks": 0, "sent": 0, "dry": 0, "obs_fail": 0, "lat_s": collections.deque(maxlen=50)}
    STATE["stream"] = stats
    threading.Thread(target=_stream_infer_worker, args=(TL, stats), daemon=True).start()
    dt = UMI_DT / STREAM_SPEED; nxt = time.time(); last_cid = None
    while STATE["running"]:
        now = time.time()
        hs = _hra_stop()
        if hs:
            STATE["stop_reason"] = hs; break
        with TL["lock"]:
            if not STREAM_SYNC:          # sync chunks start at the current pose: a late row is still executed, in order
                TL["rows"] = [r for r in TL["rows"] if r[0] >= now - 0.5 * dt]
            row = TL["rows"].pop(0) if TL["rows"] and TL["rows"][0][0] <= now + 0.5 * dt else None
            if row is not None and STREAM_PAUSE_S > 0 and not STREAM_SYNC and last_cid is not None and row[4] != last_cid:
                # chunk boundary: put the row back, shift everything pending by the pause, hold
                TL["rows"].insert(0, row)
                TL["rows"] = [(r[0] + STREAM_PAUSE_S,) + tuple(r[1:]) for r in TL["rows"]]
                last_cid = row[4]; row = "PAUSE"
        if row == "PAUSE":
            stats["pauses"] = stats.get("pauses", 0) + 1
            time.sleep(STREAM_PAUSE_S); nxt = time.time(); continue
        if row is not None:
            last_cid = row[4]
        if row is None:
            if stats["chunks"] > 0:
                stats["dry"] += 1
        else:
            tau, P, Q, W, cid, j = row
            q14 = _joints14_fast()
            if q14 is None:
                stats["obs_fail"] += 1
            else:
                try:
                    a, ikerr, ok, clipped = INF.solve_waypoint(q14, P, Q, W if INF.last.get("gripper_mode") in ("predict", "binary", "continuous", "relative") else None)
                except Exception as e:  # noqa: BLE001
                    STATE["stop_reason"] = f"IK failed: {e}"; break
                if getattr(INF, "_lik_stop", None):
                    STATE["stop_reason"] = f"IK stop: {INF._lik_stop}"; break
                r = _exec_step(a)
                if isinstance(r, dict) and r.get("error"):
                    STATE["stop_reason"] = f"execute_step: {r['error']}"; break
                stats["sent"] += 1; STATE["cycles"] = stats["sent"]; TL["last_sent_t"] = time.time()
                if STREAM_WP_PAUSE_S > 0:
                    with TL["lock"]:
                        TL["rows"] = [(r[0] + STREAM_WP_PAUSE_S,) + tuple(r[1:]) for r in TL["rows"]]
                    nxt += STREAM_WP_PAUSE_S
                try:
                    m_now, _ = INF.tcp_now(q14)
                    with open(CYCLE_LOG, "a") as f:
                        f.write(json.dumps(dict(t=round(now, 3), cycle=stats["sent"], stream_chunk=cid, stream_row=j + 1, late_ms=round((now - tau) * 1000, 1),
                                                ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")), exec_k=j + 1, stream=True,
                                                tcp_now_mm=[[round(float(x) * 1000, 1) for x in m_now[r_][:3, 3]] for r_ in (0, 1)],
                                                tcp_target_mm=[[round(float(x) * 1000, 1) for x in P[r_]] for r_ in (0, 1)], tcp_after_mm=None,
                                                ik_residual_mm=round(float(ikerr), 2), ik_ok=bool(ok), reached=None, yaw=getattr(INF, "_yaw_diag", None),
                                                q_now=[round(float(x), 4) for x in np.asarray(q14)[ARM_IDX]],
                                                # [2026-10-02 user: right jaw opens too wide] model g of this row (0 closed..1 open), the jaw command
                                                # sent (UI cmd units, 45 = full open), and the measured jaw raw deg (bridge-clipped to [-270, 0])
                                                g_row=None if W is None else [round(float(x), 3) for x in np.asarray(W).ravel()[:2]],
                                                grip_cmd_sent=[round(float(a[g_]), 1) for g_ in GRIP_IDX],
                                                jaw_raw_now=[round(float(q14[g_]), 1) for g_ in GRIP_IDX])) + "\n")
                except Exception:
                    pass
        nxt += dt; sl = nxt - time.time()
        if sl > 0:
            time.sleep(sl)
        else:
            nxt = time.time()
    STATE["running"] = False
    if STATE["stop_reason"] is None:
        STATE["stop_reason"] = "stopped by user"
    lat = list(stats["lat_s"])
    print(f"[v4-stream] ended: {STATE['stop_reason']} | chunks {stats['chunks']} sent {stats['sent']} dry {stats['dry']} "
          f"obs_fail {stats['obs_fail']} latency med {np.median(lat) if lat else float('nan'):.2f}s", flush=True)


@app.route("/run", methods=["POST"])
def run():
    if STATE["running"]:
        return jsonify({"error": "already running"})
    g = _grip_gate()
    if g:
        return jsonify(g)
    # [2026-09-29] RELCART20 anchor guard: a /run never silently re-anchors. First run after start/reset sets it from its first
    # observation; afterwards /run is refused until RESET ANCHOR is pressed (so rollouts from different start poses cannot mix).
    import infer_core_v4 as _IC
    if _IC.STATE_MODE == "relcart20" and getattr(INF, "_anchor", None) is not None and request.args.get("reuse_anchor") != "1":
        a = INF._anchor
        return jsonify({"error": "ANCHOR_ALREADY_SET -- press RESET ANCHOR (after GO TO TRAIN START) for a new episode, or /run?reuse_anchor=1",
                        "anchor_mm": {"L": [round(float(x) * 1000, 1) for x in a[0][:3, 3]], "R": [round(float(x) * 1000, 1) for x in a[1][:3, 3]]}})
    if _arm_only():
        e = _hra_latch()
        if e:
            return jsonify({"error": f"HRA latch failed: {e}"})
    STATE["running"] = True
    INF._g_prev = [None, None]; INF._g_cmd = [None, None]      # [2026-09-29] relative gripper: every /run starts from the measured jaw
    INF._te_hist = None                                         # [2026-10-01] temporal ensemble: every /run starts fresh
    INF._rtc_prev = None                                        # [2026-10-03] RTC: no inpainting prefix across /runs
    if _PF["thread"] is not None:                               # [2026-10-02] prefetch from a previous run is never used
        _PF["thread"].join(timeout=10.0)
    _PF["thread"] = None; _PF["res"] = None
    threading.Thread(target=(_plan_loop if PLAN_MODE else (_stream_loop if STREAM else _loop)), daemon=True).start()   # [10-06] PLAN mode
    return jsonify({"running": True, "mode": "stream" if STREAM else "step", "stream_speed": STREAM_SPEED if STREAM else None,
                    "max_cycles": MAX_CYCLES, "max_travel_mm": MAX_TRAVEL_MM})


@app.route("/goto_train_start", methods=["POST"])
def goto_train_start():
    """[2026-09-29] Evaluation initialisation only: drive both arms (slew-limited _goto) to the recorded training start pose
    closest to the median of the 180 episode starts (train_start_pose.json), jaws CLOSED as at those starts. Then /run."""
    if STATE["running"] or not _lock.acquire(blocking=False):
        return jsonify({"error": "busy / running"})
    try:
        import json as _j
        P = _j.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_start_pose.json")))
        tgt = np.zeros(14); tgt[ARM_IDX] = P["q12_follower_rad"]; tgt[FLIP_IDX] *= -1.0
        for g_ in GRIP_IDX:
            tgt[g_] = 0.0                                   # CLOSED, as at the training starts
        r = _goto(list(tgt))
        o = _observe(); m, _ = INF.tcp_now(o["joints14"]) if o else (None, None)
        now = None if m is None else {"L": [round(float(x) * 1000, 1) for x in m[0][:3, 3]], "R": [round(float(x) * 1000, 1) for x in m[1][:3, 3]]}
        return jsonify({"goto": r, "target_tcp_mm": P["tcp_mm"], "now_tcp_mm": now, "source": P["source"],
                        "next": "wait until still, then RESET ANCHOR (relcart20) and RUN"})
    finally:
        _lock.release()


@app.route("/hra_start_pose", methods=["POST"])
def hra_start_pose():
    """[2026-10-06 user] HRA start pose: tilt the RIGHT gripper so its approach axis points PITCH deg below horizontal (the 166 human
    training starts: p10/p50/p90 = 32/36/39 deg, roll ~ -3 deg), keeping the TCP position (+ optional dz) and the approach heading.
    Closing axis made horizontal (roll 0). Left arm + jaws hold. Walked in STEPS small IK waypoints from the measured pose; stops on an
    IK failure. Then RUN (the next observation becomes the RELCART20 anchor). Query: pitch (deg, default 35), dz_mm (0), steps (7)."""
    if STATE["running"] or not _lock.acquire(blocking=False):
        return jsonify({"error": "busy / running"})
    try:
        pitch = float(request.args.get("pitch", 35.0)); dz = float(request.args.get("dz_mm", 0.0)) / 1000.0
        steps = max(1, min(int(request.args.get("steps", 7)), 30))
        if not (0.0 <= pitch <= 60.0) or abs(dz) > 0.15:
            return jsonify({"error": "pitch must be 0..60 deg, |dz_mm| <= 150"})
        o = _observe()
        if o is None:
            return jsonify({"error": "observe failed"})
        m0, _ = INF.tcp_now(o["joints14"]); R0 = m0[1][:3, :3]; p0 = m0[1][:3, 3].copy()
        up = np.array([0.0, 0.0, 1.0])                                   # base z = up (assumes a level mount)
        h = R0[:, 0] - (R0[:, 0] @ up) * up
        if np.linalg.norm(h) < 1e-3:
            return jsonify({"error": "gripper points straight up/down: heading undefined"})
        h /= np.linalg.norm(h); pr = np.radians(pitch)
        a = np.cos(pr) * h - np.sin(pr) * up                              # approach axis, PITCH below horizontal
        y = np.cross(up, a); y /= np.linalg.norm(y)
        if y @ R0[:, 1] < 0: y = -y                                       # keep the jaw-closing side
        rr = np.radians(float(request.args.get("roll", 0.0)))           # roll about the approach axis (human median -3 deg)
        if rr:
            y = np.cos(rr) * y - np.sin(rr) * np.cross(a, y)
        Rt = np.stack([a, y, np.cross(a, y)], 1); pt = p0 + dz * up
        if request.args.get("saved") == "1":          # [10-06] the pose saved as "robot = HandUMI start" (umi_start_pose.json, offset-free frame)
            import json as _j, infer_core_v4 as _IC
            _nm = os.path.basename(request.args.get("pose", ""))      # [2026-10-07 user "새 시작점 추가"] choose from ~/c8/hra_red/start_poses/
            _pf = os.path.expanduser(f"~/c8/hra_red/start_poses/{_nm}") if _nm else os.path.expanduser("~/c8/hra_red/umi_start_pose.json")
            if not os.path.exists(_pf):
                return jsonify({"error": f"no start pose file {_pf}"})
            Ts = np.array(_j.load(open(_pf))["T_base_tcp"])
            # [2026-10-07 user "좀더 오른쪽 그리고 아래"] offsets in the ROBOT WRIST CAMERA view (hand-eye X_tcp_cam): right = camera x,
            # back = -optical axis, both projected on the horizontal plane; down = base -z. Orientation unchanged.
            _r, _b, _d = (float(request.args.get(k_, 0.0)) / 1000.0 for k_ in ("right_mm", "back_mm", "down_mm"))
            if _r or _b or _d:
                _X = np.array(_j.load(open(os.path.expanduser("~/c8/hra_red/handeye/robot_right_wrist_handeye.json")))["X_tcp_cam"]["park"])
                _C = Ts @ _X; _up = np.array([0.0, 0.0, 1.0])
                _ex = _C[:3, 0] - (_C[:3, 0] @ _up) * _up; _ex /= np.linalg.norm(_ex)
                _ez = _C[:3, 2] - (_C[:3, 2] @ _up) * _up; _ez /= np.linalg.norm(_ez)
                Ts = Ts.copy(); Ts[:3, 3] += _r * _ex - _b * _ez - _d * _up
            Rt = Ts[:3, :3] @ (_IC.R_UMI_OFF if _IC.UMI_CONV else np.eye(3))
            pt = Ts[:3, 3] + (Ts[:3, :3] @ _IC.UMI_TCP_OFF_T if _IC.UMI_CONV else 0.0)   # [10-06] same HandUMI convention as tcp_now
            if np.linalg.norm(pt - p0) > 0.35:     # [2026-10-07] 250 -> 350 mm: the HRA_A100 human-median start is 294 mm from rest
                return jsonify({"error": f"saved start {np.round(pt*1000,1)} mm is > 350 mm from the current TCP -- refused"})
        if all(k_ in request.args for k_ in ("x_mm", "y_mm", "z_mm")):   # [10-06] absolute base target (the recorded run-start position)
            pt = np.array([float(request.args[k_]) for k_ in ("x_mm", "y_mm", "z_mm")]) / 1000.0
            if np.linalg.norm(pt - p0) > 0.25:
                return jsonify({"error": f"target {np.round(pt*1000,1)} mm is > 250 mm from the current TCP -- refused"})
        Rl = m0[0][:3, :3]; pl = m0[0][:3, 3]
        rots = Rot.from_matrix(np.stack([R0, Rt]))
        from scipy.spatial.transform import Slerp
        sl = Slerp([0, 1], rots); log = []
        # [2026-10-07 user "끊겨서 가거든 한번에 가도록"] solve EVERY waypoint first (sequential IK seeded with the previous solution),
        # then stream the whole joint path at CMD_HZ without stopping, and dwell (reach check) only at the final waypoint.
        # The old loop called _goto per waypoint = slew + dwell-until-reached each time -> stop-and-go.
        o0 = _observe()
        if o0 is None:
            return jsonify({"error": "observe failed"})
        q_seed = list(o0["joints14"]); path = []
        for i in range(1, steps + 1):
            f = i / steps; Ri = sl([f]).as_matrix()[0]; pi = p0 + f * (pt - p0)
            P = np.stack([pl, pi]); Q = np.stack([Rot.from_matrix(Rl).as_quat(), Rot.from_matrix(Ri).as_quat()])
            act, ikerr, ok, clipped = INF.solve_waypoint(q_seed, P, Q, None)
            ok_r = bool(np.asarray(ok).reshape(-1, 2)[0][1]) if np.asarray(ok).size >= 2 else bool(np.all(ok))
            if not ok_r:
                return jsonify({"error": f"IK failed at waypoint {i}/{steps} -- robot NOT moved", "steps": log})
            act = list(_mask_arm(np.asarray(act)[None, :], "right", o0["joints14"])[0])
            path.append((act, P, Q, ikerr))
            for j_ in range(7, 13):                                       # next seed = this solution in the /observe frame
                q_seed[j_] = -act[j_] if j_ in FLIP_IDX else act[j_]
        cur = _leader_now()
        if cur is None:
            return jsonify({"error": "observe failed"})
        prev = list(cur); sent = 0
        for wi, (act, P, Q, ikerr) in enumerate(path):
            n = max(1, int(math.ceil(max(abs(act[j_] - prev[j_]) for j_ in ARM_IDX) / SLEW_MAX_STEP_RAD)))
            for k in range(1, n + 1):
                if STATE["stop_reason"] == "estop":
                    return jsonify({"error": "estop", "steps": log})
                a = [prev[j_] + (act[j_] - prev[j_]) * k / n for j_ in range(14)]
                for _g in GRIP_IDX:
                    a[_g] = act[_g]
                r = _exec_step(a, k, n)
                if isinstance(r, dict) and r.get("error"):
                    return jsonify({"error": r["error"], "steps": log})
                sent += 1; time.sleep(1.0 / CMD_HZ)
            prev = list(act)
        act, P, Q, ikerr = path[-1]
        r = _goto(list(act), (P, Q), ikerr)
        log.append({"waypoints": steps, "streamed_cmds": sent, "reached": r.get("reached"), "tcp_err_mm": r.get("tcp_err_mm")})
        o = _observe(); m1, _ = INF.tcp_now(o["joints14"]); R1 = m1[1][:3, :3]
        got = float(np.degrees(np.arcsin(np.clip(-R1[:, 0] @ up, -1, 1))))
        roll = float(np.degrees(np.arcsin(np.clip(R1[:, 1] @ up, -1, 1))))
        # [10-06 user] no implicit re-anchor here: the episode starts with RESET ANCHOR -> RUN loop, as before
        return jsonify({"ok": True, "pitch_target_deg": pitch, "pitch_now_deg": round(got, 1), "roll_now_deg": round(roll, 1),
                        "tcp_mm_before": [round(float(x) * 1000, 1) for x in p0], "tcp_mm_now": [round(float(x) * 1000, 1) for x in m1[1][:3, 3]],
                        "steps": log, "next": "place the cube ~5 cm ahead / 15-20 cm inward of the gripper, then RUN"})
    finally:
        _lock.release()


@app.route("/wrist_zoom", methods=["GET", "POST"])
def wrist_zoom_route():
    """[2026-10-07 user "feed 영상을 확대"] right-wrist focal zoom s (f' = s f about the C922 principal point), applied to the model input only."""
    import infer_core_v4 as _IC
    if "s" in request.args:
        v = float(request.args["s"])
        if not (0.7 <= v <= 3.0):
            return jsonify({"error": "s must be 0.7 .. 3.0"})
        _IC.WRIST_ZOOM["s"] = v
        try:
            open(os.path.expanduser(f"~/umi_bridge/.auto_ui_d20_e5/{os.environ.get('V4_UI_PORT', '8056')}.wristzoom"), "w").write(f"{v}\n")
        except Exception:
            pass
    return jsonify({"wrist_zoom": _IC.WRIST_ZOOM["s"]})


@app.route("/start_poses")
def start_poses():
    """[2026-10-07] saved start poses (~/c8/hra_red/start_poses/*.json) for the UMI start button."""
    import json as _j, glob as _g
    out = []
    for f in sorted(_g.glob(os.path.expanduser("~/c8/hra_red/start_poses/*.json"))):
        try:
            T = np.array(_j.load(open(f))["T_base_tcp"]); out.append({"name": os.path.basename(f), "tcp_mm": [round(float(x) * 1000, 1) for x in T[:3, 3]]})
        except Exception:
            pass
    return jsonify({"poses": out})


@app.route("/cube_stop", methods=["GET", "POST"])
def cube_stop_route():
    """[2026-10-07] cube-size stop: ?px=<sqrt(area) px in the raw 640x480 right-wrist frame> (0 = off); GET returns the live size."""
    if "px" in request.args:
        CUBE_STOP["px"] = max(0.0, float(request.args["px"])); CUBE_STOP["hits"] = 0
        try:
            open(os.path.expanduser(f"~/umi_bridge/.auto_ui_d20_e5/{os.environ.get('V4_UI_PORT', '8056')}.cubestop"), "w").write(f"{CUBE_STOP['px']}\n")
        except Exception:
            pass
    try:
        import cv2
        r = requests.get(ROBOT + "/frame/right", timeout=2.0)
        now = round(cube_px(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)), 1)
    except Exception as e:
        now = None
    if "follow" in request.args:
        CUBE_STOP["follow"] = request.args["follow"] == "1"
        if not CUBE_STOP["follow"]:
            CUBE_STOP["waiting"] = False
    return jsonify({"stop_px": CUBE_STOP["px"], "cube_px_now": now, "follow": CUBE_STOP["follow"], "waiting": CUBE_STOP["waiting"],
                    "resume_below_px": round(CUBE_STOP["resume_frac"] * CUBE_STOP["px"], 1), "resumes": CUBE_STOP["resumes"]})


@app.route("/wrist_snap", methods=["POST"])
def wrist_snap():
    """[2026-10-07] save the raw right-wrist frame + the zoomed model input + the right TCP (measure cube size: train end vs robot stop)."""
    import infer_core_v4 as _IC, json as _j, cv2
    o = _observe()
    if o is None or "right" not in o["images"]:
        return jsonify({"error": "observe failed / no right image"})
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(o["images"]["right"]), np.uint8), cv2.IMREAD_COLOR)
    z = cv2.resize(_IC.wrist_zoom(bgr, _IC.WRIST_ZOOM["s"]), (224, 224), interpolation=cv2.INTER_AREA)
    d = os.path.expanduser("~/c8/hra_a100/wrist_snaps"); os.makedirs(d, exist_ok=True); t = time.strftime("%H%M%S")
    cv2.imwrite(f"{d}/{t}_raw.png", bgr); cv2.imwrite(f"{d}/{t}_in224_s{_IC.WRIST_ZOOM['s']:.2f}.png", z)
    m, _ = INF.tcp_now(o["joints14"])
    _j.dump({"t": t, "zoom": _IC.WRIST_ZOOM["s"], "T_base_tcp_right": m[1].tolist()}, open(f"{d}/{t}.json", "w"))
    ok, e = cv2.imencode(".jpg", z)
    return jsonify({"saved": f"{d}/{t}_raw.png", "zoom": _IC.WRIST_ZOOM["s"], "tcp_mm": [round(float(x) * 1000, 1) for x in m[1][:3, 3]],
                    "img": base64.b64encode(e.tobytes()).decode()})


@app.route("/reset_anchor", methods=["POST"])
def reset_anchor():
    INF.reset_anchor(); INF._rtc_prev = None
    return jsonify({"anchor": "reset; the next observation becomes the RELCART20 anchor"})


@app.route("/stop", methods=["POST"])
def stop():
    STATE["running"] = False; CUBE_STOP["waiting"] = False      # [2026-10-07] a user STOP also ends cube-follow
    STATE["stop_reason"] = "stopped by user"
    return jsonify({"running": False})


@app.route("/status")
def status():
    import infer_core_v4 as _IC
    s = _robot("/status")
    hra = None
    if _IC.ARM_ONLY:
        h = STATE.get("hra") or {}
        hra = dict(gate=HRA_GATE, mode="HRA RIGHT-ONLY", task=INF.task if INF else None, left="LOCKED", jaws="HOLD" if _IC.ARM_ONLY_GRIP == "hold" else "RIGHT MODEL",
                   max_run_s=HRA_MAX_RUN_S, still_mm=HRA_STILL_MM, still_n=HRA_STILL_N, gate_refusals=h.get("gate_refusals", 0),
                   max_left_dev_rad=round(h.get("max_left_dev", 0.0), 5), last_pred_mm=h.get("last_pred_mm"), still=h.get("still"),
                   travel_mm=round(h.get("travel_mm", 0.0), 1), max_cycle_mm=round(h.get("max_cycle_mm", 0.0), 1),
                   max_travel_cap=HRA_MAX_TRAVEL_MM, max_cycle_cap=HRA_MAX_CYCLE_MM,
                   ik=f"{_IC.IK_BACKEND} lock={','.join(_IC.PINK_LOCK) or 'none'} ori={_IC.PINK_ORI} profile={os.environ.get('V4_HRA_IK_PROFILE', '?')}")
    return jsonify({"hra": hra, "robot": s, "ui": {k: STATE[k] for k in ("busy", "presses", "arm", "err", "running", "cycles",
                                                 "travel_mm", "stop_reason", "task")},
                    "last": STATE["last"], "ab": STATE.get("ab"), "observe_retries": dict(_OBS_FAILS),
                    "profile": STATE.get("profile"), "denoise": STATE.get("denoise"),
                    "replay": STATE.get("replay"), "sent": STATE.get("sent"),
                    "model_loaded": INF is not None,
                    # which weights and which decode are live. Two action semantics now exist (delta and
                    # UMI current-anchor) and the same chunk commands different targets under each, so this
                    # is not cosmetic -- it is how you check the robot is driven by what you think it is.
                    "model": {"ckpt": os.path.basename(_IC.CKPT.rstrip("/")),
                              "action_mode": _IC.ACTION_MODE, "clamp": dict(_IC.LIMITS),
                              "gripper": _IC.GRIPPER_MODE,
                              "n_action": getattr(INF, "n_action", _IC.N_ACTION),
                              "exec_k": _IC.EXEC_K, "dtype": _IC.DTYPE,
                              "temporal_ensemble": {"w": _IC.TE_W.tolist(), "rot": _IC.TE_ROT, "grip": _IC.TE_GRIP} if _IC.TE_ON else False,
                              "adaptive_k": None if _IC.ADAPTIVE_K is None else {"k": _IC.ADAPTIVE_K, "far_mm": _IC.AK_FAR_MM, "near_mm": _IC.AK_NEAR_MM},
                              "deadband": {"xy_mm": _IC.DB_XY_MM, "z_mm": _IC.DB_Z_MM, "ref_k": _IC.DB_REF_K, "axis_mm": None if _IC.DEADBAND_MM is None else _IC.DEADBAND_MM.tolist()},
                              "global_rot180": _IC.GLOBAL_ROT180,
                              "global_mirror": _IC.GLOBAL_MIRROR,
                              "state_mode": _IC.STATE_MODE,
                              "state_width": {"umi76": 76, "umi94": 94, "cart20": 20, "relcart20": 20}.get(_IC.STATE_MODE, 20),
                              "history_dt_ms": round(_IC.UMI_HISTORY_DT * 1000, 2) if _IC.STATE_MODE in ("umi76", "umi94", "cart20", "relcart20") else None,
                              "chunk": getattr(INF, "chunk", None),
                              "ctl": {"k": getattr(INF, "n_action", None), "stream": STREAM, "speed_pct": round(STREAM_SPEED * 100, 1),
                                      "stream_k": STREAM_K, "prefetch": PREFETCH, "prefetch_lead": PREFETCH_LEAD, "plan": PLAN_MODE, "plan_ema": PLAN_EMA, "plan_xfade": PLAN_XFADE, "plan_min_exec": PLAN_MIN_EXEC, "plan_smooth_deg": PLAN_SMOOTH_DEG, "plan_stats": {k_: (list(v_) if isinstance(v_, collections.deque) else v_) for k_, v_ in (STATE.get("plan") or {}).items()}, "prefetch_auto": {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in _PFT.items()}, "pass_last": PASS_LAST, "pass": PASS_THROUGH, "dwell_s": DWELL_TIMEOUT_S, "rtc": _IC.RTC_ON, "rtc_last": (getattr(INF, "last", {}) or {}).get("rtc"), "ik": _IC.IK_BACKEND, "remote": bool(_IC.REMOTE_INFER), "remote_ckpt": _IC.REMOTE_CKPT,
                                      "remote_last": dict(_IC.REMOTE_LAST), "goal_interp": (s or {}).get("goal_interp") if isinstance(s, dict) else None}}})


@app.route("/connect", methods=["POST"])
def connect():
    return jsonify(_robot("/connect", "post", json={}))


@app.route("/estop", methods=["POST"])
def estop():
    STATE["busy"] = False
    STATE["running"] = False
    STATE["stop_reason"] = "estop"
    return jsonify(_robot("/estop", "post", json={}))


@app.route("/clear_estop", methods=["POST"])
def clear_estop():
    return jsonify(_robot("/clear_estop", "post", json={}))


@app.route("/reenable", methods=["POST"])
def reenable():
    return jsonify(_robot("/reenable", "post", json={}))


@app.route("/gripper_zero/<arm>/<stage>", methods=["POST"])
def gripper_zero(arm, stage):
    """[2026-09-30 user] re-zero one jaw from the page (the left zero re-latches after every left-arm power drop).
    stage=release drops that jaw's torque -> close the jaw fully BY HAND -> stage=set stores zero there (robot_service
    /gripper_zero). Refused while a loop runs. After set, the page shows both raw jaw readings (closed should be ~0)."""
    if arm not in ("left", "right") or stage not in ("release", "set"):
        return jsonify({"error": f"bad arm/stage {arm}/{stage}"})
    if STATE["running"] or STATE.get("busy"):
        return jsonify({"error": "a loop is running -- STOP first"})
    r = _robot("/gripper_zero", "post", json={"arm": arm, "stage": stage})
    if stage == "set":
        ob = _observe()
        if ob and "joints14" in ob:
            r = {**(r if isinstance(r, dict) else {"result": r}), "raw_now_LR": [round(float(ob["joints14"][g]), 2) for g in GRIP_IDX]}
    return jsonify(r)


@app.route("/gripper_probe/<cmd>", methods=["POST"])
def gripper_probe(cmd):
    """Measure today's gripper command->observation scale. Arms do not move: the whole leader vector is the
    current pose and only the jaw element changes. The raw zero re-latches per power cycle, so this is a
    measurement of THIS session, not a constant."""
    lead = _leader_now()
    if lead is None:
        return jsonify({"error": "observe failed"})
    before = _observe()["joints14"]
    out = list(lead)
    for gi in GRIP_IDX:
        out[gi] = float(cmd)
    r = _exec_step(out)
    if isinstance(r, dict) and r.get("error"):
        return jsonify(r)
    time.sleep(1.5)                                   # hold to a settled readback; mid-travel reads lie
    after = _observe()["joints14"]
    obs_b = [round(float(before[g]), 2) for g in GRIP_IDX]
    obs_a = [round(float(after[g]), 2) for g in GRIP_IDX]
    ratio = [round(obs_a[i] / float(cmd), 3) if float(cmd) else None for i in range(2)]
    return jsonify({"cmd": float(cmd), "obs_before": obs_b, "obs_after": obs_a, "obs_per_cmd": ratio})


@app.route("/arm_ident", methods=["POST"])
def arm_ident():
    """Which wrist feed belongs to which arm, decided by moving one jaw and watching.

    Two different things can look like "left and right are swapped": the camera indices can come up in a
    different order after a service restart, or the arm-to-USB-location mapping can be wrong. Guessing
    between them from the pictures is how the wrong one gets "fixed". So: close ONE gripper, see which feed
    changes and which jaw element of the state vector moves, and put the jaw back.

    No arm joint moves -- the leader vector sent is the current pose with a single jaw element replaced,
    which is the same primitive /gripper_probe uses.
    """
    import cv2 as _cv2
    which = (request.get_json(silent=True) or {}).get("arm", "left")
    gi = GRIP_IDX[0 if which == "left" else 1]
    lead = _leader_now()
    if lead is None:
        return jsonify({"error": "observe failed"})

    def _grab():
        out = {}
        for cam in ("left", "right", "middle"):
            try:
                r = requests.get(f"{ROBOT}/frame/{cam}", timeout=10)
                a = np.frombuffer(r.content, np.uint8)
                out[cam] = _cv2.imdecode(a, _cv2.IMREAD_GRAYSCALE)
            except Exception:
                out[cam] = None
        return out

    before_img = _grab()
    before_q = _observe()["joints14"]
    # the jaw element of an execute_step vector is a COMMAND in 0..45, not an observation in 0..-270
    # (obs = cmd * -6). Sending -120 here is outside the command range, gets clipped, and the jaw never
    # moves -- which reads as "the probe found nothing" rather than "the probe was wrong".
    target = 20.0 if abs(float(before_q[gi])) < 60 else 0.0
    out = list(lead)
    out[gi] = target
    r = _exec_step(out)
    if isinstance(r, dict) and r.get("error"):
        return jsonify(r)
    time.sleep(2.0)
    after_img = _grab()
    after_q = _observe()["joints14"]

    out[gi] = float(lead[gi])                                      # put the jaw back where it was
    _exec_step(out)

    diff = {}
    for cam in ("left", "right", "middle"):
        a, b = before_img.get(cam), after_img.get(cam)
        diff[cam] = None if a is None or b is None or a.shape != b.shape else \
            round(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()), 3)
    jaw = {"left": round(float(after_q[GRIP_IDX[0]]) - float(before_q[GRIP_IDX[0]]), 2),
           "right": round(float(after_q[GRIP_IDX[1]]) - float(before_q[GRIP_IDX[1]]), 2)}
    moved_cam = max((c for c in ("left", "right") if diff[c] is not None), key=lambda c: diff[c], default=None)
    moved_jaw = max(jaw, key=lambda k: abs(jaw[k]))
    return jsonify({"commanded_arm": which, "jaw_delta_raw": jaw, "frame_mean_abs_diff": diff,
                    "feed_that_moved": moved_cam, "jaw_that_moved": moved_jaw,
                    "cameras_swapped": moved_cam is not None and moved_cam != which,
                    "state_jaw_swapped": moved_jaw != which})


@app.route("/zlog")
def zlog():
    """The vertical decomposition, newest last. `n` rows, or `clear=1` to start a fresh window."""
    if request.args.get("clear"):
        STATE["zlog"].clear()
        return jsonify({"cleared": True})
    n = int(request.args.get("n", "40"))
    rows = list(STATE["zlog"])[-n:]
    out = {"rows": rows}
    for arm in ("L", "R"):
        v = [r for r in rows if r["arm"] == arm]
        if v:
            import statistics as st
            a1 = [r["a1_body_dz"] for r in v]
            tg = [r["tgt_dz"] for r in v]
            ac = [r["act_dz"] for r in v if r["act_dz"] is not None]
            out[arm] = {"n": len(v),
                        "a1_body_dz_mean": round(st.fmean(a1), 2),
                        "a1_body_dz_frac_up": round(sum(x > 0 for x in a1) / len(a1), 2),
                        "tgt_dz_mean": round(st.fmean(tg), 2),
                        "tgt_dz_frac_up": round(sum(x > 0 for x in tg) / len(tg), 2),
                        "act_dz_mean": round(st.fmean(ac), 2) if ac else None,
                        "tcp_z_first": v[0]["tcp_z"], "tcp_z_last": v[-1]["tcp_z"],
                        "tcp_z_drift": round(v[-1]["tcp_z"] - v[0]["tcp_z"], 1)}
    return jsonify(out)


@app.route("/routingline", methods=["POST"])
def routingline():
    """No motion. One inference, one line: which prediction slot is large, and which physical arm it drives.

    The chain printed here is the whole L/R question end to end --

        pred[0:10] / pred[10:20]   the model's two arm slots, raw
        -> IK target index 0 / 1
        -> joint block [0..5] / [6..11]
        -> 14-vector indices [0,1,2,3,4,5] / [7,8,9,10,11,12]
        -> physical LEFT / RIGHT arm

    The last hop is the one no code inspection can settle, so it was measured: /jog/0/8 moves the LEFT wrist
    feed 17.7x more than the right, /jog/7/8 moves the RIGHT feed 9.3x more. Those ratios are quoted in the
    response so the line is self-contained.
    """
    K = int(request.args.get("k", "6"))
    prompt = request.args.get("prompt", "RBP")
    text = dict(TASKS).get(prompt)
    if text is None:
        return jsonify({"error": f"unknown prompt {prompt}"})
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    keep = INF.task
    INF.task = text
    slots, dq_blocks = [], []
    try:
        for j in range(K):
            torch.manual_seed(1234 + 17 * j)
            INF.infer([h0, h1])
            L = INF.last
            k = max(0, int(L["valid_step"]) - 1)
            slots.append([float(np.linalg.norm((L["tgt_pos"][k, r] - L["cur_mat"][r][:3, 3]) * 1000.0))
                          for r in (0, 1)])
            # q_cmd is (T, 12): timesteps x joints. Slice the JOINT axis -- d[:6] would compare the first
            # six timesteps against the rest and read out as a left/right difference that is not there.
            d = np.degrees(np.abs(np.atleast_2d(L["q_cmd"]) - L["q_now"][ARM_IDX]))
            dq_blocks.append([float(d[:, :6].max()), float(d[:, 6:].max())])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        INF.task = keep
    sl = np.asarray(slots).mean(0)
    dq = np.asarray(dq_blocks).mean(0)
    dom = 0 if sl[0] > sl[1] else 1
    return jsonify({
        "prompt": prompt, "draws": K,
        "line": (f"pred slot0(LEFT)={sl[0]:.1f} mm, slot1(RIGHT)={sl[1]:.1f} mm "
                 f"-> dominant slot {dom} -> joint block {'[0..5]' if dom == 0 else '[6..11]'} "
                 f"-> 14-vector {ARM_IDX[:6] if dom == 0 else ARM_IDX[6:]} "
                 f"-> physical {'LEFT' if dom == 0 else 'RIGHT'} arm"),
        "pred_slot_mm": {"slot0_LEFT": round(float(sl[0]), 1), "slot1_RIGHT": round(float(sl[1]), 1)},
        "commanded_joint_delta_deg": {"left_block": round(float(dq[0]), 3), "right_block": round(float(dq[1]), 3)},
        "physical_verification": {"jog_idx0_wrist_ratio": "17.74x LEFT", "jog_idx7_wrist_ratio": "9.26x RIGHT"},
        "global_rot180": bool(__import__("infer_core_v4").GLOBAL_ROT180)})


@app.route("/wristswap", methods=["POST"])
def wristswap():
    """No motion. Does swapping the two wrist feeds flip which arm the policy drives?

    The global feed has been shown to locate the requested colour correctly, so the open question is the
    next link: target position -> left/right arm. If the live wrist feeds are crossed relative to training,
    the policy can read the right target and still commit to the wrong hand, which is exactly the shape of
    "the opposite arm keeps moving".

    One frozen observation, one seed sequence, global and state untouched; only the two wrist images trade
    places. Arm flips with the swap -> wrist assignment is the bug. Arm stays -> the wrists are not the
    cause and the next suspect is the state/action channel order.
    """
    K = int(request.args.get("k", "10"))
    prompt = request.args.get("prompt", "RBP")
    text = dict(TASKS).get(prompt)
    if text is None:
        return jsonify({"error": f"unknown prompt {prompt}"})
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    im = h1["images"]
    variants = {"normal": im,
                "wrist_swapped": {**im, "left": im["right"], "right": im["left"]}}
    keep_task = INF.task
    INF.task = text
    out, horizon = {}, None
    try:
        for name, imgs in variants.items():
            hh = [h0, {**h1, "images": imgs}]
            keeps = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)
                INF.infer(hh)
                L = INF.last
                keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_mat": L["cur_mat"].copy()})
                v = int(L["valid_step"]) - 1
                horizon = v if horizon is None else min(horizon, v)
            out[name] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        INF.task = keep_task
    KS = [k for k in (4, 8, 16, 31) if k <= horizon]
    rows = []
    for name, keeps in out.items():
        D = np.stack([np.stack([[np.linalg.norm((L["tgt_pos"][k, r] - L["cur_mat"][r][:3, 3]) * 1000.0)
                                 for r in (0, 1)] for k in KS]) for L in keeps])      # (K, nk, 2)
        m = D.mean(0)
        rows.append({"variant": name,
                     "per_k": [{"k": k, "L_mm": round(float(m[i][0]), 1), "R_mm": round(float(m[i][1]), 1),
                                "acting": "LEFT" if m[i][0] > m[i][1] else "RIGHT"} for i, k in enumerate(KS)]})
    return jsonify({"prompt": prompt, "draws": K, "horizon": horizon, "rows": rows})


@app.route("/jog/<int:idx>/<deg>", methods=["POST"])
def jog(idx, deg):
    """Move ONE arm joint by a few degrees and report which half of the global frame changed.

    This is the only question the pictures could not answer: the live global view shows no arm at rest, so
    "image left = robot left?" is undecidable from a still frame. Moving a known joint decides it, and the
    answer says whether V4_GLOBAL_ROT180 puts the policy's picture in the same physical correspondence the
    training data had, or mirrors it.

    Deliberately small and reversible: arm joints only, |deg| <= 10, through _goto so the slew cap and the
    E-STOP check apply, and the caller is expected to send the negative straight after. Never a jaw index --
    the jaw is a 0..45 command in the same vector and does not belong on this path.
    """
    d = float(deg)
    if idx not in ARM_IDX:
        return jsonify({"error": f"joint {idx} is not an arm joint; ARM_IDX = {ARM_IDX}"})
    # 10 deg is the default cap, but at rest NEITHER arm is inside the global camera's field of view, so a
    # small jog changes nothing there: measured 2026-09-25, an 8 deg shoulder_pan gave a whole-frame diff of
    # max 19.7 / p99 5.0 with ZERO pixels over 30 -- pure sensor noise, and the left/right split of that
    # noise was mistaken for the arm. Bringing an arm into view needs a bigger move, so anything past 10 deg
    # is allowed only with confirm=1.
    if abs(d) > 10.0 and request.args.get("confirm") != "1":
        return jsonify({"error": f"refusing {d} deg without confirm=1; the default cap is 10"})
    if abs(d) > 45.0:
        return jsonify({"error": f"refusing {d} deg; hard cap is 45"})
    if STATE["running"]:
        return jsonify({"error": "stop the running loop first"})

    def half_diff(b64_before, b64_after, rot):
        import cv2
        def dec(b):
            im = cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_COLOR)
            return cv2.rotate(im, cv2.ROTATE_180) if rot else im
        a, b = dec(b64_before), dec(b64_after)
        dif = cv2.absdiff(a, b).mean(axis=2)
        w = dif.shape[1] // 2
        return round(float(dif[:, :w].mean()), 3), round(float(dif[:, w:].mean()), 3)

    o0 = _observe()
    if o0 is None:
        return jsonify({"error": "observe failed"})
    before = o0["images"]["middle"]
    cur = _leader_now()
    if cur is None:
        return jsonify({"error": "observe failed"})
    tgt = list(cur)
    tgt[idx] = cur[idx] + math.radians(d)
    r = _goto(tgt)
    if isinstance(r, dict) and r.get("error"):
        return jsonify({"error": r["error"]})
    time.sleep(0.6)
    o1 = _observe()
    after = o1["images"]["middle"] if o1 else None
    back = _goto(list(cur))                                  # straight back, same ramp
    if after is None:
        return jsonify({"error": "observe failed after the move"})
    import infer_core_v4 as _IC3
    rot = bool(_IC3.GLOBAL_ROT180)
    import cv2 as _cv
    def _dec(b, rot=False):
        im = _cv.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), _cv.IMREAD_COLOR)
        return _cv.rotate(im, _cv.ROTATE_180) if rot else im
    _d = _cv.absdiff(_dec(before), _dec(after)).mean(axis=2)
    _peak, _npx = _d.max(), (_d > 30).sum()
    # The wrist cameras are bolted to the arms, so they answer the routing question the global view cannot:
    # move 14-vector index 0 and the LEFT wrist feed must be the one that changes. /arm_ident checked this
    # for the jaw channel only; the arm joint block is a separate path and is checked here.
    def feed_diff(cam):
        import cv2 as _c
        def dec(b):
            return _c.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), _c.IMREAD_COLOR)
        return round(float(_c.absdiff(dec(o0["images"][cam]), dec(o1["images"][cam])).mean()), 3)
    wl, wr = feed_diff("left"), feed_diff("right")
    raw_l, raw_r = half_diff(before, after, False)
    mod_l, mod_r = half_diff(before, after, rot)
    return jsonify({
        "joint": idx, "deg": d, "returned": not (isinstance(back, dict) and back.get("error")),
        "global_transform": "rot180" if rot else "none",
        "raw_frame": {"left_half": raw_l, "right_half": raw_r,
                      "changed": "LEFT" if raw_l > raw_r else "RIGHT"},
        "model_frame": {"left_half": mod_l, "right_half": mod_r,
                        "changed": "LEFT" if mod_l > mod_r else "RIGHT"},
        "peak_diff": round(float(_peak), 1), "pixels_over_30": int(_npx),
        "wrist_diff": {"left": wl, "right": wr,
                       "changed": "LEFT" if wl > wr else "RIGHT",
                       "ratio": round(max(wl, wr) / max(1e-6, min(wl, wr)), 2)},
        "note": ("verdict uses model_frame; raw_frame must be its mirror image if rot180 is on. "
                 "pixels_over_30 near zero means the arm never entered the frame and the halves are noise")})


@app.route("/grounding", methods=["POST"])
def grounding():
    """No motion. The whole colour-grounding question in ONE frozen observation.

    Every earlier attempt flipped its answer because conditions were collected across separate calls with
    the arms in different poses, or because object coordinates were read in the captured frame while the
    policy consumes the rot180 one. Both are closed here: one observation, one seed sequence, one frame
    convention (model input, i.e. after rot180), and every cell evaluated inside the common valid_step.

    Cells: 4 image conditions (orig, R<->B, R<->P, B<->P) x 3 prompts whose first colour differs.

    Arm choice is NOT used as the target proxy -- it was measured to stay on the right arm in all six
    cells of the previous test, so it carries no target information in this scene. What is reported per
    cell is the requested first colour's image x IN THE MODEL FRAME next to the approach vector, so the
    two can be checked for covariation across cells. That needs no camera calibration: if the policy
    grounds colour, moving the requested colour across the image must move the reach with it, and the
    relationship has to hold with one consistent sign over all twelve cells.
    """
    import cv2
    K = int(request.args.get("k", "8"))
    PROMPTS = [p for p in ("RBP", "BRP", "PRB")]
    TX = dict(TASKS)
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    import infer_core_v4 as _IC2
    rot = bool(_IC2.GLOBAL_ROT180)
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(h1["images"]["middle"]), np.uint8), cv2.IMREAD_COLOR)
    if rot:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)          # model frame from here on
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (S > 90) & (V > 50)
    BANDS = {"R": (H <= 10) | (H >= 168), "B": (H >= 95) & (H <= 115), "P": (H >= 118) & (H <= 145)}
    masks, cen = {}, {}
    for c, f in BANDS.items():
        m = cv2.morphologyEx((sat & f).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if int(m.sum()) < 150:
            return jsonify({"error": f"cube {c} not found ({int(m.sum())} px)"})
        masks[c] = m
        ys, xs = np.nonzero(m)
        cen[c] = (float(xs.mean()), float(ys.mean()))

    allm = ((masks["R"] | masks["B"] | masks["P"]) * 255).astype(np.uint8)
    clean = cv2.inpaint(bgr, cv2.dilate(allm, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)

    def build(mapping):
        """mapping: colour -> centroid it should be drawn at"""
        out = clean.copy()
        for c, to in mapping.items():
            m = masks[c].astype(bool)
            dx, dy = int(round(to[0] - cen[c][0])), int(round(to[1] - cen[c][1]))
            ys, xs = np.nonzero(m)
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < out.shape[0]) & (nx >= 0) & (nx < out.shape[1])
            out[ny[ok], nx[ok]] = bgr[ys[ok], xs[ok]]
        return out

    ident = {c: cen[c] for c in "RBP"}
    def sw(a, b):
        m = dict(ident); m[a], m[b] = cen[b], cen[a]; return m
    CONDS = {"orig": ident, "R<->B": sw("R", "B"), "R<->P": sw("R", "P"), "B<->P": sw("B", "P")}

    def enc(img):
        if rot:
            img = cv2.rotate(img, cv2.ROTATE_180)      # back to capture orientation; _img rotates again
        ok, e = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("encode failed")
        return base64.b64encode(e.tobytes()).decode()

    imgs = {name: enc(build(m)) for name, m in CONDS.items()}
    KS = [2, 4, 8, 16]
    keep_task = INF.task
    cells, horizon = {}, None
    try:
        for cond, b64 in imgs.items():
            hh = [h0, {**h1, "images": {**h1["images"], "middle": b64}}]
            for pr in PROMPTS:
                INF.task = TX[pr]
                keeps = []
                for j in range(K):
                    torch.manual_seed(1234 + 17 * j)
                    INF.infer(hh)
                    L = INF.last
                    keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_mat": L["cur_mat"].copy()})
                    v = int(L["valid_step"]) - 1
                    horizon = v if horizon is None else min(horizon, v)
                cells[(cond, pr)] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        INF.task = keep_task

    ks = [k for k in KS if k <= horizon]
    rows = []
    for (cond, pr), keeps in cells.items():
        first = pr[0]
        where = CONDS[cond][first]                     # where the requested colour actually IS in this image
        D = np.stack([np.concatenate([(L["tgt_pos"][ks, r] - L["cur_mat"][r][:3, 3]) * 1000.0
                                      for r in (0, 1)], axis=1) for L in keeps])
        mean = D.mean(0)
        sem = np.linalg.norm(D - mean, axis=2).mean(0) / max(1.0, K ** 0.5)
        rows.append({"cond": cond, "prompt": pr, "first": first,
                     "requested_xy_model_frame": [round(where[0], 1), round(where[1], 1)],
                     "per_k": [{"k": k, "L": [round(float(x), 1) for x in mean[i][:3]],
                                "R": [round(float(x), 1) for x in mean[i][3:]],
                                "sem_mm": round(float(sem[i]), 2)} for i, k in enumerate(ks)]})
    return jsonify({"frame": "model input (rot180 applied)" if rot else "as captured",
                    "global_transform": "rot180" if rot else "none",
                    "reported_object_coords": "model_input_frame",
                    "draws": K, "horizon": horizon, "ks": ks,
                    "camera_raw_xy": {c: [round(bgr.shape[1] - cen[c][0], 1), round(bgr.shape[0] - cen[c][1], 1)]
                                      for c in "RBP"} if rot else {c: [round(v[0], 1), round(v[1], 1)] for c, v in cen.items()},
                    "model_input_xy_after_rot180": {c: [round(v[0], 1), round(v[1], 1)] for c, v in cen.items()},
                    "image_width": bgr.shape[1], "rows": rows})


@app.route("/swaptest", methods=["POST"])
def swaptest():
    """No motion. Does the policy follow the cube's COLOUR or its PLACE?

    The live scene has red and blue on the opposite sides from the training median, so "always goes to
    blue" could be colour grounding or a spatial prior. Editing the picture separates them causally:
    swap the two cubes in the global frame and see whether the reach follows.

    Three variants on ONE frozen observation with one shared seed sequence, global feed only, wrists and
    state untouched:

      orig    the frame as captured
      noop    both cubes cut out, the holes inpainted, then pasted back WHERE THEY WERE
      swap    same cut and inpaint, then pasted at each other's centroid

    `noop` is the control that makes the result readable. Cutting and repasting changes edges, shadows and
    JPEG texture, and a policy can react to that alone, so the comparison that means anything is swap vs
    noop -- not swap vs orig. If noop already moves the output as much as swap does, the test says nothing
    and the masks or the inpainting need work.

    Objects are moved by MASK, not by bounding box, so background does not travel with them.
    """
    import cv2
    K = int(request.args.get("k", "12"))
    prompt = request.args.get("prompt", "RBP")
    text = dict(TASKS).get(prompt)
    if text is None:
        return jsonify({"error": f"unknown prompt {prompt}"})
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    # Work in the frame the MODEL sees. _img applies GLOBAL_ROT180 downstream, so masks and centroids
    # measured on the captured frame are mirrored with respect to the policy's input, and left/right come
    # out backwards -- that inversion produced a "the policy chases blue, not the prompted colour" reading
    # that was the exact opposite of the truth. Rotate here, and rotate back before handing the edit over.
    raw = np.frombuffer(base64.b64decode(h1["images"]["middle"]), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    import infer_core_v4 as _IC2
    _rot = bool(_IC2.GLOBAL_ROT180)
    if _rot:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (S > 90) & (V > 50)
    masks = {"R": (sat & ((H <= 10) | (H >= 168))).astype(np.uint8),
             "B": (sat & (H >= 95) & (H <= 115)).astype(np.uint8)}
    for c, m in masks.items():
        m[:] = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if int(m.sum()) < 150:
            return jsonify({"error": f"cube {c} not found in the global frame ({int(m.sum())} px)"})
    cen = {}
    for c, m in masks.items():
        ys, xs = np.nonzero(m)
        cen[c] = (float(xs.mean()), float(ys.mean()))

    both = ((masks["R"] | masks["B"]) * 255).astype(np.uint8)
    clean = cv2.inpaint(bgr, cv2.dilate(both, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)

    def paste(dst, c, to):
        m = masks[c].astype(bool)
        dx, dy = int(round(to[0] - cen[c][0])), int(round(to[1] - cen[c][1]))
        ys, xs = np.nonzero(m)
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < dst.shape[0]) & (nx >= 0) & (nx < dst.shape[1])
        dst[ny[ok], nx[ok]] = bgr[ys[ok], xs[ok]]

    noop = clean.copy(); paste(noop, "R", cen["R"]); paste(noop, "B", cen["B"])
    swap = clean.copy(); paste(swap, "R", cen["B"]); paste(swap, "B", cen["R"])

    def enc(img):
        if _rot:                       # back to capture orientation; _img will rotate it again
            img = cv2.rotate(img, cv2.ROTATE_180)
        ok, e = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("encode failed")
        return base64.b64encode(e.tobytes()).decode()

    variants = {"orig": h1["images"]["middle"], "noop": enc(noop), "swap": enc(swap)}
    KS = [1, 2, 4, 8, 16, 31]
    keep_task = INF.task
    INF.task = text
    per, horizon = {}, None
    try:
        for name, b64 in variants.items():
            hh = [h0, {**h1, "images": {**h1["images"], "middle": b64}}]
            keeps = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)
                INF.infer(hh)
                L = INF.last
                keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_mat": L["cur_mat"].copy(),
                              "valid_step": int(L["valid_step"])})
                v = int(L["valid_step"]) - 1
                horizon = v if horizon is None else min(horizon, v)
            per[name] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        INF.task = keep_task

    ks = [k for k in KS if k <= horizon]
    res = {}
    for name, keeps in per.items():
        D = np.stack([np.concatenate([(L["tgt_pos"][ks, r] - L["cur_mat"][r][:3, 3]) * 1000.0
                                      for r in (0, 1)], axis=1) for L in keeps])   # (K, nk, 6)
        mean = D.mean(0)
        res[name] = {"mean": mean, "sem": np.linalg.norm(D - mean, axis=2).mean(0) / max(1.0, K ** 0.5)}

    rows = []
    for i, k in enumerate(ks):
        o, n, w = res["orig"]["mean"][i], res["noop"]["mean"][i], res["swap"]["mean"][i]
        def cmp(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return (round(float(np.linalg.norm(a - b)), 2),
                    round(float(a @ b / (na * nb)), 4) if na > 1e-9 and nb > 1e-9 else None)
        d_no, c_no = cmp(o, n)
        d_sw, c_sw = cmp(n, w)
        rows.append({"k": k,
                     # left xyz then right xyz, mm, base frame -- the components are what says whether the
                     # reach tracked the cube that moved or merely changed
                     "orig_vec": [round(float(x), 1) for x in o],
                     "noop_vec": [round(float(x), 1) for x in n],
                     "swap_vec": [round(float(x), 1) for x in w],
                     "orig_mm": round(float(np.linalg.norm(o)), 2),
                     "noop_mm": round(float(np.linalg.norm(n)), 2),
                     "swap_mm": round(float(np.linalg.norm(w)), 2),
                     "edit_artifact_mm": d_no, "cos_orig_noop": c_no,
                     "swap_effect_mm": d_sw, "cos_noop_swap": c_sw,
                     "sem_mm": round(float(np.mean([res[v]["sem"][i] for v in res])), 2)})
    return jsonify({"prompt": prompt, "draws": K, "horizon": horizon,
                    "frame": "model view (rot180 applied)" if _rot else "as captured",
                    "cube_centroids_px": {c: [round(x, 1) for x in v] for c, v in cen.items()},
                    "cube_side": {c: ("LEFT" if v[0] < bgr.shape[1] / 2 else "RIGHT") for c, v in cen.items()},
                    "cube_mask_px": {c: int(m.sum()) for c, m in masks.items()},
                    "rows": rows})


@app.route("/promptstruct", methods=["POST"])
def promptstruct():
    """No motion. Does the chunk respect the prompt's FIRST colour, and from which k does it stop?

    Coordinate-free by construction. Cube positions are never used: the six prompts already carry the
    structure, because three pairs share a first colour --

        R: RBP RPB      B: BRP BPR      P: PRB PBR

    A policy that grounds the first colour must keep each pair together while the first cube is being
    approached, and may only separate later, when the second colour starts to matter. So the statistic is,
    at every k, the mean distance between the two members of a pair against the mean distance across pairs.
    Reconstructing cube coordinates from successful-grasp TCPs was considered and rejected: the target would
    then be built in the policy's own frame from the policy's own grasp geometry, and object grounding could
    no longer be told apart from kinematics.

    Every prompt sees ONE frozen observation and the same seed sequence, so the sampler is differenced out.
    The per-k draw spread is reported next to the ratio -- a ratio computed on top of noise means nothing.
    """
    K = int(request.args.get("k", "12"))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    hh = [h0, h1]
    PAIRS = {"R": ("RBP", "RPB"), "B": ("BRP", "BPR"), "P": ("PRB", "PBR")}
    TEXT = dict(TASKS)
    keep_task = INF.task
    per_prompt = {}
    try:
        for label, text in TASKS:
            INF.task = text
            draws = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)      # identical noise for every prompt
                INF.infer(hh)
                L = INF.last
                m_now = L["cur_mat"]
                T = int(L["valid_step"])
                # both arms' world displacement from the frozen pose, per k -> (T,6)
                d = np.concatenate([(L["tgt_pos"][:T, r] - m_now[r][:3, 3]) * 1000.0 for r in (0, 1)], axis=1)
                draws.append(d)
            n = min(len(d) for d in draws)
            D = np.stack([d[:n] for d in draws])       # (K, n, 6)
            per_prompt[label] = D
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        INF.task = keep_task

    n = min(v.shape[1] for v in per_prompt.values())
    mean = {k: v[:, :n].mean(0) for k, v in per_prompt.items()}
    spread = {k: np.linalg.norm(v[:, :n] - v[:, :n].mean(0), axis=2).mean(0) for k, v in per_prompt.items()}
    rows = []
    for k in range(n):
        within = [np.linalg.norm(mean[a][k] - mean[b][k]) for a, b in PAIRS.values()]
        between = []
        for (c1, (a1, b1)), (c2, (a2, b2)) in itertools.combinations(PAIRS.items(), 2):
            for x in (a1, b1):
                for y in (a2, b2):
                    between.append(np.linalg.norm(mean[x][k] - mean[y][k]))
        w, bt = float(np.mean(within)), float(np.mean(between))
        noise = float(np.mean([spread[p][k] for p in mean])) / max(1.0, K ** 0.5)
        rows.append({"k": k, "within_mm": round(w, 2), "between_mm": round(bt, 2),
                     "ratio": round(bt / w, 2) if w > 1e-6 else None,
                     "sem_mm": round(noise, 2),
                     "per_pair_mm": {c: round(float(np.linalg.norm(mean[a][k] - mean[b][k])), 2)
                                     for c, (a, b) in PAIRS.items()}})
    return jsonify({"draws": K, "steps": n, "rows": rows})


@app.route("/camcheck", methods=["POST"])
def camcheck():
    """No motion. Does rotating the GLOBAL camera 180 deg change what the policy wants?

    B663 mixed two camera domains: the HEAD pool untouched and the FRONT (R675) pool with its global feed
    rotated 180 deg. The live robot has one orientation, and if it is read as the wrong domain the policy
    can be confidently wrong. This is the deterministic test for that, and it moves nothing.

    Both variants run on ONE frozen observation pair, and each draw uses the same seed in both variants --
    flow matching starts from noise, so without that the difference between A and B is mostly the sampler.
    Only the global (middle) image differs; wrists, state and task are byte-identical.

      A  live global as-is
      B  rot180(live global)

    Reported per arm and per k: the world-frame step vector and its magnitude, averaged over K draws, plus
    the spread across draws so a difference can be compared against the sampler's own noise. Magnitude
    alone does not decide anything -- a small output is natural on a still scene -- so the direction is
    reported next to it.
    """
    K = int(request.args.get("k", "4"))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    import cv2
    def rot180_b64(b64):
        a = np.frombuffer(base64.b64decode(b64), np.uint8)
        bgr = cv2.imdecode(a, cv2.IMREAD_COLOR)
        ok, enc = cv2.imencode(".jpg", cv2.rotate(bgr, cv2.ROTATE_180))
        if not ok:
            raise RuntimeError("re-encode failed")
        return base64.b64encode(enc.tobytes()).decode()

    variants = {"A_live": h1["images"],
                "B_rot180": {**h1["images"], "middle": rot180_b64(h1["images"]["middle"])}}
    KS_REQ = [1, 2, 4, 8, 16, 31]
    # valid_step is not constant across draws, so the k list is fixed ONCE from the first draw and reused
    # for every draw and both variants -- otherwise the per-draw arrays have different lengths and, worse,
    # A and B would be compared at different horizons.
    KS = KS_REQ
    out = {"draws": K, "variants": {}, "valid_step": {}}
    try:
        for name, imgs in variants.items():
            hh = [h0, {**h1, "images": imgs}]
            per_draw, vsteps = [], []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)          # same noise sequence in BOTH variants
                INF.infer(hh)
                L = INF.last
                vsteps.append(int(L["valid_step"]))
                m_now = L["cur_mat"]
                # valid_step is the first chunk step IK cannot reach within FK_MAX_MM, and it is NOT constant
                # across draws on the same observation. Steps past it are recorded as NaN rather than dropped,
                # so every draw keeps the same shape and the share of draws that could reach each k becomes a
                # reported number instead of a crash.
                blk = np.full((2, len(KS), 3), np.nan)
                for r in (0, 1):
                    for i, k in enumerate(KS):
                        if k < L["valid_step"]:
                            blk[r, i] = (L["tgt_pos"][k, r] - m_now[r][:3, 3]) * 1000.0
                per_draw.append(blk)
            D = np.stack(per_draw)                         # (K, 2, nk, 3), NaN past valid_step
            out["valid_step"][name] = {"min": min(vsteps), "max": max(vsteps),
                                       "median": int(np.median(vsteps)), "all": vsteps}
            arms = {}
            for r, side in ((0, "left"), (1, "right")):
                rows = []
                for i, k in enumerate(KS):
                    v = D[:, r, i, :]
                    ok = ~np.isnan(v).any(1)
                    if not ok.any():
                        rows.append({"k": k, "reachable_frac": 0.0}); continue
                    vv = v[ok]
                    mean = vv.mean(0)
                    rows.append({"k": k,
                                 "dxyz_mm": [round(float(x), 2) for x in mean],
                                 "norm_mm": round(float(np.linalg.norm(mean)), 2),
                                 "draw_spread_mm": round(float(np.linalg.norm(vv - mean, axis=1).mean()), 2),
                                 "reachable_frac": round(float(ok.mean()), 3)})
                arms[side] = rows
            out["variants"][name] = arms
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})

    # direction agreement between the two variants, per arm and k: cos ~ 1 means rotating the camera did
    # not change WHERE the policy wants to go, only how far.
    cmp = {}
    for side in ("left", "right"):
        rows = []
        for a, b in zip(out["variants"]["A_live"][side], out["variants"]["B_rot180"][side]):
            if "dxyz_mm" not in a or "dxyz_mm" not in b:
                rows.append({"k": a["k"], "unreachable": True}); continue
            va, vb = np.array(a["dxyz_mm"]), np.array(b["dxyz_mm"])
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            rows.append({"k": a["k"], "norm_A": a["norm_mm"], "norm_B": b["norm_mm"],
                         "ratio_B_over_A": round(float(nb / na), 3) if na > 1e-9 else None,
                         "cos_AB": round(float(va @ vb / (na * nb)), 4) if na > 1e-9 and nb > 1e-9 else None})
        cmp[side] = rows
    out["compare"] = cmp
    out["ks"] = KS
    return jsonify(out)



def _chunk_end(L, horizon=None):
    """Both arms' displacement at the LAST VALID chunk step, in mm, as a 6-vector.

    The rule, after 2026-09-25: every trajectory comparison stays inside each draw's `valid_step`, and a
    comparison ACROSS prompts or models uses the minimum valid_step of everything being compared.
    `tgt_pos[-1]` ignores it. DELTA32 290k truncates at 25 of 32 while REL32 runs the full 32, so reading
    index 31 for both fed the unreachable tail into the comparison and produced a "DELTA does not respect
    the prompt's first colour" verdict that a valid-horizon test then contradicted outright.
    Pass `horizon` to pin the common step; leave it None to use this draw's own last valid step.
    """
    k = (L["valid_step"] - 1) if horizon is None else min(horizon, L["valid_step"] - 1)
    k = max(0, int(k))
    return np.concatenate([L["tgt_pos"][k, r] - L["cur_tcp"][r][:3] for r in (0, 1)]) * 1000.0


@app.route("/framecheck", methods=["POST"])
def framecheck():
    """No motion. Is the chunk composed into the base frame with the right sign?

    The running telemetry shows pred_body (model output, BODY frame) next to cmd_world (the commanded step,
    BASE frame). Those are different frames, so a sign difference between them proves nothing -- a TCP that
    points back at the base legitimately flips two axes. The only honest check rotates the model output into
    the base frame first, and that is what this does:

        pred_world[k] = R_now @ act[k][:3]                 model output, rotated, NOT re-composed
        cmd_world[k]  = tgt_pos[k] - tcp_now                what the arm is actually asked to do

    Under UMI current-anchor every step hangs off the same measured T_now, so the two must agree to within
    rotation-representation error at every k. cos < 0 on a step with real magnitude is a compose/sign bug;
    cos ~ +1 with matching magnitude means the decode is right and a wrong direction on the robot is the
    policy or the frame adapter, not the composition.
    """
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    try:
        INF.infer([h0, h1])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    L = INF.last
    m_now = L["cur_mat"]
    out = {"action_mode": L.get("action_mode"), "valid_step": int(L["valid_step"]), "arms": {}}
    for r, side in ((0, "left"), (1, "right")):
        R_now = m_now[r][:3, :3]
        tcp0 = m_now[r][:3, 3]
        rows = []
        for k in (0, 3, 7, 15, 31):
            if k >= L["valid_step"]:
                continue
            pb = np.asarray(L["act"][k, r * 10:r * 10 + 3], dtype=float)
            pw = (R_now @ pb) * 1000.0
            cw = (np.asarray(L["tgt_pos"][k, r], dtype=float) - tcp0) * 1000.0
            npw, ncw = float(np.linalg.norm(pw)), float(np.linalg.norm(cw))
            cos = float(pw @ cw / (npw * ncw)) if npw > 1e-9 and ncw > 1e-9 else None
            rows.append({"k": k,
                         "pred_body_mm": [round(float(x) * 1000, 2) for x in pb],
                         "pred_world_mm": [round(float(x), 2) for x in pw],
                         "cmd_world_mm": [round(float(x), 2) for x in cw],
                         "norm_pred_mm": round(npw, 2), "norm_cmd_mm": round(ncw, 2),
                         "cos": None if cos is None else round(cos, 4)})
        w = np.asarray(L["widths"][:, r], dtype=float) * 1000.0      # mm, whole chunk
        out["arms"][side] = {"tcp_xyz_mm": [round(float(x) * 1000, 1) for x in tcp0], "steps": rows,
                             # the jaw channel over the WHOLE chunk. The dataset's width spans 0..114 mm
                             # (mean 29, std 40), so a chunk that never leaves single digits means the
                             # policy is holding the jaw shut and can never grasp -- a different failure
                             # from the arm going the wrong way, and invisible in the per-step telemetry
                             # which only shows the one executed step.
                             "width_mm": {"min": round(float(w.min()), 2), "max": round(float(w.max()), 2),
                                          "mean": round(float(w.mean()), 2),
                                          "first8": [round(float(x), 2) for x in w[:8]],
                                          "last8": [round(float(x), 2) for x in w[-8:]]}}
    return jsonify(out)


@app.route("/zprobe", methods=["POST"])
def zprobe():
    """Repeat inference on ONE frozen observation, and report the vertical component's mean and spread.

    The robot does not move. This separates a policy that is biased upward from a loop that walks upward:
    if E[dz] is clearly positive here, on a single fixed input, no amount of control tuning will fix it.
    Each repeat draws its own denoising noise, so the spread is the sampler's, not the robot's.
    """
    n = int((request.get_json(silent=True) or {}).get("n", 24))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    import numpy as _np
    KS = [int(x) for x in (request.get_json(silent=True) or {}).get("ks", [1, 2, 4, 8, 16])]
    out = {"n": n, "task": STATE["task"], "ks": KS}
    per = {(r, k): [] for r in (0, 1) for k in KS}
    for _ in range(n):
        try:
            INF.infer([h0, h1])
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"inference failed: {e}"})
        L = INF.last
        for r in (0, 1):
            for k in KS:
                a = L["act"][k - 1, r * 10:r * 10 + 3]
                world = L["cur_mat"][r][:3, :3] @ a
                per[(r, k)].append((float(world[2]) * 1000, float(_np.linalg.norm(a)) * 1000))
    for r, arm in ((0, "L"), (1, "R")):
        out[arm] = {}
        for k in KS:
            v = _np.array(per[(r, k)])
            # the fraction that points the same way as the mean: a direction that is stable across draws is
            # a prediction, one that is half up and half down is the sampler
            agree = float(((v[:, 0] > 0) == (v[:, 0].mean() > 0)).mean())
            out[arm][f"k{k}"] = {"world_dz_mean": round(float(v[:, 0].mean()), 2),
                                 "world_dz_std": round(float(v[:, 0].std()), 2),
                                 "dz_sign_agree": round(agree, 2),
                                 "step_norm_mean": round(float(v[:, 1].mean()), 2),
                                 "dz_frac_of_norm": round(float(abs(v[:, 0].mean()) /
                                                               max(v[:, 1].mean(), 1e-6)), 2)}
    return jsonify(out)


@app.route("/grip/<arm>/<cmd>", methods=["POST"])
def grip_one(arm, cmd):
    """Drive ONE jaw to an explicit command and leave it there.

    Command units are 0..45, not the raw count /observe reports (obs = cmd * -6). 45 is the open end,
    0 the closed one. The arm joints do not move: the vector sent is the current leader pose with a single
    element replaced. /gripper_probe moves both jaws and puts them back, which is a different job.
    """
    gi = GRIP_IDX[0 if arm == "left" else 1]
    lead = _leader_now()
    if lead is None:
        return jsonify({"error": "observe failed"})
    before = _observe()["joints14"]
    out = list(lead)
    out[gi] = float(cmd)
    r = _exec_step(out)
    if isinstance(r, dict) and r.get("error"):
        return jsonify(r)
    time.sleep(1.5)
    after = _observe()["joints14"]
    return jsonify({"arm": arm, "cmd": float(cmd),
                    "raw_before": [round(float(before[g]), 2) for g in GRIP_IDX],
                    "raw_after": [round(float(after[g]), 2) for g in GRIP_IDX]})


@app.route("/grip_calib", methods=["POST"])
def grip_calib():
    """Measure each jaw's command -> raw-count law separately, because they are not the same law.

    The core has been assuming one constant for both arms (obs = cmd * -6, commands 0..45). Measured on this
    robot the left follows roughly that and the right does not: it responds to the opposite sign and a
    different gain, so every gripper command the policy issued for the right arm was wrong. This sweeps a
    command ladder per arm, waits for the jaw to settle at each point, and fits obs = a*cmd + b over the
    points that are not against a stop.

    Arms do not move; only the one jaw element of the leader vector changes.
    """
    import numpy as _np
    body = request.get_json(silent=True) or {}
    cmds = [float(x) for x in body.get("cmds", [0, -45, -30, -20, -10, 10, 20, 30, 45, 0])]
    settle = float(body.get("settle_s", 1.6))
    out = {}
    for arm, gi in (("left", GRIP_IDX[0]), ("right", GRIP_IDX[1])):
        pts = []
        for c in cmds:
            lead = _leader_now()
            if lead is None:
                return jsonify({"error": "observe failed"})
            v = list(lead)
            v[gi] = c
            r = _exec_step(v)
            if isinstance(r, dict) and r.get("error"):
                return jsonify(r)
            time.sleep(settle)
            pts.append((c, float(_observe()["joints14"][gi])))
        # The readback trails the command by exactly ONE command, not by a settling time -- 3.5 s of dwell
        # does not remove it. Pairing obs[i] with cmd[i] therefore measures the PREVIOUS command and makes
        # the gain look arm-dependent and non-linear; pairing obs[i+1] with cmd[i] gives -6.00 on both arms.
        a = _np.array([(pts[i][0], pts[i + 1][1]) for i in range(len(pts) - 1)])
        # drop the ends that sit against a stop: a command that changes nothing carries no information
        lo, hi = a[:, 1].min(), a[:, 1].max()
        span = hi - lo
        keep = a[(a[:, 1] > lo + 0.02 * span) & (a[:, 1] < hi - 0.02 * span)] if span > 1 else a
        fit = None
        if len(keep) >= 2:
            A = _np.vstack([keep[:, 0], _np.ones(len(keep))]).T
            (g, b), res, *_ = _np.linalg.lstsq(A, keep[:, 1], rcond=None)
            pred = A @ _np.array([g, b])
            ss = float(1 - ((keep[:, 1] - pred) ** 2).sum() /
                       max(((keep[:, 1] - keep[:, 1].mean()) ** 2).sum(), 1e-9))
            fit = {"obs_per_cmd": round(float(g), 4), "offset": round(float(b), 3), "r2": round(ss, 5),
                   "n_fit": len(keep)}
        out[arm] = {"points": [[round(c, 1), round(o, 2)] for c, o in pts],
                    "lag_paired": [[round(c, 1), round(o, 2)] for c, o in a], "raw_min": round(float(lo), 2),
                    "raw_max": round(float(hi), 2), "fit": fit}
    return jsonify(out)


@app.route("/replay2x2", methods=["POST"])
def replay2x2():
    """image x state, dataset x live: which half of the observation costs the amplitude?

    Held-out frames are replayed through the deployment core itself -- same preprocessing, same checkpoint,
    same UMI decode -- and each of the four combinations is scored against the ground-truth chunk stored with
    that frame. The robot does not move.

      dataset / dataset   the core on data the offline evaluator agrees about: the control
      dataset / live      the pictures the policy was trained on, the proprioception of right now
      live    / dataset   the pictures of right now, the proprioception of a demonstration
      live    / live      a rollout step, minus the rollout

    A ratio near the offline one in the control and far from it in live/live puts the fault in the
    observation; a control that is already far puts it in this inference path.
    """
    import numpy as _np
    body = request.get_json(silent=True) or {}
    n_frames = int(body.get("frames", 6))
    KS = [int(x) for x in body.get("ks", [1, 4, 16])]
    path = os.path.expanduser(body.get("pack", "~/holobrain-mac-model/replay_frames.npy"))
    if not os.path.isfile(path):
        return jsonify({"error": f"no frame pack at {path}"})
    pack = list(_np.load(path, allow_pickle=True))[:n_frames]

    live = _observe()
    if live is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    live2 = _observe()
    if live2 is None:
        return jsonify({"error": "observe failed"})
    live_state, _, _ = INF.build_state([live, live2])
    live_imgs = live2["images"]

    g = torch.Generator().manual_seed(4242)
    noise = torch.randn(1, 16, INF.policy.config.max_action_dim, generator=g)

    cells = {"ds_img/ds_state": (True, True), "ds_img/live_state": (True, False),
             "live_img/ds_state": (False, True), "live_img/live_state": (False, False)}
    out = {"n_frames": len(pack), "ks": KS, "cells": {}}
    for name, (ds_img, ds_state) in cells.items():
        acc = {(arm, k): [] for arm in (0, 1) for k in KS}
        gtacc = {(arm, k): [] for arm in (0, 1) for k in KS}
        for fr in pack:
            imgs = ({"middle": fr["jpg_middle"], "left": fr["jpg_left"], "right": fr["jpg_right"]}
                    if ds_img else live_imgs)
            st = fr["state20"] if ds_state else live_state
            act = INF.infer_raw(imgs, st, task=str(fr["task"]), noise=noise)
            gt = _np.asarray(fr["action"], _np.float64)
            for arm in (0, 1):
                for k in KS:
                    o = arm * 10
                    acc[(arm, k)].append(float(_np.linalg.norm(act[k - 1, o:o + 3])) * 1000)
                    gtacc[(arm, k)].append(float(_np.linalg.norm(gt[k - 1, o:o + 3])) * 1000)
        cell = {}
        for arm, lab in ((0, "L"), (1, "R")):
            for k in KS:
                pn, gn = float(_np.mean(acc[(arm, k)])), float(_np.mean(gtacc[(arm, k)]))
                cell[f"{lab}{k}"] = {"pred_norm_mm": round(pn, 2), "gt_norm_mm": round(gn, 2),
                                     "ratio": round(pn / max(gn, 1e-6), 2)}
        out["cells"][name] = cell
    return jsonify(out)


@app.route("/gripper_mode/<mode>", methods=["POST"])
def gripper_mode(mode):
    import infer_core_v4 as C
    if mode in ("hold", "predict", "binary"):
        C.GRIPPER_MODE = mode
    return jsonify({"gripper": C.GRIPPER_MODE})


def _jaw_torque(on):
    """Torque the two gripper motors on or off, without touching the arm motors.

    At rest the jaws are commanded to 0, which presses them into the closed stop, and FORCE_POS with a 10%
    torque limit cannot settle there -- it hunts. Measured: 7.6 deg of oscillation against a stop, 2.5 deg
    off it, and the motor runs a few degrees hotter. That hunt is the buzzing. Cutting torque to the jaw
    alone stops it; `/torque` cannot be used for this because it switches every motor of an arm, and
    dropping the arm motors would let the arm fall. `/gripper_zero` with stage=release disables exactly the
    one motor (it does not re-zero anything until stage=set).
    """
    out = []
    if os.environ.get("V4_JAW_TORQUE", "1") == "0":   # [2026-10-01] MIT bridge: /torque re-ran the arm enable and stalled the bus
        return [{"skipped": "V4_JAW_TORQUE=0"}]
    for arm in ("left", "right"):
        if on:
            out.append(_robot("/torque", "post", json={"arm": arm, "on": True}))
        else:
            out.append(_robot("/gripper_zero", "post", json={"arm": arm, "stage": "release"}))
    return out


@app.route("/rest", methods=["POST"])
def rest():
    STATE["running"] = False
    STATE["stop_reason"] = "rest"
    r = _robot("/rest", "post", json={})
    jaw = _jaw_torque(False)           # nothing to hold at rest, so nothing to buzz about
    return jsonify({"rest": r, "jaw_torque_off": jaw})


@app.route("/train_start", methods=["POST"])
def train_start():
    """Move to the training data's frame-0 median pose. Rollouts start here so the start pose is in distribution."""
    if STATE["running"]:
        return jsonify({"error": "stop the loop first"})
    st = _robot("/status")
    if not st.get("connected"):
        return jsonify({"error": "robot not connected"})
    if st.get("estop"):
        return jsonify({"error": "E-STOP engaged"})
    tgt = np.asarray(TRAIN_START_DEG, float).copy()
    for i in ARM_IDX:
        tgt[i] = math.radians(TRAIN_START_DEG[i])
    tgt[GRIP_IDX[0]] = tgt[GRIP_IDX[1]] = 0.0            # jaws at raw 0
    lead = tgt.copy()
    for k in FLIP_IDX:
        lead[k] = -lead[k]
    STATE["stop_reason"] = None
    _jaw_torque(True)                  # rest leaves them unpowered; a pose move commands them again
    with _lock:
        r = _goto([float(x) for x in lead])
    return jsonify(r)


def _train_start_move():
    tgt = np.asarray(TRAIN_START_DEG, float).copy()
    for i in ARM_IDX:
        tgt[i] = math.radians(TRAIN_START_DEG[i])
    tgt[GRIP_IDX[0]] = tgt[GRIP_IDX[1]] = 0.0
    lead = tgt.copy()
    for k in FLIP_IDX:
        lead[k] = -lead[k]
    return _goto([float(x) for x in lead])


def _ab_loop(chunks, fixed=None):
    """Same scene, same start pose, six prompts: does the instruction change what the robot does?

    Every prompt is run from a freshly re-commanded TRAIN START pose, so the only thing that differs between
    rows is the text. Both the PREDICTED chunk endpoint (what the policy wanted) and the MEASURED one (what
    the arm did) are recorded -- the first is the language test, the second says whether execution preserved
    the difference.
    """
    keep = INF.task
    # `fixed` repeats ONE prompt through the identical procedure: that is the noise floor (sampler draw +
    # start-pose repeatability + scene drift) the six-prompt spread has to beat before it means anything.
    seq = [(f"{fixed}#{i+1}", dict(TASKS)[fixed]) for i in range(len(TASKS))] if fixed else list(TASKS)
    STATE["ab"] = {"running": True, "rows": [], "pairs": {}, "mode": fixed or "six-prompt"}
    ends = {}
    try:
        for label, text in seq:
            if not STATE["ab"]["running"]:
                break
            _train_start_move()
            INF.task = text
            STATE["task"] = label
            agg = None
            for _ in range(chunks):
                tel = _one_press()
                if tel.get("error"):
                    STATE["ab"]["rows"].append({"prompt": label, "error": tel["error"]})
                    agg = None
                    break
                agg = tel
            if agg is None:
                continue
            L = INF.last
            pred_end = _chunk_end(L)
            ends[label] = pred_end
            row = {"prompt": label,
                   "pred_mm": [r_["pred_mm"] for r_ in agg["rows"]],
                   "actual_mm": [r_["actual_mm"] for r_ in agg["rows"]],
                   "dir_cos": [r_["dir_cos"] for r_ in agg["rows"]],
                   "pred_LR_ratio": round(agg["rows"][0]["pred_mm"] / max(agg["rows"][1]["pred_mm"], 1e-6), 2),
                   "grip_width_mm": agg.get("grip_width_mm"),
                   "pred_endpoint_mm": [round(float(x), 1) for x in pred_end]}
            STATE["ab"]["rows"].append(row)
        labels = [k for k in ends]
        STATE["ab"]["pairs"] = {f"{a}-{b}": round(float(np.linalg.norm(ends[a] - ends[b])), 2)
                                for i, a in enumerate(labels) for b in labels[i + 1:]}
        if STATE["ab"]["pairs"]:
            v = list(STATE["ab"]["pairs"].values())
            STATE["ab"]["mean_pair_mm"] = round(sum(v) / len(v), 2)
            STATE["ab"]["max_pair_mm"] = max(v)
    finally:
        INF.task = keep
        STATE["ab"]["running"] = False
    print(f"[v4-smoke] prompt A/B done: {STATE['ab'].get('mean_pair_mm')} mm mean pairwise", flush=True)


def _profile_loop(n):
    """Profiler only -- it measures, it does not compensate. Latency is reported in action steps so the skip
    decision is made from p50/p95, not from a guess."""
    STATE["profile"] = {"running": True, "cycles": [], "n": n}
    try:
        for _ in range(n):
            if not STATE["profile"]["running"]:
                break
            tel = _one_press()
            if tel.get("error"):
                STATE["profile"]["cycles"].append({"error": tel["error"]})
                continue
            STATE["profile"]["cycles"].append({**tel["latency"], "skip": tel["skip_steps_would_be"]})
        rows = [c for c in STATE["profile"]["cycles"] if "error" not in c]
        if rows:
            def pct(key, q):
                v = sorted(r[key] for r in rows if r.get(key) is not None)
                return round(v[min(int(q * len(v)), len(v) - 1)], 1) if v else None
            STATE["profile"]["summary"] = {
                k: {"p50": pct(k, 0.5), "p95": pct(k, 0.95),
                    "n": sum(1 for r in rows if r.get(k) is not None)}
                for k in ("observe_ms", "infer_ms", "ik_ms", "arm_latency_ms", "gripper_latency_ms",
                          "total_ms", "skip")}
    finally:
        STATE["profile"]["running"] = False
    print(f"[v4-smoke] profile done: {STATE['profile'].get('summary')}", flush=True)


@app.route("/profile", methods=["POST"])
def profile():
    if STATE["running"] or STATE.get("ab", {}).get("running") or STATE.get("profile", {}).get("running"):
        return jsonify({"error": "stop the other run first"})
    n = int(request.args.get("n", "20"))
    threading.Thread(target=_profile_loop, args=(n,), daemon=True).start()
    return jsonify({"started": True, "cycles": n})


def _replay_loop(mode, n_action, stride, max_queries, use_grip, anchor="source", ep=0):
    """Drive the robot through the deployment loop with the POLICY REPLACED BY GROUND TRUTH.

    Same decode, same per-waypoint IK re-solve, same dwell as a real rollout -- only the chunk comes from the
    dataset instead of the model. So if this does not reproduce the demonstration's motion, the fault is in the
    decoder or the execution path, and no checkpoint can fix it.

      delta : T_k = T_{k-1} @ dT_k   anchored on the MEASURED pose  (cumulative, v4)
      umi   : T_k = T_measured @ A_k                                 (current-anchor, upstream UMI)

    The anchor is the measured pose, not the demo's absolute start, so the arm reproduces the demonstration's
    RELATIVE motion from wherever it currently is. That is what deployment does, and it avoids a jump.
    """
    import numpy as _np
    from scipy.spatial.transform import Rotation as _Rot
    path = os.path.expanduser(f"~/holobrain-mac-model/chunks_ep{ep}.npz")
    d = _np.load(path, allow_pickle=True)
    SRC = d["src"]
    JOINTS = None
    if mode == "joints":
        # the recorded joint trajectory: what the robot physically did during the demo, with no IK, no FK and
        # no frame convention in the way. The purest possible reference for the other three modes.
        jp = os.path.expanduser(f"~/holobrain-mac-model/joints_ep{ep}.npz")
        JOINTS = _np.load(jp)["joints"]                      # (frames+1, 2, 4, 4) absolute source TCP poses, the teleop itself
    C = d["delta"] if mode == "delta" else (d["umi"] if mode == "umi" else d["delta"])
    if JOINTS is not None:
        C = JOINTS[:max(len(JOINTS) - 1, 1)]
    total = len(C) if max_queries <= 0 else min(len(C), max_queries * stride)
    STATE["replay"] = {"running": True, "mode": mode, "anchor": anchor, "episode": int(ep), "rows": [],
                       "file": path,
                       "done": 0, "total_queries": int(np.ceil(total / stride)), "frames": int(len(C))}
    q = 0
    try:
        # The replay targets the demonstration's absolute poses, so the arm has to be AT its start first --
        # otherwise the very first step is the whole distance to the demo (128.7 mm on the first attempt) and
        # the safety guard stops it, correctly. This approach move is slew-limited like every other motion.
        o0 = _observe()
        if o0 is None:
            STATE["replay"]["rows"].append({"error": "observe failed before approach"})
            raise RuntimeError("no observation")
        if JOINTS is not None:
            tgt0 = list(JOINTS[0])
            for _j in ARM_IDX:
                tgt0[_j] = math.radians(tgt0[_j])
            for _f in FLIP_IDX:
                tgt0[_f] = -tgt0[_f]
            STATE["replay"]["approach"] = {"mode": "joints"}
            print("[v4-smoke] replay approach: recorded joints of frame 0 (no IK)", flush=True)
            r0 = _goto(tgt0)
            if isinstance(r0, dict) and r0.get("error"):
                STATE["replay"]["rows"].append({"error": f"approach failed: {r0['error']}"})
                raise RuntimeError("approach failed")
            approached = True
        else:
            approached = False
        if not approached:
          start_pos = _np.stack([SRC[0, r][:3, 3] for r in (0, 1)])
          start_quat = _np.stack([_Rot.from_matrix(SRC[0, r][:3, :3]).as_quat() for r in (0, 1)])
          m0 = INF.tcp_now(o0["joints14"])[0]
          gap = max(float(_np.linalg.norm(start_pos[r] - m0[r][:3, 3])) * 1000 for r in (0, 1))
          a0, ik0, ok0, _c = INF.solve_waypoint(o0["joints14"], start_pos, start_quat,
                                              C[0, 0, [9, 19]] if use_grip else None)
          STATE["replay"]["approach"] = {"gap_mm": round(gap, 1), "ik_fk_err_mm": round(ik0, 2), "ik_ok": bool(ok0)}
          print(f"[v4-smoke] replay approach: {gap:.1f} mm to episode start, ik residual {ik0:.2f} mm", flush=True)
          r0 = _goto(list(a0), (start_pos, start_quat), ik0)
          if isinstance(r0, dict) and r0.get("error"):
              STATE["replay"]["rows"].append({"error": f"approach failed: {r0['error']}"})
              raise RuntimeError("approach failed")
        while STATE["replay"]["running"] and q < total:
            o = _observe()
            if o is None:
                STATE["replay"]["rows"].append({"error": "observe failed"}); break
            m_now = INF.tcp_now(o["joints14"])[0]
            # anchor=source replays the demonstration itself: every representation decodes against the pose the
            # demo was at, so all three must trace the same path. anchor=measured is what deployment does, and
            # it is where a cumulative representation can drift on execution lag.
            if mode == "joints":
                # send the recorded joints straight through the leader-frame path, slew limited as always
                tgt = list(JOINTS[q])
                for _j in ARM_IDX:
                    tgt[_j] = math.radians(tgt[_j])          # the source dataset stores degrees
                for _f in FLIP_IDX:
                    tgt[_f] = -tgt[_f]
                if not use_grip:
                    lead_now = list(o["joints14"])
                    for _f in FLIP_IDX:
                        lead_now[_f] = -lead_now[_f]
                    for _g in GRIP_IDX:
                        tgt[_g] = lead_now[_g]
                r_ = _goto(tgt)
                o3 = _observe()
                m_after = INF.tcp_now(o3["joints14"])[0] if o3 else None
                row = {"q": int(q), "sent": 1}
                for r, nm in ((0, "L"), (1, "R")):
                    got = (m_after[r][:3, 3] - m_now[r][:3, 3]) if m_after is not None else _np.zeros(3)
                    row[nm] = {"want_mm": None, "got_mm": [round(float(x) * 1000, 2) for x in got], "cos": None}
                STATE["replay"]["rows"].append(row)
                STATE["replay"]["done"] += 1
                q += stride
                continue
            base = m_now if anchor == "measured" else SRC[q]
            ch = C[q]
            pos = _np.zeros((16, 2, 3)); quat = _np.zeros((16, 2, 4)); wid = _np.zeros((16, 2))
            for r in (0, 1):
                cur = base[r].copy()
                for k in range(16):
                    M = _np.eye(4)
                    d6 = ch[k, r * 10 + 3:r * 10 + 9]
                    b1 = d6[:3] / _np.linalg.norm(d6[:3])
                    b2 = d6[3:] - b1 * (b1 @ d6[3:]); b2 /= _np.linalg.norm(b2)
                    M[:3, :3] = _np.stack([b1, b2, _np.cross(b1, b2)])
                    M[:3, 3] = ch[k, r * 10:r * 10 + 3]
                    if mode == "source":
                        Tk = SRC[min(q + k + 1, len(SRC) - 1), r]      # the recorded pose, no representation
                    elif mode == "delta":
                        Tk = cur @ M
                        cur = Tk
                    else:
                        Tk = base[r] @ M
                    pos[k, r] = Tk[:3, 3]
                    quat[k, r] = _Rot.from_matrix(Tk[:3, :3]).as_quat()
                    wid[k, r] = ch[k, r * 10 + 9]
            step_mm = max(float(_np.linalg.norm(pos[0, r] - m_now[r][:3, 3])) * 1000 for r in (0, 1))
            if REPLAY_MAX_STEP_MM and step_mm > REPLAY_MAX_STEP_MM:
                STATE["replay"]["rows"].append({"q": int(q), "abort": f"first step {step_mm:.1f} mm > "
                                                f"{REPLAY_MAX_STEP_MM}"})
                break
            sent = 0
            for k in range(min(n_action, 16)):
                if not STATE["replay"]["running"]:
                    break
                o2 = _observe()
                if o2 is None:
                    break
                a, ikerr, ok, clip = INF.solve_waypoint(o2["joints14"], pos[k], quat[k],
                                                        wid[k] if use_grip else None)
                r_ = _goto(list(a), (pos[k], quat[k]), ikerr)
                if isinstance(r_, dict) and r_.get("error"):
                    STATE["replay"]["rows"].append({"q": int(q), "error": r_["error"]}); STATE["replay"]["running"] = False
                    break
                sent += 1
            o3 = _observe()
            m_after = INF.tcp_now(o3["joints14"])[0] if o3 else None
            row = {"q": int(q), "sent": sent}
            for r, nm in ((0, "L"), (1, "R")):
                want = pos[max(sent - 1, 0), r] - m_now[r][:3, 3]
                got = (m_after[r][:3, 3] - m_now[r][:3, 3]) if m_after is not None else _np.zeros(3)
                row[nm] = {"want_mm": [round(float(x) * 1000, 2) for x in want],
                           "got_mm": [round(float(x) * 1000, 2) for x in got],
                           "cos": round(float(_np.dot(want, got) / max(_np.linalg.norm(want) * _np.linalg.norm(got), 1e-12)), 3)}
            STATE["replay"]["rows"].append(row)
            STATE["replay"]["done"] += 1
            q += stride
    finally:
        STATE["replay"]["running"] = False
    print(f"[v4-smoke] replay {mode} done: {len(STATE['replay']['rows'])} queries", flush=True)


@app.route("/replay", methods=["POST"])
def replay():
    if STATE["running"] or STATE.get("replay", {}).get("running"):
        return jsonify({"error": "stop the other run first"})
    mode = request.args.get("mode", "delta")
    if mode not in ("delta", "umi", "source", "joints"):
        return jsonify({"error": "mode must be joints, source, delta or umi"})
    anchor = request.args.get("anchor", "source")
    ep = int(request.args.get("ep", "0"))
    n_action = int(request.args.get("n_action", "4"))
    stride = int(request.args.get("stride", str(n_action)))
    max_q = int(request.args.get("queries", "0"))          # 0 = the whole episode
    grip = request.args.get("gripper", "1") != "0"
    threading.Thread(target=_replay_loop, args=(mode, n_action, stride, max_q, grip, anchor, ep),
                     daemon=True).start()
    return jsonify({"started": True, "mode": mode, "anchor": anchor, "episode": ep,
                    "n_action": n_action, "stride": stride, "queries": max_q, "gripper": grip})


@app.route("/replay_stop", methods=["POST"])
def replay_stop():
    if STATE.get("replay"):
        STATE["replay"]["running"] = False
    return jsonify({"stopped": True})


@app.route("/dump", methods=["POST"])
def dump():
    """Freeze one live observation and this path's raw prediction for it. No motion.

    Everything the policy saw is written out -- the three JPEGs exactly as robot_service sent them, the state
    the core built, the prompt, the pose and joints -- plus the raw (16,20) chunk at a fixed seed. Feeding the
    same file to the offline path answers 'is it the same model seeing the same thing' before anyone argues
    about domain shift.
    """
    import numpy as _np
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    seed = int(request.args.get("seed", "99"))
    st, m_now, poses = INF.build_state([h0, h1])
    # torch.manual_seed does NOT give MPS and CUDA the same numbers, and this policy starts from noise, so a
    # cross-device comparison needs the SAME tensor, generated on the CPU and handed to both paths.
    noise = torch.randn(1, INF.policy.config.chunk_size, 20,
                        generator=torch.Generator().manual_seed(seed))
    try:
        INF.infer([h0, h1], noise=noise)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    L = INF.last
    path = os.path.expanduser(request.args.get("path", "~/live_dump.npz"))
    _np.savez_compressed(
        path,
        jpg_global=_np.frombuffer(base64.b64decode(h1["images"]["middle"]), _np.uint8),
        jpg_left=_np.frombuffer(base64.b64decode(h1["images"]["left"]), _np.uint8),
        jpg_right=_np.frombuffer(base64.b64decode(h1["images"]["right"]), _np.uint8),
        state20=st.astype(_np.float32), prompt=_np.array(INF.task),
        joints_prev=_np.asarray(h0["joints14"], _np.float64),
        joints_now=_np.asarray(h1["joints14"], _np.float64),
        tcp=_np.asarray(m_now, _np.float64), raw_action=L["act"].astype(_np.float64),
        noise=noise.numpy().astype(_np.float32),      # the exact tensor both paths must start from
        seed=_np.array(seed), ckpt=_np.array(os.path.basename(str(INF_CKPT).rstrip("/"))),
        denoise=_np.array(int(INF.policy.config.num_denoising_steps)))
    return jsonify({"wrote": path, "seed": seed,
                    "ckpt": os.path.basename(str(INF_CKPT).rstrip("/")),
                    "denoise": int(INF.policy.config.num_denoising_steps),
                    "state20": [round(float(x), 5) for x in st],
                    "raw_step0_L": [round(float(x), 5) for x in L["act"][0, 0:3]],
                    "raw_step0_R": [round(float(x), 5) for x in L["act"][0, 10:13]],
                    "prompt": INF.task})


@app.route("/denoise/<int:n>", methods=["POST"])
def set_denoise(n):
    INF.policy.config.num_denoising_steps = max(1, min(n, 50))
    return jsonify({"num_denoising_steps": INF.policy.config.num_denoising_steps})


def _denoise_ablation(ns, draws):
    """Does a cheaper sampler change the action, or only the clock?

    One observation pair is reused for every setting, and every setting is scored against the 10-step mean
    endpoint, so the comparison is sampler-vs-sampler with the policy, the scene and the state held fixed.
    """
    keep = INF.policy.config.num_denoising_steps
    STATE["denoise"] = {"running": True, "rows": []}
    try:
        h0 = _observe()
        time.sleep(OBS_GAP_S)
        h1 = _observe()
        if h0 is None or h1 is None:
            STATE["denoise"] = {"running": False, "error": "observe failed"}
            return
        ref = None
        for n in ns:
            INF.policy.config.num_denoising_steps = n
            ends, ms = [], []
            for j in range(draws):
                torch.manual_seed(4321 + 13 * j)
                t = time.perf_counter()
                INF.infer([h0, h1])
                ms.append((time.perf_counter() - t) * 1000.0)
                L = INF.last
                ends.append(_chunk_end(L))
            E = np.stack(ends)
            mean_end = E.mean(axis=0)
            spread = float(np.linalg.norm(E - mean_end, axis=-1).mean())
            if ref is None:
                ref = mean_end
            STATE["denoise"]["rows"].append({
                "steps": n,
                "infer_ms_p50": round(float(np.median(ms)), 1),
                "infer_ms_max": round(float(np.max(ms)), 1),
                "draw_spread_mm": round(spread, 2),
                "delta_vs_10step_mm": round(float(np.linalg.norm(mean_end - ref)), 2),
                "endpoint_mm": [round(float(x), 1) for x in mean_end],
                "latency_steps": round(float(np.median(ms)) / DT_STEP_MS, 2)})
    finally:
        INF.policy.config.num_denoising_steps = keep
        STATE["denoise"]["running"] = False
    print(f"[v4-smoke] denoise ablation: {STATE['denoise']['rows']}", flush=True)


@app.route("/denoise_ablation", methods=["POST"])
def denoise_ablation():
    if STATE["running"] or STATE.get("profile", {}).get("running"):
        return jsonify({"error": "stop the other run first"})
    ns = [int(x) for x in request.args.get("steps", "10,5,4,2").split(",")]
    draws = int(request.args.get("draws", "4"))
    threading.Thread(target=_denoise_ablation, args=(ns, draws), daemon=True).start()
    return jsonify({"started": True, "steps": ns, "draws": draws})


@app.route("/prompt_ab", methods=["POST"])
def prompt_ab():
    if STATE["running"] or STATE.get("ab", {}).get("running"):
        return jsonify({"error": "stop the loop first"})
    g = _grip_gate()
    if g:
        return jsonify(g)
    chunks = int(request.args.get("chunks", "1"))
    fixed = request.args.get("fixed")
    if fixed and fixed not in dict(TASKS):
        return jsonify({"error": f"unknown prompt {fixed}"})
    threading.Thread(target=_ab_loop, args=(chunks, fixed), daemon=True).start()
    return jsonify({"started": True, "mode": fixed or "six-prompt", "chunks": chunks})


@app.route("/prompt_ab_stop", methods=["POST"])
def prompt_ab_stop():
    if STATE.get("ab"):
        STATE["ab"]["running"] = False
    return jsonify({"stopped": True})


@app.route("/clamp/<mm>/<deg>", methods=["POST"])
def clamp(mm, deg):
    return jsonify(set_limits(mm, deg))


# [2026-10-03 user] "k 수도 내가 선택 + streaming / 보간 켜기 + streaming % 직접": live mode control (refused while running).
# k -> step mode n_action AND streaming STREAM_K (rows of each chunk that enter the timeline); stream on/off; speed % -> STREAM_SPEED;
# interp -> MIT bridge goal interpolation (/set_speed goal_interp -- ONE bridge, so this is shared by every UI on :8021).
@app.route("/mode", methods=["POST"])
def mode():
    global STREAM, STREAM_SPEED, STREAM_K, PREFETCH, PASS_THROUGH, DWELL_TIMEOUT_S, PREFETCH_LEAD, PASS_LAST, PLAN_MODE, PLAN_EMA, PLAN_XFADE, PLAN_MIN_EXEC, PLAN_SMOOTH_DEG
    if STATE["running"] or STATE["busy"]:
        return jsonify({"ok": False, "err": "running -- STOP first"})
    a = request.args; out = {"ok": True}
    if "k" in a:
        k = max(1, min(int(a["k"]), 16)); INF.n_action = k; STREAM_K = k
    if "stream" in a:
        STREAM = a["stream"] == "1"
    if "speed" in a:
        STREAM_SPEED = max(0.05, min(float(a["speed"]) / 100.0, 1.0))
    if "pass" in a:                                     # [2026-10-03 user] step mode: intermediate waypoints pass at V4_PASS_TCP_MM
        PASS_THROUGH = a["pass"] == "1"
    if "dwell" in a:                                    # reach-test timeout per (last) waypoint, seconds
        DWELL_TIMEOUT_S = max(0.05, min(float(a["dwell"]), 5.0))
    if "rtc" in a:                                      # [2026-10-03 user] Real-Time Chunking (infer_core_v4.RTC_ON), any mode
        import infer_core_v4 as _IC
        _IC.RTC_ON = a["rtc"] == "1"; INF._rtc_prev = None
    if "prefetch" in a:                                 # [2026-10-03 user] step mode: next observe+infer during the last waypoint
        PREFETCH = a["prefetch"] == "1"; _PF["thread"] = None; _PF["res"] = None
    if "plan" in a:                                     # [2026-10-06] PLAN mode (always-one-forward + 100 Hz playback)
        PLAN_MODE = a["plan"] == "1"
    if "planema" in a:
        PLAN_EMA = max(0.0, min(float(a["planema"]), 1.0))
    if "planxf" in a:
        PLAN_XFADE = max(0, min(int(a["planxf"]), 16))
    if "plansm" in a:
        PLAN_SMOOTH_DEG = max(0, min(int(a["plansm"]), 5))
    if "planmin" in a:
        PLAN_MIN_EXEC = max(1, min(int(a["planmin"]), 16))
    if "passlast" in a:                                 # [2026-10-04 user] last waypoint passes through while a prefetch runs
        PASS_LAST = a["passlast"] == "1"
    if "pflead" in a:                                   # [2026-10-04 user] prefetch starts this many waypoints before the last
        PREFETCH_LEAD = max(-1, min(int(a["pflead"]), 15))
    if "remote" in a:                                   # [2026-10-03] policy forward on the 5090 server (V4_REMOTE_CKPT from the loader)
        import infer_core_v4 as _IC
        if a["remote"] == "1":
            if not _IC.REMOTE_CKPT:
                out.update(ok=False, err="no V4_REMOTE_CKPT (UI not started with V4_REMOTE_INFER)")
            else:
                _IC.REMOTE_INFER = _REMOTE_URL0 or os.environ.get("V4_REMOTE_INFER") or "http://100.64.0.5:8791"   # [10-04] the URL the UI was started with (a remote=0 pops the env)
        else:
            _IC.REMOTE_INFER = ""; os.environ.pop("V4_REMOTE_INFER", None)   # a later checkpoint pick (ckpt_select) stays local too
    if "interp" in a:
        try:
            r = _SESS.post(ROBOT + "/set_speed", json={"goal_interp": a["interp"] == "1"}, timeout=5).json()
            out["bridge_goal_interp"] = r.get("goal_interp")
        except Exception as e:
            out.update(ok=False, err=f"bridge /set_speed failed: {e}")
    out.update(k=INF.n_action, stream=STREAM, speed_pct=round(STREAM_SPEED * 100, 1), stream_k=STREAM_K)
    print(f"[v4-smoke] mode -> {out}", flush=True)
    return jsonify(out)


# [2026-10-03 user] "pink numerical learned curobo 중 하나 선택": live IK backend switch (infer_core_v4.IK_BACKEND, read per solve).
@app.route("/ik/<name>", methods=["POST"])
def ik_backend(name):
    import infer_core_v4 as _IC
    if name not in ("pink", "numerical", "learned", "curobo"):
        return jsonify({"ok": False, "err": f"unknown IK backend {name}"})
    if STATE["running"] or STATE["busy"]:
        return jsonify({"ok": False, "err": "running -- STOP first"})
    if name == "curobo":
        try:
            h = _SESS.get(_IC.CUROBO_URL + "/health", timeout=5).json()
        except Exception as e:
            return jsonify({"ok": False, "err": f"cuRobo server {_IC.CUROBO_URL} not reachable: {e}"})
    _IC.IK_BACKEND = name; os.environ["IK_BACKEND"] = name
    print(f"[v4-smoke] IK backend -> {name}", flush=True)
    return jsonify({"ok": True, "ik_backend": name})


@app.route("/steps/<n>", methods=["POST"])
def steps(n):
    INF.n_action = max(1, min(int(n), 16))
    return jsonify({"n_action": INF.n_action})


@app.route("/prompt_sweep", methods=["POST"])
def prompt_sweep():
    """No motion. One scene, all six prompts, K draws each: does the instruction change what the model wants?

    The sampler starts from noise, so a single draw per prompt cannot answer this -- two runs of the SAME
    prompt already differ. So the comparison is between-prompt distance of the MEAN 16-step endpoint against
    the within-prompt spread of the draws. Anything below the noise floor is not a language effect.
    """
    K = int(request.args.get("k", os.environ.get("V4_SWEEP_K", "4")))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    keep_task, keep_n = INF.task, INF.n_action
    INF.n_action = 1                                   # nothing is sent; this only limits the returned slice
    # Two passes on purpose. The endpoint must be the SAME step for every prompt, and valid_step is not
    # constant across prompts or draws, so the common horizon can only be known after all draws are in.
    # Taking each draw's own last index instead is what produced the retracted "DELTA ignores the prompt's
    # first colour" reading; see _chunk_end.
    raw, ends = {}, {}
    horizon = None
    try:
        for label, text in TASKS:
            INF.task = text
            keeps = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)       # same noise sequence for every prompt: only text varies
                INF.infer([h0, h1])
                L = INF.last
                keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_tcp": L["cur_tcp"].copy(),
                              "valid_step": int(L["valid_step"])})
                v = int(L["valid_step"]) - 1
                horizon = v if horizon is None else min(horizon, v)
            raw[label] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        INF.task, INF.n_action = keep_task, keep_n
    for label, keeps in raw.items():
        ends[label] = np.stack([_chunk_end(L, horizon) for L in keeps])   # (K, 6) mm at the common step

    labels = [t[0] for t in TASKS]
    means = {k: v.mean(axis=0) for k, v in ends.items()}
    within = float(np.mean([np.linalg.norm(v - v.mean(axis=0), axis=-1).mean() for v in ends.values()]))
    sem = float(np.mean([v.std(axis=0, ddof=1).mean() / np.sqrt(K) for v in ends.values()])) if K > 1 else 0.0
    pairs = {f"{a}-{b}": round(float(np.linalg.norm(means[a] - means[b])), 2)
             for i, a in enumerate(labels) for b in labels[i + 1:]}
    bt = float(np.mean(list(pairs.values())))
    return jsonify({
        "dry_run": True, "draws_per_prompt": K,
        "endpoint_mm_per_prompt": {k: [round(float(x), 1) for x in v] for k, v in means.items()},
        "between_prompt_mean_mm": round(bt, 2),
        "within_prompt_spread_mm": round(within, 2),
        "sem_of_means_mm": round(sem, 2),
        "bt_over_sem": round(bt / max(sem, 1e-9), 2),
        "verdict": ("prompt changes the intended trajectory" if bt > 3 * sem
                    else "NOT distinguishable from sampling noise"),
        "common_horizon_step": horizon,
        "pairs_mm": pairs})


@app.route("/task/<int:i>", methods=["POST"])
def set_task(i):
    import infer_core_v4 as _IC
    if _IC.ARM_ONLY:                                             # [2026-10-03] one-arm model: one training instruction only
        return jsonify({"task_label": "ARM_ONLY", "task": INF.task, "error": "V4_ARM_ONLY model: task fixed to V4_TASK (no stacking prompts)"})
    if 0 <= i < len(TASKS):
        INF.task = TASKS[i][1]
        STATE["task"] = TASKS[i][0]
    return jsonify({"task_label": STATE.get("task"), "task": INF.task})


@app.route("/arm/<which>", methods=["POST"])
def set_arm(which):
    if which in ("left", "right", "both"):
        STATE["arm"] = which
    return jsonify({"arm": STATE["arm"]})


@app.route("/observe_only", methods=["POST"])
def observe_only():
    """Dry run: infer and report, send nothing."""
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(OBS_GAP_S)
    h1 = _observe()
    try:
        INF.infer([h0, h1])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    L = INF.last
    return jsonify({"dry_run": True, "kept": int(L["kept"]), "valid": int(L["valid_step"]),
                    "pred_mm_step0": [round(float(L["pred_mm"][0, r]), 3) for r in (0, 1)],
                    "pred_deg_step0": [round(float(L["pred_deg"][0, r]), 3) for r in (0, 1)],
                    "cmd_mm_step0": [round(float(L["cmd_mm"][0, r]), 3) for r in (0, 1)],
                    "fkerr_mm": [round(float(L["fkerr_mm"][0, r]), 2) for r in (0, 1)],
                    "state20": [round(float(x), 4) for x in L["state"]]})


@app.route("/step", methods=["POST"])
def step():
    g = _grip_gate()
    if g:
        return jsonify(g)
    if not _lock.acquire(blocking=False):
        return jsonify({"error": "busy"})
    try:
        STATE["busy"] = True
        return jsonify(_one_press())
    finally:
        STATE["busy"] = False
        _lock.release()


_LAST_FRAME = {}   # [2026-09-30] last good jpeg per cam, display only


@app.route("/frame/<cam>")
def frame(cam):
    """[2026-09-30 user: feeds flicker] robot_service /frame does a full get_observation, so one camera timing out (right wrist,
    ~9 % of reads after the 15:51 USB re-enumeration) makes EVERY feed return 500 and blink. For DISPLAY only, a failed read
    serves this cam's last good jpeg (header X-Frame-Stale: 1). Inference does not use this route."""
    try:
        r = requests.get(f"{ROBOT}/frame/{cam}", timeout=5)
        if r.status_code == 200 and r.content:
            _LAST_FRAME[cam] = r.content
            return (r.content, 200, {"Content-Type": r.headers.get("Content-Type", "image/jpeg")})
    except Exception:  # noqa: BLE001
        pass
    if cam in _LAST_FRAME:
        return (_LAST_FRAME[cam], 200, {"Content-Type": "image/jpeg", "X-Frame-Stale": "1"})
    return ("frame unavailable", 500)


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>v4 smoke</title><style>
body{font-family:-apple-system,sans-serif;margin:18px;background:#111;color:#eee}
button{font-size:15px;padding:9px 14px;margin:3px;border-radius:7px;border:1px solid #444;background:#222;color:#eee}
button.big{font-size:19px;padding:14px 22px}
#estop{background:#a11;border-color:#f55;font-weight:700}
.on{background:#2a5;border-color:#6d9}
table{border-collapse:collapse;margin-top:10px}td,th{border:1px solid #444;padding:5px 9px;font-size:13px}
pre{background:#181818;padding:10px;border-radius:6px;font-size:12px;max-height:230px;overflow:auto}
img{width:220px;border:1px solid #333;border-radius:5px}
</style></head><body>
<div id=hra style="display:none;background:#7a1010;color:#fff;font:bold 15px monospace;padding:8px 12px;margin:0 0 8px;border-radius:6px"></div>
<div style="margin:0 0 8px;color:#fc9">HRA 출발 자세: 손목 <input id=hrapitch value=35 style="width:3em">° 아래, 높이 <input id=hradz value=0 style="width:3em"> mm
  <button class=big onclick="hrastart()">HRA 출발 자세로 이동</button> <span id=hrainfo style="font-size:12px"></span>
  <span style="font-size:11px;color:#aaa">(학습 166개 시작 자세 p50 36°; 이동 후 큐브를 그리퍼 앞 5 cm·안쪽 15-20 cm에 놓고 RUN)</span>
  <br><b>UMI 출발 자세</b>: <select id=spose><option value="">umi_start_pose.json</option></select> 오른쪽 <input id=soR value=0 style="width:3em"> 뒤 <input id=soB value=0 style="width:3em"> 아래 <input id=soD value=0 style="width:3em"> mm <span style="font-size:11px;color:#aaa">(로봇 손목 카메라 화면 기준, 음수 = 반대)</span>
  <button class=big onclick="umistart(false)">UMI 출발 자세로 이동</button>
  <button class=big style="background:#2a6" onclick="umistart(true)">UMI 출발 자세 → RESET ANCHOR → RUN loop</button> <span id=umiinfo style="font-size:12px"></span>
  <br><b>손목 영상 확대</b> (모델 입력만, f'=s·f, C922 주점 기준): s <input id=wzoom value=1.00 style="width:4em">
  <button onclick="setwz()">적용</button> <button onclick="wsnap()">손목 사진 저장</button> <span id=wzinfo style="font-size:12px"></span>
  <br><b>큐브 크기 정지</b> (오른손 C922 원본, 빨간 큐브 √면적 px ≥ 값이면 RUN 정지, 0 = 끔): <input id=cstop value=0 style="width:4em"> px
  <label><input type=checkbox id=cfollow checked> 따라가기(작아지면 재시작)</label>
  <button onclick="setcs()">적용</button> <button onclick="getcs()">지금 큐브 크기</button> <span id=csinfo style="font-size:12px"></span>
  <img id=wzimg style="width:112px;vertical-align:middle;display:none"></div>
<h2>reBot real-robot smoke <span id=model style="font-size:13px;color:#9ad">loading model…</span></h2>
<div id=modelbar style="font-size:13px;margin:-6px 0 8px 0;padding:4px 8px;border-radius:4px;background:#122;color:#9ad"></div>
<div id=sent style="font-size:12px;margin:0 0 8px 0;padding:4px 8px;border-radius:4px;background:#121;color:#9d9;font-family:ui-monospace,monospace"></div>
<div>
  <button id=estop class=big onclick="post('/estop')">E-STOP</button>
  <button onclick="post('/clear_estop')">clear estop</button>
  <button onclick="post('/reenable')">reenable</button>
  <button onclick="post('/connect')">connect</button>
  <button onclick="post('/rest')">REST</button>
  <button onclick="post('/train_start')">TRAIN START pose</button>
  <button onclick="post('/gripper_mode/predict')">gripper: PREDICT</button>
  <button onclick="post('/gripper_mode/hold')">gripper: HOLD</button>
</div>
<div style="margin-top:6px;font-size:12px;color:#888">gripper zero (jaw closed should read ~0):
  L <button onclick="gz('left','release')">1 release</button><button onclick="gz('left','set')">2 set zero</button>
  &nbsp; R <button onclick="gz('right','release')">1 release</button><button onclick="gz('right','set')">2 set zero</button>
</div>
<div style="margin-top:10px" id=tasks></div>
<div style="margin-top:8px">
  <span style="font-size:12px;color:#888">safety: arm
  <button onclick="post('/arm/left')">L</button><button onclick="post('/arm/right')">R</button>
  <button onclick="post('/arm/both')">both</button></span>
  &nbsp;|&nbsp;
  <button onclick="post('/observe_only')">DRY RUN (no motion)</button>
  <button class=big onclick="post('/prompt_sweep')">PROMPT SWEEP (no motion)</button>
</div>
<div style="margin-top:8px">
  clamp: <button onclick="post('/clamp/3/1')">3mm/1&deg;</button>
  <button onclick="post('/clamp/10/2')">10mm/2&deg;</button>
  <button onclick="post('/clamp/25/4')">25mm/4&deg;</button>
  &nbsp;steps: <button onclick="post('/steps/1')">1</button>
  <button onclick="post('/steps/4')">4</button>
  <button onclick="post('/steps/8')">8</button>
  <button class=big onclick="post('/step')">MOVE 1 CHUNK</button>
  <button class=big onclick="post('/goto_train_start')">GO TO TRAIN START</button>
  <button onclick="post('/reset_anchor')">RESET ANCHOR</button>
  <button class=big id=run onclick="post('/run')">RUN loop</button>
  <button class=big onclick="post('/stop')">STOP</button>
  <button class=big onclick="post('/prompt_ab')">PROMPT A/B (6 prompts)</button>
  <button class=big onclick="post('/prompt_ab?fixed=RBP')">CONTROL (RBP x6)</button>
  <button onclick="post('/prompt_ab_stop')">A/B stop</button>
  <button class=big onclick="post('/profile?n=20')">LATENCY PROFILE (20)</button>
</div>
<div style="margin-top:8px">
  <span style="color:#9ad">replay episode</span>
  <button id=ep0 onclick="setep(0)">ep0</button><button id=ep7 onclick="setep(7)">ep7</button>
  <button id=ep42 onclick="setep(42)">ep42</button>
  &nbsp;anchor <button id=ancsource onclick="setanc('source')">source</button>
  <button id=ancmeasured onclick="setanc('measured')">measured</button>
  &nbsp;
  <button class=big onclick="rep('joints')">REPLAY joints (IK 없음)</button>
  <button class=big onclick="rep('source')">REPLAY teleop TCP→IK</button>
  <button class=big onclick="rep('delta')">REPLAY delta</button>
  <button class=big onclick="rep('umi')">REPLAY relative(umi)</button>
  <button onclick="post('/replay_stop')">replay stop</button>
  <span id=repcfg style="color:#9ad;font-size:12px"></span>
  <button class=big onclick="post('/denoise_ablation')">DENOISE ABLATION (no motion)</button>
</div>
<div style="margin-top:8px;color:#9fc">mode: k <select id=ctk><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option><option>9</option><option>10</option><option>11</option><option>12</option><option>13</option><option>14</option><option>15</option><option>16</option></select>
  &nbsp;<label><input type=checkbox id=ctstream> streaming</label>
  speed <input id=ctspeed type=number min=5 max=100 step=5 style="width:4em">%
  &nbsp;<label><input type=checkbox id=ctinterp> 보간 (bridge, 모든 UI 공통)</label>
  &nbsp;<label><input type=checkbox id=ctpass> pass-through</label>
  dwell <input id=ctdwell type=number min=0.05 max=5 step=0.05 style="width:4em">s
  &nbsp;<label><input type=checkbox id=ctrtc> RTC</label>
  &nbsp;<label style="color:#6f6"><input type=checkbox id=ctplan> PLAN(연속)</label>
  &nbsp;<label><input type=checkbox id=ctprefetch> prefetch (step: 마지막 waypoint 동안 다음 추론)</label>
  &nbsp;<label><input type=checkbox id=ctremote> 5090 추론</label>
  &nbsp;IK <select id=ctik><option>pink</option><option>numerical</option><option>learned</option><option>curobo</option></select>
  <button class=big onclick="ctapply()">APPLY MODE</button> <span id=ctinfo style="font-size:12px"></span></div>
<div style="margin-top:8px;color:#fc9">checkpoint: <select id=cksel></select>
  <button class=big onclick="ckload()">LOAD CKPT</button> <span id=ckinfo style="font-size:12px"></span></div>
<div id=trialbox style="margin-top:8px;padding:8px 10px;border:1px solid #365;border-radius:6px;background:#0f1a14;color:#cfe">
  <b>TRIAL 기록</b> (시도 1회 끝날 때마다 저장)
  &nbsp;layout <input id=trlayout placeholder="예: L1" style="width:5em">
  &nbsp;실제로 쌓은 순서 <select id=trorder><option value="">-</option><option>RBP</option><option>RPB</option><option>BRP</option><option>BPR</option><option>PRB</option><option>PBR</option></select>
  &nbsp;쌓은 단수 <button id=trl0 onclick="trlv(0)">0</button><button id=trl1 onclick="trlv(1)">1</button><button id=trl2 onclick="trlv(2)">2</button><button id=trl3 onclick="trlv(3)">3</button>
  &nbsp;<label><input type=checkbox id=trwrong> 색 틀림</label>
  &nbsp;실패 이유 <select id=trreason><option value="">-</option><option value=grasp_miss>잡기 실패</option><option value=wrong_cube>다른 큐브/색</option>
    <option value=drop>운반 중 떨어뜨림</option><option value=place_off>놓는 위치 어긋남</option><option value=knocked_over>쌓은 것 무너뜨림</option>
    <option value=stall>멈춤/진행 안 함</option><option value=ik_limit>IK/한계 정지</option><option value=collision>충돌</option><option value=estop>E-STOP</option><option value=other>기타</option></select>
  &nbsp;메모 <input id=trnote style="width:14em">
  <button class=big onclick="trsave()">SAVE TRIAL</button>
  <div id=trsum style="font-size:12px;margin-top:4px;color:#9d9;font-family:ui-monospace,monospace"></div>
</div>
<div id=clamp style="margin-top:6px;color:#9ad;font-size:13px"></div>
<table id=t><thead><tr><th>arm</th><th>pred mm</th><th>pred deg</th><th>cmd mm</th><th>cmd deg</th>
<th>actual mm</th><th>actual deg</th><th>dir cos</th><th>FK err mm</th></tr></thead><tbody></tbody></table>
<pre id=out>press DRY RUN first</pre>
<div style="display:flex;gap:10px;margin-top:8px">
  <div><div>middle → global</div><img src=/frame/middle id=m></div>
  <div><div>left wrist</div><img src=/frame/left id=l></div>
  <div><div>right wrist</div><img src=/frame/right id=r></div>
</div>
<script>
const REP={ep:0,anchor:'source'};
function rep(m){post('/replay?mode='+m+'&ep='+REP.ep+'&anchor='+REP.anchor);}
// update the DOM in the click itself: routing this through refresh() made the buttons look dead whenever the
// /status fetch was slow or failed, because the label never got as far as being rewritten
function repui(){
  [0,7,42].forEach(function(e){var b=document.getElementById('ep'+e); if(b) b.className=(REP.ep===e)?'on':'';});
  ['source','measured'].forEach(function(a){var b=document.getElementById('anc'+a);
    if(b) b.className=(REP.anchor===a)?'on':'';});
  var rc=document.getElementById('repcfg'); if(rc) rc.textContent='  → ep'+REP.ep+' / anchor='+REP.anchor;
}
function setep(e){REP.ep=e;repui();}
function setanc(a){REP.anchor=a;repui();}
const TASKS=[["RBP", "Stack the red cube on the bottom, blue cube in the middle, and purple cube on the top, on the plate."], ["RPB", "Stack the red cube on the bottom, purple cube in the middle, and blue cube on the top, on the plate."], ["BRP", "Stack the blue cube on the bottom, red cube in the middle, and purple cube on the top, on the plate."], ["BPR", "Stack the blue cube on the bottom, purple cube in the middle, and red cube on the top, on the plate."], ["PRB", "Stack the purple cube on the bottom, red cube in the middle, and blue cube on the top, on the plate."], ["PBR", "Stack the purple cube on the bottom, blue cube in the middle, and red cube on the top, on the plate."]];
document.getElementById('tasks').innerHTML = 'prompt: ' + TASKS.map(function(t,i){
  return '<button id=tk' + i + ' title="' + t[1] + '">' + t[0] + '</button>';}).join('');
TASKS.forEach(function(t,i){
  document.getElementById('tk'+i).addEventListener('click', function(){ post('/task/'+i); });});
async function post(p){document.getElementById('out').textContent='...';
  const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  const j=await r.json(); document.getElementById('out').textContent=JSON.stringify(j,null,1); refresh();}
async function gz(arm,stage){
  if(stage==='set' && !confirm(arm+' jaw fully closed by hand? store zero here')) return;
  await post('/gripper_zero/'+arm+'/'+stage);
  if(stage==='release') alert(arm+' jaw torque OFF: close the jaw fully by hand, then press "2 set zero"');}
async function refresh(){const j=await (await fetch('/status')).json();
  {const b=document.getElementById('hra'); if(j.hra){b.style.display='block'; b.textContent='MODE: '+j.hra.mode+'  |  TASK: '+j.hra.task+'  |  LEFT ARM: '+j.hra.left+'  |  JAWS: '+j.hra.jaws+'  |  stop: '+(j.hra.max_run_s>0?j.hra.max_run_s+' s or ':'no time limit, ')+'<'+j.hra.still_mm+' mm x'+j.hra.still_n+'  |  travel '+j.hra.travel_mm+'/'+j.hra.max_travel_cap+' mm  cycle max '+j.hra.max_cycle_mm+'/'+j.hra.max_cycle_cap+' mm  |  IK '+j.hra.ik+'  |  gate refusals '+j.hra.gate_refusals+'  left dev '+j.hra.max_left_dev_rad+' rad'+(j.hra.last_pred_mm!=null?'  |  pred '+j.hra.last_pred_mm+' mm':'');} else b.style.display='none';}
  if(j.model){
    const m=j.model, off=(m.clamp && m.clamp.mm<=0);
    document.getElementById('model').textContent = m.ckpt+'  ·  '+m.action_mode
      +(m.action_mode==='umi' ? ' (current-anchor: T_now @ A[k])' : ' (sequential delta)');
    // the decode and the clamp change what reaches the arm, so they are stated on the page, not buried in
    // /status: two checkpoints that differ only in these have looked identical here before
    document.getElementById('modelbar').innerHTML =
      '<b>ckpt</b> '+m.ckpt+' &nbsp; <b>decode</b> '+m.action_mode
      +' &nbsp; <b>clamp</b> '+(off?'<span style="color:#f96">OFF</span>':(m.clamp.mm+' mm / '+m.clamp.deg+'&deg; per step'))
      +' &nbsp; <b>exec k</b> '+m.exec_k+' &nbsp; <b>n_action</b> '+m.n_action
      +' &nbsp; <b>gripper</b> '+m.gripper+' &nbsp; <b>dtype</b> '+m.dtype;
  }
  document.getElementById('clamp').textContent='arm='+j.ui.arm+'   '+(j.last&&j.last.clamp?j.last.clamp:'')
    +'   presses='+j.ui.presses+'   running='+j.ui.running+'   cycles='+j.ui.cycles
    +'   travel='+JSON.stringify((j.ui.travel_mm||[]).map(x=>Math.round(x)))+'mm'
    +(j.ui.stop_reason?('   ['+j.ui.stop_reason+']'):'')
    +(j.replay?('   REPLAY '+j.replay.mode+' '+j.replay.done+'/'+j.replay.total_queries
                +(j.replay.running?' running':' stopped')):'')
    +'   robot='+JSON.stringify(j.robot);
  document.getElementById('run').className = j.ui.running ? 'big on' : 'big';
  repui();
  TASKS.forEach((t,i)=>{const b=document.getElementById('tk'+i); if(b) b.className = (t[0]===j.ui.task)?'on':'';});
  if(j.sent){
    const NM=['L_pan','L_lift','L_elbow','L_wflex','L_wyaw','L_wroll',
              'R_pan','R_lift','R_elbow','R_wflex','R_wyaw','R_wroll'];
    document.getElementById('sent').innerHTML =
      '<b>last /execute_step</b> &nbsp; substep '+j.sent.substep+'/'+j.sent.of
      + ' &nbsp; grip cmd L '+j.sent.grip_cmd[0]+' R '+j.sent.grip_cmd[1]+' <span style="color:#777">(0..45)</span><br>'
      + NM.map(function(nm,i){return nm+' <b>'+j.sent.arm_deg[i].toFixed(1)+'</b>&deg;';}).join(' &nbsp; ');
  }
  const tb=document.querySelector('#t tbody'); tb.innerHTML='';
  ((j.last&&j.last.rows)||[]).forEach(x=>{const tr=document.createElement('tr');
    tr.innerHTML='<td>'+x.arm+'</td><td>'+x.pred_mm+'</td><td>'+x.pred_deg+'</td><td>'+x.cmd_mm+'</td><td>'
      +x.cmd_deg+'</td><td>'+x.actual_mm+'</td><td>'+x.actual_deg+'</td><td>'+x.dir_cos+'</td><td>'+x.fkerr_mm+'</td>';
    tb.appendChild(tr);});
  ['m','l','r'].forEach((id,i)=>{const cam=['middle','left','right'][i];
    document.getElementById(id).src='/frame/'+cam+'?t='+Date.now();});}
async function cklist(){const j=await (await fetch('/ckpts')).json(); const el=document.getElementById('cksel');
  const prev=el.value; el.innerHTML='<option value=auto>auto (newest)</option>'+j.steps.slice().reverse().map(s=>`<option value=${s}>${parseInt(s,10)/1000}k</option>`).join('');
  el.value=prev||j.pinned||'auto'; if(!el.value) el.value='auto';
  const rl=j.remote_url?(Array.isArray(j.remote_loaded)?j.remote_loaded.join(', '):j.remote_loaded):null;
  document.getElementById('ckinfo').innerHTML=`<b style="font-size:15px;color:#6f6">▶ 실제 로드: ${j.serving_k}</b>`
    +` | 선택(pinned): ${j.pinned?parseInt(j.pinned,10)/1000+'k':'auto'}`
    +(j.loading?` | <b style="color:#ff6">⏳ ${parseInt(j.loading,10)/1000}k 로딩 중</b>`:'')
    +(j.remote_url?` | 원격 ${j.remote_url.replace('http://','')}: ${rl}`:' | 추론 Mac')+` | ${j.run}`;}
async function hrastart(){document.getElementById('hrainfo').textContent='이동 중...';
  const j=await (await fetch(`/hra_start_pose?pitch=${document.getElementById('hrapitch').value}&dz_mm=${document.getElementById('hradz').value}`,{method:'POST'})).json();
  document.getElementById('hrainfo').textContent=j.error?('실패: '+j.error):`완료: 손목 ${j.pitch_now_deg}° 아래, roll ${j.roll_now_deg}°, TCP ${j.tcp_mm_now} mm`;
  document.getElementById('out').textContent=JSON.stringify(j,null,1);}
function csmsg(j){return `정지 기준 ${j.stop_px} px (재시작 < ${j.resume_below_px} px, 따라가기 ${j.follow?'ON':'OFF'}${j.waiting?' · 대기 중':''}, 재시작 ${j.resumes}회) | 지금 큐브 ${j.cube_px_now} px`;}
async function setcs(){const j=await (await fetch('/cube_stop?px='+document.getElementById('cstop').value+'&follow='+(document.getElementById('cfollow').checked?1:0),{method:'POST'})).json();
  document.getElementById('csinfo').textContent=csmsg(j);}
async function getcs(){const j=await (await fetch('/cube_stop')).json(); document.getElementById('csinfo').textContent=csmsg(j);}
fetch('/cube_stop').then(r=>r.json()).then(j=>{document.getElementById('cstop').value=j.stop_px; document.getElementById('cfollow').checked=j.follow; document.getElementById('csinfo').textContent=csmsg(j);});
async function setwz(){const j=await (await fetch('/wrist_zoom?s='+document.getElementById('wzoom').value,{method:'POST'})).json();
  document.getElementById('wzinfo').textContent=j.error?('실패: '+j.error):('현재 확대 '+j.wrist_zoom.toFixed(2)+'x');}
async function wsnap(){const j=await (await fetch('/wrist_snap',{method:'POST'})).json();
  if(j.error){document.getElementById('wzinfo').textContent='실패: '+j.error;return;}
  document.getElementById('wzinfo').textContent=`저장 (확대 ${j.zoom.toFixed(2)}x, TCP ${j.tcp_mm} mm)`; const im=document.getElementById('wzimg'); im.src='data:image/jpeg;base64,'+j.img; im.style.display='inline';}
fetch('/wrist_zoom').then(r=>r.json()).then(j=>{if(j.wrist_zoom){document.getElementById('wzoom').value=j.wrist_zoom.toFixed(2); document.getElementById('wzinfo').textContent='현재 확대 '+j.wrist_zoom.toFixed(2)+'x';}});
fetch('/start_poses').then(r=>r.json()).then(j=>{const sel=document.getElementById('spose'); (j.poses||[]).forEach(p=>{const o=document.createElement('option'); o.value=p.name; o.textContent=p.name.replace('.json','')+' ('+p.tcp_mm.join(', ')+' mm)'; sel.appendChild(o);});});
const UMI_START='saved=1&steps=10';
async function umistart(andRun){const el=document.getElementById('umiinfo'); el.textContent='UMI 출발 자세로 이동 중...';
  const j=await (await fetch('/hra_start_pose?'+UMI_START+'&pose='+encodeURIComponent(document.getElementById('spose').value)+'&right_mm='+document.getElementById('soR').value+'&back_mm='+document.getElementById('soB').value+'&down_mm='+document.getElementById('soD').value,{method:'POST'})).json();
  document.getElementById('out').textContent=JSON.stringify(j,null,1);
  if(j.error){el.textContent='실패: '+j.error; return;}
  el.textContent=`도착: TCP ${j.tcp_mm_now} mm, 손목 ${j.pitch_now_deg}°, roll ${j.roll_now_deg}°`;
  if(andRun){await new Promise(r=>setTimeout(r,1000));
    const ra=await (await fetch('/reset_anchor',{method:'POST'})).json(); el.textContent+=' → RESET ANCHOR';
    const rr=await (await fetch('/run',{method:'POST'})).json(); el.textContent+=rr.error?(' → RUN 실패: '+rr.error):' → RUN loop';
    document.getElementById('out').textContent=JSON.stringify({start:j,reset_anchor:ra,run:rr},null,1);}}
async function ckload(){const v=document.getElementById('cksel').value;
  const j=await (await fetch('/ckpt_select/'+v,{method:'POST'})).json(); document.getElementById('out').textContent=JSON.stringify(j,null,1);}
cklist(); setInterval(cklist,5000);
let CTINIT=false, CTSIG='';
async function ctsync(force){const j=await (await fetch('/status')).json(); const c=j.model&&j.model.ctl; if(!c) return;
  document.getElementById('ctinfo').textContent=`now: k ${c.k} | ${c.stream?'STREAM '+c.speed_pct+'%':'STEP'} | 보간 ${c.goal_interp?'ON':'OFF'} | RTC ${c.rtc?'ON'+(c.rtc_last?' (prev k'+(c.rtc_last.prev_idx+1)+', delay '+c.rtc_last.delay+', left '+c.rtc_last.leftover+')':''):'OFF'} | prefetch ${c.prefetch?'ON':'OFF'} | pass ${c.pass?'ON':'OFF'} dwell ${c.dwell_s}s | IK ${c.ik} | 추론 ${c.remote?'원격'+(c.remote_last&&c.remote_last.rt_ms?' '+c.remote_last.rt_ms+' ms':''):'Mac MPS'}`;
  // [2026-10-03] re-sync the form whenever the SERVER state changes (UI restarted / another tab applied), not only on page load:
  // a stale form + APPLY MODE silently turned 5090 inference off after a UI restart
  const sig=JSON.stringify([c.k,c.stream,c.speed_pct,c.goal_interp,c.ik,c.remote,c.prefetch,c.rtc,c.pass,c.dwell_s]);
  if(CTINIT&&!force&&sig===CTSIG) return; CTINIT=true; CTSIG=sig;
  document.getElementById('ctk').value=String(c.k); document.getElementById('ctstream').checked=!!c.stream;
  document.getElementById('ctspeed').value=c.speed_pct; document.getElementById('ctik').value=c.ik; document.getElementById('ctremote').checked=!!c.remote; document.getElementById('ctprefetch').checked=!!c.prefetch; document.getElementById('ctrtc').checked=!!c.rtc; document.getElementById('ctplan').checked=!!c.plan; document.getElementById('ctpass').checked=!!c.pass; document.getElementById('ctdwell').value=c.dwell_s; document.getElementById('ctinterp').checked=!!c.goal_interp;}
async function ctapply(){const q=new URLSearchParams({k:document.getElementById('ctk').value,
  stream:document.getElementById('ctstream').checked?'1':'0', speed:document.getElementById('ctspeed').value,
  interp:document.getElementById('ctinterp').checked?'1':'0', remote:document.getElementById('ctremote').checked?'1':'0', prefetch:document.getElementById('ctprefetch').checked?'1':'0', rtc:document.getElementById('ctrtc').checked?'1':'0', plan:document.getElementById('ctplan').checked?'1':'0',
  pass:document.getElementById('ctpass').checked?'1':'0', dwell:document.getElementById('ctdwell').value});
  const j=await (await fetch('/mode?'+q,{method:'POST'})).json();
  const k=await (await fetch('/ik/'+document.getElementById('ctik').value,{method:'POST'})).json();
  document.getElementById('out').textContent=JSON.stringify({mode:j,ik:k},null,1); ctsync(true);}
ctsync(true); setInterval(()=>ctsync(false),5000);
repui(); setInterval(refresh,3000); refresh();

var TR={lv:null};
function trlv(v){TR.lv=v;[0,1,2,3].forEach(function(i){var b=document.getElementById('trl'+i); if(b) b.className=(i===v)?'on':'';});}
async function trsum(){try{const j=await (await fetch('/trials')).json(); const c=(j.by_ckpt||[]).find(function(b){return b.ckpt===j.current_ckpt;});
  document.getElementById('trsum').textContent='log '+j.log+' | 전체 '+j.n+'회 | 현재 ckpt '+(c?(c.n+'회, 3단 성공 '+c.success+' ('+Math.round(100*c.success/c.n)+'%), 단수 0/1/2/3 = '+c.levels.join('/')+', 색 틀림 '+c.wrong_color):'기록 없음');}catch(e){}}
async function trsave(){if(TR.lv===null){alert('쌓은 단수(0-3)를 고르세요');return;}
  const body={stack_level:TR.lv, wrong_color:document.getElementById('trwrong').checked, failure_reason:document.getElementById('trreason').value,
    layout_id:document.getElementById('trlayout').value, order_executed:document.getElementById('trorder').value, note:document.getElementById('trnote').value};
  const r=await (await fetch('/trial',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  document.getElementById('out').textContent=JSON.stringify(r,null,1);
  if(r.saved){trlv(null);document.getElementById('trwrong').checked=false;document.getElementById('trreason').value='';document.getElementById('trnote').value='';document.getElementById('trorder').value='';trsum();}}
trsum();
</script></body></html>"""


# [2026-10-03 user] "ckpoint 쌓인거 내가 ui에서 선택할수 있게 해줘": pick any archived step of this UI's run. A pick PINS the port
# (auto_ui_d20_e5.sh skips pinned ports) and re-runs load_relonly_run_ckpt_to_ui.sh with this process's env (same deploy contract),
# which restarts this UI on the chosen weights (~30-60 s). "auto" removes the pin and makes the swapper load the newest again.
CKSEL_SSD = "/Volumes/PortableSSD/rebot_ckpts_archive"
CKSEL_ST = os.path.expanduser("~/umi_bridge/.auto_ui_d20_e5")


CKSEL_LOADER = os.environ.get("CKSEL_LOADER", os.path.expanduser("~/umi_bridge/load_relonly_run_ckpt_to_ui.sh"))
CKSEL_NODE = os.environ.get("CKSEL_NODE", "")          # 5090 runs: "<ssh host>:<checkpoints dir>" (no SSD archive, list the node)
_CKSEL_CACHE = {"t": 0.0, "v": set()}


def _cksel_local_step(name, run):
    pre = f"ckpt_UI_{run}_"
    if name.startswith(pre) and name.endswith("k_pinklockwy") and name[len(pre):-len("k_pinklockwy")].isdigit():
        return f"{int(name[len(pre):-len('k_pinklockwy')]) * 1000:06d}"
    return None


def _cksel_steps():
    run = os.environ.get("RUN", ""); d = os.path.join(CKSEL_SSD, os.environ.get("CKSEL_SSD_PREFIX", "trackb_") + run); out = set()
    if run and os.path.isdir(d):
        out |= {s for s in os.listdir(d) if s.isdigit() and len(s) == 6 and os.path.isfile(os.path.join(d, s, ".archived"))}
    # [2026-10-03 user "새 체크포인트 나올때마다 드롭다운 업데이트"]: list the training node too (4090 runs before SSD archiving)
    node = CKSEL_NODE or f"bh-aiteam@100.64.0.2:/home/bh-aiteam/holobrain-data/trainB/{run}/checkpoints"
    jump = [] if CKSEL_NODE else ["-J", "head-lp"]
    if run:
        if time.time() - _CKSEL_CACHE["t"] > 25:
            import subprocess
            host, cdir = node.split(":", 1)
            try:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", *jump, host,
                                    f"cd {cdir} && for c in [0-9]*; do [ -f $c/pretrained_model/model.safetensors ] && echo $c; done"],
                                   capture_output=True, text=True, timeout=20)
                _CKSEL_CACHE.update(t=time.time(), v={x for x in r.stdout.split() if x.isdigit() and len(x) == 6})
            except Exception:
                pass
        out |= _CKSEL_CACHE["v"]
    hm = os.path.expanduser("~/holobrain-mac-model"); pre = f"ckpt_UI_{run}_"
    for n in os.listdir(hm):
        st = _cksel_local_step(n, run) if run else None
        if st and os.path.isfile(os.path.join(hm, n, "model.safetensors")): out.add(st)
    return sorted(out)


@app.route("/ckpts")
def ckpts():
    pin = os.path.join(CKSEL_ST, str(UI_PORT) + ".pin")
    # [2026-10-06 user "pinned 말고 실제 load된 ckpt도 표기"] serving step (this UI's model), a load in progress, the remote server's models
    import re as _re
    serving_dir = os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")); m_ = _re.search(r"_(\d+)k_", serving_dir)
    lk = os.path.join(CKSEL_ST, str(UI_PORT) + ".loading"); loading = None
    if os.path.exists(lk):
        try:
            pid_, st_ = open(lk).read().split()[:2]; os.kill(int(pid_), 0); loading = st_
        except Exception:
            loading = None
    import infer_core_v4 as _IC
    remote = None
    if _IC.REMOTE_INFER:
        try:
            h = requests.get(_IC.REMOTE_INFER + "/health", timeout=1.0).json()
            remote = [(_re.search(r"_(\d+)k_", c[0]).group(1) + "k") if _re.search(r"_(\d+)k_", c[0]) else c[0] for c in h.get("loaded", [])]
        except Exception:
            remote = "busy (loading a checkpoint)"
    return jsonify({"serving_k": (m_.group(1) + "k") if m_ else serving_dir, "loading": loading, "remote_url": _IC.REMOTE_INFER, "remote_loaded": remote,
                    "run": os.environ.get("RUN", ""), "steps": _cksel_steps(),
                    "pinned": open(pin).read().strip() if os.path.exists(pin) else None,
                    "serving": os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/"))})


@app.route("/ckpt_select/<sel>", methods=["POST"])
def ckpt_select(sel):
    import subprocess
    run = os.environ.get("RUN", "")
    if not run or not os.environ.get("DS_EXPECT"):
        return jsonify({"ok": False, "err": "this UI was not started by load_relonly_run_ckpt_to_ui.sh (no RUN/DS_EXPECT)"})
    if STATE.get("running") or STATE.get("busy"):
        return jsonify({"ok": False, "err": "episode running -- STOP first"})
    os.makedirs(CKSEL_ST, exist_ok=True); pin = os.path.join(CKSEL_ST, str(UI_PORT) + ".pin")
    lk = os.path.join(CKSEL_ST, str(UI_PORT) + ".loading")    # one load at a time (a double click raced two rsyncs into one .part)
    if os.path.exists(lk):
        try:
            os.kill(int(open(lk).read().split()[0]), 0)
            return jsonify({"ok": False, "err": "a checkpoint load is already running for this port -- wait"})
        except (OSError, ValueError, IndexError):
            pass
    if sel == "auto":
        if os.path.exists(pin):
            old = open(pin).read().strip()
            with open(os.path.join(CKSEL_ST, "pending_delete"), "a") as f: f.write(f"{run} {old}\n")   # the pinned copy goes too
            os.remove(pin)
        st = os.path.join(CKSEL_ST, str(UI_PORT))
        if os.path.exists(st): os.remove(st)          # swapper reloads the newest on its next pass (<= 2 min)
        return jsonify({"ok": True, "msg": "auto: newest checkpoint loads within ~2 min"})
    if sel not in _cksel_steps():
        return jsonify({"ok": False, "err": f"step {sel} not archived for {run}"})
    with open(pin, "w") as f: f.write(sel + "\n")
    pd = os.path.join(CKSEL_ST, "pending_delete")     # never let the swapper's cleanup delete the copy we are about to serve
    if os.path.exists(pd):
        keep = [l for l in open(pd) if l.split() != [run, sel]]
        open(pd, "w").writelines(keep)
    cs = _cksel_local_step(os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")), run)
    if cs and cs != sel:                              # old copy: swapper deletes it (4090 runs once archived, 5090 runs at once)
        with open(pd, "a") as f: f.write(f"{run} {cs}\n")
    log = os.path.expanduser(f"~/ckpt_select_{UI_PORT}.log")
    _pl = subprocess.Popen(["/bin/bash", CKSEL_LOADER, str(int(sel)), str(UI_PORT)],
                     cwd=os.path.expanduser("~/umi_bridge"), env=dict(os.environ), stdout=open(log, "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    with open(lk, "w") as f: f.write(f"{_pl.pid} {sel}\n")
    return jsonify({"ok": True, "msg": f"loading {run} {sel} (pinned) -- UI restarts in ~30-60 s, log {log}"})


# [2026-10-04] closed-loop TRIAL LOG: one JSON line per real-robot trial, written by the operator after the trial.
# Auto fields come from the served checkpoint / UI state; outcome fields come from the form. Success rates are only ever
# computed from this file (the per-cycle log has no outcome). File: V4_TRIAL_LOG (default ~/rebot_trials.jsonl).
TRIAL_LOG = os.path.expanduser(os.environ.get("V4_TRIAL_LOG", "~/rebot_trials.jsonl"))
_TRIAL_LOCK = threading.Lock()


def _trial_ckpt_info():
    import re as _re
    name = os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/"))
    m = _re.match(r"ckpt_(?:UI_)?(.+?)_(\d+)k(?:_.*)?$", name)
    run, step = (m.group(1), int(m.group(2)) * 1000) if m else (os.environ.get("RUN", "") or name, None)
    u = run.upper()
    if u.startswith("HRA"): model_type = "hra_rightonly"
    elif u.startswith("COTRAIN"): model_type = "cotrain"
    elif "ROBOT100" in u and u.startswith("FT-"): model_type = "robot100_init"
    elif u.startswith("FT-") and "EGO" in u: model_type = "ego_init"
    elif u.startswith("EGO"): model_type = "ego_only"
    else: model_type = "scratch"
    data = next((d for d in ("R30", "R90", "R384", "R312C", "R150") if _re.search(r"(^|-)" + d + r"(-|$)", u)), None)
    size = {"R30": 30, "R90": 90, "R384": 384, "R312C": 312, "R150": 150}.get(data)
    return dict(ckpt=name, run=run, step=step, model_type=model_type, robot_dataset=data, robot_episodes=size)


@app.route("/trial", methods=["POST"])
def trial():
    import json as _json
    f = request.get_json(silent=True) or {}
    try:
        level = int(f.get("stack_level"))
    except (TypeError, ValueError):
        return jsonify({"error": "stack_level 0-3 is required"})
    if not 0 <= level <= 3:
        return jsonify({"error": "stack_level must be 0, 1, 2 or 3"})
    task_label = STATE.get("task"); instr = getattr(INF, "task", None) if INF else None
    rec = dict(t=round(time.time(), 3), time_local=time.strftime("%Y-%m-%d %H:%M:%S"), ui_port=UI_PORT, **_trial_ckpt_info(),
               task=task_label, instruction=instr,
               order_executed=(str(f.get("order_executed") or "").strip().upper() or None),
               layout_id=(str(f.get("layout_id") or "").strip() or None),
               stack_level=level, success=(level == 3 and not f.get("wrong_color")),
               wrong_color=bool(f.get("wrong_color")), failure_reason=(f.get("failure_reason") or None) if level < 3 or f.get("wrong_color") else None,
               note=(str(f.get("note") or "").strip() or None), operator=(str(f.get("operator") or "").strip() or None),
               last_run=dict(cycles=STATE.get("cycles"), travel_mm=STATE.get("travel_mm"), stop_reason=STATE.get("stop_reason")))
    try:
        import infer_core_v4 as _IC
        rec["ctl"] = dict(ik=_IC.IK_BACKEND, k=getattr(INF, "n_action", None), stream=STREAM, rtc=_IC.RTC_ON, dtype=_IC.DTYPE)
    except Exception:  # noqa: BLE001
        pass
    with _TRIAL_LOCK:
        n = 0
        if os.path.exists(TRIAL_LOG):
            with open(TRIAL_LOG) as fh:
                n = sum(1 for _ in fh)
        rec["trial_id"] = n + 1
        with open(TRIAL_LOG, "a") as fh:
            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    return jsonify({"saved": True, "trial_id": rec["trial_id"], "log": TRIAL_LOG, "record": rec})


@app.route("/trials")
def trials():
    import json as _json
    rows = []
    if os.path.exists(TRIAL_LOG):
        with open(TRIAL_LOG) as fh:
            for line in fh:
                try: rows.append(_json.loads(line))
                except ValueError: pass
    by = {}
    for r in rows:
        k = r.get("ckpt") or "?"
        b = by.setdefault(k, dict(ckpt=k, model_type=r.get("model_type"), robot_episodes=r.get("robot_episodes"), step=r.get("step"),
                                  n=0, success=0, levels=[0, 0, 0, 0], wrong_color=0))
        b["n"] += 1; b["success"] += bool(r.get("success")); b["wrong_color"] += bool(r.get("wrong_color"))
        lv = r.get("stack_level")
        if isinstance(lv, int) and 0 <= lv <= 3: b["levels"][lv] += 1
    cur = _trial_ckpt_info()["ckpt"]
    return jsonify({"log": TRIAL_LOG, "n": len(rows), "current_ckpt": cur, "by_ckpt": sorted(by.values(), key=lambda b: (b["model_type"] or "", b["step"] or 0)),
                    "recent": rows[-5:]})


@app.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    INF_CKPT = os.environ.get("V4_CKPT", "")
    INF = V4Inferencer()
    print(f"[v4-smoke] http://localhost:{UI_PORT}  robot={ROBOT}", flush=True)
    app.run(host="0.0.0.0", port=UI_PORT, threaded=True)
