"""[2026-10-01 user] MIT-mode robot service for the bimanual reBot, built on the colleague's bh_indy7_LeRobot reBot plugin
("내 동료가 mit를 구현한게 있어 그걸 사용하자" / "중력보상 말고 integral버전으로 해줘").

Same HTTP contract as robot_service.py (POS_VEL, untouched), so mac_v4_smoke_ui.py works unchanged with V4_ROBOT pointed here.
Runs in the colleague's env:  ~/bh_indy7_LeRobot/.venv/bin/python ~/robot-cockpit/robot_service_mit.py   (port MIT_PORT, 8021)
The two services cannot own the arm ports at the same time: stop robot_service.py (8020) first.

Per arm: lerobot_plugin_rebot.ReBot(arm_control_mode="mit", arm_feedforward_law=INTEGRAL) -> connect -> integral law armed exactly
as bh_indy7_lerobot_gui/infer_cli.py:_arm_rebot_feedforward does it (gravity model DISARMED, IntegralOnlyLaw terms, Ki from the
config, default 28 N*m/(rad*s)) -> set_arm_torque(True). The colleague's kp/kd/vel-FF/collision guard are used as shipped.
Known cost (colleague's measurement): t_ff starts at 0, so a loaded joint sags until the integral winds up (8.9 deg at the elbow
at kp 64, recovered in ~6.5 s). Arm with the arms low / near rest.

ONE control thread owns both buses at MIT_TICK_HZ (100): reads each arm, ramps the commanded pose toward the goal at most
MIT_MAX_DEG_S (45) per joint, sends one MIT frame per arm. HTTP handlers only swap goals and read the published snapshot.
If either arm's collision guard trips (or a tick raises), BOTH arms hold in place and the service latches E-STOP.

Frames: /execute_step takes the robot_service vector (14 = [L j1..j6 rad, L jaw CMD 0..45, R j1..j6 rad, R jaw CMD]) in the
LEADER frame the Seeed plugin expects (pan/lift/roll negated). It is mapped back to the raw motor frame with the Seeed
joint_directions, which is also the colleague's canonical frame (REBOT_JOINT_SIGNS all +1, raw motor degrees). Jaw: raw =
cmd * -6 deg in [-270, 0] -> colleague's normalised gripper (0 = open = -270 deg, 1 = closed = 0 deg).
MIT_MOCK=1: both arms on the plugin's MockReBotArmBackend and synthetic camera frames (no hardware) for testing.
"""
import base64, math, os, sys, threading, time
import numpy as np, cv2
from flask import Flask, Response, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MOCK = os.environ.get("MIT_MOCK", "0") == "1"
PORT = int(os.environ.get("MIT_PORT", "8021"))
TICK_HZ = float(os.environ.get("MIT_TICK_HZ", "100"))
MAX_DEG_S = float(os.environ.get("MIT_MAX_DEG_S", "45"))
REST_DEG_S = float(os.environ.get("MIT_REST_DEG_S", "15"))
MAX_ACC_DEG_S2 = float(os.environ.get("MIT_MAX_ACC_DEG_S2", "180"))   # [2026-10-01] colleague's POLICY_MAX_JOINT_ACC_DEG_S2
# [2026-10-02 user: "브리지로 해줘"] GOAL INTERPOLATION. Streaming sends a new goal every DT; the plain profile reached each one early
# and braked to a stop in between (stop-go at the row rate). With MIT_GOAL_INTERP=1 a goal that arrives within INTERP_MAX_GAP_S of
# the previous one becomes a SEGMENT: a reference point moves from where the previous segment's reference was to the new goal over
# the EMA of the goal inter-arrival time, and the commanded pose tracks it with velocity feed-forward + KP on the error. The
# stopping-distance cap is taken to the segment END (no early braking); acceleration limit and "never past the goal" are kept.
# Any other writer of arm["goal"] (hold/rest/recover/connect) replaces the array, which drops the segment -> plain profile.
# Live toggle: POST /set_speed {"goal_interp": true|false, "interp_kp": 8}.
GOAL_INTERP = os.environ.get("MIT_GOAL_INTERP", "0") == "1"
INTERP_KP = float(os.environ.get("MIT_INTERP_KP", "8.0"))          # 1/s
INTERP_MAX_GAP_S = float(os.environ.get("MIT_INTERP_MAX_GAP_S", "1.5"))
# [2026-10-02 user: "rest 자세로 다 온 후에는 torque를 꺼줘"] PARK: when /rest has arrived, both arms are marked parked and their arm
# torque is switched off. A parked arm is still READ every tick but gets no frames and no auto-recover (the recover path would
# otherwise re-enable a "disabled" drive within 1 s). The next motion command (/execute_step, /rest, /torque on, /reenable)
# re-enables it in place: torque on (guard paused), hold re-seeded at the MEASURED pose, then unparked -> no jump.
REST_TORQUE_OFF = os.environ.get("MIT_REST_TORQUE_OFF", "0") == "1"
GRIP_DEG_S = float(os.environ.get("MIT_GRIP_DEG_S", "90"))   # [2026-10-01 user: "그리퍼가 너무 빨라"] FORCE_POS jaw speed (colleague default 900)
CAMERAS = {"global": int(os.environ.get("CAM_GLOBAL", 0)), "left_wrist": int(os.environ.get("CAM_LEFT", 1)),
           "right_wrist": int(os.environ.get("CAM_RIGHT", 2))}
CAM_MAP = {"global": "middle", "left_wrist": "left", "right_wrist": "right"}
CAM_W, CAM_H = 640, 480
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
SEEED_DIR = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0])            # Seeed DM follower joint_directions (leader -> raw)
SEEED_LIM = np.array([(-145, 145), (-170, 0), (-200, 0), (-80, 90), (-90, 90), (-130, 130)], float)   # raw-frame clip (deg)
GRIP_OPEN_DEG, GRIP_CLOSE_DEG, GRIP_CMD_SCALE = -270.0, 0.0, -6.0
ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
SIDES = ("left", "right")

app = Flask(__name__)
S = {"connected": False, "estop": False, "status": "idle", "error": None, "arms": {}, "cams": {}, "ticks": 0,
     "tick_overruns": 0, "max_tick_ms": 0.0, "rest_pose_raw": None}
glock = threading.Lock()                     # goals / snapshot only; the bus belongs to the control thread


def raw_jaw_to_norm(raw_deg):
    return float(np.clip((raw_deg - GRIP_OPEN_DEG) / (GRIP_CLOSE_DEG - GRIP_OPEN_DEG), 0.0, 1.0))


def norm_to_raw_jaw(norm):
    return GRIP_OPEN_DEG + float(norm) * (GRIP_CLOSE_DEG - GRIP_OPEN_DEG)


def leader_vec_to_goals(act):
    """robot_service /execute_step vector -> per-side (raw arm deg (6), jaw norm)."""
    out = {}
    for r, side in enumerate(SIDES):
        o = r * 7
        arm_leader_deg = np.degrees(np.asarray(act[o:o + 6], float))
        raw = np.clip(arm_leader_deg * SEEED_DIR, SEEED_LIM[:, 0], SEEED_LIM[:, 1])
        jaw_raw = float(np.clip(float(act[o + 6]) * GRIP_CMD_SCALE, GRIP_OPEN_DEG, GRIP_CLOSE_DEG))
        out[side] = (raw, raw_jaw_to_norm(jaw_raw))
    return out


def _arm_ports():
    if MOCK:
        return {"left": "mock-left", "right": "mock-right"}
    from usb_arm_ports import arm_ports
    m = arm_ports(); L = os.environ.get("LEFT_LOC"); R = os.environ.get("RIGHT_LOC")
    if not (L and R):
        raise RuntimeError("set LEFT_LOC and RIGHT_LOC (USB locationIDs, see usb_arm_ports.py) -- ports are never guessed here")
    l, r = m.get(int(L)), m.get(int(R))
    if not (l and r):
        raise RuntimeError(f"arm ports not found for LEFT_LOC={L} RIGHT_LOC={R}: {m}")
    return {"left": l, "right": r}


def _build_arm(side, port):
    from bh_indy7_lerobot.feedforward.law import FeedforwardLaw, build_law
    from bh_indy7_lerobot_gui.workers.rebot_gui_adapter import ReBotGuiFollower
    from lerobot_plugin_rebot import ReBot
    from lerobot_plugin_rebot.config import ReBotConfig
    cfg = ReBotConfig(id=f"bh_bimanual_{side}", port=port, can_adapter="damiao_serial", cameras={},
                      arm_control_mode="mit", arm_feedforward_law=FeedforwardLaw.INTEGRAL, gripper_velocity_deg_s=GRIP_DEG_S,
                      control_tick_hz=TICK_HZ)   # [2026-10-01] the plugin's integral / vel-FF laws step per configured tick
    backend = None
    if MOCK:
        from lerobot_plugin_rebot.arm_backend.mock_backend import MockReBotArmBackend
        backend = MockReBotArmBackend()
    rebot = ReBot(cfg, arm_backend=backend)
    if not MOCK:
        # [2026-10-01] the colleague's transport takes ONE host-wide flock (/tmp/bh_indy7_rebot.lock: one reBot per computer),
        # so the second arm was refused. Its _lockfile_path() is a per-instance seam: one lock per arm.
        rebot.arm_backend._transport._lockfile_path = (lambda _p=f"/tmp/bh_indy7_rebot_{side}.lock": _p)
    rebot.connect()
    if os.environ.get("MIT_COLLISION_GUARD", "0") != "1":
        # [2026-10-04 user: "충돌 정지 이런것도 없애줘 오히려 연결이 안되"] the plugin's collision guard keeps running but can no
        # longer LATCH: a would-be trip is only recorded in /status auto_events. Motor-level faults (drive disabled) still happen
        # in hardware. MIT_COLLISION_GUARD=1 restores the latch.
        g = rebot._guard
        def _no_latch(reason, _side=side):
            if _NOGUARD.get(_side) != str(reason)[:120]:
                _NOGUARD[_side] = str(reason)[:120]
                AUTO_EVENTS.append({"t": time.time(), "arm": _side, "reason": "collision guard (latch disabled): " + str(reason)[:160], "result": "ignored"})
                print(f"[mit] {_side}: collision guard would trip ({str(reason)[:120]}) -- latch disabled, ignored", flush=True)
        g._latch = _no_latch; g.reset()
        print(f"[mit] {side}: collision-guard latch DISABLED (MIT_COLLISION_GUARD!=1)", flush=True)
    fol = ReBotGuiFollower(rebot)
    # == infer_cli._arm_rebot_feedforward, integral branch: BEFORE the enable, so the arm is never energised at t_ff = 0 with
    # the wrong law installed. The model is disarmed explicitly (the integral law uses no calibration).
    law = build_law(cfg.arm_feedforward_law, friction_supplier=None)
    assert not law.arms_gravity, "integral law expected"
    try:
        fol.set_gravity_feedforward(None, False)
        fol.set_feedforward_terms(law.extra_terms())
        fol.set_gain_schedule(None, float(cfg.arm_integral_ki_nm_per_rad_s))
        armed = f"law={cfg.arm_feedforward_law} terms={law.term_names()} ki={cfg.arm_integral_ki_nm_per_rad_s}"
    except Exception as e:                    # the mock backend has no torque path
        if not MOCK:
            raise
        armed = f"mock (no torque path: {e.__class__.__name__})"
    fol.set_arm_torque(True)
    print(f"[mit] {side} arm on {port}: mode={_mode(fol, cfg)} {armed} kp={cfg.arm_mit_kp} kd={cfg.arm_mit_kd}", flush=True)
    return {"rebot": rebot, "fol": fol, "cfg": cfg, "goal": None, "jaw_goal": None, "cmd": None, "q": None, "jaw_norm": None,
            "t_read": None, "blocked": None}


def _mode(arm_or_fol, cfg=None):
    fol = arm_or_fol["fol"] if isinstance(arm_or_fol, dict) else arm_or_fol
    try:
        return fol.arm_control_mode
    except Exception:                         # the mock backend has no mode register
        return f"{(cfg or arm_or_fol['cfg']).arm_control_mode} (mock)"


def _read(arm):
    fol = arm["fol"]
    fol.refresh_arm_state()
    obs = fol.read_observation()
    q = np.array([float(obs[f"joint_{i}.pos"]) for i in range(6)])
    return q, float(obs.get("gripper.pos", 0.0))


_NOGUARD = {}                                   # [2026-10-04] last ignored collision reason per arm (log once per reason)
AUTO_EVENTS = []                               # [2026-10-01 user: "e stop이나 이런거 없애줘"] automatic trips, recovered in place


def _recover_in_place(side, reason):
    """No automatic E-STOP (user). The colleague's guard latches and then suppresses every frame, which under MIT lets the
    drives time out and drop the arm. Instead: clear the latch, re-seed the hold at the measured pose (fresh enable if a drive
    dropped out), keep streaming. Logged + counted in /status (auto_events)."""
    arm = S["arms"][side]; fol = arm["fol"]
    if arm.get("parked"):                                  # [2026-10-02] parked (torque off at rest): never auto re-enable
        return
    now = time.time()
    if now - arm.get("last_recover", 0.0) < 1.0:          # at most one recovery per second per arm (no busy loop)
        return
    arm["last_recover"] = now
    try:
        fol.clear_collision_stop()
        # a drive that dropped out of Enable Mode is NOT seen by fol.arm_torque_enabled (a software flag that stays True),
        # so a MOTOR FAULT / disabled report forces a real re-enable (guard paused so the enable's bus hold cannot latch STALE)
        if (not fol.arm_torque_enabled) or ("FAULT" in str(reason)) or ("disabled" in str(reason)) or ("latched" in str(reason)):
            with fol.guard_paused():
                fol.set_arm_torque(True)
        q, jn = _read(arm)
        with glock:
            arm["q"], arm["jaw_norm"] = q, jn; arm["goal"] = q.copy(); arm["cmd"] = q.copy(); arm["vel"] = None
        arm["blocked"] = None; res = "recovered"
    except Exception as e:
        res = f"recover failed: {str(e)[:120]}"
    AUTO_EVENTS.append({"t": time.time(), "arm": side, "reason": str(reason)[:200], "result": res})
    print(f"[mit] AUTO-RECOVER {side}: {reason} -> {res}", flush=True)


def _hold_all(reason):
    for arm in S["arms"].values():
        try:
            arm["fol"].stop_motion()
        except Exception as e:
            print("[mit] hold failed:", e, flush=True)
        if arm["q"] is not None:
            arm["goal"] = arm["q"].copy(); arm["cmd"] = arm["q"].copy(); arm["vel"] = None
    S["estop"] = True; S["status"] = "ESTOP"; S["error"] = reason
    print(f"[mit] BOTH ARMS HOLD: {reason}", flush=True)


def _profile_step(arm, vmax, amax, dt):
    """[2026-10-01 user: "부드럽고 자연스럽게"] SYNCHRONISED, ACCELERATION-LIMITED motion of the commanded pose toward the goal.
    All joints of an arm move along the straight joint-space line to the goal (they start and finish together, scaled by the joint
    with the farthest to go); the speed along it is capped by vmax and by the stopping distance sqrt(2*amax*d), and each joint's
    velocity changes by at most amax*dt per tick (no velocity steps). A new goal mid-move keeps the present velocity and turns
    toward it. Returns the per-joint velocity (deg/s), also sent to the drive as the MIT velocity setpoint."""
    v = arm.get("vel");  v = np.zeros(6) if v is None else v
    e = arm["goal"] - arm["cmd"]; d = float(np.abs(e).max())
    if d < 1e-3 and float(np.abs(v).max()) < amax * dt:
        arm["cmd"] = arm["goal"].copy(); arm["vel"] = np.zeros(6); return arm["vel"]
    speed = min(vmax, math.sqrt(2.0 * amax * d)) if d > 0 else 0.0
    v_des = e / d * speed if d > 0 else np.zeros(6)
    v = v + np.clip(v_des - v, -amax * dt, amax * dt)
    step = v * dt
    # [2026-10-02 fix] the old test np.sign(e) * (|step| - |e|) > 0 was TRUE on every tick for e < 0, so any move in the negative
    # raw direction snapped straight to the goal (no velocity/acceleration limit). Past the goal = same direction AND longer.
    over = (step * e > 0) & (np.abs(step) > np.abs(e))          # never step past the goal on a joint
    step[over] = e[over]; v[over] = 0.0
    arm["cmd"] = arm["cmd"] + step; arm["vel"] = v
    return v


def _profile_step_interp(arm, vmax, amax, dt):
    """[2026-10-02] segment-tracking variant of _profile_step (see GOAL_INTERP). Same return contract (per-joint deg/s)."""
    v = arm.get("vel");  v = np.zeros(6) if v is None else v
    g0, g1, T = arm["seg_start"], arm["seg_end"], arm["seg_T"]
    s_ = min(1.0, (time.time() - arm["t_seg"]) / T) if T > 0 else 1.0
    gi = g0 + (g1 - g0) * s_
    v_seg = (g1 - g0) / T if (T > 0 and s_ < 1.0) else np.zeros(6)
    e_end = g1 - arm["cmd"]; d_end = float(np.abs(e_end).max())
    if s_ >= 1.0 and d_end < 1e-3 and float(np.abs(v).max()) < amax * dt:
        arm["cmd"] = g1.copy(); arm["vel"] = np.zeros(6); return arm["vel"]
    v_des = v_seg + INTERP_KP * (gi - arm["cmd"])
    cap = min(vmax, math.sqrt(2.0 * amax * d_end)) if d_end > 0 else 0.0
    m = float(np.abs(v_des).max())
    if m > cap > 0:
        v_des = v_des * (cap / m)
    elif cap == 0:
        v_des = np.zeros(6)
    v = v + np.clip(v_des - v, -amax * dt, amax * dt)
    step = v * dt
    over = (step * e_end > 0) & (np.abs(step) > np.abs(e_end))     # never step past the segment end on a joint (see fix in _profile_step)
    step[over] = e_end[over]; v[over] = 0.0
    arm["cmd"] = arm["cmd"] + step; arm["vel"] = v
    return v


def _park_torque_off():
    for side, arm in S["arms"].items():
        try:
            with arm["fol"].guard_paused():
                arm["fol"].set_arm_torque(False)
            res = "torque off"
        except Exception as e:
            res = f"torque off failed: {str(e)[:120]}"
        AUTO_EVENTS.append({"t": time.time(), "arm": side, "reason": "rest reached (park)", "result": res})
        print(f"[mit] PARK {side}: {res}", flush=True)


def _unpark(side):
    arm = S["arms"][side]
    if not arm.get("parked"):
        return
    fol = arm["fol"]
    with fol.guard_paused():
        fol.set_arm_torque(True)
    q, jn = _read(arm)
    with glock:
        arm["q"], arm["jaw_norm"] = q, jn; arm["goal"] = q.copy(); arm["cmd"] = q.copy(); arm["vel"] = None; arm["seg_end"] = None
        arm["parked"] = False
    AUTO_EVENTS.append({"t": time.time(), "arm": side, "reason": "unpark (motion command)", "result": "torque on, hold at measured pose"})
    print(f"[mit] UNPARK {side}: torque on, hold at measured pose", flush=True)


def _control_loop():
    dt = 1.0 / TICK_HZ; nxt = time.perf_counter()
    while S["connected"]:
        t0 = time.perf_counter()
        try:
            if PLAN["session"] is not None and not S["estop"]:
                with glock:
                    _plan_update(time.time())
            for side, arm in S["arms"].items():
                q, jn = _read(arm)
                with glock:
                    arm["q"], arm["jaw_norm"], arm["t_read"] = q, jn, time.time()
                    if arm["cmd"] is None:
                        arm["cmd"] = q.copy()
                    if arm["goal"] is None:
                        arm["goal"] = q.copy()
                    if arm["jaw_goal"] is None:
                        arm["jaw_goal"] = jn
                    if arm.get("parked"):          # [2026-10-02] torque off at rest: read only, no frames, no recover
                        continue
                    if S["estop"]:
                        continue
                    if (GOAL_INTERP or PLAN["session"] is not None) and S["status"] != "resting" and arm.get("seg_end") is not None and arm["seg_end"] is arm["goal"]:
                        v_ff = _profile_step_interp(arm, MAX_DEG_S, MAX_ACC_DEG_S2, dt)
                    else:
                        v_ff = _profile_step(arm, REST_DEG_S if S["status"] == "resting" else MAX_DEG_S, MAX_ACC_DEG_S2, dt)
                    action = {f"joint_{i}.pos": float(arm["cmd"][i]) for i in range(6)}
                    action["gripper.pos"] = float(arm["jaw_goal"])
                arm["fol"].send_joint_action(action, vel_ff_deg_s=[float(x) for x in v_ff])
                if S["status"] == "resting" and all(float(np.abs(x["goal"] - x["cmd"]).max()) < 0.05
                                                    for x in S["arms"].values() if x["cmd"] is not None):
                    S["status"] = "connected"          # rest reached: back to the normal speed limit
                    if REST_TORQUE_OFF:
                        with glock:
                            for x in S["arms"].values():
                                x["parked"] = True
                        threading.Thread(target=_park_torque_off, daemon=True).start()
                blocked = arm["fol"].streaming_blocked_reason()
                arm["blocked"] = blocked
                if blocked and not S["estop"]:
                    _recover_in_place(side, blocked)
        except Exception as e:
            if not S["estop"]:
                for side_ in S["arms"]:
                    _recover_in_place(side_, f"control tick raised: {e}")
        S["ticks"] += 1
        el = (time.perf_counter() - t0) * 1000.0; S["max_tick_ms"] = max(S["max_tick_ms"] * 0.999, el)
        nxt += dt; sl = nxt - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
        else:
            S["tick_overruns"] += 1; nxt = time.perf_counter()


# ---- cameras (never on the control thread) ----
def _open_cameras():
    if MOCK:
        return {}
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    cams = {}
    for name, idx in CAMERAS.items():
        c = OpenCVCamera(OpenCVCameraConfig(index_or_path=idx, width=CAM_W, height=CAM_H, fps=30, warmup_s=4.0))
        c.connect(); cams[name] = c
    return cams


def _frame(name):
    """(rgb, age_ms): newest frame without waiting for a new one (the colleague's camera_frame.latest_frame peek)."""
    if MOCK:
        img = np.full((CAM_H, CAM_W, 3), 60, np.uint8); cv2.putText(img, f"MOCK {name}", (40, 240), 0, 2, (255, 255, 255), 3)
        return img, 0.0
    from bh_indy7_lerobot.camera_frame import latest_frame
    cam = S["cams"].get(name)
    if cam is None:
        return None, None
    img = latest_frame(cam)
    ts = getattr(cam, "latest_timestamp", None)
    return img, (None if ts is None else round((time.perf_counter() - ts) * 1000.0, 1))


def _joints14_raw():
    out = []
    for side in SIDES:
        arm = S["arms"][side]
        out += [float(x) for x in arm["q"]] + [norm_to_raw_jaw(arm["jaw_norm"])]
    return out


# ---- HTTP (robot_service contract) ----
@app.route("/connect", methods=["POST"])
def connect():
    if S["connected"]:
        return jsonify({"ok": True, "msg": "already connected"})
    try:
        S["status"] = "connecting"
        ports = _arm_ports()
        S["arms"] = {}
        try:
            for side in SIDES:
                S["arms"][side] = _build_arm(side, ports[side])
        except Exception:
            # [2026-10-01] a partial connect must not leave an energised arm without a control loop: whatever connected is
            # held by streaming its present pose (E-STOP latched), so the operator can still /disconnect it.
            if S["arms"]:
                for arm in S["arms"].values():
                    q, jn = _read(arm); arm.update(q=q, jaw_norm=jn, goal=q.copy(), cmd=q.copy(), jaw_goal=jn, t_read=time.time())
                S["connected"] = True; S["estop"] = True; S["status"] = "ESTOP"
                threading.Thread(target=_control_loop, daemon=True, name="mit-control").start()
            raise
        S["cams"] = _open_cameras()
        for arm in S["arms"].values():
            q, jn = _read(arm); arm.update(q=q, jaw_norm=jn, goal=q.copy(), cmd=q.copy(), jaw_goal=jn, t_read=time.time())
        S["rest_pose_raw"] = _joints14_raw()
        S["connected"] = True; S["estop"] = False; S["status"] = "connected"; S["error"] = None
        threading.Thread(target=_control_loop, daemon=True, name="mit-control").start()
        return jsonify({"ok": True, "mode": "mit/integral", "ports": ports, "tick_hz": TICK_HZ, "max_deg_s": MAX_DEG_S})
    except Exception as e:
        S["error"] = str(e)
        if not S["connected"]:
            S["status"] = "error"
        return jsonify({"ok": False, "error": str(e), "held_arms": list(S["arms"]) if S["connected"] else []}), 500


@app.route("/reopen_cameras", methods=["POST"])
def reopen_cameras():
    """[2026-10-03 user] cameras re-plugged (used elsewhere): release the old OpenCV handles and open them again; arms untouched."""
    old = S.get("cams") or {}
    for c in old.values():
        try:
            c.disconnect()
        except Exception:
            pass
    try:
        S["cams"] = _open_cameras()
        return jsonify({"ok": True, "cams": list(S["cams"])})
    except Exception as e:
        S["cams"] = {}
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/status")
def status():
    arms = {s: {"blocked": a.get("blocked"), "mode": _mode(a),
                "goal_minus_cmd_deg": None if a["cmd"] is None else round(float(np.abs(a["goal"] - a["cmd"]).max()), 3),
                "read_age_ms": None if a["t_read"] is None else round((time.time() - a["t_read"]) * 1000.0, 1)}
            for s, a in S["arms"].items()} if S["arms"] else {}
    return jsonify({"connected": S["connected"], "estop": S["estop"], "status": S["status"], "n_joints": 14, "service": "mit",
                    "error": S["error"], "ticks": S["ticks"], "tick_overruns": S["tick_overruns"],
                    "max_tick_ms": round(S["max_tick_ms"], 2), "tick_hz": TICK_HZ, "max_deg_s": MAX_DEG_S, "max_acc_deg_s2": MAX_ACC_DEG_S2, "grip_deg_s": GRIP_DEG_S, "auto_events": len(AUTO_EVENTS), "last_auto_event": AUTO_EVENTS[-1] if AUTO_EVENTS else None, "arms": arms,
                    "goal_interp": GOAL_INTERP, "rest_torque_off": REST_TORQUE_OFF, "parked": [sd for sd, x in S["arms"].items() if x.get("parked")]})


@app.route("/observe")
def observe():
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    with glock:
        joints = _joints14_raw()
    imgs, ages = {}, {}
    for name, mapped in CAM_MAP.items():
        img, age = _frame(name)
        if img is None:
            return jsonify({"error": f"camera {name} has no frame"}), 500
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        imgs[mapped] = base64.b64encode(buf.tobytes()).decode(); ages[mapped] = age
    jr = list(joints)
    for i in ARM_IDX:
        jr[i] = math.radians(joints[i])
    return jsonify({"joints_deg": joints, "joints_rad": jr, "images": imgs, "cam_age_ms": ages, "t_wall": time.time()})


@app.route("/frame/<cam>")
def frame(cam):
    inv = {v: k for k, v in CAM_MAP.items()}
    img, _ = _frame(inv.get(cam, cam)) if S["connected"] else (None, None)
    if img is None:
        img = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/motor_states")
def motor_states():
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    out = {}
    with glock:
        for side in SIDES:
            a = S["arms"][side]
            d = {n: {"status": 1, "pos_deg": round(float(a["q"][i]), 2)} for i, n in enumerate(MOTORS)}
            d["gripper"] = {"status": 1, "pos_deg": round(norm_to_raw_jaw(a["jaw_norm"]), 1)}
            out[f"{side}_arm"] = d
    return jsonify(out)


@app.route("/execute_step", methods=["POST"])
def execute_step():
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    if S["estop"]:
        return jsonify({"error": f"E-STOP ({S['error']})"}), 400
    if PLAN["session"] is not None:
        return jsonify({"error": f"PLAN session {PLAN['session']} open -- /plan/close first"}), 409
    act = (request.json or {}).get("action")
    if not act or len(act) != 14:
        return jsonify({"error": "bad action"}), 400
    goals = leader_vec_to_goals(act)
    for side in goals:                                     # [2026-10-02] a motion command re-enables a parked arm in place first
        if S["arms"][side].get("parked"):
            try:
                _unpark(side)
            except Exception as e:
                return jsonify({"error": f"unpark {side} failed: {str(e)[:160]}"}), 500
    with glock:
        now_ = time.time()
        for side, (raw, jn) in goals.items():
            arm_ = S["arms"][side]
            if GOAL_INTERP:
                gap = now_ - arm_.get("t_goal", -1e9)
                if gap < INTERP_MAX_GAP_S and arm_.get("seg_end") is not None and arm_["seg_end"] is arm_["goal"]:
                    T0 = arm_["seg_T"]; s0 = min(1.0, (now_ - arm_["t_seg"]) / T0) if T0 > 0 else 1.0
                    start = arm_["seg_start"] + (arm_["seg_end"] - arm_["seg_start"]) * s0       # the reference where it is now
                    ema = gap if arm_.get("gap_ema") is None else 0.6 * arm_["gap_ema"] + 0.4 * gap
                elif gap < INTERP_MAX_GAP_S:
                    start = arm_["goal"].copy() if arm_.get("goal") is not None else raw; ema = gap
                else:
                    start = raw; ema = None                         # first goal after a pause: plain step to it
                arm_["gap_ema"] = ema; arm_["seg_start"] = np.asarray(start, float); arm_["seg_T"] = float(np.clip(ema, 0.02, 1.0)) if ema else 0.0
                arm_["t_seg"] = now_; arm_["t_goal"] = now_
                raw = np.asarray(raw, float); arm_["seg_end"] = raw
            arm_["goal"] = raw; arm_["jaw_goal"] = jn
    sent = list(act)
    for i in ARM_IDX:
        sent[i] = math.degrees(act[i])
    return jsonify({"ok": True, "sent_deg": sent, "error": None})


# ---- [2026-10-06 user: "추론에서_모터까지" port] PLAN playback ------------------------------------------------------------------
# The UI hands numbered, timed rows (knots); the 100 Hz control loop plays the straight segment between consecutive knots with the
# existing segment tracker (_profile_step_interp: position on the segment + its slope as velocity feed-forward, accel-limited).
# Rows are accepted only in order (session + row number), a knot that jumps more than MIT_PLAN_MAX_JUMP_DEG on any joint closes
# the session and holds, PACE scales time (1.0 = the rows' own durations), and when the queue runs dry the arm settles on the last
# knot. Without an open session nothing here runs and /execute_step behaves exactly as before.
import collections as _collections
PLAN = {"session": None, "knots": _collections.deque(), "last_row": 0, "exec_row": 0, "pace": 1.0, "seg": None, "dry": True,
        "rows_in": 0, "glitch": None, "max_jump_deg": float(os.environ.get("MIT_PLAN_MAX_JUMP_DEG", "90"))}


def _plan_seg_T(seg):
    return seg["T_nom"] / max(PLAN["pace"], 1e-3)


def _plan_update(now):
    """(control thread, under glock) advance to the next knot when the current segment is done."""
    P = PLAN
    if P["session"] is None:
        return
    seg = P["seg"]
    if seg is not None and now < seg["t0"] + _plan_seg_T(seg):
        return
    if not P["knots"]:
        P["dry"] = True
        return
    row, goals, dur = P["knots"].popleft()
    t0 = now if seg is None else max(now - 0.02, seg["t0"] + _plan_seg_T(seg))     # chain segments without dropping time
    start = {side: (seg["end"][side][0].copy(), seg["end"][side][1]) for side in S["arms"]} if seg is not None else \
        {side: (a["cmd"].copy(), a["jaw_goal"]) for side, a in S["arms"].items()}
    P["seg"] = seg = dict(row=row, t0=t0, T_nom=float(dur), start=start, end=goals)
    P["exec_row"] = row; P["dry"] = False
    T = _plan_seg_T(seg)
    for side, a in S["arms"].items():
        raw, jn = goals[side]
        a["seg_start"] = np.asarray(start[side][0], float).copy(); a["seg_end"] = np.asarray(raw, float).copy()
        a["goal"] = a["seg_end"]; a["seg_T"] = T; a["t_seg"] = t0; a["jaw_goal"] = jn


def _plan_close(reason):
    with glock:
        PLAN["session"] = None; PLAN["knots"].clear(); PLAN["seg"] = None; PLAN["dry"] = True
    print(f"[mit] PLAN closed: {reason}", flush=True)


def _hold_goals():
    with glock:
        for a in S["arms"].values():
            a["goal"] = a["q"].copy(); a["cmd"] = a["q"].copy(); a["vel"] = None


@app.route("/stop", methods=["POST"])
def stop():
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    _hold_goals(); S["status"] = "connected"
    return jsonify({"ok": True, "msg": "stop (hold the current pose)"})


@app.route("/plan/open", methods=["POST"])
def plan_open():
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    if S["estop"]:
        return jsonify({"error": f"E-STOP ({S['error']})"}), 400
    sid = (request.json or {}).get("session")
    if sid is None:
        return jsonify({"error": "session id required"}), 400
    for side in S["arms"]:
        if S["arms"][side].get("parked"):
            _unpark(side)
    with glock:
        if S["status"] == "resting":
            S["status"] = "connected"
        for a in S["arms"].values():
            a["goal"] = a["q"].copy(); a["cmd"] = a["q"].copy(); a["vel"] = None; a["seg_end"] = None
        PLAN.update(session=sid, last_row=0, exec_row=0, pace=1.0, seg=None, dry=True, rows_in=0, glitch=None)
        PLAN["knots"].clear()
    print(f"[mit] PLAN open: session {sid}", flush=True)
    return jsonify({"ok": True, "session": sid, "row0": "the measured pose"})


@app.route("/plan/rows", methods=["POST"])
def plan_rows():
    b = request.json or {}
    if PLAN["session"] is None or b.get("session") != PLAN["session"]:
        return jsonify({"error": f"no open session {b.get('session')} (open: {PLAN['session']})"}), 409
    acc = 0
    for r in b.get("rows", []):
        row = int(r["row"]); act = r["action"]; dur = float(r["dur_s"])
        if row != PLAN["last_row"] + 1:
            return jsonify({"error": f"row {row} out of order (expected {PLAN['last_row'] + 1})", "accepted": acc}), 409
        if not (0.005 <= dur <= 2.0) or not act or len(act) != 14:
            return jsonify({"error": f"bad row {row}", "accepted": acc}), 400
        goals = leader_vec_to_goals(act)
        with glock:
            prev = PLAN["knots"][-1][1] if PLAN["knots"] else (PLAN["seg"]["end"] if PLAN["seg"] is not None else
                                                                 {side: (a["cmd"], a["jaw_goal"]) for side, a in S["arms"].items()})
            jump = max(float(np.abs(np.asarray(goals[side][0]) - np.asarray(prev[side][0])).max()) for side in goals)
            if jump > PLAN["max_jump_deg"]:
                PLAN["glitch"] = f"row {row}: jump {jump:.1f} deg > {PLAN['max_jump_deg']}"
            else:
                PLAN["knots"].append((row, goals, dur)); PLAN["last_row"] = row; PLAN["rows_in"] += 1; acc += 1
        if PLAN["glitch"]:
            g = PLAN["glitch"]; _plan_close(g); _hold_goals()
            return jsonify({"error": f"GLITCH -- session closed, holding: {g}", "accepted": acc}), 409
    return jsonify({"ok": True, "accepted": acc, "last_row": PLAN["last_row"]})


@app.route("/plan/pace", methods=["POST"])
def plan_pace():
    b = request.json or {}
    if PLAN["session"] is None or b.get("session") != PLAN["session"]:
        return jsonify({"error": "no such session"}), 409
    new = float(np.clip(float(b.get("pace", 1.0)), 0.05, 1.5)); now = time.time()
    with glock:
        seg = PLAN["seg"]
        if seg is not None:
            T_old = _plan_seg_T(seg); s_ = min(1.0, (now - seg["t0"]) / T_old) if T_old > 0 else 1.0
            PLAN["pace"] = new; T_new = _plan_seg_T(seg); seg["t0"] = now - s_ * T_new
            for a in S["arms"].values():
                a["seg_T"] = T_new; a["t_seg"] = seg["t0"]
        else:
            PLAN["pace"] = new
    return jsonify({"ok": True, "pace": PLAN["pace"]})


@app.route("/plan/close", methods=["POST"])
def plan_close():
    sid = (request.json or {}).get("session")
    if PLAN["session"] is not None and (sid is None or sid == PLAN["session"]):
        _plan_close("operator / UI"); _hold_goals()
    return jsonify({"ok": True})


@app.route("/plan/progress")
def plan_progress():
    now = time.time()
    with glock:
        seg = PLAN["seg"]; left_seg = 0.0; left_seg_1x = 0.0
        if seg is not None:
            T = _plan_seg_T(seg); left_seg = max(0.0, seg["t0"] + T - now); left_seg_1x = left_seg * PLAN["pace"]
        q1x = sum(k[2] for k in PLAN["knots"])
        return jsonify({"session": PLAN["session"], "exec_row": PLAN["exec_row"], "last_row": PLAN["last_row"], "queued": len(PLAN["knots"]),
                        "remaining_s": left_seg + q1x / max(PLAN["pace"], 1e-3), "remaining_1x_s": left_seg_1x + q1x, "pace": PLAN["pace"],
                        "dry": PLAN["dry"], "rows_in": PLAN["rows_in"], "glitch": PLAN["glitch"], "estop": S["estop"], "t": now})


@app.route("/estop", methods=["POST"])
def estop():
    if S["connected"]:
        if PLAN["session"] is not None:
            _plan_close("operator E-STOP")
        _hold_goals(); _hold_all("operator E-STOP")
    return jsonify({"ok": True, "msg": "E-STOP - both arms hold"})


@app.route("/clear_estop", methods=["POST"])
def clear_estop():
    blocked = {s: a.get("blocked") for s, a in S["arms"].items() if a.get("blocked")}
    if blocked:
        return jsonify({"ok": False, "error": f"collision guard still latched: {blocked} -- /reenable after checking the arm"}), 400
    _hold_goals(); S["estop"] = False; S["status"] = "connected"; S["error"] = None
    return jsonify({"ok": True})


@app.route("/reenable", methods=["POST"])
def reenable():
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    done = {}
    for side, a in S["arms"].items():
        try:
            if a.get("parked"):
                _unpark(side); done[side] = "unparked"; continue
            if a["fol"].arm_torque_enabled:
                done[side] = "already enabled (no-op: a re-enable holds the bus > 0.5 s and stalls the control loop)"
            else:
                with a["fol"].guard_paused():
                    a["fol"].set_arm_torque(True)
                done[side] = True
        except Exception as e:
            done[side] = str(e)[:120]
    _hold_goals()
    return jsonify({"ok": True, "reenabled": done})


@app.route("/clear_collision", methods=["POST"])
def clear_collision():
    """[2026-10-01] the colleague's guard latches and then SUPPRESSES every frame, so under MIT the drives time out and drop to
    disabled (the left arm went limp). Clear the latch on every arm, re-enable (the plugin's enable re-seeds the live pose),
    hold the present pose, release E-STOP."""
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    done = {}
    for side, a in S["arms"].items():
        try:
            a["fol"].clear_collision_stop()
            with a["fol"].guard_paused():
                a["fol"].set_arm_torque(True)
            a["blocked"] = None; done[side] = "cleared+enabled"
        except Exception as e:
            done[side] = f"ERR {str(e)[:150]}"
    time.sleep(0.1); _hold_goals()
    if all(v == "cleared+enabled" for v in done.values()):
        S["estop"] = False; S["status"] = "connected"; S["error"] = None
    return jsonify({"ok": not S["estop"], "arms": done})


@app.route("/torque", methods=["POST"])
def torque():
    """robot_service compat: {arm: left|right|both, on: bool}. Off = the arm goes limp (support it first)."""
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    b = request.get_json(force=True) or {}
    which = b.get("arm", "both"); on = bool(b.get("on", False))
    sides = {"left": ["left"], "right": ["right"], "both": ["left", "right"]}.get(which, [])
    done = {}
    for side in sides:
        fol = S["arms"][side]["fol"]
        try:
            if on and S["arms"][side].get("parked"):
                _unpark(side); done[side] = "unparked: torque on, hold at measured pose"; continue
            if not on:
                S["arms"][side]["parked"] = True             # [2026-10-02] off = parked, so auto-recover does not re-enable it
            if on and fol.arm_torque_enabled:
                # [2026-10-01] the UI calls /torque on=True at every run start only to power the JAWS. Re-running the arm enable
                # here held the bus > 0.5 s and the guard latched ARM STATE STALE. Arm already on -> jaw only.
                fol.set_gripper_torque(True); done[side] = "arm already on; jaw on"
            else:
                with fol.guard_paused():
                    fol.set_arm_torque(on)
                done[side] = on
        except Exception as e:
            done[side] = f"ERR {str(e)[:120]}"
    _hold_goals()
    return jsonify({"ok": True, "torque_on": on, "arms": done})


@app.route("/rest", methods=["POST"])
def rest():
    if not S["connected"]:
        return jsonify({"error": "not connected"}), 400
    if S["estop"]:
        return jsonify({"error": "E-STOP"}), 400
    rp = S["rest_pose_raw"]
    for side in SIDES:
        if S["arms"][side].get("parked"):
            _unpark(side)
    with glock:
        for r, side in enumerate(SIDES):
            S["arms"][side]["goal"] = np.asarray(rp[r * 7:r * 7 + 6], float)
    S["status"] = "resting"
    return jsonify({"ok": True, "msg": f"rest: ramping to the connect pose at {REST_DEG_S} deg/s"})


@app.route("/set_speed", methods=["POST"])
def set_speed():
    """[2026-10-01 user: "움직임이 너무 빠른데"] per-joint ramp limit (deg/s), changed live: {"max_deg_s": 30} (5..90)."""
    global MAX_DEG_S, MAX_ACC_DEG_S2, GRIP_DEG_S, GOAL_INTERP, INTERP_KP, REST_TORQUE_OFF
    j = request.json or {}
    if "goal_interp" in j:
        GOAL_INTERP = bool(j["goal_interp"]); print(f"[mit] goal interpolation -> {GOAL_INTERP}", flush=True)
    if "rest_torque_off" in j:
        REST_TORQUE_OFF = bool(j["rest_torque_off"]); print(f"[mit] rest torque off -> {REST_TORQUE_OFF}", flush=True)
    if "interp_kp" in j:
        INTERP_KP = float(np.clip(float(j["interp_kp"]), 0.5, 40.0))
    v = float(j.get("max_deg_s", MAX_DEG_S)); acc = float(j.get("max_acc_deg_s2", MAX_ACC_DEG_S2))
    g = float(j.get("grip_deg_s", GRIP_DEG_S))
    if not 1.0 <= v <= 90.0 or not 5.0 <= acc <= 1000.0 or not 10.0 <= g <= 900.0:
        return jsonify({"ok": False, "error": "max_deg_s in [1, 90], max_acc_deg_s2 in [5, 1000], grip_deg_s in [10, 900]"}), 400
    MAX_DEG_S, MAX_ACC_DEG_S2, GRIP_DEG_S = v, acc, g
    for a in S["arms"].values():                # the transport reads cfg.gripper_velocity_deg_s on every FORCE_POS frame
        a["cfg"].gripper_velocity_deg_s = GRIP_DEG_S
    print(f"[mit] profile -> {MAX_DEG_S} deg/s, {MAX_ACC_DEG_S2} deg/s^2, jaw {GRIP_DEG_S} deg/s", flush=True)
    return jsonify({"ok": True, "max_deg_s": MAX_DEG_S, "max_acc_deg_s2": MAX_ACC_DEG_S2, "grip_deg_s": GRIP_DEG_S,
                    "goal_interp": GOAL_INTERP, "interp_kp": INTERP_KP, "rest_torque_off": REST_TORQUE_OFF})


@app.route("/gripper_zero", methods=["POST"])
def gripper_zero():
    """release = jaw motor torque off (the UI uses it at rest so the jaw does not buzz against its stop); set (re-zero) is not
    implemented here -- use robot_service.py (POS_VEL)."""
    b = request.get_json(force=True) or {}
    side, stage = b.get("arm"), b.get("stage")
    if stage == "release" and side in S["arms"]:
        try:
            S["arms"][side]["fol"].set_gripper_torque(False)
            return jsonify({"ok": True, "msg": f"{side} jaw torque off"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)[:200]}), 500
    return jsonify({"ok": False, "error": "gripper re-zero (stage=set) is not implemented in the MIT bridge -- use robot_service.py (POS_VEL)"}), 501


@app.route("/disconnect", methods=["POST"])
def disconnect():
    S["connected"] = False; time.sleep(3.0 / TICK_HZ)
    for a in S["arms"].values():
        try:
            a["fol"].stop_motion(); a["fol"].disconnect()
        except Exception as e:
            print("[mit] disconnect:", e, flush=True)
    for c in S["cams"].values():
        try:
            c.disconnect()
        except Exception:
            pass
    S["arms"] = {}; S["cams"] = {}; S["status"] = "disconnected"
    return jsonify({"ok": True, "msg": "disconnected (torque off)"})


if __name__ == "__main__":
    print(f"[mit] robot_service_mit on :{PORT} mock={MOCK} tick={TICK_HZ} Hz max={MAX_DEG_S} deg/s", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
