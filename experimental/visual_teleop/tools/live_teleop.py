"""Control process (2026-10-14): UDP hand + pinch -> DualArmController -> robot bridge, recorder, operator dashboard.
Runs in ~/xvla-mac (pyroki). PREVIEW by default: HttpBridge(live=False) reads /observe and never POSTs.

    cd ~/ego_collector && ~/xvla-mac/bin/python -m visual_teleop.tools.live_teleop --stage 1            # grippers only, preview
    ... --stage 1 --live                                                                              # grippers only, ROBOT
    ... --stage 2 --arm right [--live]   single arm free space        --stage 4 [--live]  both arms
Open http://127.0.0.1:8730 . Live robot requires: --live, the dashboard deadman (SPACE held), INIT done, no E-STOP.
Stages >= 2 additionally refuse to start while any calibration is a placeholder (config.unmeasured) -- no arbitrary offsets.

Never run two control processes against one robot (HttpBridge enforces a single live writer per URL in-process only)."""
from __future__ import annotations
import argparse
import threading
import time
from pathlib import Path
import numpy as np
from .. import config as vcfg
from ..build import build_controller, build_pinch
from ..fusion.wrist_pose_fusion import HandEstimate, HandStatus
from ..link import Latest, Publisher
from ..recording.recorder import EpisodeRecorder
from ..recording.episode_qa import episode_qa
from ..recording.export_lerobot import export_episode
from ..robot.bridge import HttpBridge
from ..robot.dual_arm_controller import Mode
from ..safety.clutch import Clutch
from ..safety.deadman import HeartbeatDeadman, EStopLatch
from ..ui.teleop_dashboard import make_app

SIDES = ("left", "right")
STAGES = {1: dict(arms=(), grippers=SIDES), 2: dict(arms=None, grippers=None), 3: dict(arms=None, grippers=None), 4: dict(arms=SIDES, grippers=SIDES)}


def hand_from_msg(m: dict | None, side: str) -> HandEstimate | None:
    if m is None: return None
    T = None if m.get("T_W_H") is None else np.asarray(m["T_W_H"], np.float64)
    return HandEstimate(side, int(m["t_ns"]), HandStatus(m["status"]), list(m.get("reasons", [])), T,
                        None if m.get("wrist_W") is None else np.asarray(m["wrist_W"]), None if T is None else T[:3, :3],
                        float(m.get("position_conf", 0.0)), m.get("orientation_source", "none"), float(m.get("position_age_ms") or 0.0))


class LiveTeleop:
    def __init__(self, c, *, stage: int, arm: str | None, live: bool, mock_robot: bool = False) -> None:
        self.c = c; self.stage = stage; self.live = live
        arms = STAGES[stage]["arms"] if STAGES[stage]["arms"] is not None else (arm,)
        grips = STAGES[stage]["grippers"] if STAGES[stage]["grippers"] is not None else (arm,)
        if stage == 1 and arm: grips = (arm,)                                # right-only MVP: --stage 1 --arm right
        if None in arms: raise SystemExit("--arm left|right is required for stage 2/3")
        if arms:
            miss = vcfg.unmeasured(c, sides=arms)                       # only the arms this stage moves
            if miss: raise SystemExit(f"stage {stage} moves arms; these calibrations are placeholders: {miss}. Measure them first (spec 44).")
        c["control"]["arms_enabled"] = {s: s in arms for s in SIDES}; c["control"]["grippers_enabled"] = {s: s in grips for s in SIDES}
        from ..robot.ik import ik_pair
        solvers = ik_pair(c); print(f"[live_teleop] IK = {c.ik.get('backend', 'pyroki')}", flush=True)
        if mock_robot:
            if live: raise SystemExit("--mock-robot and --live are exclusive")
            from ..robot.bridge import MockBridge
            self.bridge = MockBridge(q={s: solvers[s].q_full[solvers[s].idx6].astype(np.float64) for s in SIDES}); self.bridge.close = lambda: None
        else: self.bridge = HttpBridge(c.robot["base_url"], live=live, timeout_s=c.robot["request_timeout_s"])
        self.deadman = HeartbeatDeadman(c.operator["deadman"]["timeout_ms"]); self.clutch = Clutch(); self.estop = EStopLatch()
        self.ctl = build_controller(c, bridge=self.bridge, solvers=solvers, deadman=self.deadman, clutch=self.clutch, estop=self.estop)
        self.pin = build_pinch(c); self.pin_state = {s: None for s in SIDES}; self._last_pin_t = {s: None for s in SIDES}
        self.hand_rx = Latest(c.link["hand_port"], keep_kinds=("hub_event",)); self.pin_rx = Latest(c.link["pinch_port"]); self.cmd_tx = Publisher(c.link["cmd_port"])
        self.rec: EpisodeRecorder | None = None; self.hub_events: list = []; self.last = None; self.lock = threading.Lock(); self.stop = threading.Event()
        self.loop_ms: list = []; self.rec_lock = threading.Lock()

    # ---- inputs -----------------------------------------------------------------------------------------------------
    def hands(self) -> dict:
        age = self.c.link["max_age_ms"]; out = {s: hand_from_msg(self.hand_rx.get("hand", s, age), s) for s in SIDES}
        # 10-15: while the hub HOLDS the tip position (position_held), the IMU still rotates the hand. Holding the TIP turned a wrist bend
        # into a robot rotation about its tip ("wrist 전부분이 비틀고 움직여"). Hold the human WRIST point instead (pivot_human_m behind
        # the tip along the tool axis) and re-derive the tip from it with the current orientation.
        d = float(self.c.retarget.get("pivot_human_m", 0.0)); self._pivot = getattr(self, "_pivot", {})
        for s, h in out.items():
            if h is None or h.T_W_H is None or d <= 0: continue
            held = any(r.startswith("position_held") for r in h.reasons)
            if not held: self._pivot[s] = h.T_W_H[:3, 3] - d * h.T_W_H[:3, 2]
            elif s in self._pivot:
                T = h.T_W_H.copy(); T[:3, 3] = self._pivot[s] + d * T[:3, 2]; h.T_W_H = T
        return out

    def pinches(self) -> dict:
        for s in SIDES:
            m = self.pin_rx.get("pinch", s, self.c.link["max_age_ms"])
            if m is None:   # no fresh message: a dropout seen by the detector's own clock
                self.pin_state[s] = self.pin[s].update(None, 0.0, time.monotonic_ns()); continue
            if m["t_ns"] != self._last_pin_t[s]:
                self._last_pin_t[s] = m["t_ns"]; self.pin_state[s] = self.pin[s].update(m.get("metric"), float(m.get("conf") or 0.0), int(m["t_ns"]))
        return dict(self.pin_state)

    # ---- operator actions ---------------------------------------------------------------------------------------------
    def a_init(self, _):
        if self.estop.latched: raise RuntimeError("E-STOP latched: reset it first")
        arms = [s for s, on in self.ctl.cfg.arms_enabled.items() if on]
        report = {}
        if arms:
            import subprocess
            # 10-15: the push window opened silently on the click; three INITs recorded only stillness (peak 0.07 m/s^2). Cue it.
            if self.c.fusion["position"]["source"] != "tracker":          # a 6DoF tracker needs no push (its world is calibrated)
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"]); subprocess.Popen(["say", "-v", "Yuna", "지금 앞으로 밀었다가 돌아오세요"])
            n0 = len(self.hub_events); self.cmd_tx.send("cmd", cmd="align", sides=arms)
            t_end = time.monotonic() + float(self.c.init["push_window_s"]) + 3.0; done = {}
            while time.monotonic() < t_end and len(done) < len(arms):
                time.sleep(0.05)
                for e in self.hub_events[n0:]:
                    if e.get("what") in ("aligned", "align_failed", "align_tilt_warn") and e.get("side") in arms: done[e["side"]] = e
            subprocess.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])
            report = {s: {k: v for k, v in e.items() if k not in ("kind", "t_pub_ns", "t_recv_ns")} for s, e in done.items()}
            bad = [s for s in arms if done.get(s, {}).get("what") != "aligned"]
            if bad: raise RuntimeError(f"INIT yaw failed on {bad}: {report or 'no answer from the sensor hub'}")
            time.sleep(0.5)                                              # hand back at rest before anchoring
        self.ctl.initialise(self.hands()); return {"ok": True, "initialised": True, "align": report}

    def a_estop(self, _): self.estop.trip("ui"); self.ctl.trip_estop("ui"); return {"ok": True}

    def a_estop_reset(self, _):
        self.estop.source_clear("ui"); ok, why = self.estop.reset()
        if not ok: raise RuntimeError(why)
        return {"ok": True, "note": "press INIT before moving"}

    def a_clutch(self, b): self.clutch.set(bool(b.get("on"))); return {"ok": True, "clutch": self.clutch.on}
    def a_hb(self, _): self.deadman.heartbeat(); return {"ok": True}
    def a_release(self, _): self.deadman.release(); return {"ok": True}

    def a_rec_start(self, _):
        if self.rec is not None: raise RuntimeError("already recording")
        cfg = dict(self.c); self.rec = EpisodeRecorder(self.c.recording["root"], skill_id=self.c.recording["skill_id"], data_source=self.c.recording["data_source"], config=cfg)
        self.rec.meta.update(stage=self.stage, live=self.live, unmeasured=vcfg.unmeasured(self.c), hub_events=list(self.hub_events))
        self.cmd_tx.send("cmd", cmd="rec_start", dir=str(self.rec.dir)); return {"ok": True, "dir": str(self.rec.dir)}

    def a_rec_stop(self, _):
        if self.rec is None: raise RuntimeError("not recording")
        with self.rec_lock: r, self.rec = self.rec, None
        self.cmd_tx.send("cmd", cmd="rec_stop"); r.meta["hub_events_end"] = list(self.hub_events[-50:])
        d = r.close(); qa = episode_qa(d, self.c.qa)
        try: ex = export_episode(d)
        except Exception as e: ex = {"error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "dir": str(d), "qa_flags": qa["flags"], "include": qa["include"], "export": ex.get("label_source", ex.get("error"))}

    # ---- status for the dashboard ---------------------------------------------------------------------------------------
    def status(self) -> dict:
        with self.lock: r = self.last
        st = self.ctl.status(); hands = {}
        for s in SIDES:
            m = self.hand_rx.get("hand", s); p = self.pin_state[s]
            hands[s] = dict(status=(m or {}).get("status", "NO DATA"), quality=(m or {}).get("quality"), reasons=",".join((m or {}).get("reasons", [])) or "—",
                            position_conf=(m or {}).get("position_conf"), imu_ok=(m or {}).get("imu_ok"), imu_rate_hz=(m or {}).get("imu_rate_hz"),
                            aligned=(m or {}).get("aligned"), rgbd_fps=(m or {}).get("rgbd_fps"), imu_tilt_deg=(m or {}).get("imu_tilt_deg"),
                            pinch=None if p is None else p.pinch, pinch_status=None if p is None else p.status, pinch_metric=None if p is None else p.metric,
                            gripper_g=st["gripper"].get(s), human_tcp=None if r is None or r.human_T.get(s) is None else r.human_T[s][:3, 3],
                            target_tcp=None if r is None or r.cmd_T.get(s) is None else r.cmd_T[s][:3, 3],
                            exec_tcp=None if r is None or r.exec_T.get(s) is None else r.exec_T[s][:3, 3])
        mode = st["mode"]; reasons = {k: v for k, v in st["reasons"].items() if v}
        if st["estop"]: head, sev = f"E-STOP latched ({', '.join(st['estop_sources']) or 'reset pending'}) — fix, RESET, INIT", "bad"
        elif mode == "FOLLOW": head, sev = "FOLLOWING", "ok"
        elif mode in ("NOT_INITIALISED", "NONE"): head, sev = "NOT INITIALISED — press INIT with hands at the init points", "warn"
        else: head, sev = f"{mode}: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in reasons.items()), "warn"
        lat = {} if r is None else {k: round(v, 1) for k, v in r.latency_ms.items()}
        if self.loop_ms: lat["loop_p99_ms"] = round(float(np.percentile(self.loop_ms[-300:], 99)), 1)
        robot = dict(mode=mode, stage=self.stage, recording=str(self.rec.dir) if self.rec else "off", counters=st["counters"], **lat)
        ev = [{k: v for k, v in e.items() if k not in ("kind", "t_pub_ns", "t_recv_ns")} for e in self.hub_events[-6:]]
        rot = {}                                   # 10-15: "the head keeps looking at one place" -> show rotation at every stage
        try:
            from scipy.spatial.transform import Rotation as _R
            a = self.ctl.rt.anchors["right"]
            if r is not None and a.engaged:
                if r.delta_H.get("right") is not None: rot["human_since_init_deg"] = np.round(_R.from_matrix(r.delta_H["right"][:3, :3]).as_rotvec() * 57.2958, 1).tolist()
                if r.target_T.get("right") is not None: rot["target_vs_start_deg"] = np.round(_R.from_matrix(a.T_RB_R0[:3, :3].T @ r.target_T["right"][:3, :3]).as_rotvec() * 57.2958, 1).tolist()
                if r.obs_T.get("right") is not None: rot["robot_vs_start_deg"] = np.round(_R.from_matrix(a.T_RB_R0[:3, :3].T @ r.obs_T["right"][:3, :3]).as_rotvec() * 57.2958, 1).tolist()
        except Exception as e: rot["error"] = str(e)
        robot["rotation"] = rot
        return dict(headline=head, severity=sev, deadman=self.deadman.pressed(time.monotonic_ns()), clutch=self.clutch.on, live=self.live,
                    hands=hands, robot=robot, hub_events=ev)

    # ---- main loop -----------------------------------------------------------------------------------------------------
    def events_loop(self) -> None:
        while not self.stop.is_set():
            for e in self.hand_rx.pop_all(): self.hub_events.append(e)
            time.sleep(0.02)

    def loop(self) -> None:
        dt = 1.0 / float(self.c.rate_hz)
        while not self.stop.is_set():
            t0 = time.monotonic_ns()
            rec = self.ctl.tick(self.hands(), self.pinches(), t0)
            with self.lock: self.last = rec
            if self.rec is not None: self.rec.add_tick(rec)
            el = (time.monotonic_ns() - t0) / 1e9; self.loop_ms.append(el * 1e3)
            if len(self.loop_ms) > 5000: del self.loop_ms[:2500]
            time.sleep(max(0.0, dt - el))

    def top_cam_loop(self) -> None:
        """10-15 camera roles: the tripod C922 is the DATA top view (cloth state for the policy), never control. Recorded into every
        episode as frames/top_c922.mp4 + top_c922_t_ns.npy (host monotonic, same clock as the ticks); preview /tmp/vt_c922.jpg."""
        import cv2
        tc = self.c.cameras["top_cam"]
        cap = cv2.VideoCapture(int(tc["cv_index"]), cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, tc["width"]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, tc["height"]); cap.set(cv2.CAP_PROP_FPS, tc["fps"])
        try:
            from ..vision.tag_camera import TagCamera
            d = TagCamera.__new__(TagCamera); d.cfg = tc; self.top_focus = TagCamera._lock_focus(d, tc.get("focus", "auto"))
        except Exception as e: self.top_focus = f"focus lock failed: {e}"
        print(f"[live_teleop] top C922 (OpenCV {tc['cv_index']}): {self.top_focus}", flush=True); last = 0.0; self.top_fps = 0.0; n, t_f = 0, time.monotonic()
        while not self.stop.is_set():
            ok, f = cap.read(); t = time.monotonic_ns()
            if not ok: time.sleep(0.05); continue
            n += 1
            if time.monotonic() - t_f > 2.0: self.top_fps = n / (time.monotonic() - t_f); n, t_f = 0, time.monotonic()
            with self.rec_lock:
                if self.rec is not None: self.rec.add_frame("top_c922", t, f, float(tc["fps"]))
            if time.monotonic() - last > 0.2:
                last = time.monotonic(); cv2.imwrite("/tmp/vt_c922.jpg", cv2.resize(f, (640, 360)))
        cap.release()

    def run(self, port: int) -> None:
        acts = {"/init": self.a_init, "/estop": self.a_estop, "/estop_reset": self.a_estop_reset, "/clutch": self.a_clutch, "/hb": self.a_hb,
                "/release": self.a_release, "/rec_start": self.a_rec_start, "/rec_stop": self.a_rec_stop}
        app = make_app(self.status, acts, {"rgbd": "/tmp/vt_rgbd.jpg", "left": "/tmp/vt_wrist_left.jpg", "right": "/tmp/vt_wrist_right.jpg", "c922": "/tmp/vt_c922.jpg"})
        th = threading.Thread(target=self.loop, daemon=True); th.start(); threading.Thread(target=self.events_loop, daemon=True).start()
        if self.c.cameras["top_cam"].get("record", False): threading.Thread(target=self.top_cam_loop, daemon=True).start()
        print(f"[live_teleop] stage {self.stage} {'LIVE' if self.live else 'PREVIEW'} -> http://127.0.0.1:{port}", flush=True)
        try: app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
        finally:
            self.stop.set(); self.deadman.release()
            if self.rec is not None: self.a_rec_stop({})
            self.bridge.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=sorted(STAGES), required=True); ap.add_argument("--arm", choices=SIDES)
    ap.add_argument("--live", action="store_true"); ap.add_argument("--port", type=int, default=8730); ap.add_argument("--config", default=None)
    ap.add_argument("--mock-robot", action="store_true", help="in-process MockBridge (no robot service needed)")
    a = ap.parse_args(); c = vcfg.load(a.config)
    LiveTeleop(c, stage=a.stage, arm=a.arm, live=a.live, mock_robot=a.mock_robot).run(a.port)


if __name__ == "__main__":
    main()
