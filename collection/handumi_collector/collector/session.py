"""CollectorSession: the state machine the UI drives. No Qt in here (testable).

    IDLE (=READY when can_record) --START--> WAITING_FOR_STILLNESS --(every IMU still >= min_still_s, countdown)--> RECORDING
    RECORDING --STOP--> REVIEW --KEEP/DISCARD--> IDLE      (WAITING_FOR_STILLNESS --cancel_start / device error--> IDLE)
    RECORDING --required-device error (poll)--> ERROR_REVIEW --KEEP (with warning) / DISCARD--> IDLE
"""
from __future__ import annotations
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from .episode_manager import EpisodeManager
from .integrity import gripper_integrity, preliminary_qa, validate_episode
from .stillness import StillnessGate
from .autoloop import AutoLoop, Speaker
from .recorder import EpisodeRecorder
from ..config import AppConfig
from ..devices.manager import DeviceManager

log = logging.getLogger("handumi.session")


@dataclass
class CollectorSession:
    cfg: AppConfig
    devices: DeviceManager
    manager: EpisodeManager
    state: str = "IDLE"                    # IDLE | WAITING_FOR_STILLNESS | RECORDING | REVIEW | ERROR_REVIEW | HWCHECK
    order: str = ""
    gate: StillnessGate | None = None      # armed while WAITING_FOR_STILLNESS
    _countdown_start_ns: int | None = None
    _gate_meta: dict | None = None         # how the current episode was started (into episode metadata)
    _hold_until_ns: int | None = None      # RECORDING: operator must keep still until this instant (the GO cue)
    _hold_broken: bool = False
    auto: AutoLoop | None = None           # hands-free loop with spoken cues (collector/autoloop.py)
    recorder: EpisodeRecorder | None = None
    last_meta: dict | None = None
    robotlike: object | None = None        # robotlike.monitor.RobotLikeMonitor when a --protocol is active
    log_lines: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, cfg: AppConfig, *, session_dir: Path | None = None, dataset: str | None = None) -> "CollectorSession":
        dm = DeviceManager(cfg.hardware); dm.build(); dm.connect_all()
        mgr = EpisodeManager(cfg, session_dir=session_dir, dataset=dataset)
        mgr.write_session_meta(devices={n: s.detail for n, s in dm.statuses().items()}, video_listing=dm.video_listing,
                               extra=dict(gripper_calibration_version=dm.gripper_calibration_version, identity=dm.identity(),
                                          hardware_profile=cfg.hardware.profile))
        s = cls(cfg, dm, mgr, order=mgr.next_order())
        if cfg.collector.protocol:
            try:
                from ..robotlike.monitor import RobotLikeMonitor
                s.robotlike = RobotLikeMonitor(cfg.collector.protocol, dm); s.robotlike.start()
                s.say(f"protocol {cfg.collector.protocol.get('protocol')}: robot-like monitor on, AUTO stack {cfg.collector.auto_loop.get('episode_s')} s")
            except Exception as exc:                  # the monitor is advisory; recording must not depend on it
                s.robotlike = None; s.say(f"robot-like monitor unavailable: {exc}")
        for e in mgr.incomplete_episodes():
            s.say(f"WARNING: incomplete episode from a previous crash left in place: {e.name}")
        for n, err in dm.errors.items():
            s.say(f"device {n}: {err}")
        return s

    def say(self, msg: str) -> None:
        self.log_lines.append(msg); log.info(msg)

    # ------------------------------------------------------------- gates
    def disk(self) -> dict:
        """Free space on the dataset volume + rough capacity; thresholds from collector.yaml `disk`."""
        d = self.cfg.collector.disk; root = self.manager.session_dir
        try: free_gb = shutil.disk_usage(root).free / 1e9
        except Exception: free_gb = float("nan")
        state = "RED" if free_gb < float(d.get("block_gb", 10)) else "YELLOW" if free_gb < float(d.get("warn_gb", 50)) else "GREEN"
        return dict(free_gb=free_gb, state=state, capacity_min=free_gb / max(float(d.get("gb_per_minute_estimate", 0.6)), 1e-6))

    def optional_missing(self) -> list[str]:
        return [d.name for d in self.devices.all_devices() if not d.required and not d.status().connected]

    def can_record(self) -> tuple[bool, str]:
        ok, missing = self.devices.required_ok()
        if not ok: return False, "missing: " + ", ".join(missing)
        if self.disk()["state"] == "RED": return False, f"disk below {self.cfg.collector.disk.get('block_gb')} GB"
        if not self.order: return False, "no order selected"
        if self.cfg.collector.preflight_required and not getattr(self, "preflight_ok", False): return False, "preflight checklist incomplete"
        pol = self.gripper_polarity_status()
        if not pol["ok"]: return False, pol["why"]
        if self.state != "IDLE": return False, f"state {self.state}"
        return True, "ready"

    def state_label(self) -> str:
        if self.state == "IDLE": return "READY" if self.can_record()[0] else "IDLE"
        if self.state == "WAITING_FOR_STILLNESS" and self.gate is not None:
            if self._countdown_start_ns is not None:
                left = float(self.gate.cfg["countdown_s"]) - (time.monotonic_ns() - self._countdown_start_ns) / 1e9
                return f"VIO INIT READY — START in {max(int(left) + 1, 1)}"
            return self.gate.label()
        return self.state

    # ------------------------------------------------------------- VIO init readiness (stillness) gate
    @property
    def holding_still(self) -> bool: return self.state == "RECORDING" and self._hold_until_ns is not None

    def hold_left_s(self) -> float:
        return 0.0 if self._hold_until_ns is None else max((self._hold_until_ns - time.monotonic_ns()) / 1e9, 0.0)

    def stillness_enabled(self) -> bool:
        """The gate needs IMUs; a profile without them (pre-IMU, RGB-D Stage-A) starts immediately and says so in metadata."""
        return bool(self.cfg.collector.stillness.get("enabled", True)) and bool(self.devices.imus)

    def request_start(self) -> str:
        """What START does now: arm the stillness gate instead of recording. Returns the new state."""
        ok, why = self.can_record()
        if not ok: raise RuntimeError(why)
        if not self.stillness_enabled():
            self._gate_meta = dict(enabled=False, reason="no IMUs in this hardware profile" if not self.devices.imus else "disabled in collector.stillness")
            self.start(); return self.state
        rate = self.cfg.hardware.imus[0].rate_hz if self.cfg.hardware.imus else 200.0
        still_cfg = dict(self.cfg.collector.stillness)
        if self.auto is not None and self.auto.active and self.auto.cfg.get("min_still_s") is not None:
            still_cfg["min_still_s"] = float(self.auto.cfg["min_still_s"])         # AUTO: the in-REC hold is the initializer's window
        self.gate = StillnessGate(still_cfg, sides=tuple(sorted(self.devices.imus)), rate_hz=rate)
        self.gate.arm(time.monotonic_ns()); self._countdown_start_ns = None
        self.state = "WAITING_FOR_STILLNESS"
        self.say(f"START requested — waiting for {self.gate.min_still_s:.0f} s of stillness on {', '.join(self.gate.sides)} (keep both units still)")
        return self.state

    def cancel_start(self) -> None:
        if self.state != "WAITING_FOR_STILLNESS": return
        self.gate = None; self._countdown_start_ns = None; self.state = "IDLE"; self.say("START cancelled")

    def _poll_stillness(self, now_ns: int) -> str | None:
        assert self.gate is not None
        bad = [d.name for d in self.devices.all_devices() if d.required and (d.status().error or not d.status().connected)]
        if bad: self.say(f"device failed while waiting for stillness: {bad}"); self.cancel_start(); return "IDLE"
        for side, imu in self.devices.imus.items():
            self.gate.update(side, imu.buffer.snapshot(), now_ns)
        if not self.gate.ready:
            self._countdown_start_ns = None; return None          # motion during the countdown restarts the wait
        if self._countdown_start_ns is None: self._countdown_start_ns = now_ns; self.say("VIO INIT READY — starting"); return None
        if (now_ns - self._countdown_start_ns) / 1e9 >= float(self.gate.cfg["countdown_s"]):
            gate = self.gate; self._gate_meta = gate.to_meta(started_ns=now_ns); self.start()
            hold = float(gate.cfg.get("hold_after_rec_s", 0.0))
            if self.auto is not None and self.auto.active and self.auto.cfg.get("hold_after_rec_s") is not None:
                hold = float(self.auto.cfg["hold_after_rec_s"])              # AUTO counts this down aloud
            if hold > 0:                                                  # keep judging stillness INSIDE the recording
                gate.cfg["gyro_max_deg_s"] = float(gate.cfg.get("hold_gyro_max_deg_s", gate.cfg["gyro_max_deg_s"]))   # looser: judged, not gated
                self.gate = gate; self._hold_until_ns = now_ns + int(hold * 1e9); self._hold_broken = False
                self.recorder.event("hold_still_start", "session", dict(hold_s=hold))
                self.say(f"REC — HOLD STILL {hold:.0f} s, then GO")
            return "RECORDING"
        return None

    def _poll_hold(self, now_ns: int) -> None:
        """During the post-REC hold: is the operator actually still? Log it either way; never stop the recording for it."""
        assert self.gate is not None and self.recorder is not None
        # the recorder drains the device buffers now, so read its recent ring instead; judge only once a full window is in
        win_ns = int(float(self.gate.cfg["window_s"]) * 1e9)
        for side in self.devices.imus:
            recent = list(self.recorder.recent_imu.get(side, ()))
            if len(recent) >= 2 and recent[-1].host_receive_ns - recent[0].host_receive_ns >= win_ns: self.gate.update(side, recent, now_ns)
        moved = [s for s in self.gate.sides if self.gate.state[s].still_s == 0.0]
        if moved and not self._hold_broken:
            self._hold_broken = True
            self.recorder.event("moved_before_go", "session", dict(sides=moved, hold_left_s=round(self.hold_left_s(), 2),
                                                                   windows={s: self.gate.state[s].last_window for s in moved}))
            self.say(f"moved before GO ({', '.join(moved)}) — VIO may not initialize on this episode")
        if now_ns >= self._hold_until_ns:
            self.recorder.event("go_cue", "session", dict(held_still={s: round(self.gate.state[s].still_s, 2) for s in self.gate.sides}, clean=not self._hold_broken))
            self.say("GO" if not self._hold_broken else "GO (hold was broken)")
            self._hold_until_ns = None; self.gate = None

    # ------------------------------------------------------------- hands-free loop
    def auto_toggle(self, speaker: Speaker | None = None) -> bool:
        """AUTO on/off. Returns the new state. The loop drives request_start/stop/keep from poll()."""
        if self.auto is not None and self.auto.active:
            self.auto.stop("operator"); return False
        if self.auto is not None and self.auto.phase == "PAUSED":
            self.auto.resume(); return self.auto.active
        c = dict(self.cfg.collector.auto_loop)
        if not bool(c.get("enabled", True)): self.say("AUTO loop disabled in collector.yaml (auto_loop.enabled)"); return False
        self.auto = AutoLoop(self, c, speaker or Speaker(enabled=bool(c.get("voice", True)), voice=c.get("voice_name"), rate_wpm=c.get("rate_wpm")))
        self.say(f"AUTO loop on: {float(c['episode_s']):.0f} s stack / {float(c['reset_s']):.0f} s reset, spoken cues")
        self.auto.start(); return True

    def poll(self) -> str | None:
        """Call periodically (UI timer). During RECORDING a required device error stops the episode -> ERROR_REVIEW."""
        if self.auto is not None and self.auto.active: self.auto.tick()
        if self.state == "WAITING_FOR_STILLNESS": return self._poll_stillness(time.monotonic_ns())
        if self.state != "RECORDING" or not self.recorder: return None
        if self._hold_until_ns is not None and self.gate is not None: self._poll_hold(time.monotonic_ns())
        bad = [d.name for d in self.devices.all_devices() if d.required and (d.status().error or not d.status().connected)]
        if bad:
            self.recorder.event("critical_device_error", "session", dict(devices=bad))
            self.recorder.stop(); self.state = "ERROR_REVIEW"
            self.say(f"CRITICAL: {bad} failed during recording -> ERROR_REVIEW (KEEP with warning or DISCARD)")
            return "ERROR_REVIEW"
        return None

    def start(self) -> None:
        """Begin recording NOW. The UI never calls this directly any more -- it goes through request_start() and the
        stillness gate; this is the gate's exit and the headless/test path (metadata then says the gate was bypassed)."""
        waiting = self.state == "WAITING_FOR_STILLNESS"
        if waiting: self.state = "IDLE"                                    # can_record() wants IDLE
        ok, why = self.can_record()
        if not ok:
            if waiting: self.state = "WAITING_FOR_STILLNESS"
            raise RuntimeError(why)
        gate_meta = self._gate_meta or dict(enabled=False, reason="bypassed: start() called directly")
        self._gate_meta = None; self.gate = None; self._countdown_start_ns = None; self._hold_until_ns = None; self._hold_broken = False
        ep = self.manager.next_episode_dir()
        self.recorder = EpisodeRecorder(self.cfg, self.devices, ep, self.order, self.cfg.tasks.instruction(self.order), self.manager.anchor,
                                        extra_meta=dict(operator=self.cfg.collector.operator, session=self.manager.session_dir.name,
                                                        hardware_version=self.cfg.hardware.hardware_version,
                                                        calibration_version=self.cfg.hardware.calibration_version,
                                                        gripper_calibration_version=self.devices.gripper_calibration_version,
                                                        hardware_profile=self.cfg.hardware.profile, cube_set=self.cfg.tasks.cube_set,
                                                        imu_rate_hz=(self.cfg.hardware.imus[0].rate_hz if self.cfg.hardware.imus else None),
                                                        stillness_gate=gate_meta,
                                                        **self._protocol_meta()))
        self.recorder.start(); self.state = "RECORDING"; self.say(f"REC {ep.name} order={self.order}")
        if self.robotlike is not None:
            try: self.robotlike.begin_episode(ep)
            except Exception as exc: self.say(f"robot-like monitor: {exc}")

    def _protocol_meta(self) -> dict:
        p = self.cfg.collector.protocol
        return {} if not p else dict(protocol=p.get("protocol"), protocol_config=p.get("_path"),
                                     auto_episode_s=self.cfg.collector.auto_loop.get("episode_s"))

    def stop(self) -> dict:
        assert self.recorder and self.state == "RECORDING"
        if self._hold_until_ns is not None: self.recorder.event("stopped_during_hold", "session", {}); self._hold_until_ns = None; self.gate = None
        self.recorder.stop()
        if self.robotlike is not None:
            try:
                rl = self.robotlike.end_episode()
                self.recorder.extra_meta["robot_like_live"] = dict(verdict=rl["verdict"], reasons=rl["reasons"], grasps_total=rl["grasps_total"],
                    over_frac={sd: dict(ang_w=d["over_frac_ang_w"], ang_a=d["over_frac_ang_a"]) for sd, d in rl["sides"].items()},
                    summary="derived/robot_like/live_summary.json")
            except Exception as exc: self.say(f"robot-like summary: {exc}")
        dur = (self.recorder.t_stop_ns - self.recorder.t_start_ns) / 1e9
        self.state = "REVIEW"
        summ = self.recorder.summary()
        self.say(f"stopped {self.recorder.episode_dir.name}: {dur:.1f}s, events={summ['n_events']} {summ['event_kinds']}")
        return summ

    def review_summary(self) -> dict:
        """What the REVIEW screen shows before KEEP/DISCARD (no video decode; that is the HARDWARE CHECK / inspector's job)."""
        assert self.recorder and self.state in ("REVIEW", "ERROR_REVIEW")
        r = self.recorder; summ = r.summary(); dur = (r.t_stop_ns - r.t_start_ns) / 1e9
        streams = {n: dict(frames=s["frames"], drops=s["skipped_capture_indices"], fps=s["fps_measured"]) for n, s in summ["streams"].items()}
        grips = {side: summ["sensor_messages"].get(f"/{side}/gripper", 0) for side in self.devices.grippers}
        imus = {side: summ["sensor_messages"].get(f"/{side}/imu", 0) for side in self.devices.imus}
        meta_like = dict(streams=summ["streams"], duration_s=dur)
        verdict, notes = preliminary_qa(meta_like, r.events)
        for side, dev in self.devices.grippers.items():                        # raw-integrity gate on each jaw (never a VIO metric)
            cfgg = getattr(dev, "cfg", None); tc, to = getattr(cfgg, "ticks_closed", None), getattr(cfgg, "ticks_open", None)
            level, note = gripper_integrity(side, getattr(r, "grip_stats", {}).get(side), tc, to, duration_s=dur)
            if note: notes.append(note)
            if level == "REJECT": verdict = "REJECT"
            elif level == "REVIEW" and verdict == "PASS": verdict = "REVIEW"
        if dur < self.cfg.collector.min_episode_s: verdict = "REVIEW"; notes.append(f"shorter than {self.cfg.collector.min_episode_s}s")
        if self.state == "ERROR_REVIEW": verdict = "REVIEW"; notes.insert(0, "required device failed during recording")
        rl = r.extra_meta.get("robot_like_live")
        if rl:
            notes.append(f"robot-like {rl['verdict']}" + (f": {'; '.join(rl['reasons'])}" if rl["reasons"] else ""))
            if rl["verdict"] == "FAIL" and (self.cfg.collector.protocol or {}).get("verdict_on_fail", "review") == "review" and verdict == "PASS":
                verdict = "REVIEW"
        return dict(episode=r.episode_dir.name, order=self.order, duration_s=round(dur, 2), streams=streams, grippers=grips, imus=imus,
                    hw_errors=sum(1 for e in r.events if e["kind"] in ("device_error", "critical_device_error", "recorder_error")),
                    events=summ["event_kinds"], n_events=summ["n_events"], verdict=verdict, notes=notes, head_video=str(r.episode_dir / "head.mp4"))

    def keep(self, notes: str = "") -> dict:
        assert self.recorder and self.state in ("REVIEW", "ERROR_REVIEW")
        auto = self.review_summary()["verdict"]
        if self.state == "ERROR_REVIEW": notes = (notes + " | KEEP_WITH_WARNING: required device failed during recording").strip(" |")
        meta = self.recorder.finalize(status="KEEP", quality=auto, notes=notes)
        self.manager.keep(self.recorder.episode_dir, self.order)
        self.last_meta = meta; self._next(); return meta

    def discard(self, notes: str = "") -> Path:
        assert self.recorder and self.state in ("REVIEW", "ERROR_REVIEW")
        self.recorder.finalize(status="DISCARD", quality="REJECT", notes=notes)
        dst = self.manager.discard(self.recorder.episode_dir)
        self._next(); return dst

    def _next(self) -> None:
        self.recorder = None; self.state = "IDLE"; self.order = self.manager.next_order()

    # ------------------------------------------------------------- HOME protocol marks (M2)
    PROTOCOL_MARKS = ("home_leave", "home_return")

    def mark(self, kind: str | None = None) -> str:
        """Operator protocol mark written to events (raw, timestamped). With kind=None it alternates home_leave -> home_return.
        Optional: the pose QA auto-detects still HOME windows from the IMU when marks are absent."""
        assert self.recorder and self.state == "RECORDING", "marks only while recording"
        if kind is None:
            done = [e["kind"] for e in self.recorder.events if e["kind"] in self.PROTOCOL_MARKS]
            kind = "home_leave" if "home_leave" not in done else "home_return"
        assert kind in self.PROTOCOL_MARKS
        self.recorder.event(kind, "operator", {})
        self.say(f"mark {kind}"); return kind

    # ------------------------------------------------------------- hardware check (M1.5)
    def hardware_check(self, seconds: float = 5.0) -> dict:
        """Record a short test episode into <session>/_hwcheck/ with every connected stream, then validate it (video frame
        counts == frame_meta, monotonic timestamps, gripper samples). Result: ready + problems. Never counted as data."""
        ok, why = self.can_record()
        if not ok: return dict(ready=False, problems=[f"cannot record: {why}"])
        d = self.manager.session_dir / "_hwcheck"; d.mkdir(exist_ok=True)
        ep = d / time.strftime("hwcheck_%H%M%S")
        rec = EpisodeRecorder(self.cfg, self.devices, ep, self.order or "HWCHECK", "hardware check", self.manager.anchor, extra_meta=dict(hardware_check=True))
        self.state = "HWCHECK"
        try:
            rec.start(); time.sleep(seconds); rec.stop(); rec.finalize(status="HWCHECK", quality=None, notes="hardware check")
        finally:
            self.state = "IDLE"
        res = validate_episode(ep, expect_streams=rec.streams, expect_grippers=list(self.devices.grippers), expect_imus=list(self.devices.imus))
        moving = {}
        for side, g in self.devices.grippers.items():
            q = g.quality(); moving[side] = dict(raw=q.raw, hz=round(q.hz, 1))
        res.update(ready=res["ok"], grippers_live=moving, optional_missing=self.optional_missing(), path=str(ep))
        self.say(("READY FOR COLLECTION" if res["ok"] else "HARDWARE CHECK FAILED: " + "; ".join(res["problems"])))
        return res

    # ------------------------------------------------------------- gripper polarity gate (2026-09-17)
    # Session 135136 ep029-038 were recorded with the left servo's tick mapping flipped (rest read as OPEN under the active calibration) and had to be
    # quarantined. Two guards, both blocking REC via can_record(): (1) every calibrated jaw must read CLOSED at rest right before REC; (2) once per
    # collector start the operator runs POLARITY TEST (open->close->open both jaws, 8 s): each side must sweep norm <=0.3 .. >=0.7 at least twice
    # and end closed. cfg.collector.polarity_test_required (default False since 2026-09-19) turns guard (2) off.
    POL_REST_MAX = 0.35; POL_WINDOW_S = 8.0

    def gripper_polarity_status(self) -> dict:
        for side, g in self.devices.grippers.items():
            ls = getattr(g, "last_sample", None); n = getattr(ls, "normalized", None)
            if ls is None or n is None or n != n: continue                                       # NaN = not calibrated -> integrity gate handles it
            if n > self.POL_REST_MAX:
                return dict(ok=False, why=f"gripper_{side} reads OPEN at rest (norm {n:.2f}, raw {ls.raw_position_mod}) -- close the jaw; if it stays OPEN the servo polarity/zero moved: SET CLOSED + SET OPEN + SAVE, then POLARITY TEST")
        if bool(getattr(self.cfg.collector, "polarity_test_required", False)) and not getattr(self, "polarity_ok", False):
            return dict(ok=False, why="POLARITY TEST not passed yet (button in LEFT HAND / RIGHT HAND: open->close->open both jaws)")
        return dict(ok=True, why="")

    def polarity_test_begin(self) -> None:
        self._pol_t0 = time.monotonic(); self._pol_hist = {side: [] for side in self.devices.grippers}; self.polarity_ok = False
        self.say(f"POLARITY TEST: open and close BOTH jaws fully, 3 times, within {self.POL_WINDOW_S:.0f} s -- then leave them closed")

    def polarity_test_tick(self) -> bool:
        """Called ~5x/s by the UI while the test runs; returns True while still collecting."""
        for side, g in self.devices.grippers.items():
            ls = getattr(g, "last_sample", None); n = getattr(ls, "normalized", None)
            if ls is not None and n is not None and n == n: self._pol_hist[side].append((ls.sample_ns, float(n), int(ls.raw_position_mod)))
        return (time.monotonic() - self._pol_t0) < self.POL_WINDOW_S

    def polarity_test_finish(self) -> dict:
        res = {}; ok_all = True
        for side, h in self._pol_hist.items():
            if len(h) < 10: res[side] = dict(ok=False, why="no samples (uncalibrated or servo not answering)"); ok_all = False; continue
            ns = [x[1] for x in h]; mn, mx, last = min(ns), max(ns), ns[-1]; cycles = 0; state = 0
            for v in ns:
                if state == 0 and v >= 0.7: state = 1
                elif state == 1 and v <= 0.3: state = 0; cycles += 1
            ok = mx >= 0.7 and mn <= 0.3 and cycles >= 2 and last <= self.POL_REST_MAX
            why = ("OK" if ok else ("never read OPEN (max %.2f): opening moves the wrong way -> polarity inverted, recalibrate" % mx if mx < 0.7 else
                   "never read CLOSED (min %.2f)" % mn if mn > 0.3 else "only %d open-close cycles" % cycles if cycles < 2 else "ended OPEN (%.2f): close the jaw" % last))
            res[side] = dict(ok=ok, min=round(mn, 2), max=round(mx, 2), cycles=cycles, end=round(last, 2), why=why); ok_all = ok_all and ok
        self.polarity_ok = ok_all
        self.say(("POLARITY TEST PASSED" if ok_all else "POLARITY TEST FAILED -- REC blocked") + " | " + "; ".join(f"{s}: {r['why']}" for s, r in res.items()))
        return dict(ok=ok_all, sides=res)

    # ------------------------------------------------------------- gripper calibration (M1)
    def calibrate_gripper(self, side: str, which: str) -> int:
        """Capture the current raw (unwrapped) ticks of one gripper as its `closed` or `open` reference. Returns the ticks."""
        assert which in ("closed", "open") and self.state == "IDLE"
        g = self.devices.grippers[side]
        s = g.last_sample
        if s is None: raise RuntimeError(f"gripper_{side}: no sample yet")
        self._pending_cal = getattr(self, "_pending_cal", {})
        self._pending_cal.setdefault(side, {})[f"ticks_{which}"] = int(s.raw_position)
        self.say(f"gripper_{side} {which} = {s.raw_position} ticks (pending save)")
        return int(s.raw_position)

    def save_gripper_calibration(self, notes: str = "") -> str:
        """Write the next gripper_vNNN.yaml from the pending SET CLOSED/OPEN captures (both sides, both ends required) and apply it."""
        from ..calibration import save_gripper_calibration
        pend = getattr(self, "_pending_cal", {})
        cur = {side: dict(ticks_closed=g.cfg.ticks_closed, ticks_open=g.cfg.ticks_open) for side, g in self.devices.grippers.items()}
        for side, d in pend.items(): cur.setdefault(side, {}).update(d)
        missing = [f"{s}.{k}" for s, d in cur.items() for k in ("ticks_closed", "ticks_open") if d.get(k) is None]
        if missing: raise RuntimeError("calibration incomplete: " + ", ".join(missing))
        p = save_gripper_calibration(cur, hardware_version=self.cfg.hardware.hardware_version, notes=notes)
        ver = self.devices.apply_gripper_calibration(p.stem)
        self._pending_cal = {}
        self.say(f"saved {p.name}; active gripper calibration = {ver}")
        return ver

    def close(self) -> None:
        if self.auto is not None and self.auto.active: self.auto.phase = "OFF"        # no speech on shutdown
        if self.state == "WAITING_FOR_STILLNESS": self.cancel_start()
        if self.state == "RECORDING":
            self.stop(); self.keep(notes="auto-kept on close")
        elif self.state in ("REVIEW", "ERROR_REVIEW"):
            self.keep(notes="auto-kept on close (was in review)")
        if self.robotlike is not None:
            try: self.robotlike.close()
            except Exception: pass
        self.devices.close_all()
