"""Hands-free loop. (1) The state machine in FAKE time against a fake session: the exact cue order, that the countdown and
GO are recorded after rec_start, that a late still gate is waited for and never skipped, PASS -> keep / REJECT -> pause /
REVIEW -> policy, order balancing on or off, transitions written with timestamps. (2) With the real CollectorSession and mock
devices in short real time: two episodes recorded by themselves, cues in the episode events, clean stop from any phase."""
import json
import time
import pytest
from handumi_collector.collector.autoloop import DEFAULTS, AutoLoop, Speaker, order_words
from handumi_collector.collector.session import CollectorSession
from handumi_collector.collector.stillness import DEFAULTS as STILL
from handumi_collector.config import DEFAULT_CONFIG_DIR, AppConfig, CameraCfg, CollectorCfg, GripperCfg, HardwareCfg, ImuCfg, TasksCfg, load_config


# ----------------------------------------------------------------------------------------------------- fake time
class Clock:
    def __init__(self): self.t = 1000.0
    def now(self): return self.t
    def now_ns(self): return int(self.t * 1e9)
    def advance(self, s): self.t += s


class FakeRecorder:
    def __init__(self): self.events = []
    def event(self, kind, dev, detail=None, t_ns=None): self.events.append(dict(kind=kind, detail=detail or {}, t_ns=t_ns))


class FakeManager:
    def __init__(self, orders, target): self.counts = {o: 0 for o in orders}; self.orders = orders; self.target = target; self.raw_total = 0
    def next_order(self): return min(self.orders, key=lambda o: (self.counts[o], self.orders.index(o)))


class FakeSession:
    """Just enough of CollectorSession: the gate opens `gate_delay` s after request_start, the hold lasts `hold` s."""
    def __init__(self, clock, orders=("RBP", "RPB"), target=1, verdicts=("PASS",), gate_delay=0.5, hold=3.0):
        self.clock = clock; self.state = "IDLE"; self.recorder = None; self.log = []; self.manager = FakeManager(list(orders), target)
        self.order = self.manager.next_order(); self.verdicts = list(verdicts); self.gate_delay = gate_delay; self.hold = hold
        self._req_t = None; self._rec_t = None; self.kept = []; self.cancelled = 0
        class T: pass
        self.cfg = T(); self.cfg.tasks = T(); self.cfg.tasks.target_per_order = target; self.cfg.tasks.orders = list(orders)
    def say(self, m): self.log.append(m)
    def can_record(self): return (self.state == "IDLE", "ready" if self.state == "IDLE" else self.state)
    def request_start(self): self.state = "WAITING_FOR_STILLNESS"; self._req_t = self.clock.now(); return self.state
    def cancel_start(self): self.state = "IDLE"; self.cancelled += 1
    def advance_hw(self):
        """the part poll() does: open the gate after gate_delay, run the hold"""
        if self.state == "WAITING_FOR_STILLNESS" and self.clock.now() - self._req_t >= self.gate_delay:
            self.state = "RECORDING"; self.recorder = FakeRecorder(); self._rec_t = self.clock.now()
            self.recorder.event("hold_still_start", "session", dict(hold_s=self.hold), self.clock.now_ns())
    @property
    def holding_still(self): return self.state == "RECORDING" and self.clock.now() - self._rec_t < self.hold
    def hold_left_s(self): return max(self.hold - (self.clock.now() - self._rec_t), 0.0)
    def stop(self): self.state = "REVIEW"
    def review_summary(self): v = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]; return dict(verdict=v, notes=[] if v == "PASS" else [f"fake {v}"])
    def keep(self, notes=""):
        self.kept.append((self.order, notes)); self.manager.counts[self.order] += 1; self.manager.raw_total += 1
        self.state = "IDLE"; self.recorder = None; self.order = self.manager.next_order(); return dict(episode_dir=f"episode_{len(self.kept):06d}", quality="PASS")


def _loop(sess, clock, **cfg):
    c = dict(DEFAULTS, episode_s=10.0, reset_s=10.0, voice=False); c.update(cfg)
    lp = AutoLoop(sess, c, Speaker(enabled=False), _now=clock.now, _now_ns=clock.now_ns); return lp


def _drive(lp, sess, clock, seconds, step=0.1):
    n = int(round(seconds / step))
    for _ in range(n):
        clock.advance(step); sess.advance_hw(); lp.tick()


def test_state_machine_cue_order_and_timing_in_fake_time():
    clock = Clock(); sess = FakeSession(clock, gate_delay=4.0, hold=3.0); lp = _loop(sess, clock); lp.start()
    said = lambda: [t for _, t in lp.speaker.spoken]
    assert lp.phase == "RESET" and said() == [f"리셋하세요. 에피소드 1. 다음 순서: {order_words('RBP')}"]
    _drive(lp, sess, clock, 7.9); assert "준비하세요" not in said()
    _drive(lp, sess, clock, 0.2); assert said()[-1] == "준비하세요"                       # ready_cue_s before the reset ends
    _drive(lp, sess, clock, 2.0); assert lp.phase == "ARMING" and sess.state == "WAITING_FOR_STILLNESS"
    _drive(lp, sess, clock, 2.0); assert lp.phase == "ARMING", "a slow still gate is WAITED for, never skipped"
    _drive(lp, sess, clock, 2.1); assert lp.phase == "HOLD" and sess.state == "RECORDING"
    _drive(lp, sess, clock, 3.1); assert lp.phase == "EPISODE"
    assert said()[-4:] == ["3", "2", "1", "고"]
    ev = sess.recorder.events; cues = [(e["kind"], e["detail"].get("cue"), e["detail"].get("before_rec", False)) for e in ev]
    assert cues[:3] == [("hold_still_start", None, False), ("auto_loop", "reset_start", True), ("auto_loop", "ready_cue", True)]
    assert [c for k, c, _ in cues if k == "auto_loop"][2:] == ["arming", "rec_start", "countdown_3", "countdown_2", "countdown_1", "go"]
    t = {e["detail"].get("cue"): e["t_ns"] for e in ev if e["kind"] == "auto_loop"}
    assert t["reset_start"] < t["arming"] < t["rec_start"] < t["countdown_3"] < t["countdown_1"] < t["go"], "GO and the countdown are recorded AFTER rec_start"
    _drive(lp, sess, clock, 10.1)
    assert said()[-1].startswith("스톱") or said()[-1].startswith("저장 완료")
    assert sess.kept and sess.kept[0][0] == "RBP" and lp.phase == "RESET" and sess.order == "RPB", "least-collected balancing picked the other order"
    assert f"저장 완료. 리셋하세요. 에피소드 2. 다음 순서: {order_words('RPB')}" in said()
    _drive(lp, sess, clock, 40.0)
    assert lp.phase == "DONE" and len(sess.kept) == 2 and said()[-1].startswith("목표 달성")


def test_reject_pauses_and_keeps_nothing_review_follows_policy():
    clock = Clock(); sess = FakeSession(clock, verdicts=("REJECT", "PASS"), gate_delay=0.2, hold=1.0)
    lp = _loop(sess, clock, reset_s=1.0, episode_s=1.0, ready_cue_s=0.5); lp.start()
    _drive(lp, sess, clock, 4.0)
    assert lp.phase == "PAUSED" and sess.state == "REVIEW" and sess.kept == [], "a failed take must not be auto-kept"
    assert [t for _, t in lp.speaker.spoken][-1] == "에피소드 실패. 확인하세요"
    sess.keep(notes="operator decided"); lp.resume(); assert lp.phase == "RESET"
    # REVIEW verdict with on_warn=pause pauses too; with keep it keeps and notes the warning
    clock2 = Clock(); s2 = FakeSession(clock2, verdicts=("REVIEW",), gate_delay=0.2, hold=1.0)
    lp2 = _loop(s2, clock2, reset_s=1.0, episode_s=1.0, on_warn="pause"); lp2.start(); _drive(lp2, s2, clock2, 4.0)
    assert lp2.phase == "PAUSED" and s2.kept == []
    clock3 = Clock(); s3 = FakeSession(clock3, verdicts=("REVIEW",), gate_delay=0.2, hold=1.0)
    lp3 = _loop(s3, clock3, reset_s=1.0, episode_s=1.0, on_warn="keep"); lp3.start(); _drive(lp3, s3, clock3, 4.0)
    assert len(s3.kept) == 1 and "WARN" in s3.kept[0][1] and lp3.phase == "RESET"


def test_balance_orders_off_repeats_the_same_order():
    clock = Clock(); sess = FakeSession(clock, target=3, gate_delay=0.2, hold=1.0)
    lp = _loop(sess, clock, reset_s=1.0, episode_s=1.0, balance_orders=False); lp.start(); _drive(lp, sess, clock, 12.0)
    assert len(sess.kept) >= 2 and {o for o, _ in sess.kept} == {"RBP"}


def test_production_config_and_feature_flag():
    c = load_config().collector.auto_loop
    assert c["enabled"] is True and c["episode_s"] == 20.0 and c["reset_s"] == 10.0 and c["hold_after_rec_s"] == 3.0 and c["min_still_s"] == 1.0 and c["voice_name"] == "Yuna"
    assert c["auto_keep_pass"] is True and c["pause_on_fail"] is True and c["on_warn"] == "keep" and c["balance_orders"] is True
    assert order_words("RBP") == "빨강, 파랑, 보라"


# ------------------------------------------------------------------------------------------ real session, mock devices
def _cfg(tmp_path, target=1, **auto):
    hw = HardwareCfg(profile="auto_test", cameras=[CameraCfg("head", "policy_obs", backend="mock", width=64, height=48, codec="libx264"),
                                                   CameraCfg("left_wrist", "pose_estimation", backend="mock", width=64, height=48, codec="libx264"),
                                                   CameraCfg("right_wrist", "pose_estimation", backend="mock", width=64, height=48, codec="libx264")],
                     imus=[ImuCfg("left", backend="mock", rate_hz=400), ImuCfg("right", backend="mock", rate_hz=400)],
                     grippers=[GripperCfg("left", backend="mock", ticks_closed=1000, ticks_open=2000), GripperCfg("right", backend="mock", ticks_closed=3000, ticks_open=2000)])
    al = dict(DEFAULTS, episode_s=0.6, reset_s=0.4, hold_after_rec_s=0.6, ready_cue_s=0.2, voice=False); al.update(auto)
    return AppConfig(hardware=hw, tasks=TasksCfg(target_per_order=target, orders=["RBP", "RPB"]), config_dir=DEFAULT_CONFIG_DIR,
                     collector=CollectorCfg(dataset_root=str(tmp_path / "raw"), session_prefix="Htest", min_episode_s=0.1,
                                            stillness=dict(STILL, min_still_s=0.3, countdown_s=0.0, hold_after_rec_s=2.0), auto_loop=al))


def _run(s, seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds: s.poll(); time.sleep(0.03)


def test_real_session_records_two_episodes_by_itself(tmp_path):
    s = CollectorSession.create(_cfg(tmp_path, target=1), dataset=None); sp = Speaker(enabled=False)
    try:
        time.sleep(0.3); assert s.auto_toggle(sp) is True and s.auto.phase == "RESET"
        _run(s, 9.0)
        assert s.auto.phase == "DONE", f"phase {s.auto.phase}, kept {s.manager.raw_total}, said {[t for _, t in sp.spoken]}"
        assert s.manager.raw_total == 2 and s.state == "IDLE"
        said = [t for _, t in sp.spoken]; i_go = said.index("고"); assert said[i_go - 1] == "1"
        eps = sorted(s.manager.session_dir.glob("episode_*")); ev = json.loads((eps[0] / "events.json").read_text())
        cues = [e["detail"].get("cue") for e in ev if e["kind"] == "auto_loop"]
        assert cues[:3] == ["reset_start", "ready_cue", "arming"] and cues[3:5] == ["rec_start", "countdown_1"] or "rec_start" in cues
        assert "go" in cues and "auto_stop" in cues and "go_cue" in [e["kind"] for e in ev]
        go = [e for e in ev if e["kind"] == "auto_loop" and e["detail"]["cue"] == "go"][0]; st = [e for e in ev if e["kind"] == "auto_loop" and e["detail"]["cue"] == "auto_stop"][0]
        assert 0.5 <= st["t_rel_s"] - go["t_rel_s"] <= 1.2 and go["t_rel_s"] >= 0.5, "the hold (still data) precedes GO inside the recording"
        # the auto loop's hold override (0.6 s) was used, not stillness.hold_after_rec_s (2.0)
        hs = [e for e in ev if e["kind"] == "hold_still_start"][0]; assert hs["detail"]["hold_s"] == 0.6
        # the second episode carries the keep record of the first
        ev2 = json.loads((eps[1] / "events.json").read_text()); assert "keep" in [e["detail"].get("cue") for e in ev2 if e["kind"] == "auto_loop"]
    finally: s.close()


def test_toggle_off_mid_episode_and_while_waiting(tmp_path):
    s = CollectorSession.create(_cfg(tmp_path, target=5), dataset=None); sp = Speaker(enabled=False)
    try:
        time.sleep(0.3); s.auto_toggle(sp); t0 = time.monotonic()
        while time.monotonic() - t0 < 6 and s.auto.phase != "EPISODE": s.poll(); time.sleep(0.03)
        assert s.auto.phase == "EPISODE" and s.state == "RECORDING"
        assert s.auto_toggle(sp) is False and s.state == "REVIEW" and s.manager.raw_total == 0 and not s.auto.active   # interrupted take: operator decides
        s.discard(notes="test"); assert s.state == "IDLE"
        s.auto_toggle(sp); t0 = time.monotonic()
        while time.monotonic() - t0 < 3 and s.state != "WAITING_FOR_STILLNESS": s.poll(); time.sleep(0.02)
        assert s.auto.phase == "ARMING"; s.auto_toggle(sp); assert s.state == "IDLE" and not list(s.manager.session_dir.glob("episode_*"))   # nothing kept
    finally: s.close()
