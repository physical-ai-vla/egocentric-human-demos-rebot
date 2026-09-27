"""Hands-free episode loop with spoken cues (2026-09-16). The operator stacks, resets, stacks, resets.

    RESET    "저장 완료. 리셋하세요. 다음 순서: 빨강, 파랑, 보라"   reset_s to lay the cubes out; "준비하세요" near the end
    ARMING   session.request_start(): the stillness gate must open (both IMUs still). No timeout -- a late reset WAITS,
             it never starts a recording on a moving hand.
    HOLD     REC has begun. The post-REC hold is counted down aloud "3" "2" "1"; the still window is INSIDE the data,
             which is what the VIO initializer needs (stillness before REC was proven useless on 2026-09-16).
    GO       "고" at the GO cue
    STACK    episode_s
    STOP     "스톱" -> session.stop() -> lightweight validation (preliminary QA verdict):
                 PASS    -> KEEP ("저장 완료")            [auto_keep_pass]
                 REVIEW  -> on_warn: keep (with a warning event) | pause
                 REJECT  -> PAUSE, "에피소드 실패. 확인하세요" -- the take stays in REVIEW for the operator  [pause_on_fail]
    next order = the existing least-collected balancing (EpisodeManager.next_order) unless balance_orders is false.

Every transition is recorded with its timestamp: those that happen while a recorder exists go straight into the episode's
events (kind auto_loop); those before REC (reset_start, ready_cue, arming) are queued and written into the NEXT episode's
events with their original t_ns the moment REC starts. Speech is macOS `say` (non-blocking); tests inject a recorder.
Driven from CollectorSession.poll(); nothing sleeps; the clock is injectable so the state machine is testable in fake time.
The manual collector is untouched: AUTO is opt-in through the button / A key."""
from __future__ import annotations
import logging
import math
import os
import platform
import pwd
import shutil
import subprocess
import time
from dataclasses import dataclass, field

log = logging.getLogger("handumi.autoloop")
DEFAULTS = dict(enabled=True, episode_s=10.0, reset_s=10.0, min_still_s=1.0, hold_after_rec_s=3.0, auto_keep_pass=True, pause_on_fail=True,
                on_warn="keep", balance_orders=True, voice=True, voice_name="Yuna", rate_wpm=None, ready_cue_s=2.0, stop_when_targets_met=True)
# min_still_s: in AUTO the pre-REC gate only has to confirm the hands ARE still; the still window the VIO initializer uses is
#   the hold_after_rec_s inside the recording. 3 s + 3 s asked for 6 s of stillness per episode -- the operator felt asked all the time.
COLOR_KO = {"R": "빨강", "B": "파랑", "P": "보라", "G": "초록", "Y": "노랑", "O": "주황", "W": "하양", "K": "검정"}
CUES = dict(reset="리셋하세요", episode_n="에피소드 {n}", next_order="다음 순서: {colors}", ready="준비하세요", go="고", stop="스톱", saved="저장 완료",
            fail="에피소드 실패. 확인하세요", warn="저장, 경고", done="목표 달성. 자동 녹화 종료", off="자동 녹화 종료", paused="일시 정지")


class Speaker:
    """Non-blocking text-to-speech via macOS `say`; a no-op elsewhere. `spoken` keeps everything ever said (tests read it)."""
    def __init__(self, enabled: bool = True, voice: str | None = None, rate_wpm: int | None = None) -> None:
        self.enabled = bool(enabled) and platform.system() == "Darwin" and shutil.which("say") is not None
        self.voice, self.rate = voice, rate_wpm
        self.spoken: list[tuple[float, str]] = []
        self._proc: subprocess.Popen | None = None
        self.prefix = self._audio_user_prefix()
        self.failures = 0

    @staticmethod
    def _audio_user_prefix() -> list[str]:
        """The collector runs under sudo (the head camera needs it). `say` as root has no audio session and is silent, so
        speak as the console user who launched sudo: launchctl asuser <uid> sudo -u <user> say ..."""
        user = os.environ.get("SUDO_USER")
        if os.geteuid() != 0 or not user: return []
        try: uid = pwd.getpwnam(user).pw_uid
        except KeyError: return []
        return ["launchctl", "asuser", str(uid), "sudo", "-u", user]

    def say(self, text: str) -> None:
        self.spoken.append((time.monotonic(), text))
        if not self.enabled: return
        cmd = self.prefix + ["say"] + (["-v", self.voice] if self.voice else []) + (["-r", str(int(self.rate))] if self.rate else [])
        try:
            if self._proc is not None:
                rc = self._proc.poll()
                if rc is None: self._proc.terminate()                                          # a new cue outranks the old one
                elif rc != 0:
                    self.failures += 1
                    if self.failures <= 3: log.warning("say exited %s (voice=%r, as %s) -- no audio?", rc, self.voice, self.prefix or "current user")
            self._proc = subprocess.Popen(cmd + [text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.warning("say failed: %s", exc)


def order_words(order: str) -> str:
    return ", ".join(COLOR_KO.get(c, c) for c in (order or ""))


@dataclass
class AutoLoop:
    session: object                      # CollectorSession, duck-typed (no import cycle)
    cfg: dict
    speaker: Speaker
    phase: str = "OFF"                   # OFF | RESET | ARMING | HOLD | EPISODE | PAUSED | DONE
    phase_start: float = 0.0
    episodes_done: int = 0
    pause_reason: str = ""
    _spoken: set = field(default_factory=set)
    _pending_events: list = field(default_factory=list)     # transitions before REC, written into the next episode's events
    _now: object = time.monotonic
    _now_ns: object = time.monotonic_ns

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self.episodes_done = 0; self.pause_reason = ""; self._enter("RESET", first=True)

    def resume(self) -> None:
        """After a PAUSE the operator has dealt with the review (KEEP/DISCARD); continue with a fresh RESET."""
        if self.phase != "PAUSED": return
        if self.session.state != "IDLE": self.session.say("AUTO: finish KEEP/DISCARD before resuming"); return
        self.pause_reason = ""; self._enter("RESET")

    def stop(self, reason: str = "operator") -> None:
        s = self.session
        if s.state == "WAITING_FOR_STILLNESS": s.cancel_start()
        elif s.state == "RECORDING":                                   # an interrupted take is the operator's call, never auto-kept
            self._event("auto_stop", dict(reason=f"loop stopped: {reason}", interrupted=True)); s.stop()
            s.say("AUTO: interrupted take left in REVIEW -- KEEP (K) or DISCARD (D)")
        self.phase = "OFF"; self.speaker.say(CUES["off"]); s.say(f"AUTO loop off ({reason})")

    @property
    def active(self) -> bool: return self.phase in ("RESET", "ARMING", "HOLD", "EPISODE")

    def elapsed(self) -> float: return self._now() - self.phase_start

    def left_s(self) -> float:
        if self.phase == "RESET": return max(float(self.cfg["reset_s"]) - self.elapsed(), 0.0)
        if self.phase == "EPISODE": return max(float(self.cfg["episode_s"]) - self.elapsed(), 0.0)
        return 0.0

    def label(self) -> str:
        return {"RESET": lambda: f"AUTO · RESET {self.left_s():.0f} s → {self.session.order} ({order_words(self.session.order)})",
                "ARMING": lambda: "AUTO · 준비 — waiting for both IMUs to be still", "HOLD": lambda: "AUTO · REC — HOLD STILL, counting down",
                "EPISODE": lambda: f"AUTO · GO — {self.left_s():.0f} s left", "PAUSED": lambda: f"AUTO · PAUSED — {self.pause_reason} (KEEP/DISCARD, then A to resume)",
                "DONE": lambda: "AUTO · DONE — every order met its target"}.get(self.phase, lambda: "")()

    # ------------------------------------------------------------------ transitions
    def _event(self, kind: str, detail: dict | None = None, t_ns: int | None = None) -> None:
        t_ns = t_ns or self._now_ns(); d = dict(cue=kind, phase=self.phase, **(detail or {}))
        r = getattr(self.session, "recorder", None)
        if r is not None and self.session.state == "RECORDING":
            try: r.event("auto_loop", "session", d, t_ns); return
            except Exception: pass
        self._pending_events.append((t_ns, d))

    def _flush_pending(self) -> None:
        r = getattr(self.session, "recorder", None)
        if r is None: return
        for t_ns, d in self._pending_events:
            try: r.event("auto_loop", "session", dict(d, before_rec=True), t_ns)
            except Exception: pass
        self._pending_events = []

    def _enter(self, phase: str, *, first: bool = False, prefix: str | None = None) -> None:
        self.phase = phase; self.phase_start = self._now(); self._spoken = set()
        s = self.session
        if phase == "RESET":
            order = getattr(s, "order", "") or ""
            n_next = self._session_kept() + 1                                     # "에피소드 12" = the one about to be recorded (session count)
            self.speaker.say(f"{prefix + '. ' if prefix else ''}{CUES['reset']}. {CUES['episode_n'].format(n=n_next)}. {CUES['next_order'].format(colors=order_words(order))}")
            self._event("reset_start", dict(order=order, reset_s=float(self.cfg["reset_s"]), episode_n=n_next))
            s.say(f"AUTO: reset {float(self.cfg['reset_s']):.0f} s, next order {order} ({order_words(order)})")
        elif phase == "ARMING":
            self._event("arming")

    def _pause(self, reason: str) -> None:
        self.phase = "PAUSED"; self.pause_reason = reason; self.speaker.say(CUES["fail"]); self.session.say(f"AUTO PAUSED: {reason}")

    # ------------------------------------------------------------------ the loop
    def tick(self) -> None:
        """From CollectorSession.poll() every UI tick. Advances phases on the wall clock; never sleeps."""
        if not self.active: return
        s = self.session
        if self.phase == "RESET":
            if self.left_s() <= float(self.cfg.get("ready_cue_s", 2.0)) and "ready" not in self._spoken:
                self._spoken.add("ready"); self.speaker.say(CUES["ready"]); self._event("ready_cue")
            if self.left_s() > 0: return
            if bool(self.cfg.get("stop_when_targets_met", True)) and self._targets_met():
                self.phase = "DONE"; self.speaker.say(CUES["done"]); s.say("AUTO: all order targets met"); return
            ok, why = s.can_record()
            if not ok:
                if "blocked" not in self._spoken: self._spoken.add("blocked"); s.say(f"AUTO: cannot start yet — {why}")
                return
            s.request_start(); self._enter("ARMING")
            return
        if self.phase == "ARMING":                                        # waits as long as it takes; never records a moving hand
            if s.state == "RECORDING":
                self._flush_pending(); self._event("rec_start"); self.phase = "HOLD"; self.phase_start = self._now(); self._spoken = set()
            elif s.state == "IDLE":
                s.say("AUTO: start was cancelled (device?), retrying after a reset"); self._enter("RESET")
            return
        if self.phase == "HOLD":
            if s.state != "RECORDING": self._pause("recording ended during the hold"); return
            if getattr(s, "holding_still", False):
                n = int(math.ceil(s.hold_left_s() - 1e-6))
                if 1 <= n <= 5 and n not in self._spoken:
                    self._spoken.add(n); self.speaker.say(str(n)); self._event(f"countdown_{n}")
            else:
                self.speaker.say(CUES["go"]); self._event("go"); self.phase = "EPISODE"; self.phase_start = self._now(); self._spoken = set()
            return
        if self.phase == "EPISODE":
            if s.state != "RECORDING": self._pause("recording ended during the episode"); return
            if self.left_s() > 0: return
            self.speaker.say(CUES["stop"]); self._event("auto_stop", dict(episode_s=float(self.cfg["episode_s"])))
            prev_order = s.order
            s.stop()
            r = s.review_summary(); verdict = r["verdict"]
            self._event("validation", dict(verdict=verdict, notes=r.get("notes", [])))       # goes to pending: state is REVIEW now
            if verdict == "PASS" and bool(self.cfg.get("auto_keep_pass", True)): self._keep(prev_order, "PASS", CUES["saved"])
            elif verdict == "PASS": self._pause("PASS but auto_keep_pass is off")
            elif verdict == "REVIEW":
                if str(self.cfg.get("on_warn", "keep")) == "keep": self._keep(prev_order, "REVIEW", CUES["warn"], warn=r.get("notes", []))
                else: self._pause(f"validation REVIEW: {'; '.join(r.get('notes', []))}")
            else:
                if bool(self.cfg.get("pause_on_fail", True)): self._pause(f"validation {verdict}: {'; '.join(r.get('notes', []))}")
                else: self._keep(prev_order, verdict, CUES["warn"], warn=r.get("notes", []))
            return

    def _keep(self, prev_order: str, verdict: str, cue: str, warn: list | None = None) -> None:
        s = self.session
        notes = f"auto_loop episode {self.episodes_done + 1}: {float(self.cfg['episode_s']):.0f} s stack, spoken cues, validation {verdict}"
        if warn: notes += " | WARN: " + "; ".join(warn)
        m = s.keep(notes=notes); self.episodes_done += 1
        self._pending_events.append((self._now_ns(), dict(cue="keep", verdict=verdict, episode=str(m.get("episode_dir", "")), phase="EPISODE")))
        if not bool(self.cfg.get("balance_orders", True)): s.order = prev_order        # else: EpisodeManager's least-collected choice stands
        s.say(f"AUTO: kept {m.get('episode_dir', '')} quality={m.get('quality')} next={s.order}")
        self._enter("RESET", prefix=cue)                                           # "저장 완료. 리셋하세요. 다음 순서: ..." as ONE utterance

    def _session_kept(self) -> int:
        """Episodes kept in this session so far (all orders) -- what the operator hears as the episode number."""
        m = getattr(self.session, "manager", None); c = getattr(m, "counts", None)
        return int(sum(c.values())) if isinstance(c, dict) else self.episodes_done

    def _targets_met(self) -> bool:
        """Targets are judged on reBot-feasible counts once the audit has run (36% of bimanual chunks
        were executable, so a raw quota finishes a session that has not actually collected enough)."""
        m = self.session.manager; t = int(self.session.cfg.tasks.target_per_order)
        c, _ = m.quota_counts() if hasattr(m, "quota_counts") else (m.counts, "valid")
        return all(c.get(o, 0) >= t for o in self.session.cfg.tasks.orders)
