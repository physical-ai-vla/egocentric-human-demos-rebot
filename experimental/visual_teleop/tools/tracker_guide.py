"""Spoken step-by-step guide for a live stereo_vio_tracker test (2026-10-16, user: "말로 단계별로"). Tails the tracker jsonl log and
speaks the next step when the tracker state allows it:
  INIT          every 8 s: "가만히 있다가 천천히 움직이세요"
  first TRACKING -> 30 s normal motion -> one fast twist -> stop + hold still -> slow motion (recovery) -> done
  HOLD          announced once per entry, recovery announced
    .venv/bin/python -m visual_teleop.tools.tracker_guide datasets/stereo_vio_live/<log>.jsonl"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path


def say(t: str) -> None:
    print("[guide]", t, flush=True); subprocess.run(["say", "-v", "Yuna", t])


def main() -> None:
    log = Path(sys.argv[1]); pos = 0; state = None; phase = "init"; t_phase = time.monotonic(); last_init_cue = 0.0
    STEPS = [("normal", 30, "좋습니다. 이제 보통 텔레옵 속도로 30초 동안 앞뒤, 좌우, 위아래로 움직이고 손목도 천천히 돌려 주세요"),
             ("fast", 4, "이제 손목을 한 번 빠르게 비트세요"),
             ("still", 4, "멈추고 가만히 계세요"),
             ("recover", 12, "이제 천천히 움직이세요. 다시 잡히는지 봅니다"),
             ("done", 0, "시험이 끝났습니다. 수고하셨습니다")]
    if len(sys.argv) > 2 and sys.argv[2] == "live80":     # 10-16 Basalt live acceptance: 60 s laundry speed, 20 s big moves, then one fast twist
        STEPS = [("laundry", 60, "좋습니다. 이제 60초 동안 평소 빨래 조작하는 속도로 움직이세요"),
                 ("big", 20, "이제 20초 동안 크게 앞뒤 좌우 위아래로 움직이고, 손목도 보통 속도로 돌리세요"),
                 ("fast", 4, "이제 손목을 한 번 빠르게 비트세요"),
                 ("still", 4, "멈추고 가만히 계세요"),
                 ("recover", 10, "이제 천천히 움직이세요. 다시 잡히는지 봅니다"),
                 ("done", 0, "시험이 끝났습니다. 유미를 든 채로 계세요")]
    k = -1
    while True:
        if log.exists():
            with open(log) as f:
                f.seek(pos); lines = f.readlines(); pos = f.tell()
            for l in lines:
                try: m = json.loads(l)
                except ValueError: continue
                if m["state"] != state:
                    if m["state"] == "HOLD" and k >= 0: print("[guide] HOLD", flush=True)
                    if state == "HOLD" and m["state"] == "TRACKING": print("[guide] 복구", flush=True)
                    state = m["state"]
        now = time.monotonic()
        if k < 0:
            if state == "TRACKING": k = 0; say(STEPS[0][2]); t_phase = now
            elif now - last_init_cue > 8 and state in ("INIT", "HOLD"): say("가만히 있다가 천천히 움직이세요"); last_init_cue = time.monotonic()
        elif STEPS[k][0] != "done" and now - t_phase >= STEPS[k][1]:
            k += 1; say(STEPS[k][2]); t_phase = time.monotonic()
            if STEPS[k][0] == "done": return
        time.sleep(0.2)


if __name__ == "__main__":
    main()
