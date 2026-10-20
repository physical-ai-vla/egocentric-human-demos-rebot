"""Labelled jaw OPEN / CLOSED frames from the running stereo_vio_tracker preview (2026-10-16, user: "그리퍼는 제멋대로 열고 닫고 해").
The orange-tape rule (jaw_metric v2) scattered 35..2000 px in robot9; the jaws are rigid to the tracking cameras, so a small classifier on
a fixed bottom ROI of both views replaces it. This tool only READS http://127.0.0.1:<port>/stream (the tracker keeps the cameras) and
speaks the protocol (Korean): ready, then N cycles of OPEN / CLOSED, the UMI moved around the room in every phase. The first second
after each cue is not labelled (transition).
    .venv/bin/python -m visual_teleop.tools.jaw_capture [--cycles 4 --phase-s 8] -> datasets/jaw_cls/<ts>.npz (left, right crops, label 1=open)"""
from __future__ import annotations
import argparse, subprocess, threading, time, urllib.request
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# ROI in ONE preview view (800 x 450, = the tracker's 960 x 540 gray frame resized); jaws sit bottom-centre in both cameras
ROI = (slice(180, 450), slice(200, 760))
DS = 2


def crops(view_pair: np.ndarray):
    """1600 x 450 preview (gray) -> (left, right) ROI crops, downsampled by DS."""
    l, r = view_pair[:, :800], view_pair[:, 800:]
    f = lambda v: cv2.resize(v[ROI], None, fx=1 / DS, fy=1 / DS, interpolation=cv2.INTER_AREA)
    return f(l), f(r)


def say(t: str) -> None:
    print("[jaw]", t, flush=True); subprocess.Popen(["say", "-v", "Yuna", t])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8746); ap.add_argument("--cycles", type=int, default=4); ap.add_argument("--phase-s", type=float, default=8.0)
    ap.add_argument("--hz", type=float, default=8.0); ap.add_argument("--looks", default=None, help="comma list of spoken view hints per cycle (10-16: front-heavy recapture)"); a = ap.parse_args()
    latest = {"img": None}

    def reader():
        s = urllib.request.urlopen(f"http://127.0.0.1:{a.port}/stream", timeout=5); buf = b""
        while True:
            buf += s.read(65536); e = buf.find(b"\xff\xd9")
            if e < 0: continue
            st = buf.rfind(b"\xff\xd8", 0, e)
            if st >= 0: latest["img"] = cv2.imdecode(np.frombuffer(buf[st:e + 2], np.uint8), cv2.IMREAD_GRAYSCALE)
            buf = buf[e + 2:]
    threading.Thread(target=reader, daemon=True).start()
    t0 = time.monotonic()
    while latest["img"] is None:
        if time.monotonic() - t0 > 5: raise SystemExit("미리보기 스트림을 못 받음 -- stereo_vio_tracker 가 떠 있는지 확인")
        time.sleep(0.1)
    phases = [(1, "집게를 활짝 연 채로, 유미를 여기저기 돌려 주세요"), (0, "이제 집게를 꽉 닫은 채로, 유미를 여기저기 돌려 주세요")]
    say("그리퍼 학습 데이터를 모읍니다. 3초 뒤 시작합니다"); time.sleep(4.0)
    L, R, Y, T = [], [], [], []
    for c in range(a.cycles):
        for y, msg in phases:
            LOOKS = ["", "앞쪽 방을 보면서", "위쪽 천장을 보면서", "책상 위를 보면서", "로봇 쪽을 보면서", "사람이나 옷 쪽을 보면서", "바닥을 보면서", "천천히 흔들면서"]
            looks = a.looks.split(",") if a.looks else LOOKS
            look = looks[c % len(looks)]
            say(msg if c == 0 else f"{look} {'열어 주세요' if y else '닫아 주세요'}"); ts = time.monotonic()   # 10-16: capture 1 only saw the floor
            while time.monotonic() - ts < a.phase_s:
                time.sleep(1 / a.hz)
                if time.monotonic() - ts < 1.0: continue                       # transition: not labelled
                l, r = crops(latest["img"]); L.append(l); R.append(r); Y.append(y); T.append(time.monotonic() - t0)
    say("끝났습니다. 수고하셨습니다")
    out = ROOT / "datasets/jaw_cls" / (time.strftime("%Y%m%d_%H%M%S") + ".npz"); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, left=np.array(L), right=np.array(R), label=np.array(Y), t=np.array(T), roi=np.array([180, 450, 200, 760]), ds=DS, cycles=a.cycles)
    cv2.imwrite(str(out.with_suffix(".jpg")), np.vstack([np.hstack([L[i], R[i]]) for i in (len(L) // 4, 3 * len(L) // 8)]))
    print(f"[jaw] 저장 {out} | 열림 {int(np.sum(Y))} / 닫힘 {len(Y) - int(np.sum(Y))} 프레임", flush=True)


if __name__ == "__main__":
    main()
