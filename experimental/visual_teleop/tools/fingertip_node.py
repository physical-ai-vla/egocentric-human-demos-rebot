"""Fingertip node (2026-10-14): wrist RGB cameras -> student fingertip detector -> pinch metric per side over UDP. Robot-free.
Runs in ~/finger_distill/.venv (ultralytics 8.4; the student checkpoint does not load in ultralytics 8.1).

    cd ~/ego_collector && ~/finger_distill/.venv/bin/python -m visual_teleop.tools.fingertip_node [--sides left right] [--record DIR]
Each side: open the UVC camera by AVFoundation name (YAML cameras.<side>_wrist.match_name), undistort like the student's labels,
detect, publish {kind: pinch, side, t_ns (capture), metric, conf, thumb, index}. Latest JPEG -> /tmp/vt_wrist_<side>.jpg for the
dashboard. --record writes frames/<side>_wrist.mp4 + timestamps (camera observations for the dataset)."""
from __future__ import annotations
import argparse
import threading
import time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def find_camera(name: str, max_index: int = 6) -> int:
    """OpenCV AVFoundation index of the camera called `name`. ffmpeg's device numbering is NOT OpenCV's (10-14: the Arducam
    was ffmpeg 0 but OpenCV 0 was a camera with no frames), so: grab one frame BY NAME with ffmpeg, grab one frame from each
    OpenCV index, and pick the index whose image matches (normalised correlation of a small grey thumbnail)."""
    import subprocess, tempfile, cv2
    ref_path = tempfile.mktemp(suffix=".png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-framerate", "30", "-pixel_format", "nv12",
                    "-i", name, "-frames:v", "1", "-y", ref_path], timeout=20)
    ref = cv2.imread(ref_path)
    if ref is None: raise RuntimeError(f"camera {name!r}: ffmpeg could not grab a frame by name")
    time.sleep(1.5)                                          # let ffmpeg release the device (10-15: index 1 gave no frame right after)
    th = lambda im: (lambda g: (g - g.mean()) / (g.std() + 1e-6))(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32))
    r = th(ref); scores = {}
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION); ok = False
        if cap.isOpened():
            for _ in range(60):
                ok, f = cap.read()
                if ok: break
                time.sleep(0.05)
            for _ in range(5): ok2, f2 = cap.read(); f = f2 if ok2 else f     # skip the first (often over-exposed) frames
        cap.release()
        if ok: scores[i] = float((th(f) * r).mean())
    # measured 10-14: the right camera 0.52, the others -0.30 / -0.11 (the scene moves between the two grabs, so not ~1.0)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < 0.3 or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.3):
        raise RuntimeError(f"camera {name!r}: no OpenCV index clearly matches its image (scores {scores})")
    return ranked[0][0]


def _find_camera_by_ffmpeg_order(name: str) -> int:
    """AVFoundation index by device name (cv2 has no name lookup on macOS)."""
    import subprocess, re
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""], capture_output=True, text=True).stderr
    vids = []
    for line in out.splitlines():
        if "audio devices" in line: break
        m = re.search(r"\[(\d+)\] (.+)$", line)
        if m: vids.append((int(m.group(1)), m.group(2).strip()))
    hits = [i for i, n in vids if name.lower() in n.lower()]
    if len(hits) != 1: raise RuntimeError(f"camera {name!r}: {len(hits)} matches in {vids}")
    return hits[0]


def run_side(side: str, cam_cfg: dict, wh, pub, stop: threading.Event, rec_dir: Path | None, det_cfg: dict, cam_index: int | None = None) -> None:
    import cv2
    from ..vision.fingertip_detector import make_detector, fisheye_undistorter
    idx = cam_index if cam_index is not None else find_camera(cam_cfg["match_name"])    # --cam-index: operator-verified number wins
    print(f"[{side}] OpenCV camera index {idx}", flush=True); cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wh[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wh[1]); cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
    ok = False
    for _ in range(40):                                      # UVC warm-up: the first reads after open often fail
        ok, f = cap.read()
        if ok: break
        time.sleep(0.05)
    if not ok: raise RuntimeError(f"{side}: 카메라 {cam_cfg['match_name']} 영상 없음 (터미널 카메라 권한 / 다른 프로그램이 사용 중?)")
    und = fisheye_undistorter(ROOT / cam_cfg["intrinsics"], (f.shape[1], f.shape[0]))
    det = make_detector(det_cfg, undistort=und)                    # backend from YAML gripper.detector (student | student2)
    vw = None; ts = []
    if rec_dir is not None:
        (rec_dir / "frames").mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(rec_dir / "frames" / f"{side}_wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cam_cfg["fps"], (f.shape[1], f.shape[0]))
    n = 0; t_last_jpg = 0.0
    while not stop.is_set():
        ok, img = cap.read(); t_cap = time.monotonic_ns()
        if not ok: pub.send("pinch", side, t_ns=t_cap, metric=None, conf=0.0, error="no_frame"); time.sleep(0.01); continue
        r = det.detect(img); t_done = time.monotonic_ns(); n += 1
        pub.send("pinch", side, t_ns=t_cap, metric=r.pinch_metric, conf=r.confidence if r.detected else 0.0,
                 thumb=r.thumb_tip, index=r.index_tip, det_ms=(t_done - t_cap) / 1e6)
        if vw is not None: vw.write(img); ts.append(t_cap)
        if time.monotonic() - t_last_jpg > 0.1:
            t_last_jpg = time.monotonic(); v = und(img)
            for p, col in ((r.thumb_tip, (0, 0, 255)), (r.index_tip, (0, 255, 0))):
                if p is not None: cv2.circle(v, (int(p[0]), int(p[1])), 12, col, 3)
            cv2.putText(v, f"{side} pinch={r.pinch_metric if r.pinch_metric is None else round(r.pinch_metric, 1)} conf={r.confidence:.2f}",
                        (20, 50), 0, 1.2, (0, 255, 255), 3)
            cv2.imwrite(f"/tmp/vt_wrist_{side}.jpg", cv2.resize(v, (640, int(640 * v.shape[0] / v.shape[1]))))
    cap.release()
    if vw is not None: vw.release(); np.save(rec_dir / "frames" / f"{side}_wrist_t_ns.npy", np.asarray(ts, np.int64))


def main() -> None:
    from ..link import Publisher
    ap = argparse.ArgumentParser(); ap.add_argument("--sides", nargs="+", default=["left", "right"]); ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--cam-index", type=int, default=None, help="OpenCV index of the wrist camera (ffmpeg numbering differs!)")
    ap.add_argument("--conf", type=float, default=None, help="override detector confidence (diagnostics only)")
    a = ap.parse_args()
    c = yaml.safe_load((ROOT / "visual_teleop/config/teleop_rebot.yaml").read_text())
    pub = Publisher(c["link"]["pinch_port"]); stop = threading.Event(); rec = Path(a.record).expanduser() if a.record else None
    th = [threading.Thread(target=run_side, args=(s, c["cameras"][f"{s}_wrist"], c["cameras"]["wrist_image_wh"], pub, stop, rec, dict(c["gripper"]["detector"], **({"conf": a.conf} if a.conf is not None else {})), a.cam_index),
                           daemon=True) for s in a.sides]
    for t in th: t.start()
    t0 = time.monotonic()
    try:
        while any(t.is_alive() for t in th) and (a.seconds <= 0 or time.monotonic() - t0 < a.seconds): time.sleep(0.2)
    except KeyboardInterrupt: pass
    stop.set()
    for t in th: t.join(timeout=3)


if __name__ == "__main__":
    main()
