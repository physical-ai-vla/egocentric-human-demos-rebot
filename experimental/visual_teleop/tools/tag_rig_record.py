"""Orbbec RGB-D recording for the UMI AprilTag rig (tag36h11 IDs 200-202) -- 2026-10-16, global-anchor plan (user: depth cam + 3 tags,
slow x/y/z + yaw correction of OpenVINS). Needs sudo on macOS (pyorbbecsdk). Spoken Korean protocol, live tag counts every second:
    still   UMI held still with >= 2 tags visible (tag-to-tag agreement / noise floor)
    rotate  slow turns so every PAIR of tags is seen together at several angles (rig bundle: T_rig_tag_i)
    still   end
Saves datasets/tag_rig/<ts>/: color_%05d.jpg (q95), depth.npz (uint16 aligned to colour, metres = value * depth_scale), t_ns.npy (host
monotonic at receive), meta.json (intrinsics fx fy cx cy, depth_scale, protocol with t0). Files are chowned back to $SUDO_USER.
    sudo .venv/bin/python -m visual_teleop.tools.tag_rig_record [--proto rig|pivot] [--hz 15]
10-17: --proto rig = both UMIs (still, then each UMI turned slowly); --proto pivot = jaw tip held on one desk point while the UMI is tilted
in every direction (T_rig_TCP by pivot least squares), right UMI then left UMI. Both XIAO IMUs are logged too (imu_{left,right}.npz:
device_us, host_ns, gyro rad/s, acc m/s^2) for the IMU<->rig rotation and time offset.
10-20: --proto robotcal = free recording of --secs (Ctrl+C ends and still saves) while visual_teleop.tools.robot_cam_poses moves the robot
through its calibration poses; the preview also serves /tags (latest tag ids + pixel sides, JSON) so the mover can check tag 220 per pose.
--proto axes = 6-direction translation strokes per UMI (replay parity, GPT plan step 6): forward/back -> left/right -> up/down relative to
the ROBOT base, orientation and jaw fixed, --reps times, hold 3 s between strokes (scripts/umi_tcp_traj.py + scripts/replay_parity.py)."""
from __future__ import annotations
import argparse, json, os, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAG_IDS = (200, 201, 202, 210, 211, 212, 220, 221, 222, 230, 231, 232)   # 10-17: UMI R 200-202, UMI L 210-212, robot R 220-222, robot L 230-232


PREV = {"jpg": None, "tags": {}}
COL = {200: (0, 220, 0), 210: (255, 0, 255), 220: (0, 220, 255), 230: (0, 220, 255)}


def start_preview(port: int) -> None:
    """10-17 (user: "녹화과정도 보고 싶어"): MJPEG of the Orbbec colour with tag overlays + phase/timer, http://localhost:<port>/"""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/tags"):           # 10-20: for robot_cam_poses (tag 220 visible at this pose?)
                b = json.dumps(PREV["tags"]).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b); return
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        if PREV["jpg"]: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + PREV["jpg"] + b"\r\n")
                        time.sleep(0.07)
                except (BrokenPipeError, ConnectionResetError): return
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>")
        def log_message(self, *a): pass
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever, daemon=True).start()
    print(f"[태그] 미리보기 http://localhost:{port}/", flush=True)


def annotate(img, cs, ids, title: str):
    v = img.copy()
    for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
        i = int(i); px = c.reshape(4, 2); side = float(np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1)))
        if i not in TAG_IDS: continue
        cv2.polylines(v, [px.astype(np.int32)], True, COL.get(i // 10 * 10, (255, 255, 255)), 3)
        org = tuple(int(x) for x in px[0] + [0, -8]); cv2.putText(v, f"{i} {side:.0f}px", org, 0, 0.8, (0, 0, 0), 5)
        cv2.putText(v, f"{i} {side:.0f}px", org, 0, 0.8, (0, 0, 255) if side < 30 else (255, 255, 255), 2)
    cv2.rectangle(v, (0, 0), (v.shape[1], 44), (0, 0, 0), -1); cv2.putText(v, title, (10, 31), 0, 0.85, (0, 255, 0), 2)
    return v


def say(t: str) -> None:
    print("[태그]", t, flush=True)
    try: subprocess.Popen(["say", "-v", "Yuna", t])
    except Exception: pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--still-s", type=float, default=20.0); ap.add_argument("--rotate-s", type=float, default=40.0); ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--proto", default="rig", choices=["rig", "pivot", "view", "corners", "robotcal", "axes"]); ap.add_argument("--hand", default="both", choices=["right", "left", "both"]); ap.add_argument("--port", type=int, default=8747)
    ap.add_argument("--secs", type=float, default=900.0); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--stroke-cm", type=int, default=15); ap.add_argument("--axes", default="XYZ", help="axes 프로토콜에서 할 축만 (예: X)")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720); a = ap.parse_args()
    from ego_collector.camera.capture import CaptureConfig
    from ego_collector.camera.orbbec_capture import OrbbecCapture
    cap = OrbbecCapture(CaptureConfig(index=0, width=a.width, height=a.height, fps=30)); cap.open()
    out = ROOT / "datasets/tag_rig" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    start_preview(a.port)
    if a.proto == "view":                      # 10-17: preview only (aim the Orbbec / tags), nothing saved, Ctrl+C to stop
        out.rmdir(); det_v = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)); t0v = time.monotonic()
        print("[태그] 미리보기만 (저장 안 함). 끄려면 Ctrl+C", flush=True)
        try:
            while True:
                f = cap.get(timeout=0.5)
                if f is None: continue
                cs, ids, _ = det_v.detectMarkers(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)); vis = sorted(int(i) for i in ids.ravel() if int(i) in TAG_IDS) if ids is not None else []
                PREV["jpg"] = cv2.imencode(".jpg", cv2.resize(annotate(f.image, cs, ids, f"VIEW {time.monotonic() - t0v:5.0f}s  tags {vis}"), (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        except KeyboardInterrupt: pass
        cap.close(); return
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    if a.proto == "rig":
        proto = [(5.0, "양손 유미를 오르벡 카메라 쪽으로 들고, 태그가 두 개 이상 보이게 맞추세요"), (a.still_s, "그대로 완전히 가만히 계세요"),
                 (a.rotate_s, "이제 오른손 유미를 천천히 돌리면서, 태그 두 개가 함께 보이는 여러 각도를 보여 주세요"),
                 (a.rotate_s, "이제 왼손 유미를 천천히 돌리면서, 태그 두 개가 함께 보이는 여러 각도를 보여 주세요"), (5.0, "멈추고 가만히 계세요")]
    elif a.proto == "corners":           # 10-19 baseline accuracy (GPT review): every board corner 4x with different UMI tilts, 3 s hold each
        proto = [(6.0, "모서리 시험입니다. 안내되는 모서리에 집게 끝을 대고 멈추세요")]
        for hand in (["오른손", "왼손"] if a.hand == "both" else ["오른손" if a.hand == "right" else "왼손"]):
            for corner in ("왼쪽 아래", "오른쪽 아래", "오른쪽 위", "왼쪽 위"):
                for tilt in ("태그가 카메라를 보게", "다시 대고", "살짝 다른 각도로", "다시 대고"):   # 10-20: the Orbbec sees the tags only near this pose (side tilts hid them)
                    proto += [(6.0, f"{hand}, {corner}, {tilt}, 멈추세요"), (1.5, "떼세요")]      # analysis uses the last 1.5 s of each hold
        proto += [(3.0, "끝났습니다")]
    elif a.proto == "robotcal":         # 10-20: robot base <-> Orbbec; the robot moves itself (robot_cam_poses), this only records
        proto = [(a.secs, "로봇 캘리브레이션 녹화를 시작합니다. 끝나면 컨트롤 C")]
    elif a.proto == "axes":             # 10-20 replay parity: 6 directions relative to the ROBOT base, slow (~4 s per stroke), orientation + jaw fixed
        proto = [(6.0, "방향 시험입니다. 집게를 닫은 채, 손목 각도를 고정하고, 로봇 기준 방향으로 움직이세요")]
        for hand in (["오른손", "왼손"] if a.hand == "both" else ["오른손" if a.hand == "right" else "왼손"]):
            for ax, (d1, d2) in (("X", ("앞으로", "뒤로")), ("Y", ("왼쪽으로", "오른쪽으로")), ("Z", ("위로", "아래로"))):
                if ax not in a.axes.upper(): continue
                for r in range(a.reps):
                    for dname in (d1, d2):
                        proto += [(3.0, f"{hand}, {ax}축, {r + 1}회, 멈추세요"), (4.0, f"{hand} {dname} {a.stroke_cm}센티 천천히"), (3.0, "멈추세요"), (4.0, "제자리로 천천히")]
        proto += [(3.0, "멈추세요"), (2.0, "끝났습니다")]
    else:
        proto = [(6.0, "오른손 유미의 집게 끝을 책상의 표시한 점에 대세요"), (30.0, "집게 끝을 떼지 말고, 유미를 앞뒤 좌우로 천천히 기울이세요"),
                 (6.0, "이제 왼손 유미의 집게 끝을 같은 점에 대세요"), (30.0, "집게 끝을 떼지 말고, 유미를 앞뒤 좌우로 천천히 기울이세요"), (4.0, "멈추세요")]
    imus = {}
    try:
        from visual_teleop import config as vcfg
        from visual_teleop.tools.calib_right import open_imu
        for side in ("right", "left"):
            try: imus[side] = open_imu(vcfg.load(), side); imus[side].buffer.drain()
            except Exception as e: print(f"[태그] {side} IMU 없음: {e}", flush=True)
    except Exception as e: print(f"[태그] IMU 열기 실패: {e}", flush=True)
    IMU = {sd: [] for sd in imus}
    t_start = time.monotonic(); sched, acc = [], 0.0
    for d, txt in proto: sched.append(dict(t0=acc, dur=d, text=txt)); acc += d
    T, D, k, nxt, last_print, seen = [], [], 0, 0, 0.0, []
    phase = -1; long_take = a.proto in ("robotcal", "axes")
    while True:
      try:
        el = time.monotonic() - t_start
        if el >= acc: break
        p = max(i for i, s in enumerate(sched) if el >= s["t0"])
        if p != phase: phase = p; say(sched[p]["text"])
        f = cap.get(timeout=0.5)
        if f is None or getattr(f, "depth", None) is None:
            if k == 0 and el > 6.0:      # 10-16 first run: the Orbbec opened but never delivered a frame for 85 s -> fail fast
                cap.close(); raise SystemExit(f"[태그] 6초 동안 Orbbec 프레임 0장 (읽기 실패 {getattr(cap, 'read_failures', '?')}회). "
                                              "USB 재연결(허브 말고 Mac 본체 포트) 후 다시, 계속되면 Mac 재부팅 (예전 Orbbec 멈춤 증상)")
            continue
        for sd, dv in imus.items(): IMU[sd] += dv.buffer.drain()
        if time.monotonic() < nxt: continue
        nxt = time.monotonic() + 1.0 / a.hz
        img = f.image; cv2.imwrite(str(out / f"color_{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if long_take: cv2.imwrite(str(out / f"depth_{k:05d}.png"), np.asarray(f.depth, np.uint16))   # 10-20: 15 min x 15 Hz in RAM = 25 GB
        else: D.append(np.asarray(f.depth, np.uint16))
        T.append(time.monotonic_ns()); k += 1
        cs, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)); ids_raw = ids
        sides = [float(np.mean(np.linalg.norm(c.reshape(4, 2) - np.roll(c.reshape(4, 2), 1, 0), axis=1))) for c, i in zip(cs, ids.ravel()) if int(i) in TAG_IDS] if ids is not None else []
        ids = sorted(int(i) for i in ids.ravel() if int(i) in TAG_IDS) if ids is not None else []
        seen.append(len(ids))
        PREV["tags"] = {"t_ns": T[-1], "k": k - 1, "ids": ids, "side_px": {str(int(i)): round(float(np.mean(np.linalg.norm(c.reshape(4, 2) - np.roll(c.reshape(4, 2), 1, 0), axis=1))), 1)
                                                                        for c, i in zip(cs, ids_raw.ravel()) if int(i) in TAG_IDS} if ids_raw is not None else {}}
        ph_n = (["READY", "STILL", "ROTATE RIGHT UMI", "ROTATE LEFT UMI", "STOP"] if a.proto == "rig" else ["RIGHT TIP ON DOT", "RIGHT: TILT, KEEP TIP", "LEFT TIP ON DOT", "LEFT: TILT, KEEP TIP", "STOP"]
                if a.proto == "pivot" else ["ROBOT CAL (Ctrl+C = stop)"] if a.proto == "robotcal" else [("HOLD " + x["text"].split(",")[1]) if "멈추세요" in x["text"] and "," in x["text"]
              else "MOVE" if "천천히" in x["text"] else "HOLD" for x in sched] if a.proto == "axes" else ["CORNER TOUCH / HOLD" if "멈추세요" in x["text"] else "LIFT" for x in sched])
        title = f"{a.proto.upper()} {phase + 1}/{len(sched)} {ph_n[min(phase, len(ph_n) - 1)]}  {el - sched[phase]['t0']:4.1f}/{sched[phase]['dur']:.0f}s  tags {ids}"
        PREV["jpg"] = cv2.imencode(".jpg", cv2.resize(annotate(img, cs, ids_raw, title), (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        if time.monotonic() - last_print > 1.0:
            last_print = time.monotonic(); print(f"[태그] {el:5.1f}s 프레임 {k} | 보이는 태그 {ids} 크기 {[round(x) for x in sides]}px (30px 이상 권장)  (2개 이상 비율 {np.mean(np.array(seen) >= 2):.2f})", flush=True)
      except KeyboardInterrupt:
        print("[태그] Ctrl+C -- 지금까지 저장합니다", flush=True); break
    cap.close(); say("끝났습니다")
    # 10-17: the pivot take lost meta/t_ns/depth (crash after the frames) -> write the small, essential files FIRST, each step guarded
    np.save(out / "t_ns.npy", np.array(T, np.int64))
    (out / "meta.json").write_text(json.dumps(dict(intrinsics=cap.intrinsics, depth_scale=cap.depth_scale, size=[a.width, a.height], hz=a.hz, tag_ids=TAG_IDS,
                                                    protocol=sched, proto=a.proto, frames=k, t_start_monotonic_ns=int(t_start * 1e9)), ensure_ascii=False, indent=1))
    for sd, dv in imus.items():
        try:
            IMU[sd] += dv.buffer.drain(); dv.stop(); dv.close(); S_ = IMU[sd]
            np.savez(out / f"imu_{sd}.npz", device_us=np.array([x.device_timestamp_us for x in S_]), host_ns=np.array([x.host_receive_ns for x in S_]),
                     gyro=np.array([[x.gx, x.gy, x.gz] for x in S_]), acc=np.array([[x.ax, x.ay, x.az] for x in S_]))
            print(f"[태그] {sd} IMU 샘플 {len(S_)}", flush=True)
        except Exception as e: print(f"[태그] {sd} IMU 저장 실패: {e}", flush=True)
    try:
        if D: np.savez_compressed(out / "depth.npz", depth=np.stack(D))
    except Exception as e: print(f"[태그] depth 저장 실패: {e}", flush=True)
    (out / "meta.json").write_text(json.dumps({**json.loads((out / "meta.json").read_text()), "depth_files": "png" if long_take else "npz"}, ensure_ascii=False, indent=1))
    u = os.environ.get("SUDO_UID"); g = os.environ.get("SUDO_GID")
    if u and g:
        for pth in [out, *out.iterdir()]: os.chown(pth, int(u), int(g))
        os.chown(out.parent, int(u), int(g))
    print(f"[태그] 저장 {out} | 프레임 {k} | 태그 2개 이상 보인 비율 {np.mean(np.array(seen) >= 2):.2f}, 1개 이상 {np.mean(np.array(seen) >= 1):.2f}", flush=True)


if __name__ == "__main__":
    main()
