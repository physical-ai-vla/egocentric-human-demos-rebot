"""Right-arm MVP calibrations (2026-10-14). Each writes config/calibration.yaml (auto-merged by config.load) + calibration_log.jsonl.

  imu-mount   R_I_H for one wrist IMU from two still poses (gravity only, no camera):
                A  palm DOWN, fingers pointing FORWARD, horizontal       -> gravity = +x_H
                B  fingers pointing straight DOWN                        -> gravity = +z_H
              H axes (= robot IK TCP axes): z = forearm -> fingertips, y = index -> thumb, x = y cross z.
              Gate: the two gravity directions 90 +- 15 deg apart, both windows still.
      .venv/bin/python -m visual_teleop.tools.calib_right imu-mount --side right

  sync        camera<->IMU time offset: wrist camera frame-to-frame rotation (LK + essential matrix) vs gyro, cross-correlated
              (imu/time_sync.py). Wave / twist the wrist for the whole window. Saved only when graded OK (--accept-warn for WARN).
      .venv/bin/python -m visual_teleop.tools.calib_right sync --side right --seconds 12

  rgbd-level  T_W_C with W := gravity-levelled frame at the fixed camera (z up from the Orbbec accelerometer, x = optical axis
              projected horizontal, origin = camera). Relative teleop needs no table origin; the robot workspace lives in RB.
              Orbbec needs sudo:
      sudo .venv/bin/python -m visual_teleop.tools.calib_right rgbd-level"""
from __future__ import annotations
import argparse
import json
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .. import config as vcfg

G = 9.80665


def open_imu(c, side):
    from handumi_collector.config import ImuCfg
    from handumi_collector.devices.teensy_imu import TeensyImu
    sn = c.imu[side]["serial_number"]
    if not sn: sys.exit(f"imu.{side}.serial_number 가 설정되지 않았습니다")
    d = TeensyImu(ImuCfg(side=side, serial_number=sn, rate_hz=int(c.imu[side]["rate_hz"]))); d.open(); d.start(); return d


class CueBoard:
    """Big on-screen cue (http://127.0.0.1:8731): the voice alone was missed twice (10-14), so every cue is also shown."""
    PAGE = """<!doctype html><html><head><meta charset=utf-8><title>calib cue</title><style>body{margin:0;font-family:-apple-system,sans-serif;
    background:#222;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;text-align:center}
    #t{font-size:6vw;font-weight:700;padding:4vw}#c{font-size:14vw}</style></head><body><div><div id=t>…</div><div id=c></div></div>
    <script>setInterval(async()=>{try{const s=await (await fetch('/s')).json();t.textContent=s.text;c.textContent=s.count;
    document.body.style.background=s.color}catch(e){}},150)</script></body></html>"""

    def __init__(self, port: int = 8731) -> None:
        import http.server, threading, json as _j
        self.state = dict(text="준비", count="", color="#222"); board = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = (_j.dumps(board.state) if self.path.startswith("/s") else board.PAGE).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json" if self.path.startswith("/s") else "text/html; charset=utf-8")
                self.end_headers(); self.wfile.write(body)
            def log_message(self, *a): pass
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H); threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        import subprocess; subprocess.run(["open", f"http://127.0.0.1:{port}"]); time.sleep(1.5)

    def show(self, text: str, count="", color: str = "#222") -> None: self.state = dict(text=text, count=str(count), color=color)


BOARD: CueBoard | None = None
TOOL = "hand"                      # hand | umi (what the operator holds; changes the spoken pose instructions only)


def show(text: str, count="", color: str = "#222") -> None:
    if BOARD is not None: BOARD.show(text, count, color)


def say(msg: str, spoken: str | None = None) -> None:
    """Print the English instruction; SPEAK the full instruction (Korean voice when given) -- a pose name alone is not enough."""
    print(spoken or msg, flush=True)
    try:
        import subprocess; subprocess.run(["say", "-v", "Yuna", spoken] if spoken else ["say", msg], timeout=20)   # blocking: a background say got lost (10-14)
    except Exception: pass


def beep(name: str = "Glass") -> None:
    import subprocess
    try: subprocess.run(["afplay", f"/System/Library/Sounds/{name}.aiff"], timeout=3)
    except Exception: pass


def still_window(dev, prompt: str, settle_s: float = 6.0, hold_s: float = 2.0, tries: int = 3, spoken: str | None = None) -> np.ndarray:
    show(spoken or prompt, "", "#333"); say(prompt, spoken)
    for attempt in range(tries):
        for k in range(int(settle_s if attempt == 0 else 3), 0, -1):
            print(f"  {k}...", flush=True); show(spoken or prompt, k, "#333"); time.sleep(1.0)
        show("멈추세요 — 측정 중", "STOP", "#b00"); beep("Glass"); dev.buffer.drain(); time.sleep(hold_s); S = dev.buffer.drain(); beep("Pop")
        show("OK", "", "#2e7d32")
        acc = np.array([[s.ax, s.ay, s.az] for s in S]); gyr = np.array([[s.gx, s.gy, s.gz] for s in S])
        if len(S) < 50: sys.exit(f"측정 구간의 IMU 샘플이 {len(S)}개뿐입니다 (IMU 연결 확인)")
        gstd = float(np.linalg.norm(gyr.std(0))); an = np.linalg.norm(acc, axis=1)
        print(f"  샘플 {len(S)}개, 자이로 흔들림 {gstd:.3f} rad/s, 가속도 {an.mean():.2f}±{an.std():.2f}")
        if gstd <= 0.08 and an.std() <= 0.3:                          # hand-held: tremor averages out of the 2 s gravity mean
            return -acc.mean(0) / np.linalg.norm(acc.mean(0))         # accelerometer reads +up at rest -> gravity = down
        show("움직였습니다 — 같은 자세로 다시", "", "#b58900"); say("Not still. Hold still.", "움직이지 마세요. 다시 잽니다.")
    sys.exit("  3번 모두 움직임이 감지됨 -- 처음부터 다시 실행하세요")


def imu_mount(c, side: str) -> None:
    dev = open_imu(c, side)
    try:
        if TOOL == "umi":   # H on the UMI: z = along the jaws (tool direction), y = jaw opening axis, x = y cross z
            dA = still_window(dev, "Pose A: UMI level, jaws pointing forward, jaws side by side (left-right). Hold still.",
                              spoken="첫 번째 자세. 유미를 수평으로, 집게 끝이 앞을 향하고 두 집게가 좌우로 나란히. 띵부터 뽁까지 멈추세요.")
            dB = still_window(dev, "Pose B: UMI jaws pointing straight down. Hold still.",
                              spoken="두 번째 자세. 유미 집게 끝이 바닥을 향하게 세우세요. 띵부터 뽁까지 멈추세요.")
        else:
            dA = None
        if dA is None:
          dA = still_window(dev, f"Pose A: {side} hand palm down, fingers pointing forward, horizontal. Hold still.",
                          spoken="첫 번째 자세. 손바닥을 아래로, 손가락은 앞으로 수평. 띵 소리부터 뽁 소리까지 멈추세요.")
          dB = still_window(dev, f"Pose B: {side} arm hanging relaxed at the side, fingers pointing at the floor. Hold still.",
                            spoken="두 번째 자세. 팔을 몸 옆으로 편하게 늘어뜨려서 손가락이 바닥을 향하게 하세요. 띵 소리부터 뽁 소리까지 멈추세요.")
    finally: dev.stop(); dev.close()
    ang = float(np.degrees(np.arccos(np.clip(dA @ dB, -1, 1))))
    print(f"자세 A와 B의 중력 방향 차이: {ang:.1f}도 (정상 90도, 75~105도 통과)")
    if abs(ang - 90) > 15: sys.exit("  두 자세가 90도 정도 차이 나지 않음 -- 다시 (A: 손가락 앞으로 수평, B: 팔 늘어뜨려 손가락이 바닥)")
    x = dA; z = dB - (dB @ x) * x; z /= np.linalg.norm(z); y = np.cross(z, x)
    R_I_H = np.column_stack([x, y, z])
    rv = np.degrees(Rotation.from_matrix(R_I_H).as_rotvec())
    print("R_I_H (열 = IMU 좌표에서 본 손 좌표축):\n", np.round(R_I_H, 3), "\n회전벡터(도)", np.round(rv, 2))
    vcfg.save_calibration({"imu": {"R_I_H": {side: {"measured": True, "rotvec_deg": [round(float(v), 3) for v in rv]}}}},
                          dict(tool="calib_right imu-mount", side=side, gravity_angle_deg=ang, serial=c.imu[side]["serial_number"]))
    print("저장됨: imu.R_I_H." + side)


def sync(c, side: str, seconds: float, accept_warn: bool, cam_index: int | None = None) -> None:
    import cv2, yaml
    from pathlib import Path
    from .fingertip_node import find_camera
    from ..imu.time_sync import estimate_side, fit_device_to_host
    cam = c.cameras[f"{side}_wrist"]; K_y = yaml.safe_load((Path(__file__).resolve().parents[2] / cam["intrinsics"]).read_text())
    K = np.asarray(K_y["K"], float); D = np.asarray(K_y["D"], float).reshape(4, 1); W0, H0 = K_y["image_size"]
    cap = cv2.VideoCapture(cam_index if cam_index is not None else find_camera(cam["match_name"]), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W0); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H0); cap.set(cv2.CAP_PROP_FPS, cam["fps"])
    got = 0
    for _ in range(30):
        ok, _f = cap.read(); got += int(ok)
        if got >= 5: break
        time.sleep(0.05)
    if got < 5:
        sys.exit(f"카메라 '{cam['match_name']}'에서 영상이 안 들어옵니다. 보통 원인: 이 터미널 앱에 카메라 권한이 없음 "
                 "(시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널 허용 후 터미널 재시작), 또는 다른 프로그램"
                 "(fingertip_node / collector)이 카메라를 쓰는 중.")
    dev = open_imu(c, side); dev.buffer.drain()
    show(f"{seconds:.0f}초 동안 손목을 여러 방향으로 계속 비틀고 흔드세요 (카메라는 책상/물건 쪽)", "", "#333")
    say(f"Sync: twist and wave the {side} wrist continuously for {seconds:.0f} seconds, camera seeing a textured scene.",
        f"곧 시작합니다. {seconds:.0f}초 동안 {'유미를' if TOOL == 'umi' else '손목을'} 여러 방향으로 계속 비틀고 흔드세요.")
    for k in (3, 2, 1): show("곧 시작 — 손목 흔들 준비", k, "#333"); time.sleep(1.0)
    beep("Glass"); dev.buffer.drain()
    frames, t_cam = [], []; t0 = time.monotonic(); S = []; last_shown = None
    while time.monotonic() - t0 < seconds:
        rem = int(seconds - (time.monotonic() - t0)) + 1
        if rem != last_shown: show("계속 흔드세요!", rem, "#1565c0"); last_shown = rem
        ok, f = cap.read(); t = time.monotonic_ns()
        if ok: frames.append(cv2.cvtColor(cv2.resize(f, (W0 // 3, H0 // 3)), cv2.COLOR_BGR2GRAY)); t_cam.append(t)
        S += dev.buffer.drain()
    S += dev.buffer.drain(); dev.stop(); dev.close(); cap.release(); beep("Pop"); show("끝 — 계산 중", "", "#2e7d32")
    print(f"영상 {len(frames)}프레임, IMU 샘플 {len(S)}개")
    if len(frames) < 60 or len(S) < 500: sys.exit("데이터가 너무 적음 (카메라 멈춤 또는 IMU 무응답) -- 저장 안 함")
    Ks = K.copy(); Ks[:2] /= 3.0
    Rs = [np.eye(3)]; t_ok = [t_cam[0]]; n_bad = 0
    for a, b, tb in zip(frames[:-1], frames[1:], t_cam[1:]):
        p0 = cv2.goodFeaturesToTrack(a, 400, 0.01, 8)
        R_rel = None
        if p0 is not None and len(p0) >= 30:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
            m = st.ravel() == 1
            if m.sum() >= 30:
                n0 = cv2.fisheye.undistortPoints(p0[m].reshape(-1, 1, 2), Ks, D); n1 = cv2.fisheye.undistortPoints(p1[m].reshape(-1, 1, 2), Ks, D)
                E, inl = cv2.findEssentialMat(n0, n1, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
                if E is not None and E.shape == (3, 3):
                    _, R_rel, _, _ = cv2.recoverPose(E, n0, n1, np.eye(3), mask=inl)
        if R_rel is None: n_bad += 1; continue
        Rs.append(Rs[-1] @ R_rel.T); t_ok.append(tb)
    print(f"영상 회전 추정: {len(Rs)}개 (실패한 프레임 쌍 {n_bad}개)")
    du = np.array([s.device_timestamp_us for s in S]); hn = np.array([s.host_receive_ns for s in S]); gyr = np.array([[s.gx, s.gy, s.gz] for s in S])
    clk = fit_device_to_host(du, hn); t_imu = clk.to_host_ns(du)
    est = estimate_side(side, np.asarray(t_ok), np.asarray(Rs), t_imu, gyr, c.time_sync)
    print(json.dumps(est.to_dict(), indent=1))
    if est.verdict == "OK" or (est.verdict == "WARN" and accept_warn):
        vcfg.save_calibration({"time_sync": {"camera_imu_offset_ms": {side: round(est.offset_ms, 2)}}},
                              dict(tool="calib_right sync", estimate=est.to_dict(), clock_fit=clk.to_dict(), camera=cam["match_name"]))
        print(f"저장됨: 카메라↔IMU 시간차 {side} = {est.offset_ms:.2f} ms (판정 {est.verdict}, 상관 {est.peak_corr:.2f})")
    else: print(f"저장 안 함 (판정 {est.verdict}: {est.reason}) -- 손목을 더 크고 빠르게 비틀면서 다시")


def rgbd_level(c) -> None:
    # Follows pyorbbecsdk examples/beginner/07_imu.py: an IMU-ONLY pipeline (accel + gyro, FULL_FRAME_REQUIRE) and
    # frames.get_accel_frame(). The first version streamed colour + accel in one pipeline and received 0 accel frames (10-14).
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFrameAggregateOutputMode
    pipe = Pipeline()
    try:
        acc_prof = pipe.get_stream_profile_list(OBSensorType.ACCEL_SENSOR).get_stream_profile_by_index(0)
        col_prof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
        ext = acc_prof.get_extrinsic_to(col_prof); R_C_A = np.asarray(ext.rot, np.float64).reshape(3, 3)
    except Exception as e:
        sys.exit(f"Orbbec 가속도계 또는 가속도계→컬러 변환을 못 읽음 ({type(e).__name__}: {e}) -- 이 회전은 추측하지 않습니다")
    if abs(np.linalg.det(R_C_A) - 1) > 1e-3: sys.exit(f"가속도계→컬러 변환이 회전행렬이 아님 (det {np.linalg.det(R_C_A):.4f})")
    cfg = Config(); cfg.enable_accel_stream(); cfg.enable_gyro_stream(); cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipe.start(cfg); time.sleep(1.0); A = []
    say("Level: keep the fixed camera untouched for 3 seconds.", "고정 카메라를 3초 동안 건드리지 마세요.")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        fs = pipe.wait_for_frames(100)
        if fs is None: continue
        f = fs.get_accel_frame()
        if f is not None: v = f.get_value(); A.append([v.x, v.y, v.z])
    pipe.stop()
    A = np.asarray(A, np.float64)
    if len(A) < 20: sys.exit(f"가속도 샘플이 {len(A)}개뿐입니다")
    if np.linalg.norm(A.std(0)) > 0.1: sys.exit("카메라가 흔들렸습니다 -- 건드리지 말고 다시")
    up_C = R_C_A @ (A.mean(0) / np.linalg.norm(A.mean(0)))           # accelerometer at rest reads +up
    ang = lambda v: float(np.degrees(np.arccos(np.clip(up_C @ v, -1, 1))))
    a_imgup, a_back = ang(np.array([0, -1.0, 0])), ang(np.array([0, 0, -1.0]))   # level camera: up = image-up; top-down camera: up = behind
    print(f"가속도 평균(센서 좌표) {np.round(A.mean(0), 3)}, 크기 {np.linalg.norm(A.mean(0)):.3f}; 컬러 카메라 좌표의 '위' 방향 {np.round(up_C, 3)}\n"
          f"'위'와 영상 위쪽 사이 {a_imgup:.1f}도, '위'와 카메라 뒤쪽 사이 {a_back:.1f}도")
    if min(a_imgup, a_back) > 60:
        sys.exit("저장 안 함: '위' 방향이 영상 위쪽(수평 카메라)에도, 카메라 뒤쪽(위에서 내려다보는 카메라)에도 가깝지 않음 -- "
                 "가속도계 부호 규약이 반대이거나 특이한 장착. 어느 쪽인지 정한 뒤 저장 (숫자를 맞추려고 뒤집지 않음)")
    # heading axis = whichever camera axis is more horizontal: optical axis (level / oblique camera) or image-up (top-down camera)
    cands = {"optical_axis": np.array([0, 0, 1.0]), "image_up": np.array([0, -1.0, 0])}
    name, v = max(cands.items(), key=lambda kv: np.linalg.norm(kv[1] - (kv[1] @ up_C) * up_C))
    xw = v - (v @ up_C) * up_C
    if np.linalg.norm(xw) < 0.2: sys.exit("방향을 정할 수평 카메라 축이 없음")
    xw /= np.linalg.norm(xw); zw = up_C; yw = np.cross(zw, xw)
    R_C_W = np.column_stack([xw, yw, zw]); T = np.eye(4); T[:3, :3] = R_C_W.T
    print(f"장착: {'위에서 내려다봄' if a_back < a_imgup else '수평/비스듬'}; W의 x축 = {'광축' if name == 'optical_axis' else '영상 위쪽'}을 수평으로 투영\nT_W_C:\n{np.round(T, 4)}")
    vcfg.save_calibration({"cameras": {"T_W_C": {"measured": True, "matrix": T.round(6).tolist()}}},
                          dict(tool="calib_right rgbd-level", n=len(A), accel_mean=A.mean(0).tolist(), R_C_A=R_C_A.tolist(),
                               W=f"gravity-levelled at the camera: z up, x = {name} projected horizontal, origin = camera",
                               angle_up_imageup_deg=a_imgup, angle_up_back_deg=a_back))
    print("저장됨: cameras.T_W_C")


def pinch(c, side: str, seconds: float) -> None:
    """Per-mount pinch range from two STILL holds (10-15): fingers wide open 3 s, then squeezed 3 s; pinch_open / pinch_closed =
    their medians. Replaces the 10-13 'open/close repeatedly, p5/p95' range: on the back-of-wrist mount that range (2..43) was
    motion noise while open-still vs closed-still read 17.2 vs 15.8 -- the gripper followed noise and felt reversed. Refuses to
    save unless open > closed by >= min_sep (this is also the polarity check). Needs fingertip_node running, control STOPPED."""
    from ..link import Latest
    rx = Latest(c.link["pinch_port"]); min_sep = 10.0
    def hold(text, spoken, secs=3.0):
        show(text, "", "#333"); say(text, spoken)
        for k in (3, 2, 1): show(text, k, "#333"); time.sleep(1.0)
        beep("Glass"); show(text + " — 그대로 유지", "", "#b00"); vals = []; last = None; t0 = time.monotonic(); n = n_any = 0; confs = []
        while time.monotonic() - t0 < secs:
            m = rx.get("pinch", side)
            if m and m["t_ns"] != last:
                last = m["t_ns"]; n += 1; cf = float(m.get("conf") or 0)
                if m.get("metric") is not None: n_any += 1; confs.append(cf)
                if m.get("metric") is not None and cf >= c.tracking["min_fingertip_conf"]: vals.append(float(m["metric"]))
            time.sleep(0.005)
        beep("Pop")
        print(f"  [{text}] 프레임 {n}, 손 검출 {n_any} (신뢰도 중앙값 {np.median(confs) if confs else 0:.2f}), 기준 통과 {len(vals)}")
        return np.asarray(vals)
    o = hold("손가락 활짝 벌리기", "엄지와 검지를 활짝 벌리고 그대로 멈추세요.")
    cl = hold("손가락 꽉 붙이기", "엄지와 검지를 꽉 붙이고 그대로 멈추세요.")
    rx.close()
    print(f"벌림: 값 {len(o)}개, 중앙값 {np.median(o) if len(o) else float('nan'):.1f} | 붙임: 값 {len(cl)}개, 중앙값 {np.median(cl) if len(cl) else float('nan'):.1f}")
    if len(o) < 20 or len(cl) < 20: show("손이 잘 안 잡힘 — 다시", "", "#b58900"); sys.exit("손이 거의 안 잡혔음 (엄지·검지가 손목 카메라에 보이는지 확인) -- 저장 안 함")
    mo, mc = float(np.median(o)), float(np.median(cl))
    if mo - mc < min_sep:
        show("벌림과 붙임이 구분 안 됨 — 저장 안 함", "", "#b00")
        sys.exit(f"벌림({mo:.1f})과 붙임({mc:.1f})이 {min_sep} 이상 차이 나지 않음 -- 이 장착에서는 손가락 벌림이 안 보임. 저장 안 함")
    rng = mo - mc
    vcfg.save_calibration({"gripper": {"pinch_closed": round(mc, 2), "pinch_open": round(mo, 2),
                                       "close_threshold": round(mc + 0.4 * rng, 2), "open_threshold": round(mc + 0.6 * rng, 2)}},
                          dict(tool="calib_right pinch (still holds)", side=side, n_open=len(o), n_closed=len(cl), open_median=mo, closed_median=mc,
                               open_iqr=np.percentile(o, [25, 75]).tolist(), closed_iqr=np.percentile(cl, [25, 75]).tolist()))
    show("완료", "", "#2e7d32")
    print(f"저장됨: 붙임 {mc:.1f} = 완전히 닫힘, 벌림 {mo:.1f} = 완전히 열림 (차이 {rng:.1f})")


def c922_level(c, seconds: float = 3.0) -> None:
    """T_W_C for the fixed C922 (no accelerometer): the UMI AprilTag lies FLAT on the table; its plane normal = up. Sign: the
    camera is above the table, so up must point from the tag toward the camera side (up . (-t_C_tag) > 0). W: z up,
    x = optical axis projected horizontal, origin = camera. The table is assumed level (state it if it is not)."""
    from ..vision.tag_camera import TagCamera
    from ..vision.tag_pose import UmiTagTracker
    tg = c.fusion["tag"]; cam = TagCamera(c.cameras["tag_cam"]); print("카메라:", cam.index, cam.focus)
    tr = UmiTagTracker(np.eye(4), family=tg["family"], tag_id=tg["id"], size_m=tg["size_m"], min_margin=tg["min_margin"], use_depth=False)
    show("태그를 책상 위에 평평하게 놓고 손을 떼세요", "", "#333"); say("Lay the tag flat on the table.", "태그를 책상 위에 평평하게 놓고 손을 떼세요.")
    for k in (3, 2, 1): show("태그를 책상에 평평하게", k, "#333"); time.sleep(1)
    beep("Glass"); zs, ts = [], []; t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        f = cam.read()
        if f is None: continue
        ob = tr.detect(f)
        if ob is not None: zs.append(ob.T_W_tag[:3, 2]); ts.append(ob.T_W_tag[:3, 3])
    beep("Pop"); cam.close()
    if len(zs) < 20: show("태그가 안 보임", "", "#b00"); sys.exit(f"태그 검출 {len(zs)}회뿐 -- C922에 태그가 보이는지, 크기 설정({tg['size_m']} m)이 맞는지 확인")
    Z = np.asarray(zs); n = Z.mean(0); n /= np.linalg.norm(n); spread = float(np.degrees(np.arccos(np.clip(Z @ n, -1, 1))).max())
    t = np.mean(ts, axis=0)
    up = n if n @ (-t) > 0 else -n
    if spread > 3.0: sys.exit(f"태그 방향이 흔들림 (최대 {spread:.1f}도) -- 태그가 평평하게 놓였는지, 조명 확인")
    opt = np.array([0, 0, 1.0]); xw = opt - (opt @ up) * up
    if np.linalg.norm(xw) < 0.2: sys.exit("카메라가 거의 수직으로 내려다봄: 방향을 정할 수평 축이 없음")
    xw /= np.linalg.norm(xw); yw = np.cross(up, xw); R_C_W = np.column_stack([xw, yw, up]); T = np.eye(4); T[:3, :3] = R_C_W.T
    tilt = float(np.degrees(np.arccos(np.clip(up @ np.array([0, -1.0, 0]), -1, 1))))
    print(f"태그 검출 {len(zs)}회, 법선 흔들림 최대 {spread:.2f}도, 태그까지 거리 {np.linalg.norm(t):.3f} m\n"
          f"카메라 기울기(수평 기준) {tilt:.1f}도\nT_W_C:\n{np.round(T, 4)}")
    vcfg.save_calibration({"cameras": {"T_W_C": {"measured": True, "matrix": T.round(6).tolist()}}},
                          dict(tool="calib_right c922-level", camera=c.cameras["tag_cam"]["match_name"], n=len(zs), normal_spread_deg=spread,
                               tag_dist_m=float(np.linalg.norm(t)), focus=cam.focus, assumption="table level"))
    show("완료", "", "#2e7d32"); print("저장됨: cameras.T_W_C (C922)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["imu-mount", "sync", "rgbd-level", "pinch", "c922-level"]); ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--seconds", type=float, default=12.0); ap.add_argument("--accept-warn", action="store_true")
    ap.add_argument("--no-board", action="store_true", help="no on-screen cue page")
    ap.add_argument("--tool", choices=["hand", "umi"], default="hand", help="umi: pose instructions for the hand-held HandUMI")
    ap.add_argument("--cam-index", type=int, default=None, help="OpenCV index of the wrist camera (ffmpeg numbering differs)")
    a = ap.parse_args(); c = vcfg.load()
    global BOARD, TOOL
    TOOL = a.tool
    if not a.no_board: BOARD = CueBoard()
    if a.what == "imu-mount": imu_mount(c, a.side)
    elif a.what == "sync": sync(c, a.side, a.seconds, a.accept_warn, a.cam_index)
    elif a.what == "pinch": pinch(c, a.side, a.seconds if a.seconds != 12.0 else 8.0)
    elif a.what == "c922-level": c922_level(c)
    else: rgbd_level(c)
    print(f"{a.side} 남은 캘리브레이션:", vcfg.unmeasured(vcfg.load(), (a.side,)) or "없음 — 완료")


if __name__ == "__main__":
    main()
