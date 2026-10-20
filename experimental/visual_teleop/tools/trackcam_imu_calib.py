"""Tracking fisheye <-> UMI IMU: rotation R_CI + time offset, measured per session (2026-10-15, DIY wrist tracker step 2).
Hub must be OFF (this reads the Teensy). 30 s, spoken: mixed yaw/pitch/roll, slow AND fast. Saves raw video + IMU for the VIO eval.

  visual: LK on the 1080p frames downscaled /3, KB4 undistortion (configs/calibration/trackcam_fisheye_v001.yaml), essential + recoverPose
          -> per-pair camera rotation (camera frame)
  offset: reuse imu/time_sync.estimate_side (rotation-magnitude cross-correlation; never a hard-coded value)
  R_CI:   per pair, gyro integrated over the same (offset-shifted) interval in the IMU frame vs the visual rotation vector in the
          camera frame -> Kabsch, with the rotation sign convention chosen by fit quality; residual angular-rate RMS reported.
    .venv/bin/python -m visual_teleop.tools.trackcam_imu_calib [--index 4] [--location 0x00140000] [--seconds 30] [--exposure 15]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from .. import config as vcfg
from .wrist_frontend_check import lock_exposure, say

ROOT = Path(__file__).resolve().parents[2]
VIO_PROTO = [   # DIY tracker spec tests A-E (~72 s)
    (10, "A. 십 초 동안 가만히 들고 계세요"),
    (4, "B. 자를 따라 정확히 십 센티 앞으로 옮기고 멈추세요"), (4, "다시 십 센티 뒤로 옮기고 멈추세요"), (4, "앞으로 십 센티, 멈추세요"), (4, "뒤로 십 센티, 멈추세요"),
    (4, "앞으로 십 센티, 멈추세요"), (4, "뒤로 십 센티, 멈추세요"),
    (12, "C. 위치는 그대로 두고 좌우, 끄덕, 비틀기를 삼십 도씩 하세요"),
    (5, "D. 이리저리 움직이세요"), (3, "멈추세요"), (5, "다시 움직이세요"), (3, "멈추세요"),
    (10, "E. 천 쪽으로 손을 뻗어 천을 잡았다 놓으세요")]
BOARD_PROTO = [   # Basalt camera / camera-IMU calibration against a FIXED AprilGrid (~75 s); the board must stay in view
    (5, "보드가 화면 가운데 오게 들고 가만히 계세요"), (10, "보드를 보면서 천천히 앞뒤 좌우 위아래로 움직이세요"),
    (10, "보드를 보면서 천천히 좌우로 돌리고 끄덕이고 비트세요"), (10, "보드가 화면 귀퉁이에도 오도록 움직이세요"),
    (10, "가깝게 그리고 멀게 움직이세요"), (15, "이제 빠르게, 모든 방향으로 흔들고 돌리세요. 보드는 화면 안에"), (10, "다시 천천히 움직이세요"), (5, "가만히 계세요")]
BOARD_SLOW_PROTO = [   # VIO-vs-board: STILL start (VIO init assumes zero velocity), slow motion only, still end (~70 s)
    (8, "보드가 화면 가운데 오게 들고, 완전히 가만히 계세요"), (12, "보드를 보면서 천천히 앞뒤 좌우 위아래로 움직이세요"),
    (12, "천천히 좌우로 돌리고 끄덕이고 비트세요"), (12, "보드가 화면 귀퉁이에도 오도록 천천히 움직이세요"),
    (12, "천천히 가깝게 그리고 멀게 움직이세요"), (8, "가만히 계세요")]
BOARD_TRANS_PROTO = [   # VIO scale test: still start, brisk TRANSLATION (scale needs acceleration), little rotation, still end (~62 s)
    (8, "보드가 화면 가운데 오게 들고, 완전히 가만히 계세요"), (12, "앞뒤로 빠르게 왕복하세요. 10에서 20센티, 회전은 하지 마세요"),
    (12, "좌우로 빠르게 왕복하세요"), (12, "위아래로 빠르게 왕복하세요"), (10, "모든 방향을 섞어서 빠르게 왕복하세요"), (8, "가만히 계세요")]
PROTO = [(3, "가만히 들고 계세요"), (7, "좌우로 돌리세요, 천천히 그리고 빠르게"), (7, "위아래로 끄덕이세요, 천천히 그리고 빠르게"),
         (7, "손목을 비트세요, 천천히 그리고 빠르게"), (6, "여러 방향을 섞어서 흔드세요")]


def kabsch(a, b):
    """R minimising |b - R a| over rows."""
    U, _, Vt = np.linalg.svd(b.T @ a); d = np.sign(np.linalg.det(U @ Vt)); return U @ np.diag([1, 1, d]) @ Vt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=4); ap.add_argument("--location", default="0x00140000"); ap.add_argument("--exposure", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=30.0); ap.add_argument("--side", default="right")
    ap.add_argument("--board", action="store_true", help="~75 s AprilGrid recording for basalt_calibrate / basalt_calibrate_imu")
    ap.add_argument("--board-slow", action="store_true", help="~64 s still-start slow-only board take for VIO vs ground truth"); ap.add_argument("--board-trans", action="store_true", help="~62 s still-start brisk-translation board take (VIO scale test)"); ap.add_argument("--save-scale", type=int, default=3, help="saved video = 1080p / this (3 -> 640x360, 2 -> 960x540)")
    ap.add_argument("--vio-test", action="store_true", help="record the ~72 s VIO evaluation protocol (A-E) instead of the 30 s calibration"); ap.add_argument("--intrinsics", default="configs/calibration/trackcam_fisheye_v001.yaml")
    a = ap.parse_args(); c = vcfg.load()
    y = yaml.safe_load((ROOT / a.intrinsics).read_text()); K = np.asarray(y["K"], float).reshape(3, 3); D = np.asarray(y["D"], float).reshape(4, 1); W0, H0 = y["image_size"]
    out = ROOT / "datasets" / "trackcam_imu" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FRAME_WIDTH, W0); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H0); cap.set(cv2.CAP_PROP_FPS, 30)
    print("[camimu]", lock_exposure(a.location, a.exposure), flush=True)
    for _ in range(10): cap.read()
    from .calib_right import open_imu
    dev = open_imu(c, a.side); dev.buffer.drain()
    sc = a.save_scale; vw = cv2.VideoWriter(str(out / "tracking_cam.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 30.0, (W0 // sc, H0 // sc))
    sched, acc = [], 0.0; proto = BOARD_TRANS_PROTO if a.board_trans else BOARD_SLOW_PROTO if a.board_slow else BOARD_PROTO if a.board else (VIO_PROTO if a.vio_test else PROTO)
    for d, s in proto: sched.append((acc, s)); acc += d
    if a.vio_test or a.board or a.board_slow or a.board_trans: a.seconds = acc
    (out / "protocol.json").write_text(json.dumps([dict(t0=st, text=tx) for st, tx in sched], ensure_ascii=False)); (out / "meta.json").write_text(json.dumps(dict(save_scale=sc, size=[W0 // sc, H0 // sc])))
    say("카메라와 아이엠유 맞춤 녹화를 시작합니다"); time.sleep(3); dev.buffer.drain()
    frames, t_cam, S = [], [], []; t0 = time.monotonic(); ph = -1
    while (el := time.monotonic() - t0) < a.seconds:
        k = max(i for i, (st, _) in enumerate(sched) if st <= el)
        if k != ph: ph = k; say(sched[k][1]); print(f"[camimu] {el:4.1f}s {sched[k][1]}", flush=True)
        ok, f = cap.read(); t = time.monotonic_ns()
        if ok:
            g = cv2.cvtColor(cv2.resize(f, (W0 // sc, H0 // sc), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); frames.append(g); t_cam.append(t); vw.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        S += dev.buffer.drain()
    S += dev.buffer.drain(); dev.stop(); dev.close(); cap.release(); vw.release(); say("녹화가 끝났습니다. 계산합니다.")
    if a.vio_test or a.board or a.board_slow or a.board_trans:
        np.save(out / "tracking_cam_t_ns.npy", np.asarray(t_cam, np.int64))
        np.savez(out / f"imu_{a.side}.npz", device_us=np.array([s.device_timestamp_us for s in S]), host_ns=np.array([s.host_receive_ns for s in S]),
                 gyro=np.array([[s.gx, s.gy, s.gz] for s in S]), acc=np.array([[s.ax, s.ay, s.az] for s in S]))
        print(f"[camimu] VIO 평가 녹화 저장 {out} (영상 {len(frames)}, IMU {len(S)})", flush=True); return
    du = np.array([s.device_timestamp_us for s in S]); hn = np.array([s.host_receive_ns for s in S]); gyr = np.array([[s.gx, s.gy, s.gz] for s in S]); acc_ = np.array([[s.ax, s.ay, s.az] for s in S])
    np.save(out / "tracking_cam_t_ns.npy", np.asarray(t_cam, np.int64)); np.savez(out / f"imu_{a.side}.npz", device_us=du, host_ns=hn, gyro=gyr, acc=acc_)
    print(f"[camimu] 영상 {len(frames)}프레임 ({len(frames) / a.seconds:.1f} fps), IMU {len(S)}개 ({len(S) / a.seconds:.0f} Hz)", flush=True)
    Ks = K.copy(); Ks[:2] /= 3.0; Rs = [np.eye(3)]; t_ok = [t_cam[0]]; pairs = []; n_bad = 0
    for i0 in range(len(frames) - 1):
        f0, f1 = frames[i0], frames[i0 + 1]; p0 = cv2.goodFeaturesToTrack(f0, 400, 0.01, 8); R_rel = None
        if p0 is not None and len(p0) >= 30:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(f0, f1, p0, None); m = st.ravel() == 1
            if m.sum() >= 30:
                n0 = cv2.fisheye.undistortPoints(p0[m].reshape(-1, 1, 2), Ks, D); n1 = cv2.fisheye.undistortPoints(p1[m].reshape(-1, 1, 2), Ks, D)
                E, inl = cv2.findEssentialMat(n0, n1, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
                if E is not None and E.shape == (3, 3): _, R_rel, _, _ = cv2.recoverPose(E, n0, n1, np.eye(3), mask=inl)
        if R_rel is None: n_bad += 1; continue
        Rs.append(Rs[-1] @ R_rel.T); t_ok.append(t_cam[i0 + 1]); pairs.append((t_cam[i0], t_cam[i0 + 1], R_rel))
    from ..imu.time_sync import estimate_side, fit_device_to_host
    clk = fit_device_to_host(du, hn); t_imu = clk.to_host_ns(du)
    est = estimate_side(a.side, np.asarray(t_ok), np.asarray(Rs), t_imu, gyr, c.time_sync)
    print(f"[camimu] 시간 차 {est.offset_ms} ms, 상관 {est.peak_corr:.2f}, 판정 {est.verdict} (영상 회전 실패 쌍 {n_bad})", flush=True)
    off_ns = (est.offset_ms or 0.0) * 1e6; tI = t_imu + off_ns
    A, B, Wts = [], [], []
    for ta, tb, Rr in pairs:                                        # gyro integrated over the same interval (small-angle sum)
        m = (tI >= ta) & (tI < tb)
        if m.sum() < 3: continue
        dt = np.diff(np.r_[tI[m], tb]) / 1e9; th_i = (gyr[m] * dt[:, None]).sum(0); th_c = R.from_matrix(Rr).as_rotvec()
        if np.linalg.norm(th_i) > np.radians(0.5): A.append(th_i); B.append(th_c)
    A, B = np.array(A), np.array(B); best = None
    for sgn in (1.0, -1.0):                                         # recoverPose's R maps points 0->1, i.e. the INVERSE camera motion
        Rci = kabsch(A, sgn * B); res = sgn * B - A @ Rci.T; rms = float(np.degrees(np.sqrt(np.mean(np.sum(res ** 2, 1)))))
        if best is None or rms < best[1]: best = (Rci, rms, sgn)
    Rci, rms, sgn = best; mag = float(np.degrees(np.median(np.linalg.norm(A, axis=1))))
    rep = dict(offset_ms=est.offset_ms, offset_corr=round(est.peak_corr, 3), offset_verdict=est.verdict, n_pairs=int(len(A)), visual_fail_pairs=n_bad,
               R_CI=Rci.round(6).tolist(), R_CI_rotvec_deg=np.degrees(R.from_matrix(Rci).as_rotvec()).round(2).tolist(), sign=sgn,
               residual_rot_per_frame_deg_rms=round(rms, 3), median_rot_per_frame_deg=round(mag, 3),
               fps=round(len(frames) / a.seconds, 1), imu_hz=round(len(S) / a.seconds, 1), intrinsics=a.intrinsics, clock_fit=clk.to_dict())
    (out / "camimu.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps({k: rep[k] for k in rep if k != "clock_fit"}, indent=1, ensure_ascii=False)); print(f"[camimu] 저장 {out}")


if __name__ == "__main__":
    main()
