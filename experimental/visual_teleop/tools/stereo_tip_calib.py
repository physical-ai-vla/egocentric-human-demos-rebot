"""Jaw-tip TCP translation for the stereo VIO tracker by STEREO TRIANGULATION of the orange jaw tape (2026-10-16).
Replaces `calib_tracker pivot` for this tracker: a pivot keeps rotating at 80-150 deg/s around a close point and the stereo fisheyes lost all
features (tracking HOLD for 36 s, live 10-16). Both cameras see the orange tips directly, so one still moment is enough:
  per camera: HSV orange mask -> largest blob in the lower image half -> the tip = mean of the blob's top 8 % rows (jaws point up/away)
  KB4 bearings (stereo_trackcam_v004) -> midpoint triangulation with T_CR_CL -> p_C0 (tip in the left camera)
  r = T_I_C0 p_C0 (tracker body = IMU frame, same T_I_C0 the tracker's OpenVINS config uses: R_C0I from stereo_trackcam_imu_v002 + --t-ci)
Saved as fusion.tracker.T_TB_TCP translation with pivot_measured=True (method stereo_tip). The tracker must be STOPPED (camera access).
    .venv/bin/python -m visual_teleop.tools.stereo_tip_calib [--left-index 3 --right-index 4] [--frames 15] [--dry-run]"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def tip_px(img_bgr: np.ndarray):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 80), (22, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)); h = m.shape[0]
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    cand = [k for k in range(1, n) if cen[k][1] > h * 0.45 and st[k, 4] > 3000]
    if not cand: return None, m
    k = max(cand, key=lambda k: st[k, 4]); ys, xs = np.nonzero(lab == k); y0 = ys.min(); hh = ys.max() - y0
    # 10-16: the "top 8 % of the blob" was ONE jaw's tip -- the left camera picked the right jaw and vice versa (2.6 cm error).
    # Split the upper part at the median column into the two jaws, take each jaw's top, and use their MIDPOINT (same physical point in both views).
    band = ys <= y0 + 0.35 * hh; xm = np.median(xs[band]); tips = []
    for side in (xs < xm, xs >= xm):
        yy, xx = ys[band & side], xs[band & side]
        if len(yy) < 30: return None, None
        t = yy <= yy.min() + 0.10 * (hh * 0.35); tips.append([xx[t].mean(), yy[t].mean()])
    return np.mean(tips, 0), (lab == k)


def tip_base_px(img_bgr: np.ndarray):
    """(tip, base) pixels of the orange jaw tape: mean of the blob's top 8 % rows and bottom 8 % rows."""
    tip, blob = tip_px(img_bgr)
    if tip is None: return None, None
    ys, xs = np.nonzero(blob); y1 = ys.max(); hh = y1 - ys.min(); bot = ys >= y1 - 0.08 * hh
    return tip, np.array([xs[bot].mean(), ys[bot].mean()])


def bearing(px, c):
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]); D = np.array([c["k1"], c["k2"], c["k3"], c["k4"]]).reshape(4, 1)
    u = cv2.fisheye.undistortPoints(np.asarray(px, np.float64).reshape(1, 1, 2), K, D).reshape(2); v = np.r_[u, 1.0]; return v / np.linalg.norm(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-index", type=int, default=3); ap.add_argument("--right-index", type=int, default=4); ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--t-ci", default="-0.0413,-0.0275,0.0137"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mount", action="store_true", help="also the TCP axes from the tape (tip - base = z, camera->jaws perpendicular = x = gravity in pose A); replaces calib_tracker mount")
    ap.add_argument("--world-identity", action="store_true", help="T_W_TW = identity (the VIO world is gravity-aligned z-up; yaw comes from the INIT push)"); a = ap.parse_args()
    st = yaml.safe_load((ROOT / "configs/calibration/stereo_trackcam_v005.yaml").read_text()); ci = yaml.safe_load((ROOT / "configs/calibration/stereo_trackcam_imu_v004.yaml").read_text())
    T_RL = np.asarray(st["T_CR_CL"], float); R_LR = T_RL[:3, :3].T; o_R = -R_LR @ T_RL[:3, 3]           # right camera centre / rotation in C0
    caps = [cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) for i in (a.left_index, a.right_index)]
    for c in caps: c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    from visual_teleop.tools.stereo_imu_record import lock          # same exposure as the tracker (auto exposure washed the tape out)
    print("[tip] 노출", lock("0x03110000", 10), lock("0x03140000", 10), flush=True)
    for _ in range(15): [c.read() for c in caps]
    P, vis = [], None
    for k in range(a.frames):
        fl, fr = caps[0].read()[1], caps[1].read()[1]; (pl, ml), (pr, mr) = tip_px(fl), tip_px(fr)
        if pl is None or pr is None: print(f"  프레임 {k}: 주황 집게 끝을 못 찾음"); continue
        dL = bearing(pl, st["left"]); dR = R_LR @ bearing(pr, st["right"])
        A = np.c_[dL, -dR]; s, t = np.linalg.lstsq(A, o_R, rcond=None)[0]; p = (s * dL + (o_R + t * dR)) / 2; gap = np.linalg.norm(s * dL - (o_R + t * dR))
        P.append((p, gap)); vis = (fl, fr, pl, pr)
        time.sleep(0.05)
    [c.release() for c in caps]
    if len(P) < 5: sys.exit("집게 끝 검출 프레임이 5개 미만 -- 주황 테이프가 두 화면 아래쪽에 보이는지 확인")
    pts = np.array([p for p, _ in P]); gaps = np.array([g for _, g in P]); p = np.median(pts, 0)
    print(f"집게 끝 (왼쪽 카메라 좌표, cm): {np.round(p * 100, 2)}  거리 {np.linalg.norm(p) * 100:.1f} cm | 프레임 간 흔들림 {np.linalg.norm(pts.std(0)) * 1000:.1f} mm | 두 광선 간격 중앙 {np.median(gaps) * 1000:.1f} mm")
    T_C0I = np.eye(4); T_C0I[:3, :3] = np.asarray(ci["R_C0I"], float); T_C0I[:3, 3] = [float(v) for v in a.t_ci.split(",")]
    r = (np.linalg.inv(T_C0I) @ np.r_[p, 1.0])[:3]; print(f"r (트래커 몸체 = IMU 좌표, m): {np.round(r, 4)}  |r| = {np.linalg.norm(r) * 100:.1f} cm")
    if vis is not None:
        fl, fr, pl, pr = vis; im = [cv2.circle(f.copy(), tuple(int(v) for v in q), 18, (0, 0, 255), 4) for f, q in ((fl, pl), (fr, pr))]
        out = ROOT / "datasets/stereo_vio_live/stereo_tip.jpg"; cv2.imwrite(str(out), np.hstack([cv2.resize(x, (960, 540)) for x in im])); print("확인 이미지", out)
    if not (0.05 < np.linalg.norm(p) < 0.6) or np.median(gaps) > 0.01: sys.exit("  거리/광선 간격이 비정상 -- 저장 안 함")
    R_TCP = None
    if a.mount:
        caps = [cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) for i in (a.left_index, a.right_index)]
        for c_ in caps: c_.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c_.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        for _ in range(10): [c_.read() for c_ in caps]
        Z = []
        for _ in range(a.frames):
            fl, fr = caps[0].read()[1], caps[1].read()[1]; (tl_, bl_), (tr_, br_) = tip_base_px(fl), tip_base_px(fr)
            if tl_ is None or tr_ is None: continue
            pts3 = []
            for ql, qr in ((tl_, tr_), (bl_, br_)):
                dL = bearing(ql, st["left"]); dR = R_LR @ bearing(qr, st["right"]); s_, t_ = np.linalg.lstsq(np.c_[dL, -dR], o_R, rcond=None)[0]
                pts3.append((s_ * dL + o_R + t_ * dR) / 2)
            Z.append(pts3)
        [c_.release() for c_ in caps]
        if len(Z) < 5: sys.exit("집게 테이프 끝/뿌리 검출 실패")
        tipm, basem = np.median([z[0] for z in Z], 0), np.median([z[1] for z in Z], 0)
        z = tipm - basem; L = np.linalg.norm(z); z /= L; mid = (tipm + basem) / 2; x = mid - (mid @ z) * z; x /= np.linalg.norm(x); y = np.cross(z, x)
        R_C0_TCP = np.c_[x, y, z]; R_TCP = np.asarray(ci["R_C0I"], float).T @ R_C0_TCP          # TCP axes in the IMU (= tracker body) frame
        print(f"집게 테이프 길이 {L * 100:.1f} cm | TCP z (집게 방향, 왼쪽 카메라 좌표) {np.round(z, 3)} | x (카메라→집게, 수직) {np.round(x, 3)}")
        print("R_TB_TCP (열 = 트래커 좌표에서 본 TCP 축):\n", np.round(R_TCP, 3))
        if not 0.01 < L < 0.12: sys.exit("  테이프 길이가 비정상 -- 저장 안 함")
    if a.dry_run: print("dry-run: 저장 안 함"); return
    from visual_teleop import config as vcfg
    from visual_teleop.tools.calib_tracker import tcp_matrix, save_tcp
    c = vcfg.load(); T = tcp_matrix(c); T[:3, 3] = r; flags = dict(pivot_measured=True, pivot_rms_mm=None)
    if R_TCP is not None: T[:3, :3] = R_TCP; flags["mount_measured"] = True
    vcfg.save_calibration({"fusion": {"tracker": {"T_TB_TCP": save_tcp(c, T, **flags)}}},
                          dict(tool="stereo_tip_calib", method="stereo_tip (replaces pivot)", p_C0_m=p.tolist(), r_m=r.tolist(), jitter_mm=float(np.linalg.norm(pts.std(0)) * 1000), ray_gap_mm=float(np.median(gaps) * 1000)))
    print("저장됨: fusion.tracker.T_TB_TCP 이동" + (" + 회전 (stereo mount)" if R_TCP is not None else "") + " 성분")
    if a.world_identity:
        vcfg.save_calibration({"fusion": {"tracker": {"T_W_TW": {"measured": True, "matrix": np.eye(4).tolist()}}}},
                              dict(tool="stereo_tip_calib --world-identity", reason="OpenVINS world is gravity-aligned z-up; teleop heading comes from the INIT push (translation_frame heading)"))
        print("저장됨: fusion.tracker.T_W_TW = identity")


if __name__ == "__main__":
    main()
