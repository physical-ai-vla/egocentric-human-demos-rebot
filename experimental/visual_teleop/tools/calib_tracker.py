"""6DoF tracker calibrations (2026-10-15, tracker path). Order: pivot -> world -> mount. Each writes config/calibration.yaml.
The hub must NOT be running (same UDP port); the vendor bridge (or mock_tracker) must be streaming.

  pivot   tracker body -> UMI jaw tip translation r:  p_pivot = p_tracker,i + R_tracker,i r  (least squares over all poses).
          Put the jaw tip in a small dent / on one fixed point and rock + twist the UMI around it (>= 30 deg every way).
          Prints the residual RMS; PASS <= 2 mm, WARN <= 4 mm (--accept-warn), else refused.
      .venv/bin/python -m visual_teleop.tools.calib_tracker pivot --seconds 20

  world   T_W_TW (tracker world -> W, explicit 4x4): touch the jaw tip on 3 table points, still each:
          P0 origin, P1 ~20 cm FORWARD of it (robot +x direction), P2 ~20 cm LEFT of P0 (+y). z = table normal (table level).
      .venv/bin/python -m visual_teleop.tools.calib_tracker world

  mount   R_TB_TCP (tracker body -> TCP axes), two still poses, same as imu-mount --tool umi:
          A  UMI level, jaws forward, jaws side by side -> gravity = +x_TCP;  B  jaws straight down -> gravity = +z_TCP.
      .venv/bin/python -m visual_teleop.tools.calib_tracker mount"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .. import config as vcfg
from ..vision.tracker6dof import (Tracker6DoFProvider, pivot_calibration, table_frame_from_points, R_TB_TCP_from_gravity,
                                  rotation_spread_deg, make_T)
from . import calib_right as cr

PIVOT_PASS_M, PIVOT_WARN_M = 0.002, 0.004


def open_tracker(c) -> Tracker6DoFProvider:
    tc = c.fusion["tracker"]
    # raw poses only (no extrinsics needed); a wide jump gate so hand rocking is never rejected during calibration
    tr = Tracker6DoFProvider(int(tc["port"]), stale_warn_ms=tc["stale_warn_ms"], hold_ms=tc["hold_ms"], v_max_m_s=10.0, margin_m=0.5)
    try: tr.start()
    except OSError as e: sys.exit(f"UDP {tr.port} 포트를 열 수 없음 ({e}) -- 허브가 켜져 있으면 끄고 다시 실행하세요")
    t0 = time.monotonic()
    while tr.latest_raw("right") is None:
        if time.monotonic() - t0 > 3.0: tr.stop(); sys.exit(f"3초 동안 트래커 데이터 없음 (UDP {tr.port}) -- 브리지/mock_tracker 실행 확인")
        time.sleep(0.05)
    return tr


def collect(tr: Tracker6DoFProvider, side: str, seconds: float) -> list[np.ndarray]:
    """Valid raw T_TW_TB, one per new seq (or per new receive time when the bridge sends no seq)."""
    out, last = [], None; t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        tp = tr.latest_raw(side)
        if tp is not None and tp.valid and tp.conf >= tr.min_conf and (tp.seq, tp.t_ns) != last: out.append(tp.T_TW_TB.copy()); last = (tp.seq, tp.t_ns)
        time.sleep(0.004)
    return out


def still_pose(tr, side: str, prompt: str, spoken: str, r_tip: np.ndarray | None = None, hold_s: float = 1.5, tries: int = 3) -> np.ndarray:
    """Countdown, then hold_s of still samples -> mean pose (rotation by chordal mean). Still = tip std <= 2 mm, rotation std <= 1 deg."""
    cr.show(spoken, "", "#333"); cr.say(prompt, spoken)
    for attempt in range(tries):
        for k in range(5 if attempt == 0 else 3, 0, -1): print(f"  {k}...", flush=True); cr.show(spoken, k, "#333"); time.sleep(1.0)
        cr.show("멈추세요 — 측정 중", "STOP", "#b00"); cr.beep("Glass"); S = collect(tr, side, hold_s); cr.beep("Pop"); cr.show("OK", "", "#2e7d32")
        if len(S) < 20: sys.exit(f"측정 구간 트래커 샘플 {len(S)}개뿐 (스트림/conf 확인)")
        r = np.zeros(3) if r_tip is None else r_tip
        P = np.array([T[:3, :3] @ r + T[:3, 3] for T in S]); rv = Rotation.from_matrix([T[:3, :3] for T in S])
        Rm = rv.mean().as_matrix(); rs = float(np.degrees(np.linalg.norm((rv * Rotation.from_matrix(Rm).inv()).as_rotvec(), axis=1).std()))
        ps = float(np.linalg.norm(P.std(0))) * 1000
        print(f"  샘플 {len(S)}개, 끝점 흔들림 {ps:.2f} mm, 회전 흔들림 {rs:.2f}도")
        if ps <= 2.0 and rs <= 1.0: return make_T(Rm, P.mean(0))
        cr.show("움직였습니다 — 같은 자세로 다시", "", "#b58900"); cr.say("Not still.", "움직이지 마세요. 다시 잽니다.")
    sys.exit("  3번 모두 움직임 감지 -- 처음부터 다시")


def tcp_matrix(c) -> np.ndarray: return np.asarray(c.fusion["tracker"]["T_TB_TCP"]["matrix"], np.float64)


def save_tcp(c, T: np.ndarray, **flags) -> None:
    cur = dict(c.fusion["tracker"]["T_TB_TCP"]); cur.update(flags); cur["matrix"] = T.round(6).tolist()
    cur["measured"] = bool(cur.get("pivot_measured")) and bool(cur.get("mount_measured"))
    return cur


def pivot(c, side: str, seconds: float, accept_warn: bool) -> None:
    tr = open_tracker(c)
    try:
        cr.show("집게 끝을 한 점에 고정하고 이리저리 돌리세요", "", "#333")
        cr.say("Pivot: jaw tip on one fixed point, rock and twist the UMI around it.",
               "피벗 캘리브레이션. 집게 끝을 한 점에 고정하고, 끝이 움직이지 않게 유미를 앞뒤 좌우로 기울이고 비트세요. 띵 소리부터 시작.")
        cr.beep("Glass"); S = collect(tr, side, seconds); cr.beep("Pop"); cr.show("계산 중", "", "#333")
    finally: tr.stop()
    S = S[:: max(1, len(S) // 600)]
    if len(S) < 30: sys.exit(f"포즈 {len(S)}개뿐 -- 스트림 확인")
    spread = rotation_spread_deg(S)
    print(f"포즈 {len(S)}개, 회전 범위 최대 {spread:.1f}도")
    if spread < 30: sys.exit("  회전이 30도 미만 -- 더 크게 기울이고 비틀어서 다시")
    t, pp, rms = pivot_calibration(S)
    res = np.array([np.linalg.norm(T[:3, :3] @ t + T[:3, 3] - pp) for T in S]); keep = res <= max(3 * np.median(res), 0.002)
    if keep.sum() < len(S):                                         # one pass of outlier removal (a slipped tip), then re-solve
        print(f"  이상치 {int((~keep).sum())}개 제거 후 재계산"); S = [T for T, k in zip(S, keep) if k]; t, pp, rms = pivot_calibration(S)
    grade = "PASS" if rms <= PIVOT_PASS_M else ("WARN" if rms <= PIVOT_WARN_M else "FAIL")
    print(f"집게 끝 오프셋 r (트래커 좌표, m): {np.round(t, 4)}  |r| = {np.linalg.norm(t) * 100:.1f} cm\n"
          f"피벗 잔차 RMS: {rms * 1000:.2f} mm  -> {grade} (통과 ≤ {PIVOT_PASS_M * 1000:.0f} mm, 경고 ≤ {PIVOT_WARN_M * 1000:.0f} mm)")
    if grade == "FAIL" or (grade == "WARN" and not accept_warn):
        cr.show(f"실패 RMS {rms * 1000:.1f} mm", "", "#b00"); sys.exit("  저장 안 함 -- 끝점이 미끄러지지 않게 다시 (WARN 수용은 --accept-warn)")
    T = tcp_matrix(c); T[:3, 3] = t
    vcfg.save_calibration({"fusion": {"tracker": {"T_TB_TCP": save_tcp(c, T, pivot_measured=True, pivot_rms_mm=round(rms * 1000, 3))}}},
                          dict(tool="calib_tracker pivot", side=side, n=len(S), rms_mm=rms * 1000, rot_spread_deg=spread, grade=grade, r=t.tolist()))
    cr.show(f"완료 RMS {rms * 1000:.1f} mm", "", "#2e7d32"); print("저장됨: fusion.tracker.T_TB_TCP 이동 성분")


def world(c, side: str) -> None:
    tc = c.fusion["tracker"]
    if not tc["T_TB_TCP"].get("pivot_measured"): sys.exit("pivot을 먼저 하세요 (집게 끝 위치가 필요)")
    r = tcp_matrix(c)[:3, 3]; tr = open_tracker(c)
    try:
        P0 = still_pose(tr, side, "Touch P0 (origin) on the table.", "첫 번째 점. 집게 끝을 테이블 위 기준점에 대세요.", r)[:3, 3]
        P1 = still_pose(tr, side, "Touch P1, 20 cm FORWARD of P0.", "두 번째 점. 기준점에서 로봇 앞쪽 방향으로 20센티 떨어진 곳에 대세요.", r)[:3, 3]
        P2 = still_pose(tr, side, "Touch P2, 20 cm LEFT of P0.", "세 번째 점. 기준점에서 왼쪽으로 20센티 떨어진 곳에 대세요.", r)[:3, 3]
    finally: tr.stop()
    d1, d2 = np.linalg.norm(P1 - P0), np.linalg.norm(P2 - P0)
    ang = float(np.degrees(np.arccos(np.clip((P1 - P0) @ (P2 - P0) / (d1 * d2), -1, 1))))
    print(f"P0→P1 {d1 * 100:.1f} cm, P0→P2 {d2 * 100:.1f} cm, 사잇각 {ang:.1f}도 (90도 근처 정상)")
    if min(d1, d2) < 0.08 or not 45 <= ang <= 135: sys.exit("  점 간격 8 cm 미만이거나 사잇각이 45~135도 밖 -- 다시")
    T_W_TW, _ = table_frame_from_points(P0, P1, P2)
    up_TW = np.linalg.inv(T_W_TW)[:3, 2]
    print("T_W_TW:\n", np.round(T_W_TW, 4), f"\n트래커 월드에서 본 위쪽: {np.round(up_TW, 3)}  (SteamVR이면 대략 [0,1,0])")
    vcfg.save_calibration({"fusion": {"tracker": {"T_W_TW": {"measured": True, "matrix": T_W_TW.round(6).tolist()}}}},
                          dict(tool="calib_tracker world", side=side, d1_m=d1, d2_m=d2, angle_deg=ang, up_in_tracker_world=up_TW.tolist(),
                               assumption="table level"))
    cr.show("완료", "", "#2e7d32"); print("저장됨: fusion.tracker.T_W_TW")


def mount(c, side: str) -> None:
    tc = c.fusion["tracker"]
    if not tc["T_W_TW"].get("measured"): sys.exit("world를 먼저 하세요 (중력 방향이 필요)")
    T_W_TW = np.asarray(tc["T_W_TW"]["matrix"], np.float64); tr = open_tracker(c)
    try:
        A = still_pose(tr, side, "Pose A: UMI level, jaws forward, jaws side by side.",
                       "첫 번째 자세. 유미를 수평으로, 집게 끝이 앞을 향하고 두 집게가 좌우로 나란히. 띵부터 뽁까지 멈추세요.")
        B = still_pose(tr, side, "Pose B: jaws pointing straight down.", "두 번째 자세. 유미 집게 끝이 바닥을 향하게 세우세요. 띵부터 뽁까지 멈추세요.")
    finally: tr.stop()
    R, ang = R_TB_TCP_from_gravity(T_W_TW[:3, :3] @ A[:3, :3], T_W_TW[:3, :3] @ B[:3, :3])
    print(f"두 자세의 중력 방향 차이: {ang:.1f}도 (정상 90도, 75~105도 통과)")
    if abs(ang - 90) > 15: sys.exit("  90도 정도 차이 나지 않음 -- 다시 (A: 수평 앞으로, B: 아래로)")
    T = tcp_matrix(c); T[:3, :3] = R
    rv = np.degrees(Rotation.from_matrix(R).as_rotvec())
    print("R_TB_TCP (열 = 트래커 좌표에서 본 TCP 축):\n", np.round(R, 3), "\n회전벡터(도)", np.round(rv, 2))
    vcfg.save_calibration({"fusion": {"tracker": {"T_TB_TCP": save_tcp(c, T, mount_measured=True)}}},
                          dict(tool="calib_tracker mount", side=side, gravity_angle_deg=ang, rotvec_deg=rv.tolist()))
    cr.show("완료", "", "#2e7d32"); print("저장됨: fusion.tracker.T_TB_TCP 회전 성분")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["pivot", "world", "mount"]); ap.add_argument("--side", default="right", choices=["right"])
    ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--accept-warn", action="store_true")
    ap.add_argument("--no-board", action="store_true", help="화면 큐 페이지 없이"); ap.add_argument("--quiet", action="store_true", help="음성 없이")
    ap.add_argument("--dry-run", action="store_true", help="계산만, calibration.yaml에 저장 안 함 (mock 검증용)")
    a = ap.parse_args(); c = vcfg.load()
    if a.quiet: cr.say = lambda m, s=None: print(s or m, flush=True); cr.beep = lambda n="Glass": None
    if not a.no_board: cr.BOARD = cr.CueBoard()
    if a.dry_run: vcfg.save_calibration = lambda u, prov: print("[dry-run] 저장 생략:", prov.get("tool"))
    {"pivot": lambda: pivot(c, a.side, a.seconds, a.accept_warn), "world": lambda: world(c, a.side), "mount": lambda: mount(c, a.side)}[a.what]()
    print("트래커 남은 캘리브레이션:", vcfg.tracker_missing(vcfg.load()) or "없음 — 완료")


if __name__ == "__main__":
    main()
