"""Single-joint step + slow-sine test of the MIT bridge (2026-10-20, replay parity follow-up: X replay showed j2 leading ~-90 ms and
j3 lagging +150..900 ms -> Z path error 2-4 cm). Around READY, one joint at a time (default j2 then j3), all other joints held:
  step   +D (hold --hold s) -> 0 (hold) -> -D (hold) -> 0 (hold)       D = --step-deg, sent as a 0.3 s smoothstep through HttpBridge.send
  sine   amplitude --sine-deg, period --period s, --cycles cycles
Logged at the observe rate (t, q_send, q_obs). Report per joint and direction (+ / -, i.e. with / against gravity):
  step: error 0.3 s after the move, at hold end (steady state), time to settle within 0.5 deg, overshoot
  sine: lag of q_obs behind q_send on the RISING and FALLING halves separately, amplitude ratio
Bridge facts (10-20 read): MIT kp (64,64,64,28,28,28) Nm/rad, kd (3,3,3,2,2,2), integral-only torque law Ki 28 Nm/(rad s), gravity
model disarmed -> a pose change that changes the gravity torque by dT leaves an error dT/kp that the integral removes with tau ~ kp/Ki
~ 2.3 s. This test measures exactly that.
    ~/xvla-mac/bin/python -m visual_teleop.tools.joint_step_test                 # 계획만 (로봇 안 움직임)
    ~/xvla-mac/bin/python -m visual_teleop.tools.joint_step_test --go            # 실제 (사람이 옆에 있을 때만, 팔은 READY)
    ~/xvla-mac/bin/python -m visual_teleop.tools.joint_step_test --analyze datasets/joint_test/<ts>"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from ..robot.ik import ArmIk
from ..robot.ik_pink import PinkIkSolver, PinkParams
from .robot_moves import READY_Q_DEG, ramp_to, say, unpark

ROOT = Path(__file__).resolve().parents[2]


def profile(kind, D, hold, period, cycles, hz, reps=1):
    """offset (deg) of the tested joint vs time, plus segment labels"""
    seg, out = [], []
    if kind in ("step", "steps"):
        cur = 0.0
        for tgt, lab in ((D, "+"), (0.0, "+back"), (-D, "-"), (0.0, "-back")) * (reps if kind == "steps" else 1):
            n_m = int(0.3 * hz); u = np.linspace(0, 1, n_m + 1)[1:]; u = u * u * (3 - 2 * u)
            t0 = len(out); out += list(cur + (tgt - cur) * u) + [tgt] * int(hold * hz); seg.append((lab, t0, t0 + n_m, len(out))); cur = tgt
    else:
        t = np.arange(int(period * cycles * hz)) / hz; out = list(D * np.sin(2 * np.pi * t / period)); seg.append(("sine", 0, 0, len(out)))
    return np.array(out), seg


def analyze(run: Path):
    meta = json.loads((run / "meta.json").read_text()); print(f"{run}  [{meta.get('label', '')}]  (step {meta['step_deg']}°, hold {meta['hold']} s, sine ±{meta['sine_deg']}° / {meta['period']} s)")
    for f in sorted(run.glob("j*_*.npz")):
        z = np.load(f); j = int(z["joint"]); kind = str(z["kind"]); ts = z["t_send"] / 1e9; qs = z["q_send"]; to = z["t_obs"] / 1e9; qo = z["q_obs"]
        ref = np.interp(to, ts, qs); e = qo - ref
        if kind == "steps":
            res = {}
            for lab, a, b, c in json.loads(str(z["seg_json"])):
                tb, tc = ts[min(b, len(ts) - 1)], ts[min(c, len(ts) - 1)]; tgt = qs[min(c, len(qs) - 1)]; step = tgt - qs[max(a - 1, 0)]
                m = (to >= tb) & (to <= tc); tt, ee = to[m] - tb, qo[m] - tgt
                if m.sum() < 10: continue
                errs = [abs(float(np.interp(x, tt, ee))) for x in (0.5, 1.0, 2.0, tt[-1])]
                ok = np.where(np.abs(ee) > 0.3)[0]; settle = float(tt[ok[-1]]) if len(ok) else 0.0
                over = max(float(np.max(np.sign(step) * ee)), 0.0) if step != 0 else 0.0
                late = tt > tt[-1] - 2.0; osc = float(np.std(ee[late] - np.polyval(np.polyfit(tt[late], ee[late], 1), tt[late]))) if late.sum() > 5 else 0.0
                tm = (to >= ts[a]) & (to <= tc); vpk = float(np.abs(np.gradient(qo[tm], to[tm])).max()) if tm.sum() > 3 else 0.0
                res.setdefault("+" if step > 0 else "-", []).append(errs + [settle, over, osc, vpk])
            print(f"  j{j + 1} 작은 계단 ±{meta['step_deg']}° x{meta.get('reps', 1)} (중앙값; 방향 = 관절각 +/-):")
            print("      방향  |e|0.5s  |e|1s  |e|2s  |e|끝   0.3°안착  오버슛  진동RMS  속도최대")
            for d, L in sorted(res.items()):
                M = np.median(np.array(L), 0)
                print(f"       {d}    {M[0]:5.2f}°  {M[1]:5.2f}°  {M[2]:5.2f}°  {M[3]:5.2f}°  {M[4]:5.1f}s   {M[5]:5.2f}°  {M[6]:5.3f}°  {M[7]:5.1f}°/s")
            allL = np.array(sum(res.values(), [])); print(f"      전체 중앙: 끝 오차 {np.median(allL[:, 3]):.2f}°, 1초 오차 {np.median(allL[:, 1]):.2f}°, 진동 {np.median(allL[:, 6]):.3f}° (n={len(allL)})")
            continue
        if kind == "step":
            print(f"  j{j + 1} 계단:")
            for lab, a, b, c in z["seg"].tolist() if False else json.loads(str(z["seg_json"])):
                ta, tb, tc = ts[a], ts[min(b, len(ts) - 1)], ts[min(c, len(ts) - 1)]
                tgt = qs[min(c, len(qs) - 1)]; m = (to >= tb) & (to <= tc); tt, ee = to[m] - tb, qo[m] - tgt
                if not m.any(): continue
                e03 = float(np.interp(0.3, tt, ee)); e_end = float(np.median(ee[tt > tt[-1] - 0.5]))
                ok = np.where(np.abs(ee) > 0.5)[0]; settle = float(tt[ok[-1]]) if len(ok) else 0.0
                step = tgt - qs[max(a - 1, 0)]; over = float(np.max(np.sign(step) * ee)) if step != 0 else 0.0
                print(f"     {lab:6s}: 0.3초 후 오차 {e03:+5.2f}°, 정지 끝 오차 {e_end:+5.2f}°, 0.5° 안 들어온 시간 {settle:4.1f}s, 오버슛 {max(over, 0):4.2f}°")
        else:
            dq = np.gradient(ref, to); best = {}
            for nm, msk in (("올라가는 쪽(+)", dq > 0), ("내려가는 쪽(-)", dq < 0)):
                lags = np.arange(-0.3, 1.5, 0.01)
                cost = [np.mean((np.interp(to[msk] - L, ts, qs) - qo[msk]) ** 2) for L in lags]; best[nm] = (lags[int(np.argmin(cost))], float(np.sqrt(min(cost))))
            amp = np.ptp(qo[to > to[0] + meta["period"]]) / max(np.ptp(qs), 1e-6)
            print(f"  j{j + 1} 사인: " + ", ".join(f"{k} 지연 {v[0] * 1000:+4.0f} ms (잔차 {v[1]:.2f}°)" for k, v in best.items()) + f", 진폭비 {amp:.2f}, 최대 오차 {np.abs(e).max():.2f}°")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="right", choices=["right"]); ap.add_argument("--joints", type=int, nargs="+", default=[2, 3], help="1-based")
    ap.add_argument("--step-deg", type=float, default=5.0); ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--sine-deg", type=float, default=8.0); ap.add_argument("--period", type=float, default=8.0); ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--small", action="store_true", help="kp 시험: ±--step-deg 계단만 --reps 회 (사인 없음)"); ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--label", default=""); ap.add_argument("--hz", type=float, default=30.0); ap.add_argument("--min-z", type=float, default=0.12)
    ap.add_argument("--go", action="store_true"); ap.add_argument("--offline", action="store_true"); ap.add_argument("--analyze")
    a = ap.parse_args()
    if a.analyze: analyze(Path(a.analyze)); return
    if a.offline and a.go: sys.exit("--offline 과 --go 는 같이 못 씀")
    c = vcfg.load(); s = a.side; qr = np.radians(READY_Q_DEG[s])
    br = None if a.offline else HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"])
    try:
        if a.offline:
            from ego_teleop.robot.rebot_client import SIDE_SLICE
            jda = np.zeros(14); jda[SIDE_SLICE[s]] = READY_Q_DEG[s]; q_now = qr
        else:
            ob = br.observe()
            if ob.estop: sys.exit("로봇 상태가 E-STOP/미연결/읽기 지연입니다")
            jda, q_now = ob.joints_deg_all, ob.q[s]
        if np.degrees(np.abs(q_now - qr)).max() > 5: sys.exit("팔이 READY 자세가 아님 -- 먼저 visual_teleop.tools.ready_pose --go")
        pk = PinkIkSolver(s, PinkParams(**c.ik["pink"])); pk.sync(jda)
        L = ArmIk.from_cfg(c, pk).gate.cfg.limits; lo, hi = np.asarray(L.lower_rad), np.asarray(L.upper_rad)
        plans = []
        for j1 in a.joints:
            j = j1 - 1
            for kind, D in ((("steps", a.step_deg),) if a.small else (("step", a.step_deg), ("sine", a.sine_deg))):
                off, seg = profile(kind, D, a.hold, a.period, a.cycles, a.hz, a.reps); Q = np.tile(qr, (len(off), 1)); Q[:, j] += np.radians(off)
                zmin = min(pk.fk(q)[2, 3] for q in Q[::5]); lim = bool(np.any(Q < lo) or np.any(Q > hi))
                print(f"  j{j1} {kind:5s}: ±{D:.1f}°, {len(off) / a.hz:4.1f}s, TCP 최저 z {zmin:.3f}{'  관절한계 밖' if lim else ''} {'OK' if zmin >= a.min_z and not lim else '<- 불가'}")
                if zmin < a.min_z or lim: sys.exit("불가 -- --step-deg / --sine-deg 를 줄이세요")
                plans.append(dict(j=j, kind=kind, Q=Q, seg=seg))
        print(f"총 {sum(len(p['Q']) for p in plans) / a.hz / 60:.1f}분")
        if not a.go: print("계획만 출력함 (--go 로 실제 이동)"); return
        out = ROOT / "datasets/joint_test" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps(dict(side=s, label=a.label, reps=a.reps, kp=br.status().get("kp"), step_deg=a.step_deg, hold=a.hold, sine_deg=a.sine_deg, period=a.period, cycles=a.cycles, hz=a.hz, q_ready=qr.tolist())))
        say("관절 시험을 시작합니다"); unpark(br, s); ob = br.observe(); g = {x: ob.g(x) for x in ("left", "right")}; other = "left" if s == "right" else "right"
        for p in plans:
            ramp_to(br, s, p["Q"][0], 10.0, g=g); time.sleep(1.5); say(f"{p['j'] + 1}번 관절 {'사인' if p['kind'] == 'sine' else '계단'}")
            q_oth = br.observe().q[other]; ts, to, qo = [], [], []; t0 = time.monotonic()
            for k, q in enumerate(p["Q"]):
                r = br.send({s: q, other: q_oth}, g); ts.append(time.monotonic_ns())
                if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
                o = br.observe()
                if o.estop: br.estop(); sys.exit("E-STOP/읽기 지연 -- 정지")
                to.append(o.t_ns); qo.append(np.degrees(o.q[s][p["j"]]))
                time.sleep(max(0.0, t0 + (k + 1) / a.hz - time.monotonic()))
            np.savez(out / f"j{p['j'] + 1}_{p['kind']}.npz", joint=p["j"], kind=p["kind"], t_send=np.array(ts), q_send=np.degrees(p["Q"][:, p["j"]]),
                     t_obs=np.array(to), q_obs=np.array(qo), seg_json=json.dumps(p["seg"]))
        ramp_to(br, s, qr, 10.0, g=g); say("관절 시험 끝났습니다"); print(f"저장 {out}"); analyze(out)
    finally:
        if br is not None: br.close()


if __name__ == "__main__":
    main()
