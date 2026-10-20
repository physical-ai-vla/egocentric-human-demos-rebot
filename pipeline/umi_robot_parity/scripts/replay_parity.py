"""UMI -> robot replay parity (2026-10-20, GPT plan step 6). Per outward stroke (hold before -> hold after, displacements relative to the
stroke start, robot base frame): A = UMI measured (umi_tcp_traj p_B), B = robot commanded (FK of the replayed q), C = robot executed (FK of
the observed q). Compared on NORMALIZED PROGRESS s in [0,1] (projection on A's chord, each series normalised by its own length, monotone),
so lag / speed differences do not count as error.
Per stroke: main-axis ratio C/A (and B/A), cross-axis leakage |perp(C(s)-A(s))| / |A main|, endpoint |dC-dA|, RMSE / p95 |C(s)-A(s)|,
B-A (transform + IK) and C-B (robot tracking) split, A's own straightness and angle to the robot axis (human, informational).
Per direction: 3D endpoint repeat std of A and C over the repeats.
Gates: ratio 0.9-1.1, leakage < 10% (better < 5%), endpoint < 20 mm, RMSE < 20 mm, p95 < 30 mm, repeat std C < 10 mm.
    .venv/bin/python scripts/replay_parity.py datasets/tag_rig/<take> datasets/replay/<run>        # A/B/C
    .venv/bin/python scripts/replay_parity.py datasets/tag_rig/<take> --umi-only [--hand 오른손]     # A only (before the robot), axes from the data
    .venv/bin/python scripts/replay_parity.py --synthetic                                            # self-test"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
S = np.linspace(0, 1, 101)
GATE = dict(ratio=(0.9, 1.1), leak=0.10, end_mm=20, rmse_mm=20, p95_mm=30, rep_mm=10)


def on_progress(P, u):
    """displacements P (n,3) relative to their start, resampled at S by the monotone projection on u, normalised by its own end"""
    P = P - P[0]; x = P @ u; L = x[-1] if abs(x[-1]) > 1e-6 else 1e-6; s = np.maximum.accumulate(np.clip(x / L, 0, None))
    s, idx = np.unique(s, return_index=True); P = P[idx]
    if len(s) < 3: return None
    return np.stack([np.interp(S, s, P[:, j]) for j in range(3)], 1)


def perp(v, e): return v - np.outer(v @ e, e)


def stroke(A, e, B=None, C=None):
    u = A[-1] - A[0]; u = u / max(np.linalg.norm(u), 1e-9); a = on_progress(A, u); r = {}
    if a is None: return None
    mA = float((A[-1] - A[0]) @ e); r.update(main_A_mm=mA * 1e3, leak_A=float(np.linalg.norm(perp(a, e), axis=1).max() / abs(mA)),
                                             angle_A_deg=float(np.degrees(np.arccos(np.clip(abs(u @ e), 0, 1)))), end_A=(A[-1] - A[0]))
    for nm, X in (("B", B), ("C", C)):
        if X is None: continue
        x = on_progress(X, u)
        if x is None: continue
        d = np.linalg.norm(x - a, axis=1) * 1e3
        r.update({f"ratio_{nm}": float((X[-1] - X[0]) @ e / mA), f"leak_{nm}": float(np.linalg.norm(perp(x - a, e), axis=1).max() / abs(mA)),
                  f"end_{nm}_mm": float(np.linalg.norm((X[-1] - X[0]) - (A[-1] - A[0])) * 1e3), f"rmse_{nm}_mm": float(np.sqrt((d ** 2).mean())),
                  f"p95_{nm}_mm": float(np.percentile(d, 95)), f"end_{nm}": X[-1] - X[0]})
        if nm == "C" and "ratio_B" in r: r["rmse_CB_mm"] = float(np.sqrt((np.linalg.norm(x - on_progress(B, u), axis=1) ** 2).mean()) * 1e3)
    return r


def judge(r):
    g = GATE; f = []
    if not g["ratio"][0] <= r["ratio_C"] <= g["ratio"][1]: f.append("비율")
    if r["leak_C"] >= g["leak"]: f.append("새어나감")
    if r["end_C_mm"] >= g["end_mm"]: f.append("끝점")
    if r["rmse_C_mm"] >= g["rmse_mm"] or r["p95_C_mm"] >= g["p95_mm"]: f.append("경로")
    return f


def summarize(rows, mode):
    print(f"\n{'구간':14s} {'A 이동':>7s} {'A 각도':>6s} {'A 휨':>5s}" + ("" if mode == "A" else f" {'C/A':>5s} {'B/A':>5s} {'새어남':>6s} {'끝점':>6s} {'RMSE':>6s} {'p95':>6s} {'C-B':>6s}  판정"))
    fails = 0
    for k, r in rows:
        line = f"{k:14s} {r['main_A_mm']:6.0f}mm {r['angle_A_deg']:5.1f}° {r['leak_A'] * 100:4.0f}%"
        if mode != "A" and "ratio_C" in r:
            f = judge(r); fails += bool(f)
            line += (f" {r['ratio_C']:5.2f} {r.get('ratio_B', np.nan):5.2f} {r['leak_C'] * 100:5.1f}% {r['end_C_mm']:5.1f}mm {r['rmse_C_mm']:5.1f}mm {r['p95_C_mm']:5.1f}mm "
                     f"{r.get('rmse_CB_mm', np.nan):5.1f}mm  " + ("PASS" if not f else "FAIL(" + ",".join(f) + ")"))
        print(line)
    print("\n방향별 반복 (끝점 3D 표준편차; A = 사람 동작 편차, 참고):")
    groups = {}
    for k, r in rows: groups.setdefault(k.rsplit(" ", 1)[0], []).append(r)
    rep_bad = 0
    for g, L in groups.items():
        if len(L) < 2: continue
        sA = float(np.sqrt(np.trace(np.cov(np.array([x["end_A"] for x in L]).T))) * 1e3)
        msg = f"  {g:12s} 반복 {len(L)}회: A {sA:5.1f} mm"
        if mode != "A" and all("end_C" in x for x in L):
            # 10-20: each repeat replays a DIFFERENT human stroke, so C's own spread just follows A's; the robot's repeatability is the
            # spread of the endpoint ERROR (C - A) over the repeats
            sC = float(np.sqrt(np.trace(np.cov(np.array([x["end_C"] - x["end_A"] for x in L]).T))) * 1e3); rep_bad += sC >= GATE["rep_mm"]
            msg += f", 로봇 끝점오차(C-A) {sC:5.1f} mm" + ("" if sC < GATE["rep_mm"] else "  <- 10 mm 이상")
        print(msg)
    if mode != "A":
        allr = [r for _, r in rows if "ratio_C" in r]
        d = np.array([r["rmse_C_mm"] for r in allr])
        print(f"\n전체: 구간 {len(allr)}개, 통과 {len(allr) - fails}, RMSE 중앙 {np.median(d):.1f} mm, 최대 {d.max():.1f} mm, 반복 기준 넘은 방향 {rep_bad}")
        print("최종:", "PASS" if fails == 0 and rep_bad == 0 else "FAIL",
              "| B-A 가 크면 좌표변환/IK 문제, C-B 가 크면 로봇 추종 문제, A 각도가 크면 사람이 로봇 축과 다르게 움직인 것(오류 아님)")
        return fails == 0 and rep_bad == 0
    return True


def windows(sg):
    a = (sg["hold_pre"][0] + sg["hold_pre"][1]) // 2; b = (sg["hold_mid"][0] + sg["hold_mid"][1]) // 2; return a, b


def sel(t, P, a, b, h0=None):
    m = (t >= a) & (t <= b)
    if m.sum() < 5: return None
    X = P[m].copy()
    if h0 is not None:                          # start = median of the hold before the stroke (operator jitter)
        hm = (t >= h0[0]) & (t <= h0[1])
        if hm.sum() >= 3: X[0] = np.median(P[hm], 0)
    return X


def run_files(a):
    take = Path(a.take); side = {"오른손": "right", "왼손": "left"}[a.hand]
    d = take if (take / f"umi_traj_{side}.npz").exists() else take.parent / (take.name + "_derived")
    z = np.load(d / f"umi_traj_{side}.npz", allow_pickle=True); segs = json.loads((d / "segments.json").read_text())
    tA = z["t_ns"].astype(np.int64); rows = []
    if a.umi_only:
        PA = z["p_B"] if "p_B" in z else z["p_C"]; segs = [g for g in segs if g["hand"] == a.hand]
        E = dict(AX) if "p_B" in z else None
        if E is None:                            # no robot calibration: axes from the data (sign-corrected mean chords, Gram-Schmidt)
            raw = {}
            for g in segs:
                X = sel(tA, PA, *windows(g))
                if X is not None: raw.setdefault(g["axis"], []).append(g["sign"] * (X[-1] - X[0]) / np.linalg.norm(X[-1] - X[0]))
            raw = {k: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for k, v in raw.items()}
            if len(raw) == 3:
                ang = lambda p, q: np.degrees(np.arccos(np.clip(raw[p] @ raw[q], -1, 1)))
                print(f"데이터에서 잡은 축 사이 각도 (90° 가 이상적): X-Y {ang('X', 'Y'):.1f}°, X-Z {ang('X', 'Z'):.1f}°, Y-Z {ang('Y', 'Z'):.1f}° (카메라 좌표, 로봇 캘리브 전)")
            E = raw
        for g in segs:
            X = sel(tA, PA, *windows(g), g["hold_pre"])
            if X is None or g["axis"] not in E: continue
            r = stroke(X, g["sign"] * E[g["axis"]])
            if r: rows.append((f"{g['axis']} {g['dir']} {g['rep']}", r))
        return summarize(rows, "A")
    run = Path(a.run); meta = json.loads((run / "meta.json").read_text()); sp = meta["speed"]
    for j, g in enumerate(meta["segments"]):
        f = run / f"seg_{j:02d}.npz"
        if not f.exists(): continue
        R_ = np.load(f); w0, w1 = windows(g)
        src = R_["src_t_ns"].astype(np.int64); tC = src[0] + ((R_["t_obs_ns"] - R_["t_cmd_ns"][0]) * sp).astype(np.int64)
        # 10-20: C window = B window shifted by C's measured lag (was a fixed +0.5 s: the operator often kept moving in the 'hold', so the
        # extra 0.5 s of C showed up as a fake 40 mm endpoint error). lag = best match of C(t + lag) to B(t) along the stroke direction
        u_ = R_["p_cmd"][-1] - R_["p_cmd"][0]; Bf = sel(src, R_["p_cmd"], w0, w1)
        if Bf is not None and np.linalg.norm(Bf[-1] - Bf[0]) > 1e-3: u_ = (Bf[-1] - Bf[0]) / np.linalg.norm(Bf[-1] - Bf[0])
        mb = (src >= w0) & (src <= w1); xb = R_["p_cmd"][mb] @ u_; xc = R_["p_obs"] @ u_
        lag = min(np.arange(0, 1.0e9, 2e7), key=lambda L: np.mean((np.interp(src[mb] + L, tC, xc) - xb) ** 2)) if mb.sum() > 5 else 0
        A = sel(tA, z["p_B"], w0, w1, g["hold_pre"]); B = sel(src, R_["p_cmd"], w0, w1); C = sel(tC, R_["p_obs"], w0 + int(lag), w1 + int(lag))
        if A is None or B is None or C is None: print(f"  {g['axis']} {g['dir']} {g['rep']}: 데이터 부족"); continue
        r = stroke(A, g["sign"] * AX[g["axis"]], B, C)
        if r: rows.append((f"{g['axis']} {g['dir']} {g['rep']}", r))
    return summarize(rows, "ABC")


def synthetic():
    rng = np.random.default_rng(3); ok_all = True
    for case, leak, scale in (("정상 로봇 (C = 0.97배, 1.5% 새어남)", 0.015, 0.97), ("불량 로봇 (X 이동 때 Z 로 15% 새어남)", 0.15, 1.0)):
        print(f"\n=== 합성: {case}"); rows = []
        for ax in "XYZ":
            for sg in (+1, -1):
                for rep in range(3):
                    n = 60; tt = np.linspace(0, 1, n); mj = 10 * tt ** 3 - 15 * tt ** 4 + 6 * tt ** 5
                    A = np.outer(mj, AX[ax] * sg * 0.15) + np.outer(np.sin(np.pi * tt), rng.normal(0, 0.006, 3)) + rng.normal(0, 0.0015, (n, 3))
                    B = A.copy(); m2 = 10 * np.linspace(0, 1, 2 * n) ** 3 - 15 * np.linspace(0, 1, 2 * n) ** 4 + 6 * np.linspace(0, 1, 2 * n) ** 5
                    Bs = np.stack([np.interp(np.linspace(0, 1, 2 * n), tt, B[:, j]) for j in range(3)], 1)
                    lag = np.r_[np.zeros(6), m2[:-6]]                                   # executed: lags, scaled, maybe leaks
                    C = scale * np.stack([np.interp(lag, m2, Bs[:, j]) for j in range(3)], 1) + rng.normal(0, 0.0005, (2 * n, 3))
                    if ax == "X": C[:, 2] += leak * 0.15 * lag * sg * (1 if leak < 0.1 else 1)
                    r = stroke(A, sg * AX[ax], Bs, C); rows.append((f"{ax} {'+' if sg > 0 else '-'} {rep + 1}", r))
        res = summarize(rows, "ABC"); ok_all &= (res if "정상" in case else not res)
    print("\n합성 시험", "PASS (정상은 통과, 불량은 걸러냄)" if ok_all else "FAIL"); return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?"); ap.add_argument("run", nargs="?"); ap.add_argument("--hand", default="오른손", choices=["오른손", "왼손"])
    ap.add_argument("--umi-only", action="store_true"); ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
    if a.synthetic: sys.exit(0 if synthetic() else 1)
    if not a.take or (not a.run and not a.umi_only): ap.error("take 와 run (또는 --umi-only) 필요")
    run_files(a)


if __name__ == "__main__":
    main()
