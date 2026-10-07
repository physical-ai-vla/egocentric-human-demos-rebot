"""A1 — RGB-D metric 3D hand pose, no Aero motion (spec sections 27/28).

Runs the hand-pose branch over a recorded RGB-D episode or the live head camera, writes every estimate, and reports
the acceptance gate as DISTRIBUTIONS (p50/p95/max), not averages — a p95 fingertip jitter is what decides whether
this pose source can drive a hand, an average hides the frames that matter.

    .venv/bin/python -m ego_teleop.tools.a1_hand3d --episode datasets/HumanRGBD_v1/episode_000001 \
        --static 2:6 --motion 8:14 --viz --out outputs/a1

The required motions of spec section 27 (open / fist / thumb-index pinch / thumb-middle pinch / individual finger
flexion / hand translation / camera motion with the hand pose held) are operator-side: record one take per motion and
point `--static` / `--motion` at the relevant seconds. `--motion` is the camera-moves-hand-does-not window, which is
what the palm-local invariance number is computed from."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import load_teleop_cfg, HeadRgbdCfg
from ..hand3d import aero_mocap as M
from ..hand3d.head_camera import HeadRgbdCalibration, OrbbecHeadCamera, RecordedRgbdSource
from ..hand3d.interfaces import HandPoseEstimate, HandPoseHealth
from ..hand3d.supervisor import HandPoseSupervisor

TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _pct(v, q):
    v = np.asarray([x for x in np.ravel(v) if np.isfinite(x)], np.float64)
    return float(np.percentile(v, q)) if v.size else float("nan")


def _dist(v) -> dict:
    return dict(n=int(np.count_nonzero(np.isfinite(np.ravel(v)))), p50=_pct(v, 50), p95=_pct(v, 95), max=_pct(v, 100))


def _window(df: pd.DataFrame, window: tuple[float, float] | None) -> pd.DataFrame:
    if window is None or df.empty: return df.iloc[0:0]
    t0 = df["t_ns"].iloc[0]
    s = (df["t_ns"] - t0) / 1e9
    return df[(s >= window[0]) & (s <= window[1])]


def stationary_jitter_mm(df: pd.DataFrame) -> dict:
    """Per-fingertip scatter around its own mean, in the palm-local frame and in the camera frame."""
    out = {}
    for frame, key in (("local", "loc"), ("camera", "cam")):
        per_tip = {}
        for name, j in zip(TIP_NAMES, M.MP_TIPS):
            P = df[[f"lm{j}_{key}_{a}" for a in "xyz"]].to_numpy(np.float64)
            P = P[np.isfinite(P).all(1)]
            per_tip[name] = _dist(np.linalg.norm(P - P.mean(0), axis=1) * 1000.0) if len(P) > 2 else _dist([])
        out[frame] = per_tip
        out[f"{frame}_all_tips_p95_mm"] = float(np.nanmax([v["p95"] for v in per_tip.values()])) if per_tip else float("nan")
    return out


def camera_motion_invariance(df: pd.DataFrame) -> dict:
    """Palm-local articulation must not change when only the camera moves (spec section 10/28).

    Reports how far the camera-frame wrist travelled over the window (the stimulus) against the spread of the
    palm-local landmarks (the response). A large stimulus with a small response is the property we need."""
    W = df[[f"lm0_cam_{a}" for a in "xyz"]].to_numpy(np.float64)
    W = W[np.isfinite(W).all(1)]
    travel_mm = float(np.linalg.norm(W - W.mean(0), axis=1).max() * 2000.0) if len(W) > 2 else float("nan")
    drift = []
    for j in range(M.N_LANDMARKS):
        P = df[[f"lm{j}_loc_{a}" for a in "xyz"]].to_numpy(np.float64)
        P = P[np.isfinite(P).all(1)]
        if len(P) > 2: drift.append(np.linalg.norm(P - P.mean(0), axis=1) * 1000.0)
    return dict(camera_wrist_travel_mm=travel_mm, palm_local_drift_mm=_dist(np.concatenate(drift) if drift else []))


def summarize(df: pd.DataFrame, *, rate_hz: float, swaps: int, counters: dict, cfg: HeadRgbdCfg,
              static: tuple[float, float] | None, motion: tuple[float, float] | None) -> dict:
    hands = df[df["n_valid"] > 0]
    valid_cols = [f"lm{j}_valid" for j in range(M.N_LANDMARKS)]
    per_landmark = {j: float(hands[f"lm{j}_valid"].mean()) if len(hands) else float("nan") for j in range(M.N_LANDMARKS)}
    rep = dict(
        frames=int(len(df)), frames_with_hand=int(len(hands)),
        rate_hz=float(rate_hz),
        health_counts={k: int(v) for k, v in df["health"].value_counts().items()},
        supervisor_counters=dict(counters),
        identity_swaps=int(swaps),
        metric_landmark_fraction=float(hands[valid_cols].to_numpy().mean()) if len(hands) else float("nan"),
        filled_landmarks_per_frame=_dist(hands["n_filled"]) if len(hands) else _dist([]),
        per_landmark_valid_fraction=per_landmark,
        tips_valid_fraction=float(hands["n_valid_tips"].mean() / 5.0) if len(hands) else float("nan"),
        palm_scale_m=_dist(hands["palm_scale_m"]) if len(hands) else _dist([]),
        pose_latency_ms=_dist(df["latency_ms"]),
        wrist_depth_m=_dist(hands["lm0_depth_m"]) if len(hands) else _dist([]),
    )
    rep["stationary"] = stationary_jitter_mm(_window(hands, static)) if static else None
    rep["camera_motion"] = camera_motion_invariance(_window(hands, motion)) if motion else None

    g = cfg.gates
    checks = {
        "rate_hz >= min": (rep["rate_hz"] >= g.min_rate_hz, rep["rate_hz"], g.min_rate_hz),
        "metric_landmark_fraction >= min": (rep["metric_landmark_fraction"] >= g.min_metric_landmark_fraction,
                                            rep["metric_landmark_fraction"], g.min_metric_landmark_fraction),
        "identity_swaps <= max": (rep["identity_swaps"] <= g.max_identity_swaps, rep["identity_swaps"], g.max_identity_swaps),
    }
    if rep["stationary"]:
        v = rep["stationary"]["camera_all_tips_p95_mm"]
        checks["stationary tip jitter p95 <= max (mm)"] = (v <= g.max_stationary_jitter_mm_p95, v, g.max_stationary_jitter_mm_p95)
    if rep["camera_motion"]:
        v = rep["camera_motion"]["palm_local_drift_mm"]["p95"]
        checks["palm-local drift under camera motion p95 <= max (mm)"] = (v <= g.max_palm_local_drift_mm_p95, v, g.max_palm_local_drift_mm_p95)
    rep["gate"] = {k: dict(passed=bool(ok), value=val, threshold=thr) for k, (ok, val, thr) in checks.items()}
    rep["gate_passed"] = all(c[0] for c in checks.values()) if checks else False
    rep["gate_note"] = ("Thresholds are pre-hardware starting points (spec section 28): re-fit them on measured "
                        "depth behaviour before treating a PASS as evidence.")
    return rep


# ---- visualisation (never in the inference path; the raw frames stay unmirrored, spec section 19) --------------
def draw_overlay(frame, est: HandPoseEstimate, *, mirror_display: bool = False) -> np.ndarray:
    import cv2
    img = frame.color_bgr.copy()
    if est is not None and np.isfinite(est.landmarks_2d).all():
        for a, b in M.HAND_EDGES:
            pa, pb = est.landmarks_2d[a].astype(int), est.landmarks_2d[b].astype(int)
            cv2.line(img, tuple(pa), tuple(pb), (200, 200, 200), 1, cv2.LINE_AA)
        for j in range(M.N_LANDMARKS):
            p = est.landmarks_2d[j].astype(int)
            filled = est.filled is not None and est.filled[j]
            col = (0, 220, 0) if est.valid[j] else ((0, 200, 255) if filled else (0, 0, 255))
            cv2.circle(img, tuple(p), 4, col, -1, cv2.LINE_AA)
        z = est.landmarks_camera_3d[0, 2]
        cv2.putText(img, f"{est.side} {est.health.value} valid={est.n_valid}/21 filled={est.n_filled} wrist={z:.3f}m",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "no hand", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    panel = draw_palm_local(est)
    out = np.hstack([img, cv2.resize(panel, (img.shape[0], img.shape[0]))])
    return cv2.flip(out, 1) if mirror_display else out


def draw_palm_local(est: HandPoseEstimate | None, size: int = 480) -> np.ndarray:
    """Palm-local skeleton, two orthographic views. This is the representation the retargeter actually consumes."""
    import cv2
    panel = np.full((size, size, 3), 30, np.uint8)
    if est is None or not np.isfinite(est.landmarks_local_3d).all(): return panel
    L = est.landmarks_local_3d
    span = max(float(np.abs(L).max()) * 1.15, 1e-3)
    for k, (ax, ay, label, oy) in enumerate(((0, 2, "x-z", 0), (1, 2, "y-z", size // 2))):
        h = size // 2
        to_px = lambda p: (int(size / 2 + p[ax] / span * (size / 2 - 10)), int(oy + h - 10 - (p[ay] / span) * (h - 20)))
        for a, b in M.HAND_EDGES: cv2.line(panel, to_px(L[a]), to_px(L[b]), (120, 120, 120), 1, cv2.LINE_AA)
        for j in range(M.N_LANDMARKS):
            cv2.circle(panel, to_px(L[j]), 3, (0, 220, 0) if est.valid[j] else (0, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(panel, f"palm-local {label}  span {span*100:.1f} cm", (6, oy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


# ---- runner ----------------------------------------------------------------------------------------------------
def run(source, cfg: HeadRgbdCfg, *, max_frames: int | None = None, out_dir: Path | None = None, viz: bool = False,
        static: tuple[float, float] | None = None, motion: tuple[float, float] | None = None, live_window: bool = False) -> dict:
    provider = cfg.build_provider()
    sup = HandPoseSupervisor(cfg.hand_pose)
    rows: list[dict] = []
    writer = None
    t_start = time.monotonic()
    n = 0
    for frame in source:
        if frame is None: continue
        est = provider.get_hand_pose(frame)
        st = sup.update(est, frame.timestamp_ns)
        row = (st.estimate or est).to_row()
        row["supervised_health"] = st.health.value; row["health_reason"] = st.reason
        row["frame_index"] = getattr(frame, "frame_index", n)
        rows.append(row)
        if viz or live_window:
            import cv2
            img = draw_overlay(frame, est if est.n_valid or est.health != HandPoseHealth.LOST else None)
            if viz and out_dir is not None:
                if writer is None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out_dir / "a1_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             frame.calib.fps, (img.shape[1], img.shape[0]))
                writer.write(img)
            if live_window:
                cv2.imshow("A1 head RGB-D hand pose", img)
                if cv2.waitKey(1) & 0xFF == 27: break
        n += 1
        if max_frames and n >= max_frames: break
    if writer is not None: writer.release()

    df = pd.DataFrame(rows)
    elapsed = max(time.monotonic() - t_start, 1e-6)
    rep = summarize(df, rate_hz=n / elapsed, swaps=provider.identity.swap_events, counters=sup.counters, cfg=cfg,
                    static=static, motion=motion)
    rep["provider"] = provider.source
    rep["config"] = dict(side=cfg.side, selfie_mirrored=cfg.selfie_mirrored, depth=asdict(cfg.depth), fill_missing=cfg.fill_missing)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "hand_pose.parquet", index=False)
        (out_dir / "a1_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def _win(s: str | None):
    if not s: return None
    a, b = s.split(":"); return (float(a), float(b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A1 — RGB-D metric 3D hand pose (no Aero motion)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episode", help="recorded RGB-D episode dir (color/*.jpg + aligned depth/*.png)")
    src.add_argument("--live", action="store_true", help="live Orbbec head camera (macOS: needs sudo)")
    ap.add_argument("--provider", choices=("head_rgbd", "mediapipe_mono"), help="override configs/ego_teleop/head_rgbd.yaml")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", default="outputs/a1")
    ap.add_argument("--viz", action="store_true", help="write an overlay mp4")
    ap.add_argument("--window", action="store_true", help="show a live preview window")
    ap.add_argument("--static", help="seconds a:b where the hand AND camera are still (fingertip jitter)")
    ap.add_argument("--motion", help="seconds a:b where the CAMERA moves and the hand pose is held (invariance)")
    a = ap.parse_args(argv)

    cfg = load_teleop_cfg().head_rgbd
    if a.provider: cfg = type(cfg)(**{**cfg.__dict__, "provider": a.provider})
    if a.live:
        cam = OrbbecHeadCamera(fps=int(cfg.gates.min_rate_hz) or 30).open()
        def gen():
            while True: yield cam.read()
        source = gen()
    else:
        source = RecordedRgbdSource(a.episode)
    rep = run(source, cfg, max_frames=a.frames, out_dir=Path(a.out), viz=a.viz, live_window=a.window,
              static=_win(a.static), motion=_win(a.motion))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_landmark_valid_fraction"}, indent=1, default=float))
    print(f"\nA1 GATE: {'PASS' if rep['gate_passed'] else 'FAIL'}   ({rep['gate_note']})")
    for k, v in rep["gate"].items():
        print(f"  [{'ok' if v['passed'] else 'XX'}] {k}: {v['value']} (threshold {v['threshold']})")
    return 0 if rep["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
