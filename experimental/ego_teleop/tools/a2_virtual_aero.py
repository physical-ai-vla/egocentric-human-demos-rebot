"""A2 — virtual Aero, no motors (spec section 27/30).

Pushes the SAME frames through both hand-pose providers and the SAME retargeter, so any difference is attributable
to the pose source and nothing else:

    B0  official monocular MediaPipe world landmarks  ->  DexPilot  ->  virtual Aero
    B1  head RGB-D metric hand pose                   ->  DexPilot  ->  virtual Aero     <- ours
    B2  head RGB-D metric hand pose                   ->  semantic-7D mapping (existing baseline)

The load-bearing comparison is not "do the joint angles differ" but **does the virtual hand reproduce the human's
grasp**: for every frame with metric depth we know the operator's real thumb-index fingertip distance, and we can
measure the virtual Aero's thumb-index fingertip gap by forward kinematics. B0 cannot be scored that way on its own
(it has no metric scale), which is precisely the gap this branch exists to close.

    .venv/bin/python -m ego_teleop.tools.a2_virtual_aero --episode datasets/HumanRGBD_v1/episode_000001 \
        --arms b0 b1 --viz --out outputs/a2

Required motions to record before reading anything into the numbers (spec section 27): open, close, pinch, tripod,
cylindrical grasp, index-only flexion, thumb opposition."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ego_collector.hands3d.aero import AeroHandFK, default_urdf
from ..config import load_teleop_cfg, HeadRgbdCfg
from ..hand3d import aero_mocap as M
from ..hand3d.head_camera import RecordedRgbdSource
from ..hand3d.interfaces import HandPoseHealth
from ..hand3d.supervisor import HandPoseSupervisor
from ..retarget.aero_backends import AeroCommandLimiter, DexPilotAeroRetargeter, Semantic7DAeroRetargeter
from .a1_hand3d import _dist, draw_overlay

ARMS = {
    "b0": dict(provider="mediapipe_mono", backend="dexpilot", label="official monocular -> DexPilot"),
    "b1": dict(provider="head_rgbd", backend="dexpilot", label="head RGB-D metric -> DexPilot"),
    "b2": dict(provider="head_rgbd", backend="semantic7d", label="head RGB-D metric -> semantic 7D"),
}


def _cfg_for(cfg: HeadRgbdCfg, arm: str) -> HeadRgbdCfg:
    import dataclasses
    return dataclasses.replace(cfg, provider=ARMS[arm]["provider"], backend=ARMS[arm]["backend"])


def run_arm(source, cfg: HeadRgbdCfg, arm: str, *, max_frames: int | None = None) -> pd.DataFrame:
    c = _cfg_for(cfg, arm)
    provider, backend = c.build_provider(), c.build_backend()
    sup = HandPoseSupervisor(c.hand_pose)
    limiter = AeroCommandLimiter(c.aero_limits)
    fk = AeroHandFK(default_urdf(c.side), c.side)
    rows = []
    for n, frame in enumerate(source):
        if frame is None: continue
        if max_frames and n >= max_frames: break
        t0 = time.monotonic_ns()
        est = provider.get_hand_pose(frame)
        st = sup.update(est, frame.timestamp_ns)
        target = None
        if st.commandable:
            try:
                target = backend.retarget(st.estimate)
            except ValueError as e:
                st.reason = f"retarget_refused:{e}"
        cmd = limiter.step(target, frame.timestamp_ns)
        row = dict(arm=arm, frame_index=getattr(frame, "frame_index", n), t_ns=frame.timestamp_ns,
                   health=st.health.value, reason=st.reason, n_valid=est.n_valid, n_filled=est.n_filled,
                   palm_scale_m=est.palm_scale_m, pose_ms=(est.pose_done_ns - t0) / 1e6 if est.pose_done_ns else np.nan,
                   total_ms=(time.monotonic_ns() - t0) / 1e6, commanded=cmd is not None)
        # human reference (metric only): real thumb-index fingertip distance
        if est.metric and est.valid[M.MP_TIPS[0]] and est.valid[M.MP_TIPS[1]]:
            row["human_pinch_m"] = float(np.linalg.norm(est.landmarks_camera_3d[4] - est.landmarks_camera_3d[8]))
        else:
            row["human_pinch_m"] = np.nan
        if cmd is not None:
            row.update(cmd.to_row())
            tips = fk.fingertips(np.degrees(cmd.joints_rad))
            row["aero_pinch_m"] = float(np.linalg.norm(tips["thumb"] - tips["index"]))
            for f, p in tips.items():
                for k, ax in enumerate("xyz"): row[f"aero_tip_{f}_{ax}"] = float(p[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["rate_limited"] = limiter.rate_limited
    df.attrs["holds"] = limiter.holds
    df.attrs["swaps"] = provider.identity.swap_events
    return df


def compare(frames: dict[str, pd.DataFrame]) -> dict:
    rep: dict = {"arms": {}}
    for arm, df in frames.items():
        cmd = df[df["commanded"]]
        q = cmd[[f"q_{n}_rad" for n in M.AERO_JOINT_NAMES]].to_numpy(np.float64) if len(cmd) else np.zeros((0, 16))
        pinch_err = (cmd["aero_pinch_m"] - cmd["human_pinch_m"]).abs() * 1000.0 if len(cmd) else pd.Series(dtype=float)
        rep["arms"][arm] = dict(
            label=ARMS[arm]["label"], frames=int(len(df)), commanded=int(len(cmd)),
            health_counts={k: int(v) for k, v in df["health"].value_counts().items()},
            identity_swaps=int(df.attrs.get("swaps", 0)), holds=int(df.attrs.get("holds", 0)),
            rate_limited=int(df.attrs.get("rate_limited", 0)),
            pose_ms=_dist(df["pose_ms"]), total_ms=_dist(df["total_ms"]),
            saturated_joints=_dist(cmd["diag_saturated"]) if "diag_saturated" in cmd else None,
            joint_range_deg={n: dict(min=float(np.degrees(q[:, i]).min()), max=float(np.degrees(q[:, i]).max()))
                             for i, n in enumerate(M.AERO_JOINT_NAMES)} if len(q) else {},
            aero_pinch_mm=_dist(cmd["aero_pinch_m"] * 1000.0) if len(cmd) else _dist([]),
            human_pinch_mm=_dist(cmd["human_pinch_m"] * 1000.0) if len(cmd) else _dist([]),
            pinch_abs_error_mm=_dist(pinch_err),
        )
    # paired per-frame comparison on the frames where both arms produced a command
    keys = list(frames)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A, B = frames[a], frames[b]
            m = A.merge(B, on="frame_index", suffixes=("_a", "_b"))
            m = m[m["commanded_a"] & m["commanded_b"]]
            if m.empty: continue
            d = np.degrees(np.abs(m[[f"q_{n}_rad_a" for n in M.AERO_JOINT_NAMES]].to_numpy(np.float64)
                                  - m[[f"q_{n}_rad_b" for n in M.AERO_JOINT_NAMES]].to_numpy(np.float64)))
            rep[f"{a}_vs_{b}"] = dict(
                paired_frames=int(len(m)), joint_abs_diff_deg=_dist(d),
                per_joint_p95_deg={n: float(np.percentile(d[:, i], 95)) for i, n in enumerate(M.AERO_JOINT_NAMES)},
                aero_pinch_abs_diff_mm=_dist(np.abs(m["aero_pinch_m_a"] - m["aero_pinch_m_b"]) * 1000.0),
            )
    rep["note"] = ("Only the pose source differs between b0 and b1 — same DexPilot config, same scale factors, same "
                   "clip (docs/ego_teleop/AERO_A0_AUDIT.md). Never change pose estimation and retargeting together "
                   "and then attribute the result to depth (spec section 30).")
    return rep


def render(source, frames: dict[str, pd.DataFrame], out: Path, cfg: HeadRgbdCfg) -> Path:
    """Overlay video: the human hand, and one virtual Aero skeleton per arm, side by side."""
    import cv2
    fk = AeroHandFK(default_urdf(cfg.side), cfg.side)
    by_arm = {a: df.set_index("frame_index") for a, df in frames.items()}
    out.mkdir(parents=True, exist_ok=True)
    # the human panel is drawn from a fresh pass of the metric provider: the per-frame tables keep joint targets,
    # not landmarks, and a debug video is worth one extra inference pass
    human = _cfg_for(cfg, "b1").build_provider()
    w = None
    for frame in source:
        est = human.get_hand_pose(frame)
        img = draw_overlay(frame, est if est.n_valid else None)
        panels = []
        for arm, df in by_arm.items():
            row = df.loc[frame.frame_index] if frame.frame_index in df.index else None
            q = None
            if row is not None and bool(row.get("commanded", False)):
                q = np.degrees(np.array([row[f"q_{n}_rad"] for n in M.AERO_JOINT_NAMES], np.float64))
            panels.append(_draw_aero(fk, q, cfg.side, label=f"{arm}: {ARMS[arm]['label']}"))
        canvas = np.hstack([img] + panels) if panels else img
        if w is None:
            w = cv2.VideoWriter(str(out / "a2_virtual_aero.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                frame.calib.fps, (canvas.shape[1], canvas.shape[0]))
        w.write(canvas)
    if w is not None: w.release()
    return out / "a2_virtual_aero.mp4"


def _draw_aero(fk: AeroHandFK, joints_deg: np.ndarray | None, side: str, *, size: int = 480, label: str = "") -> np.ndarray:
    import cv2
    panel = np.full((size, size, 3), 24, np.uint8)
    cv2.putText(panel, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    if joints_deg is None:
        cv2.putText(panel, "hold / no command", (6, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
        return panel
    pos = fk.link_positions(joints_deg)
    P = np.array(list(pos.values()))
    span = max(float(np.abs(P).max()) * 1.2, 1e-3)
    to_px = lambda p: (int(size / 2 + p[0] / span * (size / 2 - 20)), int(size - 30 - (p[2] / span) * (size - 60)))
    for chain in fk.chains():
        pts = [to_px(pos[l]) for l in chain if l in pos]
        for a, b in zip(pts, pts[1:]): cv2.line(panel, a, b, (160, 200, 255), 2, cv2.LINE_AA)
        for p in pts: cv2.circle(panel, p, 3, (80, 255, 120), -1, cv2.LINE_AA)
    return panel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A2 — virtual Aero from both pose sources through the same retargeter")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--arms", nargs="+", default=["b0", "b1"], choices=sorted(ARMS))
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", default="outputs/a2")
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_teleop_cfg().head_rgbd
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = {arm: run_arm(RecordedRgbdSource(a.episode), cfg, arm, max_frames=a.frames) for arm in a.arms}
    for arm, df in frames.items(): df.to_parquet(out / f"a2_{arm}.parquet", index=False)
    rep = compare(frames)
    rep["episode"] = a.episode
    (out / "a2_report.json").write_text(json.dumps(rep, indent=1, default=float))
    if a.viz:
        print("video:", render(RecordedRgbdSource(a.episode), frames, out, cfg))
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
