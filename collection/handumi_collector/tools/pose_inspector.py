"""Pose inspector (design §24): per-episode plots of L/R TCP XYZ, rotation (rotvec deg), grip, tracking state, IMU/VIO
residual windows, HOME start/end windows + return drift, and a 3D trajectory view.

    python -m handumi_collector.tools.pose_inspector --episode EP [--backend opencv_vo] [--save out.png] [--no-show]"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..config import DEFAULT_CONFIG_DIR, load_pose_cfg
from ..pose.episode_io import derived_dir, read_table


def load(ep: Path, backend: str) -> dict:
    d = derived_dir(ep, backend)
    qa = json.loads((d / "pose_qa.json").read_text())
    out = dict(qa=qa, canonical=read_table(d / "canonical"), cam={})
    for s in qa["sides"]:
        try: out["cam"][s] = read_table(d / f"{s}_camera_pose")
        except FileNotFoundError: pass
    return out


def figure(data: dict, title: str):
    import matplotlib
    import matplotlib.pyplot as plt
    qa, can = data["qa"], data["canonical"]; sides = list(qa["sides"]); t = can["t_rel_s"].values
    fig = plt.figure(figsize=(16, 11)); fig.suptitle(f"{title}  —  {qa['backend']}  verdict {qa['verdict']}")
    gs = fig.add_gridspec(4, 3)
    ax_xyz = fig.add_subplot(gs[0, :2]); ax_rot = fig.add_subplot(gs[1, :2], sharex=ax_xyz); ax_grip = fig.add_subplot(gs[2, :2], sharex=ax_xyz)
    ax_state = fig.add_subplot(gs[3, :2], sharex=ax_xyz); ax3d = fig.add_subplot(gs[:2, 2], projection="3d"); ax_txt = fig.add_subplot(gs[2:, 2]); ax_txt.axis("off")
    colors = {"left": ["tab:red", "tab:orange", "tab:pink"], "right": ["tab:blue", "tab:cyan", "tab:purple"]}
    for s in sides:
        v = can[f"{s}_tcp_valid"].values.astype(bool); xyz = can[[f"{s}_tcp_x", f"{s}_tcp_y", f"{s}_tcp_z"]].values
        for k, nm in enumerate("xyz"): ax_xyz.plot(t, np.where(v, xyz[:, k], np.nan), color=colors[s][k], lw=1, label=f"{s} {nm}")
        q = can[[f"{s}_tcp_qx", f"{s}_tcp_qy", f"{s}_tcp_qz", f"{s}_tcp_qw"]].values
        rv = np.full((len(t), 3), np.nan)
        if v.any(): rv[v] = np.degrees(Rotation.from_quat(q[v]).as_rotvec())
        for k, nm in enumerate("xyz"): ax_rot.plot(t, rv[:, k], color=colors[s][k], lw=1, label=f"{s} r{nm}")
        ax_grip.plot(t, can[f"{s}_grip"].values, color=colors[s][0], label=f"{s} grip")
        cam = data["cam"].get(s)
        if cam is not None:
            st = cam["tracking_state"].values; order = ["uninitialized", "initializing", "tracking", "degraded", "lost"]
            y = np.array([order.index(x) if x in order else 0 for x in st]); tc = (cam["t_ns"].values - can["t_ns"].values[0]) / 1e9 + t[0]
            ax_state.step(tc, y + (0.1 if s == "right" else 0), where="post", color=colors[s][0], label=f"{s} state")
            ax3d.plot(xyz[v, 0], xyz[v, 1], xyz[v, 2], color=colors[s][0], lw=1, label=s)
        sq = qa["sides"][s]
        for w, c in ((sq.get("home_start"), "green"), (sq.get("home_end"), "purple")):
            if w:
                t0 = (w["t0_ns"] - can["t_ns"].values[0]) / 1e9 + t[0]; t1 = (w["t1_ns"] - can["t_ns"].values[0]) / 1e9 + t[0]
                for ax in (ax_xyz, ax_rot, ax_grip, ax_state): ax.axvspan(t0, t1, color=c, alpha=0.08)
    ax_state.set_yticks(range(5)); ax_state.set_yticklabels(["uninit", "init", "track", "degr", "lost"]); ax_state.set_xlabel("t (s)")
    ax_xyz.set_ylabel("TCP xyz (episode frame)"); ax_rot.set_ylabel("rotvec (deg)"); ax_grip.set_ylabel("grip 0=closed 1=open")
    for ax in (ax_xyz, ax_rot, ax_grip, ax_state): ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.3)
    ax3d.set_title("TCP paths — per-hand local frames\n(not a shared world)", fontsize=9); ax3d.legend(fontsize=8, loc="upper left")
    lines = []
    for s in sides:
        q = qa["sides"][s]
        lines += [f"[{s}] {q['verdict']}  valid {q['valid_ratio']:.3f}  lost {q['lost_events']}",
                  f"  HOME return: {q['return_translation_mm'] and round(q['return_translation_mm'], 1)} mm{'' if q['metric_scale'] else ' (non-metric)'} / {q['return_rotation_deg'] and round(q['return_rotation_deg'], 2)} deg",
                  f"  HOME jitter start {q['home_start_pos_std_mm'] and round(q['home_start_pos_std_mm'], 2)} mm / {q['home_start_rot_std_deg'] and round(q['home_start_rot_std_deg'], 3)} deg",
                  f"  IMU/VIO residual median {q['imu_residual_deg_median'] and round(q['imu_residual_deg_median'], 2)} deg  p95 {q['imu_residual_deg_p95'] and round(q['imu_residual_deg_p95'], 2)}",
                  f"  jumps t/r {q['jumps_translation']}/{q['jumps_rotation']}  max step {q['max_step_rotation_deg'] and round(q['max_step_rotation_deg'], 2)} deg"]
        lines += [f"  - {r}" for r in q["reasons"][:4]] + [f"  ! {f}" for f in q["flags"][:3]]
    ax_txt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7.5)
    fig.tight_layout(); return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--episode", required=True); ap.add_argument("--backend", default=None)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "pose.yaml")); ap.add_argument("--save", default=None); ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args(argv); cfg = load_pose_cfg(a.config); backend = a.backend or cfg.backend
    if a.no_show:
        import matplotlib; matplotlib.use("Agg")
    ep = Path(a.episode); data = load(ep, backend); fig = figure(data, ep.name)
    if a.save: fig.savefig(a.save, dpi=110); print("saved", a.save)
    if not a.no_show:
        import matplotlib.pyplot as plt; plt.show()
    return 0


if __name__ == "__main__": raise SystemExit(main())
