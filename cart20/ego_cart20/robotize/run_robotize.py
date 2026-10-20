#!/usr/bin/env python3
"""[2026-10-16] Robotize one UMI ego episode onto the reBot (spec: ego -> reBot robotization, modes A / B / C).

Input : <episode>/derived/<fusion-dir>/smoothed_{left,right}.npz   (t s, T_W_tcp, full_6dof; W = table world, z = 0 on the table)
        <episode>/derived/jaw/jaw_{side}.npz                          (visual gripper, 1 = open)
Steps : common native grid (1/30 s) inside the span where BOTH arms are full-6DoF -> human task frame (start TCP midpoint on the
        table, y = right->left) -> start-relative SE(3) -> robot reference (A: READY per arm; B/C: robot task frame) -> TCP axes map
        (human camera_tcp_v2 -> robot contract = virtual-C922 tilt) -> [C] IK / workspace / table / joint-step line search with ONE
        alpha for both arms -> 20 Hz resample of the final trajectory (log only; the dataset converter resamples itself).
Output: <episode>/derived/<out>/smoothed_{left,right}.npz   same keys as the smoother, poses in the ROBOT BASE frame, so the LeRobot
                                                            converter runs unchanged with --fusion-dir <out>
        robotize_log.npz   every layer of spec section 21 (+ joints, per-step alpha / ok / reason)
        robotize_qa.json   validation of spec section 20, incl. the stage-by-stage amplitude chain
        robotize_meta.json config + "tcp_map_applied" (the converter must NOT apply its own wrist-tilt pose map again)
usage: run_robotize.py <episode> ... [--mode C] [--orientation FULL] [--scale 1,1,1] [--fusion-dir umi_fusion_rcb3gate]
"""
import argparse, json, pathlib, sys, time
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.robotize import core                                     # noqa: E402
from ego_cart20.robotize.rebot import (SAFETY, ReBotChecker, c922_R_map, ready_pose, robot_task_frame)  # noqa: E402

GRID_DT, MAX_GAP_S = 1 / 30.0, 0.15


def resample(t_src, T_src, t):
    i1 = np.clip(np.searchsorted(t_src, t, side="right"), 1, len(t_src) - 1); i0 = i1 - 1
    gap = t_src[i1] - t_src[i0]; u = np.clip((t - t_src[i0]) / np.maximum(gap, 1e-9), 0, 1)
    R = np.stack([Slerp([0, 1], Rot.from_matrix(T_src[[a, b], :3, :3]))([w]).as_matrix()[0] for a, b, w in zip(i0, i1, u)])
    p = (1 - u)[:, None] * T_src[i0, :3, 3] + u[:, None] * T_src[i1, :3, 3]
    return core.make_T(R, p), (gap <= MAX_GAP_S) & (t >= t_src[0]) & (t <= t_src[-1])


def gripper(ep, side, t0, t):
    p = ep / "derived/jaw" / f"jaw_{side}.npz"
    if not p.exists(): return np.full(len(t), np.nan)
    from ego_cart20.sources.jaw_visual import binary_from_px
    z = np.load(p, allow_pickle=True); b, _ = binary_from_px(z["t_ns"], z["d_px"]); tj = (z["t_ns"] - t0) / 1e9
    return b[np.clip(np.searchsorted(tj, t), 0, len(b) - 1)]


def t0_of(ep):
    from ego_cart20.sources.hfold_export import read_mcap
    fm, _ = read_mcap(ep); cz = np.load(ep / "derived/body_track/canonical.npz", allow_pickle=True)
    return int(min(fm["left_wrist"][0, 1], fm["right_wrist"][0, 1], cz["t_ns"][0]))


def run(ep, a, cfg, checker):
    S = {s: np.load(ep / "derived" / a.fusion_dir / f"smoothed_{s}.npz") for s in ("left", "right")}
    src = {s: (S[s]["t"][S[s]["full_6dof"].astype(bool)], S[s]["T_W_tcp"][S[s]["full_6dof"].astype(bool)]) for s in S}
    lo, hi = max(src[s][0][0] for s in S), min(src[s][0][-1] for s in S)
    t = lo + GRID_DT * np.arange(int((hi - lo) / GRID_DT) + 1)
    TH, valid = {}, np.ones(len(t), bool)
    for s in S:
        TH[s], v = resample(*src[s], t); valid &= v
    k0 = int(np.argmax(valid)); t, valid = t[k0:], valid[k0:]; TH = {s: TH[s][k0:] for s in TH}       # start = first common valid sample
    T_task_H = core.task_frame(TH["left"][0, :3, 3], TH["right"][0, :3, 3], 0.0)
    T_R0 = {s: ready_pose(s) for s in ("left", "right")}; T_task_R = robot_task_frame()
    rz = core.robotize(TH, T_R0, T_task_H, T_task_R, cfg)
    if cfg.mode == "C":
        pr = core.project(rz["raw"], checker, cfg, valid=valid); final, ok, alpha = pr["feasible"], pr["ok"] & valid, pr["alpha"]
        reasons = Counter(i.get("first_fail") for i in pr["info"] if i.get("first_fail")); infos = pr["info"]
    else:                                                     # A / B: no projection, feasibility measured only
        final, alpha, infos, ok, state = rz["raw"], np.ones(len(t)), [], np.zeros(len(t), bool), None
        for i in range(len(t)):
            o, info, st = checker(rz["raw"]["left"][i], rz["raw"]["right"][i], state); infos.append(info); ok[i] = o and valid[i]
            if o: state = st
        reasons = Counter(i.get("reason") for i in infos if i.get("reason"))
    margins = [i.get("joint_margin_deg") for i in infos if i.get("joint_margin_deg") is not None]
    tips = [i.get("tip_z_min") for i in infos if i.get("tip_z_min") is not None]
    t0 = t0_of(ep); g_h = {s: gripper(ep, s, t0, t) for s in ("left", "right")}; g_r = {s: core.map_gripper(g_h[s], cfg) for s in g_h}
    t20 = t[0] + 0.05 * np.arange(int((t[-1] - t[0]) / 0.05) + 1); f20 = {s: resample(t, final[s], t20)[0] for s in final}
    out = ep / "derived" / a.out_name; out.mkdir(parents=True, exist_ok=True)
    for s in ("left", "right"):
        np.savez_compressed(out / f"smoothed_{s}.npz", t=t, T_W_tcp=final[s], full_6dof=ok)
    np.savez_compressed(out / "robotize_log.npz", t=t, valid=valid, ok=ok, alpha=alpha, t20=t20,
                        **{f"{k}_{s}": v for s in ("left", "right") for k, v in (
                            ("human_raw_pose", TH[s]), ("human_world_aligned", rz["task"][s]), ("human_start_relative", rz["delta"][s]),
                            ("robot_raw_target", rz["raw"][s]), ("robot_feasible_target", final[s]), ("robot_final_20hz", f20[s]),
                            ("orientation_clamp_deg", rz["clamp"][s]), ("gripper_human", g_h[s]), ("gripper_robot", g_r[s]))},
                        T_task_H=T_task_H, T_task_R=T_task_R,
                        workspace_violation=np.array([bool(i.get("reason") and "workspace" in str(i.get("reason") or i.get("first_fail"))) for i in infos]))
    amp = {s: {"human": core.amplitude(rz["task"][s][valid]), "robot_raw": core.amplitude(rz["raw"][s][valid]),
               "robot_final": core.amplitude(final[s][valid]), "robot_final_20hz": core.amplitude(f20[s])} for s in final}
    for s in amp:
        amp[s]["ratio_raw"] = round(amp[s]["robot_raw"]["max_disp_m"] / max(amp[s]["human"]["max_disp_m"], 1e-9), 3)
        amp[s]["ratio_final"] = round(amp[s]["robot_final"]["max_disp_m"] / max(amp[s]["human"]["max_disp_m"], 1e-9), 3)
        amp[s]["path_ratio_final"] = round(amp[s]["robot_final"]["path_m"] / max(amp[s]["human"]["path_m"], 1e-9), 3)
    M4 = core.make_T(cfg.R_map, np.zeros(3)); HM = {k: np.einsum("nij,jk->nik", TH[k], M4) for k in TH}   # human in ROBOT TCP axes
    lr_h = np.linalg.norm(TH["left"][valid, :3, 3] - TH["right"][valid, :3, 3], axis=1)
    lr_f = np.linalg.norm(final["left"][valid, :3, 3] - final["right"][valid, :3, 3], axis=1)
    qa = dict(episode=f"{ep.parent.name}/{ep.name}", mode=cfg.mode, orientation=cfg.orientation, scale=list(cfg.scale_xyz), translation_frame=cfg.tframe(),
              n=len(t), valid_frac=round(float(valid.mean()), 4), ik_ok_frac=round(float(ok[valid].mean()), 4) if valid.any() else 0.0,
              alpha=dict(mean=round(float(alpha[valid].mean()), 4), frac_lt1=round(float((alpha[valid] < 0.999).mean()), 4)) if valid.any() else None,
              fail_reasons=dict(reasons), amplitude=amp,
              lr_dist_m=dict(human_p50=round(float(np.median(lr_h)), 4), robot_p50=round(float(np.median(lr_f)), 4)),
              lr_rel_err_raw=core.rel_pose_err(HM["left"][valid], HM["right"][valid], rz["raw"]["left"][valid], rz["raw"]["right"][valid]),
              lr_rel_err_final=core.rel_pose_err(HM["left"][valid], HM["right"][valid], final["left"][valid], final["right"][valid]),
              tcp_height_above_table_m=dict(human_p1={s: round(float(np.percentile(TH[s][valid, 2, 3], 1)), 4) for s in TH},
                                            robot_p1={s: round(float(np.percentile(final[s][valid, 2, 3] - SAFETY["table_z_m"], 1)), 4) for s in final}),
              orientation_change_deg=dict(human={s: round(core.orientation_change_deg(TH[s][valid]), 1) for s in TH},
                                          robot={s: round(core.orientation_change_deg(final[s][valid]), 1) for s in final}),
              orientation_clamp_deg_p95={s: round(float(np.percentile(rz["clamp"][s], 95)), 1) for s in rz["clamp"]},
              joint_margin_deg_min=round(float(min(margins)), 2) if margins else None,
              table_clearance_m_min=round(float(min(tips) - SAFETY["table_z_m"]), 4) if tips else None,
              ik_calls=checker.n_calls)
    (out / "robotize_qa.json").write_text(json.dumps(qa, indent=1, default=float))
    (out / "robotize_meta.json").write_text(json.dumps(dict(source=a.fusion_dir, frame="reBot base (x forward, y left, z up), physical jaw-tip contract TCP",
        tcp_map=a.tcp_map, tcp_map_applied=a.tcp_map != "identity", mode=cfg.mode, orientation=cfg.orientation, scale_xyz=list(cfg.scale_xyz),
        translation_frame=cfg.tframe(), max_tilt_deg=cfg.max_tilt_deg, time_scale=a.time_scale, gripper=dict(a=cfg.grip_a, b=cfg.grip_b), ready=dict(left=list(ready_pose("left")[:3, 3]), right=list(ready_pose("right")[:3, 3])),
        safety=SAFETY, tool="ego_cart20/robotize/run_robotize.py 2026-10-16"), indent=1, default=float))
    return qa


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episodes", nargs="+")
    ap.add_argument("--mode", choices=core.MODES, default="C"); ap.add_argument("--orientation", choices=core.ORIENT, default="FULL")
    ap.add_argument("--scale", default="1,1,1"); ap.add_argument("--translation-frame", choices=["auto", "start", "task"], default="auto")
    ap.add_argument("--max-tilt-deg", type=float, default=20.0); ap.add_argument("--tcp-map", choices=["c922", "identity"], default="c922")
    ap.add_argument("--fusion-dir", default="umi_fusion_rcb3gate"); ap.add_argument("--out-name")
    ap.add_argument("--time-scale", type=float, default=2.0, help="robot seconds per dataset second at deployment (run_rebot default 2): "
                    "the joint-step limit is checked over GRID_DT * time_scale"); a = ap.parse_args()
    a.out_name = a.out_name or f"robotized_{a.mode}_{a.orientation.lower()}"
    cfg = core.RobotizeCfg(mode=a.mode, orientation=a.orientation, scale_xyz=tuple(float(v) for v in a.scale.split(",")),
                           translation_frame=a.translation_frame, max_tilt_deg=a.max_tilt_deg,
                           R_map=c922_R_map() if a.tcp_map == "c922" else np.eye(3))
    checker = ReBotChecker(GRID_DT * a.time_scale)
    for e in a.episodes:
        t1 = time.time(); qa = run(pathlib.Path(e).resolve(), a, cfg, checker)
        print(json.dumps({k: qa[k] for k in ("episode", "mode", "ik_ok_frac", "alpha", "fail_reasons")} | dict(
            amp_ratio_final={s: qa["amplitude"][s]["ratio_final"] for s in qa["amplitude"]}, lr_rel_err_final=qa["lr_rel_err_final"],
            sec=round(time.time() - t1, 1)), default=float), flush=True)


if __name__ == "__main__":
    main()
