"""Per-episode QA (2026-10-14, spec 34). Run after every teleop episode; a FLAGGED episode is not silently used for training
(qa.json `include: false` until a human overrides it with a reason).

Metrics: duration, valid tracking %, HOLD duration, IK/safety/robot reject counts, max translation/rotation per frame of the
EXECUTED trajectory, gripper transitions, camera / IMU dropouts. Flags: unexplained jump (executed step above the bound
while the controller reports no motion reason), gripper polarity (command vs observed disagree in sign of change), long HOLD,
tracking collapse, unsafe rejection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")


def _steps(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(T).all(axis=(1, 2)); T = T[ok]
    if len(T) < 2: return np.zeros(0), np.zeros(0)
    dp = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    dr = np.degrees(Rotation.from_matrix(np.einsum("nji,njk->nik", T[:-1, :3, :3], T[1:, :3, :3])).magnitude())
    return dp, dr


def _runs(mask: np.ndarray, t: np.ndarray) -> list[float]:
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1; out.append(float(t[end] - t[start] + (t[1] - t[0] if len(t) > 1 else 0))); start = None
    return out


def episode_qa(ep_dir: str | Path, qa_cfg: dict, events: list | None = None) -> dict:
    ep = Path(ep_dir).expanduser(); a = dict(np.load(ep / "ticks.npz", allow_pickle=False))
    t = (a["t_ns"] - a["t_ns"][0]) / 1e9; n = len(t); mode = a["mode"]
    ev = events if events is not None else [json.loads(l) for l in (ep / "events.jsonl").read_text().splitlines() if l.strip()]
    reasons = [r for e in ev for v in e["reasons"].values() for r in v]
    cnt = lambda pre: sum(1 for r in reasons if r.startswith(pre))
    q = dict(duration_s=float(t[-1]) if n else 0.0, ticks=n, rate_hz=float((n - 1) / t[-1]) if n > 1 and t[-1] > 0 else 0.0,
             hold_fraction=float(np.mean(mode != "FOLLOW")) if n else 1.0, estop_ticks=int(np.sum(mode == "ESTOP")),
             ik_reject=cnt("ik_") + cnt("joint_"), safety_reject=cnt("workspace") + cnt("target_jump") + cnt("nan"),
             robot_reject=cnt("robot_reject"), inter_arm_hold=cnt("tcp_separation") + cnt("wrist_separation") + cnt("crossing"), sides={})
    flags = []
    for s in SIDES:
        T = a[f"obs_T_{s}"] if np.isfinite(a[f"obs_T_{s}"]).any() else a[f"exec_T_{s}"]
        dp, dr = _steps(T); st = a[f"hand_status_{s}"]
        gt = a[f"g_exec_{s}"]; go = a[f"obs_g_{s}"]
        trans = int(np.sum(np.abs(np.diff(np.round(gt[np.isfinite(gt)]))) > 0)) if np.isfinite(gt).sum() > 1 else 0
        hold_runs = _runs(a[f"held_{s}"], t) if n else []
        d = dict(valid_tracking=float(np.mean((st == "TRACKING") | (st == "WARN"))) if n else 0.0, tracking_ok=float(np.mean(st == "TRACKING")) if n else 0.0, tracking_hold=float(np.mean(st == "HOLD")) if n else 1.0,
                 max_translation_per_frame_m=float(dp.max()) if dp.size else 0.0, max_rotation_per_frame_deg=float(dr.max()) if dr.size else 0.0,
                 gripper_transitions=trans, longest_hold_s=max(hold_runs) if hold_runs else 0.0)
        # polarity: commanded g and observed g must move together (both 0 = closed). Correlation of their changes < 0 => flipped
        ok = np.isfinite(gt) & np.isfinite(go)
        if ok.sum() > 10 and np.std(gt[ok]) > 0.05 and np.std(go[ok]) > 0.05:
            d["gripper_cmd_obs_corr"] = float(np.corrcoef(gt[ok], go[ok])[0, 1])
            if d["gripper_cmd_obs_corr"] < 0: flags.append(f"{s}:gripper_polarity")
        if d["max_translation_per_frame_m"] > qa_cfg["max_translation_per_frame_m"]: flags.append(f"{s}:translation_jump:{d['max_translation_per_frame_m']*1e3:.0f}mm")
        if d["max_rotation_per_frame_deg"] > qa_cfg["max_rotation_per_frame_deg"]: flags.append(f"{s}:rotation_jump:{d['max_rotation_per_frame_deg']:.1f}deg")
        if d["valid_tracking"] < qa_cfg["min_valid_tracking"]: flags.append(f"{s}:tracking_collapse:{d['valid_tracking']:.2f}")
        q["sides"][s] = d
    if q["hold_fraction"] > qa_cfg["max_hold_fraction"]: flags.append(f"long_hold:{q['hold_fraction']:.2f}")
    if q["robot_reject"] or q["estop_ticks"]: flags.append(f"unsafe_or_rejected:robot_reject={q['robot_reject']},estop_ticks={q['estop_ticks']}")
    for side_stream in ("imu_left.npz", "imu_right.npz"):
        f = ep / side_stream
        if f.exists():
            h = np.load(f)["host_ns"]; gaps = np.diff(np.sort(h)) / 1e6 if len(h) > 1 else np.zeros(0)
            q[side_stream.split(".")[0] + "_max_gap_ms"] = float(gaps.max()) if gaps.size else None
    for f in sorted((ep / "frames").glob("*_t_ns.npy")):
        h = np.load(f); gaps = np.diff(h) / 1e6 if len(h) > 1 else np.zeros(0)
        q[f.stem.replace("_t_ns", "") + "_max_gap_ms"] = float(gaps.max()) if gaps.size else None
    q["flags"] = flags; q["include"] = not flags
    (ep / "qa.json").write_text(json.dumps(q, indent=1)); return q
