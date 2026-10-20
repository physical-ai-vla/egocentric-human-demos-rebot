"""VIO acceptance report for a `stereo_imu_record --proto verify` take (2026-10-16, after the camera-IMU joint calibration).
Ground truth = the fixed AprilGrid (scripts/trackcam_board_lever.py -> board_traj_cam_left.npz, camera-in-board poses; board flat on the
desk so board z = vertical). The estimator trajectory (OpenVINS offline by default, or --traj TUM of IMU poses) is turned into LEFT
camera poses with the calibration under test and rotated into the board frame (orientation-mean alignment, as stereo_basalt_eval).
Per protocol step (holds = "멈춘 채로 계세요" windows, 0.8..2.6 s after the cue):
  translation moves  GT displacement d_gt vs estimate d_est between the surrounding holds:
                     distance error % = | |d_est| - |d_gt| | / |d_gt|,  leakage = |component of (d_est - d_gt) perpendicular to d_gt|,
                     vertical leakage (forward / left moves) = |(d_est - d_gt)_z|, forward leakage of the up move = horizontal part
  pure rotations     fake translation = |d_est - d_gt| of the camera centre across the rotation (GT includes the real lever motion)
  30 s still         drift = max distance from the window mean
then the full scorer (stereo_basalt_eval --external-traj): ATE p50 / drift30 / fake_rot.
Gates (user 10-16): vertical leakage < 2 cm, distance error < 10 %, still drift < 3 cm, fake translation < 3 cm.
    python scripts/vio_verify.py datasets/stereo_imu/<ts>_verify [--camimu configs/calibration/stereo_trackcam_imu_v005.yaml] [--t-ci x,y,z] [--traj traj.tum]"""
from __future__ import annotations
import argparse, contextlib, io, json, subprocess, sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
STEREO = "configs/calibration/stereo_trackcam_v005.yaml"
GATES = dict(vertical_leak_cm=2.0, dist_err_pct=10.0, still_drift_cm=3.0, fake_trans_cm=3.0)


def camimu(path: str, t_ci_arg: str | None):
    y = yaml.safe_load((ROOT / path).read_text()); T = np.eye(4); T[:3, :3] = np.asarray(y["R_C0I"], float)
    t = t_ci_arg or y.get("t_ci") or y.get("t_C0I")
    T[:3, 3] = [float(v) for v in t.split(",")] if isinstance(t, str) else (np.asarray(t, float) if t is not None else [-0.0413, -0.0275, 0.0137])
    return T, float(y["offset_ms"])                              # T_C0I (IMU -> left camera)


def run_openvins(E: Path, cam_file: str, t_ci: str, tag: str) -> Path:
    import stereo_openvins_eval as ov
    ov.CAMIMU_FILE = cam_file; ov.STEREO_FILE = STEREO; ov.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
    sys.argv = ["x", str(E), f"--t-ci={t_ci}", "--tag", tag]
    with contextlib.redirect_stdout(io.StringIO()): ov.main()
    return E / f"openvins_stereo{tag}" / "trajectory_imu.tum"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--camimu", default="configs/calibration/stereo_trackcam_imu_v004.yaml"); ap.add_argument("--t-ci", default=None)
    ap.add_argument("--traj", default=None); ap.add_argument("--stereo", default="configs/calibration/stereo_trackcam_v005.yaml"); ap.add_argument("--tag", default="_verify"); a = ap.parse_args()
    global STEREO; STEREO = a.stereo
    E = Path(a.take).resolve(); T_C0I, _ = camimu(a.camimu, a.t_ci); t_ci = ",".join(f"{v:.4f}" for v in T_C0I[:3, 3])
    if not (E / "board_traj_cam_left.npz").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/trackcam_board_lever.py"), str(E), "--stereo-cam", "cam_left", "--camimu", a.camimu], capture_output=True)
    traj = Path(a.traj) if a.traj else run_openvins(E, a.camimu, t_ci, a.tag)
    A = np.loadtxt(traj); tn = (A[:, 0] * 1e9).astype(np.int64)
    Twi = np.tile(np.eye(4), (len(A), 1, 1)); Twi[:, :3, :3] = R.from_quat(A[:, 4:8]).as_matrix(); Twi[:, :3, 3] = A[:, 1:4]; Twc = Twi @ np.linalg.inv(T_C0I)
    gt = np.load(E / "board_traj_cam_left.npz"); gi = {int(t): i for i, t in enumerate(gt["t_ns"])}
    m = np.array([int(t) in gi for t in tn]); idx = np.array([gi[int(t)] for t in tn[m]])
    P, Q, G, RG = Twc[m, :3, 3], R.from_matrix(Twc[m, :3, :3]), gt["T_B_C"][idx, :3, 3], R.from_matrix(gt["T_B_C"][idx, :3, :3])
    Rm = (RG * Q.inv()).mean().as_matrix(); Pb = P @ Rm.T                      # estimate rotated into the board frame
    t0 = np.load(E / "cam_left_t_ns.npy")[0] / 1e9; tt = tn[m] / 1e9 - t0
    proto = json.loads((E / "protocol.json").read_text())

    def hold(c):
        w = (tt >= c["t0"] + 0.8) & (tt < c["t0"] + 2.6)
        return (Pb[w].mean(0), G[w].mean(0), P[w].mean(0)) if w.sum() >= 5 else None          # aligned est, GT (board), RAW VIO-world est
    rows, holds = [], [k for k, c in enumerate(proto) if c["text"].startswith("멈춘") or c["text"].startswith("보드가")]
    for k0, k1 in zip(holds[:-1], holds[1:]):
        h0, h1 = hold(proto[k0]), hold(proto[k1]); mv = proto[k0 + 1]["text"] if k0 + 1 < len(proto) else ""
        if h0 is None or h1 is None: rows.append(dict(step=mv, note="보드 안 보임")); continue
        de, dg = h1[0] - h0[0], h1[1] - h0[1]; err = de - dg; ng = np.linalg.norm(dg); r = dict(step=mv, gt_cm=round(ng * 100, 1), est_cm=round(np.linalg.norm(de) * 100, 1))
        if "제자리에서" in mv: r.update(kind="rot", fake_trans_cm=round(np.linalg.norm(err) * 100, 1))
        else:
            u = dg / max(ng, 1e-6); perp = err - (err @ u) * u
            r.update(kind="move", dist_err_pct=round(abs(np.linalg.norm(de) - ng) / max(ng, 1e-6) * 100, 1), leak_cm=round(np.linalg.norm(perp) * 100, 1),
                     vertical_leak_cm=round(abs(err[2]) * 100, 1) if "위로" not in mv and "내려" not in mv else None,
                     horizontal_leak_cm=round(np.linalg.norm(err[:2]) * 100, 1) if ("위로" in mv or "내려" in mv) else None)
        if "제자리에서" not in mv:      # 10-16 cross-axis coupling (user): est displacement in the GT-step basis [along, horizontal-perp, vertical] / |gt|
            zb = np.array([0, 0, 1.0]); u = dg / max(ng, 1e-6); hp = np.cross(zb, u); hp = hp / max(np.linalg.norm(hp), 1e-6) if np.linalg.norm(hp) > 0.2 else np.cross(u, [1.0, 0, 0]) / max(np.linalg.norm(np.cross(u, [1.0, 0, 0])), 1e-6)
            r["coupling_aligned"] = [round(float(de @ u / ng), 3), round(float(de @ hp / ng), 3), round(float(de[2] / ng), 3)]
            draw = h1[2] - h0[2]; gz = dg[2]
            r["raw_world"] = dict(horiz_ratio=round(float(np.linalg.norm(draw[:2]) / max(np.linalg.norm(dg[:2]), 1e-6)), 3) if np.linalg.norm(dg[:2]) > 0.03 else None,
                                 vertical_cm=round(float(draw[2]) * 100, 1), gt_vertical_cm=round(float(gz) * 100, 1), tilt_deg=round(float(np.degrees(np.arctan2(draw[2], np.linalg.norm(draw[:2])))), 1))
        rows.append(r)
    st = next((c for c in proto if c["text"].startswith(("30초", "10초"))), None); still = None
    if st:
        w = (tt >= st["t0"] + 1) & (tt < st["t0"] + (29 if st["text"].startswith("30초") else 9))
        if w.sum() > 10: still = round(float(np.linalg.norm(Pb[w] - Pb[w].mean(0), axis=1).max() * 100), 2)
    buf = io.StringIO()
    import stereo_basalt_eval as be
    be.CAMIMU_FILE = a.camimu; be.STEREO_FILE = a.stereo; sys.argv = ["x", str(E), f"--t-ci= {t_ci}", "--tag", "_s" + a.tag, "--external-traj", str(traj), "--backend-label", a.tag, "--wall-ms", "3"]
    with contextlib.redirect_stdout(buf): be.main()
    s = buf.getvalue(); sc = json.loads(s[s.find("{"):])
    mv = [r for r in rows if r.get("kind") == "move"]; rt = [r for r in rows if r.get("kind") == "rot"]
    vl = [r["vertical_leak_cm"] for r in mv if r.get("vertical_leak_cm") is not None]; hl = [r["horizontal_leak_cm"] for r in mv if r.get("horizontal_leak_cm") is not None]; de = [r["dist_err_pct"] for r in mv]; ft = [r["fake_trans_cm"] for r in rt]
    summ = dict(vertical_leak_cm_max=max(vl) if vl else None, horizontal_leak_cm_max=max(hl) if hl else None, horizontal_leak_cm_median=float(np.median(hl)) if hl else None, dist_err_pct_median=float(np.median(de)) if de else None, still_drift_cm=still,
                fake_trans_cm_max=max(ft) if ft else None, ate_p50_cm=sc.get("ate_se3_cm_p50"), drift30_cm=(sc.get("drift30_cm") or {}).get("p50"), fake_rot_cm=(sc.get("fake_rot_cm") or {}).get("p50"))
    verdict = dict(vertical_leak=summ["vertical_leak_cm_max"] is not None and summ["vertical_leak_cm_max"] < GATES["vertical_leak_cm"],
                   horizontal_leak=summ["horizontal_leak_cm_max"] is not None and summ["horizontal_leak_cm_max"] < GATES["vertical_leak_cm"],
                   distance=summ["dist_err_pct_median"] is not None and summ["dist_err_pct_median"] < GATES["dist_err_pct"],
                   still_drift=still is not None and still < GATES["still_drift_cm"], fake_translation=summ["fake_trans_cm_max"] is not None and summ["fake_trans_cm_max"] < GATES["fake_trans_cm"])
    verdict["raw_world_vertical_leak"] = summ.get("raw_world_vertical_leak_cm_max") is not None and summ["raw_world_vertical_leak_cm_max"] < GATES["vertical_leak_cm"]   # 10-16: alignment hides extrinsic tilt
    verdict = {k: bool(v) for k, v in verdict.items()}
    measured = dict(raw_world_vertical_leak=summ.get("raw_world_vertical_leak_cm_max") is not None, vertical_leak=summ["vertical_leak_cm_max"] is not None, horizontal_leak=summ["horizontal_leak_cm_max"] is not None, distance=summ["dist_err_pct_median"] is not None, still_drift=still is not None, fake_translation=summ["fake_trans_cm_max"] is not None)
    verdict = {k: (v if measured[k] else None) for k, v in verdict.items()}          # None = this protocol has no such step
    axis = lambda t: "x" if "앞으로" in t or "제자리로 돌아오세요" in t and "verify_x" in E.name else ("y" if "왼쪽" in t else ("z" if "위로" in t or "내려" in t else None))
    kind = "x" if "verify_x" in E.name else "y" if "verify_y" in E.name else "z" if "verify_z" in E.name else None
    cm = [r["coupling_aligned"] for r in rows if r.get("coupling_aligned")]
    if cm: summ["coupling_aligned_mean"] = np.round(np.mean(np.abs(cm), 0) * np.sign(np.mean(cm, 0)), 3).tolist(); summ["coupling_axis"] = kind
    rw = [r["raw_world"] for r in rows if r.get("raw_world")]
    if rw: summ["raw_world_tilt_deg_median_abs"] = float(np.median([abs(x["tilt_deg"]) for x in rw])); summ["raw_world_vertical_leak_cm_max"] = float(max(abs(x["vertical_cm"] - x["gt_vertical_cm"]) for x in rw))
    rep = dict(take=E.name, camimu=a.camimu, t_ci=t_ci, steps=rows, summary=summ, gates=GATES, verdict=verdict, all_pass=all(v for v in verdict.values() if v is not None))
    (E / f"vio_verify{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"[검증] {E.name} | 캘리브 {a.camimu} | t_ci {t_ci}")
    for r in rows: print("  ", {k: v for k, v in r.items() if v is not None})
    print("  요약", summ); print("  판정", {k: ("해당 없음" if v is None else "통과" if v else "실패") for k, v in verdict.items()}, "→", "전부 통과" if rep["all_pass"] else "미통과")


if __name__ == "__main__":
    main()
