#!/usr/bin/env python3
"""[2026-10-03 user] HRA right-only OFFLINE transfer diagnostic: does an ego-only checkpoint point the right arm at the cube in
a REAL robot scene?  Read-only on the robot: /observe only, nothing is ever sent to /execute_step.

    touch   the operator puts the right TCP on the cube's top-face centre (by hand / UI jog); FK -> p_touch (base frame).
            Saved with an explicit approach offset:  p_target = p_touch + [0, 0, offset_z]   (scene json)
    probe   one observation at a start pose -> the checkpoint (served exactly like the UI: V4_ARM_ONLY=right contract, relcart20,
            anchor = this pose, task "Approach to the red cube") -> N noise seeds -> right TCP targets k = 1..16 in the BASE frame
            (the core already applies tgt = T_base_tcp @ A_k, i.e. dp_base = R_base_tcp @ dp_tcp) ->
                d_pred_k = p_pred_k - p_tcp,   d_gt = p_target - p_tcp,   cos_k = d_pred_k . d_gt / (|d_pred_k| |d_gt|)
            for k8 / k16: cos mean / std / p10 / p90 and |dp| mean, plus the motion gate (|dp_k16| vs the HRA training p50/p95).
            One row per (ckpt, scene, pose) appended to ~/hra_probe/results.csv.
    table   the comparison table (ckpt x pose), with the direction gate and the motion gate.

    PY=~/xvla-mac/bin/python
    $PY hra_direction_probe.py touch --scene s1 [--offset-z 0.025]
    $PY hra_direction_probe.py probe --scene s1 --pose center --ckpt ~/holobrain-mac-model/ckpt_UI_HRA-..._5k_rightonly [--seeds 16]
    $PY hra_direction_probe.py table
Same cube position for all poses of a scene; move only the robot's start pose (far-left / center / far-right).
Direction gate: > 0.7 strong, 0.4-0.7 worth a robot trial, 0.2-0.4 weak, < 0.2 transfer failing, < 0 wrong direction.
"""
import argparse, csv, json, math, os, pathlib, sys, time
import numpy as np

OUT = pathlib.Path.home() / "hra_probe"; SCENES = OUT / "scenes"; RES = OUT / "results.csv"
ROBOT = os.environ.get("V4_ROBOT", "http://localhost:8020")
TASK = "Approach to the red cube"
# HRA training motion reference (0.8 s right translation); overwritten from the QC summary when it exists
TRAIN_K16_P50_CM, TRAIN_K16_P95_CM = 1.1, 11.0


def observe():
    import requests
    o = requests.get(ROBOT + "/observe", timeout=10).json()
    if "error" in o: raise SystemExit(f"/observe failed: {o['error']}")
    return {"images": o["images"], "joints14": o["joints_rad"], "t": time.time()}


def right_tcp(joints14):
    """FK of the right TCP (base frame position; the same _tcp_mat the UI uses), without loading a model"""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import infer_core_v4 as IC, eef_kin
    k = object.__new__(IC.V4Inferencer); k.kin = eef_kin.Kin({"max_joint_delta": None})
    m, _ = k._tcp_mat(joints14); return m[1]


def touch(a):
    SCENES.mkdir(parents=True, exist_ok=True)
    o = observe(); T = right_tcp(o["joints14"]); p = T[:3, 3]
    sc = dict(scene=a.scene, p_touch=p.tolist(), offset_z=a.offset_z, p_target=(p + np.array([0, 0, a.offset_z])).tolist(),
              joints14=[float(x) for x in o["joints14"]], t=time.strftime("%F %T"),
              note="right TCP on the cube top-face centre; target = touch + [0,0,offset_z] (base z assumed up)")
    (SCENES / f"{a.scene}.json").write_text(json.dumps(sc, indent=1))
    print(f"scene {a.scene}: touch {np.round(p * 1000, 1)} mm -> target {np.round(np.array(sc['p_target']) * 1000, 1)} mm  ({SCENES / a.scene}.json)")


def train_ref():
    q = pathlib.Path.home() / "c8/hra_red/scale_qc_summary.json"
    try:
        p50, p95 = json.loads(q.read_text())["action_0p8s_right_translation_cm_p50_p95"]; return float(p50), float(p95)
    except Exception:
        return TRAIN_K16_P50_CM, TRAIN_K16_P95_CM


def probe(a):
    os.environ.update(V4_ARM_ONLY="right", V4_ARM_ONLY_GRIP="hold", V4_TASK=TASK, V4_RELONLY="1", V4_CKPT=str(a.ckpt),
                      V4_ACTION_MODE="umi", V4_STATE_MODE="relcart20", V4_GRIPPER="hold", V4_CLAMP_MM="0", V4_CLAMP_DEG="0",
                      V4_CHUNK="16", V4_N_ACTION="1", V4_IK_MAX_JOINT_DELTA="none", IK_BACKEND="pink",
                      V4_ARM_ONLY_MAX_PRED_MM="100000", V4_ARM_ONLY_MAX_PRED_DEG="1000",     # diagnostic: record, never refuse
                      V4_IK_LOG=str(OUT / "probe_ik.jsonl"))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import importlib, torch, infer_core_v4 as IC
    IC = importlib.reload(IC)                                    # module-level V4_* settings are read at import: apply the ones above
    assert IC.ARM_ONLY == "right" and IC.STATE_MODE == "relcart20" and IC.LIMITS["mm"] == 0, "probe contract not applied"
    sc = json.loads((SCENES / f"{a.scene}.json").read_text()); p_tgt = np.array(sc["p_target"])
    inf = IC.V4Inferencer(str(a.ckpt))
    o = observe(); h0 = dict(o, t=o["t"] - IC.UMI_HISTORY_DT); hist = [h0, o]
    D = {8: [], 16: []}; M = {8: [], 16: []}; p_tcp = None
    for s in range(a.seeds):
        torch.manual_seed(1000 + s); inf._anchor = None
        if hasattr(inf, "_buf"): del inf._buf                        # fresh history buffer per seed (same two observations)
        inf.infer(hist); L = inf.last; p_tcp = L["cur_mat"][1][:3, 3]; d_gt = p_tgt - p_tcp
        for k in (8, 16):
            d = L["tgt_pos"][k - 1][1] - p_tcp
            D[k].append(float(d @ d_gt / (np.linalg.norm(d) * np.linalg.norm(d_gt) + 1e-12))); M[k].append(float(np.linalg.norm(d)) * 100)
    p50, p95 = train_ref(); m16 = float(np.mean(M[16]))
    motion = "too small" if m16 < 0.2 * p50 else "too large" if m16 > 2.0 * p95 else "ok"
    row = dict(t=time.strftime("%F %T"), ckpt=pathlib.Path(a.ckpt).name, scene=a.scene, pose=a.pose, seeds=a.seeds,
               dist_to_target_cm=round(float(np.linalg.norm(p_tgt - p_tcp)) * 100, 1),
               **{f"k{k}_{n}": round(float(f(D[k])), 3) for k in (8, 16) for n, f in
                  (("cos_mean", np.mean), ("cos_std", np.std), ("cos_p10", lambda x: np.percentile(x, 10)), ("cos_p90", lambda x: np.percentile(x, 90)))},
               k8_dp_cm=round(float(np.mean(M[8])), 2), k16_dp_cm=round(m16, 2), motion_gate=motion,
               train_k16_p50_p95_cm=f"{p50}/{p95}")
    OUT.mkdir(parents=True, exist_ok=True); new = not RES.exists()
    with open(RES, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader() if new else None; w.writerow(row)
    print(json.dumps(row, indent=1)); print("verdict:", verdict(row["k16_cos_mean"]), "| motion:", motion)


def verdict(c):
    return "strong transfer" if c > 0.7 else "worth a robot trial" if c > 0.4 else "weak / unstable" if c > 0.2 else "transfer failing" if c >= 0 else "WRONG direction"


def table(a):
    if not RES.exists(): raise SystemExit("no results yet")
    rows = list(csv.DictReader(open(RES)))
    print("| ckpt | scene | pose | k8 cos mean | k8 std | k16 cos mean | k16 std | k16 p10/p90 | |dp k8| cm | |dp k16| cm | dist cm | direction | motion |")
    print("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|")
    for r in rows:
        print(f"| {r['ckpt'][-40:]} | {r['scene']} | {r['pose']} | {r['k8_cos_mean']} | {r['k8_cos_std']} | {r['k16_cos_mean']} | {r['k16_cos_std']} | "
              f"{r['k16_cos_p10']}/{r['k16_cos_p90']} | {r['k8_dp_cm']} | {r['k16_dp_cm']} | {r['dist_to_target_cm']} | {verdict(float(r['k16_cos_mean']))} | {r['motion_gate']} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("touch"); t.add_argument("--scene", required=True); t.add_argument("--offset-z", type=float, default=0.025)
    p = sub.add_parser("probe"); p.add_argument("--scene", required=True); p.add_argument("--pose", required=True)
    p.add_argument("--ckpt", required=True); p.add_argument("--seeds", type=int, default=16)
    sub.add_parser("table")
    a = ap.parse_args(); {"touch": touch, "probe": probe, "table": table}[a.cmd](a)
