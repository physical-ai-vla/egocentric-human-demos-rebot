#!/bin/bash
# [2026-10-03] HRA_red 200: wait for the right-wrist MASt3R run, then (CPU only, no training):
#   right_only export (cube PnP metric scale, IMU diagnostic) -> convert (labels + loss mask) -> LeRobot (right wrist 100 % robotized)
#   -> scale QC summary.  Outputs under ~/c8/hra_red/.  Log: ~/c8/hra_red/chain.log
set -u
LOG=$HOME/c8/robotlike/logs/HRA_red_152502_full.log; OUT=$HOME/c8/hra_red; PY=$HOME/xvla-mac/bin/python
SESS=$HOME/ego_collector/datasets/human_handumi_raw/HRA_red/HRA_red_20261003_152502
log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "waiting for MASt3R ($LOG)"
until grep -q "SKIP_CHECK=1: poses only" $LOG; do sleep 120; done
log "MASt3R: $(grep 'ss1.csv present' $LOG | tail -1)"
cd $HOME/ego_cart20
$PY -m ego_cart20.right_only export $SESS $OUT/raw 2>&1 | grep -vE "jaxls|INFO|Warning|warn\(" | tail -3
$PY -m ego_cart20.right_only convert $OUT/raw $OUT/processed --val-frac 0.1 2>&1 | tail -2
$PY ego_cart20/scripts/export_lerobot_right_only.py $OUT/processed $OUT/lerobot --workers 6 2>&1 | grep -E "^(train|val) \{|Error|Traceback" | tail -4
$PY - <<'PYEOF'
import json, glob, numpy as np, pathlib
Q = [json.load(open(f)) for f in sorted(glob.glob(str(pathlib.Path.home() / "c8/hra_red/raw/_scale_qc/*.json")))]
ok = [q for q in Q if q.get("valid")]; s = np.array([q["s_pnp"] for q in ok]); r = np.array([q["r_pnp_over_imu"] for q in ok if q.get("r_pnp_over_imu")])
rep = dict(episodes=len(Q), scale_valid=len(ok), rejected={q["episode"]: q["reason"] for q in Q if not q.get("valid")},
           s_pnp_p5_p50_p95=np.percentile(s, [5, 50, 95]).round(4).tolist() if len(s) else None,
           r_pnp_over_imu_p5_p50_p95=np.percentile(r, [5, 50, 95]).round(2).tolist() if len(r) else None,
           centre_resid_cm_p50=round(float(np.median([q["centre_resid_m"] for q in ok])) * 100, 2) if ok else None,
           n_valid_pnp_p50=float(np.median([q["n_valid_pnp"] for q in ok])) if ok else None)
H = pathlib.Path.home()
ex = json.loads(open(sorted(glob.glob(str(H / "c8/hra_red/raw/export_log_*.json")))[-1]).read()) if glob.glob(str(H / "c8/hra_red/raw/export_log_*.json")) else []
nd = np.array([e["net_displacement_m"] for e in ex if e.get("status") == "ok" and e.get("net_displacement_m") is not None])
rep.update(exported_ok=sum(e.get("status") == "ok" for e in ex), export_status={k: sum(e.get("status") == k for e in ex) for k in set(e.get("status") for e in ex)},
           home_to_cube_tcp_net_m_p5_p50_p95=np.percentile(nd, [5, 50, 95]).round(3).tolist() if len(nd) else None)
k16 = []
for d in sorted((H / "c8/hra_red/processed/episodes").glob("*")) if (H / "c8/hra_red/processed/episodes").exists() else []:
    a = np.load(d / "action.npy"); k16.append(np.linalg.norm(a[:, 15, 10:13], axis=-1))
if k16:
    k16 = np.concatenate(k16) * 100; rep["action_0p8s_right_translation_cm_p50_p95"] = [round(float(np.median(k16)), 2), round(float(np.percentile(k16, 95)), 2)]
    rep["train_rows_total"] = int(len(k16))
json.dump(rep, open(H / "c8/hra_red/scale_qc_summary.json", "w"), indent=1); print("SCALE QC", json.dumps(rep)[:1200])
PYEOF
log "CHAIN DONE"
