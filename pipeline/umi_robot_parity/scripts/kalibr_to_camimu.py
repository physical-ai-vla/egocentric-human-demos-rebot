"""Kalibr camera-IMU result -> configs/calibration/stereo_trackcam_imu_v005.yaml (same schema as v004) (2026-10-16).

  input   <take>/kalibr/ with calib-camchain-imucam.yaml (+ calib-results-imucam.txt, export/export_meta.json from kalibr_bag_export.py)
  R_C0I   = cam0.T_cam_imu[:3,:3]   (w_cam = R_C0I w_imu, as in v004)
  t_ci    = cam0.T_cam_imu[:3, 3]   (IMU origin in the C0 frame, metres = the tracker's --t-ci)
  offset  Kalibr: t_imu = t_cam + timeshift_cam_imu. Bag IMU stamps = host + pre_offset_ms, cam stamps = host. Ours: t_imu_host + offset_ms
          = camera host time  ->  offset_ms = pre_offset_ms - 1000 * timeshift_cam_imu(cam0).  (cam1 shift is reported; L/R are free-running)
  Default prints a comparison with v004 only; --write creates the yaml (refuses to overwrite without --force).
    python scripts/kalibr_to_camimu.py datasets/stereo_imu/<calib_take>/kalibr [--write] [--out configs/calibration/stereo_trackcam_imu_v005.yaml]"""
from __future__ import annotations
import argparse, datetime, json, re
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
PREV = "configs/calibration/stereo_trackcam_imu_v004.yaml"
STEREO = "configs/calibration/stereo_trackcam_v005.yaml"


def rot_deg(Ra, Rb) -> float: return float(np.degrees(np.linalg.norm(R.from_matrix(Ra @ Rb.T).as_rotvec())))


def parse_results(txt: Path) -> dict:
    if not txt.exists(): return {}
    s = txt.read_text(); q = {}
    for key, lab in (("reproj_px", r"Reprojection error \((cam\d)\) \[px\]"), ("gyro_rad_s", r"Gyroscope error \(imu0\) \[rad/s\]"), ("acc_m_s2", r"Accelerometer error \(imu0\) \[m/s\^2\]")):
        for m in re.finditer(lab + r":\s*mean ([\d.eE+-]+), median ([\d.eE+-]+), std: ([\d.eE+-]+)", s):
            name = f"{key}_{m.group(1)}" if m.lastindex == 4 else key; v = m.groups()[-3:]
            q[name] = dict(mean=round(float(v[0]), 4), median=round(float(v[1]), 4), std=round(float(v[2]), 4))
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kalibr_dir"); ap.add_argument("--out", default="configs/calibration/stereo_trackcam_imu_v005.yaml")
    ap.add_argument("--write", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--note", default="")
    a = ap.parse_args()
    K = Path(a.kalibr_dir).resolve(); cc = yaml.safe_load((K / "calib-camchain-imucam.yaml").read_text())
    em_p = next((p for p in (K / "export_meta.json", K / "export" / "export_meta.json") if p.exists()), None)   # K/ = snapshot pulled with the result
    em = json.loads(em_p.read_text()) if em_p else {}
    pre = float(em.get("pre_offset_ms", 0.0))
    T0 = np.asarray(cc["cam0"]["T_cam_imu"], float); R0, t0 = T0[:3, :3], T0[:3, 3]
    sh0 = float(cc["cam0"].get("timeshift_cam_imu", 0.0)); off0 = pre - 1000.0 * sh0
    out_cam1 = {}
    if "cam1" in cc:
        T1 = np.asarray(cc["cam1"]["T_cam_imu"], float); sh1 = float(cc["cam1"].get("timeshift_cam_imu", 0.0))
        T_RL = np.asarray(yaml.safe_load((ROOT / STEREO).read_text())["T_CR_CL"], float); E = np.linalg.inv(T_RL @ T0) @ T1
        out_cam1 = dict(offset_ms=round(pre - 1000.0 * sh1, 2), chain_vs_T_CR_CL_rot_deg=round(rot_deg(E[:3, :3], np.eye(3)), 3), chain_vs_T_CR_CL_mm=round(float(np.linalg.norm(E[:3, 3]) * 1000), 2))
    prev = yaml.safe_load((ROOT / PREV).read_text()); Rp = np.asarray(prev["R_C0I"], float)
    q = parse_results(K / "calib-results-imucam.txt")
    print(f"[kalibr->camimu] {K}")
    print(f"  R_C0I rotvec(deg) {np.round(np.degrees(R.from_matrix(R0).as_rotvec()), 2).tolist()}   v004 대비 회전 차 {rot_deg(R0, Rp):.2f}°")
    print(f"  t_ci (IMU 원점 @C0) {np.round(t0, 4).tolist()} m   (트래커 기본 --t-ci -0.0413,-0.0275,0.0137)")
    print(f"  offset_ms (cam0) {off0:.2f}  (v004 {prev['offset_ms']}, Kalibr shift {sh0 * 1000:.2f} ms, pre_offset {pre} ms)")
    if out_cam1: print(f"  cam1: offset_ms {out_cam1['offset_ms']}, T_CR_CL 일관성 회전 {out_cam1['chain_vs_T_CR_CL_rot_deg']}° / {out_cam1['chain_vs_T_CR_CL_mm']} mm")
    for k, v in q.items(): print(f"  {k}: 평균 {v['mean']} 중앙 {v['median']}")
    doc = dict(created=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"), source=em.get("take", str(K.parent)), method="Kalibr kalibr_calibrate_imu_camera (joint R, t, time offset); scripts/kalibr_to_camimu.py",
               imu=prev.get("imu"), reference_camera=prev.get("reference_camera"), stereo_calib=STEREO,
               R_C0I=[[round(float(x), 6) for x in r] for r in R0], R_C0I_rotvec_deg=[round(float(x), 2) for x in np.degrees(R.from_matrix(R0).as_rotvec())],
               offset_ms=round(off0, 2), offset_note="t_imu_host + offset_ms = camera host time (cam0); w_cam = R_C0I w_imu; offset_ms = pre_offset_ms - 1000*kalibr timeshift_cam_imu",
               t_C0I_m=[round(float(x), 5) for x in t0], t_C0I_note="IMU origin in the C0 (left cam) frame = tracker --t-ci (Kalibr T_cam_imu translation)",
               cam1=out_cam1, kalibr_quality=q, vs_v004=dict(rot_deg=round(rot_deg(R0, Rp), 3), offset_ms_delta=round(off0 - float(prev["offset_ms"]), 2)),
               export=dict(pre_offset_ms=pre, noise_scale=em.get("noise_scale"), clock_fit=em.get("clock_fit")), note=a.note)
    bad = []
    if max((v["median"] for k, v in q.items() if k.startswith("reproj_px")), default=99) > 1.5: bad.append("재투영 중앙 > 1.5 px")
    if np.linalg.norm(t0) > 0.15: bad.append(f"|t_ci| {np.linalg.norm(t0) * 100:.0f} cm > 15 cm")
    if abs(off0) > 150: bad.append(f"|offset| {off0:.0f} ms > 150 ms")
    if bad: print("  !! 신뢰 불가: " + ", ".join(bad))
    doc["sanity_flags"] = bad
    if not a.write:
        print("  (미기록: --write 를 주면 저장)"); return
    out = ROOT / a.out
    if out.exists() and not a.force: raise SystemExit(f"{out} 이미 있음 — 덮어쓰려면 --force")
    if bad and not a.force: raise SystemExit("sanity 실패 — 저장 안 함 (--force 로 강제)")
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=None)); print(f"  저장 -> {out}")


if __name__ == "__main__":
    main()
