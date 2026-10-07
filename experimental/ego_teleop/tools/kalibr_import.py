"""Import an offline Kalibr result into the versioned wrist calibration bundle (and the pose pipeline's per-kind files).

    python -m ego_teleop.tools.kalibr_import camchain-imucam-calib_right.yaml --side right --write-bundle \
           [--unit-id wristR-01 --mount-revision v1 --camera-serial ... --imu-serial ...]

Conventions (checked, not assumed):
  Kalibr `T_cam_imu`        = pose of the IMU in the camera frame  == our `T_camera_imu`         (no inversion)
  Kalibr `timeshift_cam_imu` (t_imu = t_cam + shift)               -> our `time_offset_ms` (camera - imu) = -shift * 1000
  Kalibr `camera_model: pinhole` + `distortion_model: equidistant` -> our `model: kannala_brandt` (cv2.fisheye)
`--write-bundle` marks the camera-IMU provenance as `kalibr`, which is what `require_production()` demands."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml

DIST_MODEL = {"equidistant": "kannala_brandt", "radtan": "pinhole", "none": "pinhole"}


def parse_camchain(path: str | Path, cam: str = "cam0") -> dict:
    txt = Path(path).read_text()
    if txt.lstrip().startswith("%YAML"): txt = txt.split("\n", 1)[1]
    d = yaml.safe_load(txt)
    if cam not in d: raise SystemExit(f"{path}: no {cam} (found {sorted(d)})")
    c = d[cam]
    intr = [float(x) for x in c["intrinsics"]]                     # fu, fv, pu, pv
    K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]], np.float64)
    D = np.asarray([float(x) for x in c.get("distortion_coeffs", [0, 0, 0, 0])], np.float64)
    w, h = (int(v) for v in c["resolution"])
    model = DIST_MODEL.get(str(c.get("distortion_model", "equidistant")).lower())
    if model is None: raise SystemExit(f"unsupported distortion_model {c.get('distortion_model')!r}")
    out = dict(camera=dict(model=model, K=K, D=D, image_size=(w, h), target=c.get("target"), reprojection_rms_px=c.get("reprojection_error")))
    if "T_cam_imu" in c:
        T = np.asarray(c["T_cam_imu"], np.float64).reshape(4, 4)   # IMU in camera frame == our T_camera_imu
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0: raise SystemExit("T_cam_imu rotation block is not a proper rotation")
        out["T_camera_imu"] = T
    if "timeshift_cam_imu" in c: out["time_offset_ms"] = -float(c["timeshift_cam_imu"]) * 1e3
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("camchain"); ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--cam", default="cam0"); ap.add_argument("--write-bundle", action="store_true"); ap.add_argument("--export-pose-files", action="store_true")
    ap.add_argument("--cal-dir", default=None); ap.add_argument("--unit-id", default=None); ap.add_argument("--mount-revision", default=None)
    ap.add_argument("--camera-serial", default=None); ap.add_argument("--imu-serial", default=None); ap.add_argument("--notes", default="")
    ap.add_argument("--keep-time-offset", action="store_true",
                    help="keep the bundle's existing time_offset_ms (a frozen measurement) instead of taking Kalibr's; "
                         "Kalibr's value is still reported and recorded in the provenance notes as a cross-check")
    a = ap.parse_args(argv)
    r = parse_camchain(a.camchain, a.cam)
    cam = r["camera"]
    print(f"camera: {cam['model']} {cam['image_size']} fx={cam['K'][0,0]:.2f} fy={cam['K'][1,1]:.2f} cx={cam['K'][0,2]:.2f} cy={cam['K'][1,2]:.2f} D={np.round(cam['D'],5).tolist()}")
    if "T_camera_imu" in r:
        T = r["T_camera_imu"]; print(f"T_camera_imu: t={np.round(T[:3,3]*1e3,2).tolist()} mm")
    if "time_offset_ms" in r: print(f"time_offset_ms (camera - imu): {r['time_offset_ms']:+.3f}")
    if a.write_bundle:
        from ..calibration.bundle import load_bundle, save_bundle
        cal_dir = Path(a.cal_dir) if a.cal_dir else None
        b = load_bundle(a.side, cal_dir=cal_dir)
        b.camera = dict(model=cam["model"], K=cam["K"], D=cam["D"], image_size=cam["image_size"], reprojection_rms_px=cam.get("reprojection_rms_px"), target=cam.get("target"))
        b.set_provenance("camera", source="kalibr", tool="kalibr_calibrate_cameras", runs=[Path(a.camchain).name], notes=a.notes)
        if "T_camera_imu" in r:
            b.T_camera_imu = r["T_camera_imu"]
            b.set_provenance("T_camera_imu", source="kalibr", tool="kalibr_calibrate_imu_camera", runs=[Path(a.camchain).name], notes=a.notes)
        if "time_offset_ms" in r:
            kal_ms = r["time_offset_ms"]
            if a.keep_time_offset and b.time_offset_ms is not None:
                prev = b.provenance.get("time_offset_ms") or {}
                prev["notes"] = (prev.get("notes", "") + f" CROSS-CHECK: Kalibr ({Path(a.camchain).name}) independently estimated "
                                 f"{kal_ms:+.3f} ms, {abs(kal_ms - b.time_offset_ms):.3f} ms from this frozen value; the frozen value is kept.").strip()
                b.provenance["time_offset_ms"] = prev
                print(f"  keeping frozen time_offset_ms {b.time_offset_ms:+.3f} ms (Kalibr said {kal_ms:+.3f}, delta {abs(kal_ms - b.time_offset_ms):.3f} ms)")
            else:
                b.time_offset_ms = kal_ms
                b.set_provenance("time_offset_ms", source="kalibr", tool="kalibr_calibrate_imu_camera", runs=[Path(a.camchain).name], notes=a.notes)
        for k, v in (("unit_id", a.unit_id), ("mount_revision", a.mount_revision), ("camera_serial", a.camera_serial), ("imu_serial", a.imu_serial)):
            if v: b.hardware[k] = v
        b.hardware.setdefault("side", a.side)
        p = save_bundle(b, cal_dir=cal_dir, notes=f"kalibr import: {Path(a.camchain).name}. {a.notes}")
        print(f"wrote {p}  hardware_hash={b.hardware_hash()}  missing={b.missing or 'nothing'}  production_ready={not b.missing and b.offline_camera_imu}")
        if a.export_pose_files:
            for k, v in b.export_pose_files(cal_dir=cal_dir, notes=f"from {p.stem}").items(): print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
