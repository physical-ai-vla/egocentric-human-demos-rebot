"""[2026-10-06 user] HRA start-pose check: the human wrist TCP at each training episode's anchor row (first valid sample) vs the
robot rest TCP. Gravity (human) = accelerometer mean over the first 1.5 s (still hold), camera frame via IMU.T_b_c1, rotated into the
SLAM world with the anchor row's camera rotation. Cube = median over valid PnP frames of T_world_cam(i) @ t_cam (metric).
TCP axes (tcp_convention_v1): x = approach, y = jaw closing, z = x cross y."""
import ast, json, pathlib, re, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import yaml
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20.geometry.transforms import pose_to_T
RAW = H / "c8/hra_red/raw"; QC = RAW / "_scale_qc"; EXP = H / "c8/robotlike/export"; P = H / "c8/hra_red/processed"
CT = yaml.safe_load(open(H / "umi_bridge/trackA_mast3r_pose_v1/data/handumi_camera_tcp_v2.yaml"))
X = np.eye(4); X[:3, :3] = np.array(CT["rotation"]["R_camera_tcp"], float); X[:3, 3] = CT["translation"]["t_camera_tcp_m"]; Xi = np.linalg.inv(X)


def angles(R, up):
    x, y = R[:, 0], R[:, 1]
    return float(np.degrees(np.arcsin(np.clip(-x @ up, -1, 1)))), float(np.degrees(np.arcsin(np.clip(y @ up, -1, 1))))


rows = []
eps = [json.loads(l)["episode_id"] for s in ("train", "val") for l in open(P / f"{s}_manifest.jsonl")]
for ep in eps:
    num = ep.split("_")[-1]; tag = f"red_20261003_152502_{num}_right"; ex = EXP / tag
    try:
        z = np.load(RAW / ep / "raw_episode.npz"); v = z["right_valid"].astype(bool)
        T = pose_to_T(z["right_position"], z["right_quaternion"]); i0 = int(np.flatnonzero(v)[0]); T0 = T[i0]
        t = (ex / "orbslam_setting.yaml").read_text()
        Tbc = np.array([float(x) for x in re.search(r"IMU.T_b_c1:.*?data: \[([^\]]+)\]", t, re.S).group(1).split(",")]).reshape(4, 4)
        acc = json.load(open(ex / "imu_data.json"))["1"]["streams"]["ACCL"]["samples"]
        a0 = np.mean([s["value"] for s in acc if s["cts"] < 1500], axis=0); up_c = Tbc[:3, :3].T @ a0; up_c /= np.linalg.norm(up_c)
        Rwc0 = (T0 @ Xi)[:3, :3]; up = Rwc0 @ up_c
        pitch, roll = angles(T0[:3, :3], up)
        qc = json.load(open(QC / f"{ep}.json")); fr = qc["frames"]; fr = ast.literal_eval(fr) if isinstance(fr, str) else fr
        vf = z["right_video_frame"]; cw = []
        for f in fr:
            if not f.get("ok") or "t" not in f: continue
            k = np.flatnonzero((vf == f["i"]) & v)
            if len(k) == 0: continue
            Twc = T[k[0]] @ Xi; cw.append(Twc[:3, :3] @ np.array(f["t"]) + Twc[:3, 3])
        c_tcp = (np.linalg.inv(T0) @ np.r_[np.median(cw, 0), 1])[:3] if len(cw) >= 3 else np.full(3, np.nan)
        # cube elevation relative to the approach axis and in gravity terms
        c_w = np.median(cw, 0) if len(cw) >= 3 else None
        drop = float((T0[:3, 3] - c_w) @ up) if c_w is not None else np.nan          # how far above the cube the TCP starts
        rows.append(dict(ep=ep, pitch=pitch, roll=roll, cx=c_tcp[0], cy=c_tcp[1], cz=c_tcp[2], dist=float(np.linalg.norm(c_tcp)), above=drop,
                         g_norm=float(np.linalg.norm(a0))))
    except Exception as e:
        print(ep, "skip:", repr(e)[:120])
json.dump(rows, open(H / "c8/hra_red/start_pose/human_start.json", "w"), indent=1)
A = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "ep"}
q = lambda a: f"p10 {np.nanpercentile(a,10):7.1f}  p50 {np.nanpercentile(a,50):7.1f}  p90 {np.nanpercentile(a,90):7.1f}"
print(f"human episodes {len(rows)} (cube located in {int(np.isfinite(A['cx']).sum())})")
print("approach pitch below horizontal [deg]  ", q(A["pitch"]))
print("closing-axis roll [deg]                ", q(A["roll"]))
for k, nm in (("cx", "cube along approach x [cm]"), ("cy", "cube along closing y [cm]"), ("cz", "cube along z [cm]"), ("dist", "cube distance [cm]"), ("above", "TCP height above cube [cm]")):
    print(f"{nm:38s} ", q(A[k] * 100))
# robot rest
sys.path.insert(0, str(H / "holobrain-mac-model"))
import requests, infer_core_v4 as IC, eef_kin
k_ = object.__new__(IC.V4Inferencer); k_.kin = eef_kin.Kin({"max_joint_delta": None})
o = requests.get("http://localhost:8021/observe", timeout=10).json()
m, _ = k_._tcp_mat(o["joints_rad"]); R = m[1][:3, :3]; up = np.array([0, 0, 1.0])
rp, rr = angles(R, up)
print(f"\nrobot right TCP now (rest?) pos {np.round(m[1][:3,3]*1000,1)} mm  joints {np.round(np.degrees(np.array(o['joints_rad'])[7:13]),1)}")
print(f"robot approach pitch below horizontal {rp:.1f} deg | closing-axis roll {rr:.1f} deg (base z = up)")
print(f"robot approach axis in base {np.round(R[:,0],3)}  closing axis {np.round(R[:,1],3)}")
