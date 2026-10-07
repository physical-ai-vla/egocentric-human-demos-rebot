"""[2026-10-07 user "실측 방향으로"] per-episode MEASURED orientation of the HandUMI at the origin hold, in G (X origin, x = tape, z = up):
  tilt (pitch / roll): the right-IMU gravity during the hold (Kalibr camera<-IMU + the 18 deg fix), per episode
  yaw: relative camera rotation to a reference origin frame from start_match (table-plane PnP), chained; the only assumption left is
       that the MEAN yaw over all measured episodes = the tape direction. Episodes whose PnP fails keep the mean yaw (flagged).
Output origin_orient.json: R_G_umitcp (3x3) per episode + pitch / yaw / flags."""
import json, pathlib, sys
import numpy as np, cv2, yaml
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_collector"))
from handumi_collector.robotlike.start_match import StartMatcher
from handumi_collector.pose.episode_io import RawEpisode
A = H / "c8/hra_a100"; C = H / "ego_collector/configs/calibration"; RAWS = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"
fy = yaml.safe_load(open(C / "fisheye_right_v002.yaml")); Rci = np.array(yaml.safe_load(open(C / "camera_imu_right_v002.yaml"))["T_camera_imu"], float)[:3, :3]
Rfix = np.load(C / "camera_imu_right_rotfix_20261006.npy"); Xu = np.load(A.parent / "hra_red/base_frame/X_cam_tcp_full.npy"); R_cam_tcp = Xu[:3, :3]
eps = []
for s in sorted(RAWS.glob("HRA_A100_*")):
    for e in sorted(s.glob("episode_*")):
        if not (e / "events.json").exists(): continue
        ev = json.load(open(e / "events.json"))
        if any(x.get("detail", {}).get("cue") == "prep_start" for x in ev): eps.append((e, ev))
def origin_obs(e, ev):
    ep = RawEpisode.load(e); g = next(x for x in ev if x["kind"] == "go_cue"); t_go = int(g["t_ns"]); t0 = None
    for _, _, cap_ns, im in ep.iter_frames("right_wrist", downscale=1):
        if t0 is None: t0 = int(cap_ns)
        if int(cap_ns) >= t0 + 1_000_000_000 or int(cap_ns) >= t_go - 300_000_000: img, t = im, int(cap_ns); break
    imu = ep.imu["right"]; hn = np.asarray(imu.host_ns); acc = np.asarray(imu.accel, float)
    a = acc[(hn > t - 300_000_000) & (hn < t + 300_000_000)].mean(0); up = Rfix @ Rci @ a; return img, up / np.linalg.norm(up)
obs = {f"{e.parent.name}_{e.name.split('_')[1]}": origin_obs(e, ev) for e, ev in eps}
ids = list(obs); ref = ids[len(ids) // 2]
sm = StartMatcher(fy["K"], fy["D"], fy["image_size"], ref_height_mm=200.0); sm.set_reference(*obs[ref])
def cam_R(up_c, heading):   # camera->G rotation from the camera-frame up vector and the horizontal heading of the optical axis
    az = float(up_c[2]); a = np.r_[np.sqrt(max(1 - az * az, 0)) * np.cos(heading), np.sqrt(max(1 - az * az, 0)) * np.sin(heading), az]
    return Rot.align_vectors([[0, 0, 1.0], a], [up_c, [0, 0, 1.0]], weights=[1e3, 1.0])[0].as_matrix()
R_ref = cam_R(obs[ref][1], 0.0)                     # provisional: reference optical axis heading 0 (along G x)
out = {}; heads = []
for k in ids:
    up = obs[k][1]
    if k == ref: Rk, m = R_ref, dict(ok=True)
    else:
        m = sm.measure(obs[k][0], obs[k][1])
        Rk = R_ref @ cv2.Rodrigues(np.array(m["rvec"]))[0].T if m.get("ok") else None      # X_cur = R X_ref  ->  R_G_cur = R_G_ref R^T
    if Rk is not None:
        a = Rk[:, 2]; h = float(np.arctan2(a[1], a[0])); heads.append(h); out[k] = dict(heading=h, up=up.tolist(), yaw_measured=True, inliers=m.get("inliers"))
    else: out[k] = dict(heading=None, up=up.tolist(), yaw_measured=False, why=m.get("why"))
# the one assumption: the mean optical-axis heading over the measured episodes = the tape direction (G x)
mean_h = float(np.arctan2(np.mean(np.sin(heads)), np.mean(np.cos(heads))))
for k, o in out.items():
    h = (o["heading"] - mean_h) if o["yaw_measured"] else 0.0
    Rc = cam_R(np.array(o["up"]), h); Rt = Rc @ R_cam_tcp; a = Rt[:, 0]
    o.update(R_G_umitcp=Rt.tolist(), yaw_deg=round(float(np.degrees(h)), 2), tcp_pitch_deg=round(float(np.degrees(np.arcsin(-a[2]))), 2),
             tcp_roll_deg=round(float(np.degrees(np.arcsin(np.clip(Rt[2, 1], -1, 1)))), 2))
json.dump(dict(reference=ref, mean_heading_rad=mean_h, episodes=out), open(A / "origin_orient.json", "w"), indent=1)
y = np.array([o["yaw_deg"] for o in out.values() if o["yaw_measured"]]); p = np.array([o["tcp_pitch_deg"] for o in out.values()]); r = np.array([o["tcp_roll_deg"] for o in out.values()])
print(f"episodes {len(out)} | yaw measured {int(sum(o['yaw_measured'] for o in out.values()))} | yaw deg p10/50/90 {np.percentile(y, [10, 50, 90]).round(1)} (std {y.std():.1f})")
print(f"TCP pitch deg p10/50/90 {np.percentile(p, [10, 50, 90]).round(1)} | roll deg p10/50/90 {np.percentile(r, [10, 50, 90]).round(1)} | assumed before: pitch 63.7, yaw 0, roll 0")
