"""[2026-10-06] HRA_A100 origin repeatability: every new-protocol episode's origin-hold frame (1 s after REC, before go_cue)
located against episode 1's origin frame (start_match v3: table points from IMU gravity, PnP). Scale assumes the camera is
~200 mm above the table at the origin (relative check; SLAM will give the metric value)."""
import json, pathlib, sys, numpy as np, cv2, yaml
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_collector"))
from handumi_collector.robotlike.start_match import StartMatcher
from handumi_collector.pose.episode_io import RawEpisode
C = H / "ego_collector/configs/calibration"; fy = yaml.safe_load(open(C / "fisheye_right_v002.yaml"))
Rci = np.array(yaml.safe_load(open(C / "camera_imu_right_v002.yaml"))["T_camera_imu"], float)[:3, :3]; Rfix = np.load(C / "camera_imu_right_rotfix_20261006.npy")
RAW = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"
eps = []
for s in sorted(RAW.glob("HRA_A100_*")):
    for e in sorted(s.glob("episode_*")):
        ev = json.load(open(e / "events.json")); cues = [x.get("detail", {}).get("cue") for x in ev]
        if "prep_start" in cues: eps.append((e, ev))
def origin(e, ev):
    ep = RawEpisode.load(e); g = next(x for x in ev if x["kind"] == "go_cue"); t_go = int(g["t_ns"])
    t0 = None; img = None
    for _, _, cap_ns, im in ep.iter_frames("right_wrist", downscale=1):
        if t0 is None: t0 = int(cap_ns)
        if int(cap_ns) >= t0 + 1_000_000_000 or int(cap_ns) >= t_go - 300_000_000: img = im; t = int(cap_ns); break
    imu = ep.imu["right"]; hn = np.asarray(imu.host_ns); acc = np.asarray(imu.accel, float)
    a = acc[(hn > t - 300_000_000) & (hn < t + 300_000_000)].mean(0); up = Rfix @ Rci @ a; up /= np.linalg.norm(up)
    return img, up
print("new-protocol episodes:", len(eps))
img0, up0 = origin(*eps[0]); sm = StartMatcher(fy["K"], fy["D"], fy["image_size"], ref_height_mm=200.0); print("ref", eps[0][0].parent.name, eps[0][0].name, sm.set_reference(img0, up0))
rows = []
for e, ev in eps[1:]:
    img, up = origin(e, ev); m = sm.measure(img, up)
    pitch = float(np.degrees(np.arcsin(np.clip(-up[2], -1, 1))))
    rows.append(dict(ep=f"{e.parent.name[-6:]}/{e.name[-3:]}", ok=m["ok"], why=m.get("why"), fwd=m.get("fwd_mm"), left=m.get("left_mm"), up=m.get("up_mm"), dist=m.get("dist_mm"), rot=m.get("rot_deg"), pitch=pitch))
json.dump(rows, open(H / "c8/hra_a100/origin_check.json", "w"), indent=1)
ok = [r for r in rows if r["ok"]]; d = np.array([r["dist"] for r in ok]); rot = np.array([r["rot"] for r in ok]); P = np.array([r["pitch"] for r in rows])
print(f"measured {len(ok)}/{len(rows)}  fails: {[r['why'] for r in rows if not r['ok']][:4]}")
print("origin displacement vs ep1 (mm) p50/p90/max", np.percentile(d, [50, 90, 100]).round(0), " rot deg p50/p90", np.percentile(rot, [50, 90]).round(1))
for k in ("fwd", "left", "up"): print(f"  {k}: p10/50/90", np.percentile([r[k] for r in ok], [10, 50, 90]).round(0))
print("camera pitch at origin (IMU) p10/50/90", np.percentile(P, [10, 50, 90]).round(1))
print("within 15 mm & 3 deg:", int(np.sum((d < 15) & (rot < 3))), "/", len(ok))
