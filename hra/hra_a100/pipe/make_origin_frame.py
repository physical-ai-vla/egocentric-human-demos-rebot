"""[2026-10-07 user option 3] ORIGIN-anchored variant: every robot-TCP label pose re-expressed in the frame F = the HandUMI fingertip
pose during the ORIGIN HOLD (fingertip on the X = G (0,0,0); orientation = the HandUMI at the origin, gravity ~ consistent, yaw +-6 deg
across takes). Pose(t) := inv(T_umi_origin) @ T_robot(t). With convert_origin.py (anchor = identity) the RELCART20 state is then the
absolute pose in F, i.e. "where the gripper is relative to the X". usage: make_origin_frame.py <raw_v2 (UMI)> <raw_v2_robotcam> <out>"""
import json, pathlib, shutil, sys, numpy as np
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20.geometry.transforms import pose_to_T
from ego_cart20.geometry.rotation6d import matrix_to_quaternion
from scipy.spatial.transform import Rotation as Rot
U, Rr, O = map(pathlib.Path, sys.argv[1:4]); SRC = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"
assert not O.exists(); shutil.copytree(Rr, O, ignore=shutil.ignore_patterns("raw_episode.npz")); rep = {}
for d in sorted(p for p in Rr.iterdir() if (p / "raw_episode.npz").exists()):
    eid = d.name; sess, ep = eid.rsplit("_", 1)
    ev = json.load(open(SRC / sess / f"episode_{ep}" / "events.json")); gc = [e for e in ev if e["kind"] == "go_cue"]
    zu = np.load(U / eid / "raw_episode.npz"); t = zu["right_t_ns"].astype(np.int64); Tu = pose_to_T(zu["right_position"], zu["right_quaternion"])
    t0 = int(t[0]); hold = (t >= t0 + 300_000_000) & (t <= int(gc[0]["t_ns"]) - 200_000_000) & np.isfinite(zu["right_position"]).all(1)
    if hold.sum() < 5: rep[eid] = dict(skipped="no origin hold rows"); shutil.rmtree(O / eid); continue
    P = Tu[hold, :3, 3]; Rm = Rot.from_matrix(Tu[hold, :3, :3]).mean().as_matrix()
    To = np.eye(4); To[:3, :3] = Rm; To[:3, 3] = np.median(P, 0); spread = float(np.linalg.norm(P - To[:3, 3], axis=1).max())
    z = dict(np.load(d / "raw_episode.npz", allow_pickle=True)); T = np.linalg.inv(To)[None] @ pose_to_T(z["right_position"], z["right_quaternion"])
    z["right_position"] = T[:, :3, 3]; z["right_quaternion"] = matrix_to_quaternion(T[:, :3, :3]); np.savez(O / eid / "raw_episode.npz", **z)
    v = z["right_valid"].astype(bool); p0 = T[v][0, :3, 3]
    rep[eid] = dict(hold_rows=int(hold.sum()), hold_spread_mm=round(spread * 1000, 2), start_rel_origin_mm=np.round(p0 * 1000, 1).tolist())
json.dump(rep, open(O / "ORIGIN_FRAME.json", "w"), indent=1)
S = np.array([r["start_rel_origin_mm"] for r in rep.values() if "start_rel_origin_mm" in r]); sp = [r["hold_spread_mm"] for r in rep.values() if "hold_spread_mm" in r]
print("episodes", len(S), "skipped", sum("skipped" in r for r in rep.values()), "| origin hold spread mm p50/max", np.median(sp).round(1), max(sp).round(1))
print("start (go) gripper pose rel. origin, mm  p10/50/90:", np.percentile(S, [10, 50, 90], axis=0).round(0).tolist())
