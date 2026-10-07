import sys, json, pathlib, glob, numpy as np
sys.path.insert(0, __import__("os").path.expanduser("~/ego_cart20"))
from ego_cart20 import cube_pnp as CP
H = pathlib.Path.home(); simu = {}
for f in glob.glob(str(H / "c8/ego_cart20_v2_raw/20260928_101010_*/raw_episode.json")):
    m = json.load(open(f)); simu[m["episode_id"]] = m["provenance"]["arms"]["right"]["scale"]
rows = []
for eid in sorted(simu)[:int(sys.argv[1]) if len(sys.argv) > 1 else 24]:
    tag = f"{eid}_right"; ex = H / "c8/robotlike/export" / tag; ss = H / "c8/robotlike/runs" / tag / "ss1.csv"
    if not ss.exists(): continue
    r = CP.episode_pnp_scale(ex / "raw_video.mp4", ex / "orbslam_setting.yaml", ss, max_frac=0.4)
    rows.append(dict(eid=eid, s_imu=simu[eid], s_pnp=r.get("s_pnp"), n=r["n_valid_pnp"], resid=r.get("centre_resid_m"), reproj=r.get("reproj_px_p50"), valid=r["valid"]))
v = [r for r in rows if r["valid"]]; rat = np.array([r["s_pnp"] / r["s_imu"] for r in v])
print("valid", len(v), "/", len(rows), "| ratio p16/p50/p84", np.percentile(rat, [16, 50, 84]).round(3), "| reproj p50", np.median([r["reproj"] for r in v]).round(2), "| resid cm p50", round(np.median([r["resid"] for r in v]) * 100, 2))
json.dump(rows, open(sys.argv[2] if len(sys.argv) > 2 else "/dev/null", "w"), default=float, indent=1)
