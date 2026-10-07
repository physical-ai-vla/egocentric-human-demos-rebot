"""[2026-10-07 user "27개 합하면 중앙값이 달라지지?" / "새 시작점 추가"] human start (robot-TCP label at the AUTO go, measured origin orientation, in base)
over every episode that passes the CT hard gates (origin protocol, still run, origin hold <= 10 mm, yaw measured); median (position median +
rotation chordal mean) and medoid (real episode minimising sum of pos mm + 3 mm/deg rot distance) -> ~/c8/hra_red/start_poses/04_*, 05_*."""
import json, pathlib
import numpy as np
H = pathlib.Path.home(); A = H / "c8/hra_a100"
exec(open(A / "pipe/table_aware_C5.py").read().split("ids = sorted(")[0])
M = np.array(json.load(open(A / "raw_v2_robotcam/ROBOTCAM.json"))["M_robot_tcp_to_umi_tcp"])
ids = sorted(p.name for p in (A / "raw_v2").iterdir() if (p / "raw_episode.npz").exists()) + [f"{p.name}|sanity" for p in (A / "raw_v2/_rejected_sanity").iterdir() if (p / "raw_episode.npz").exists()]
S, why = {}, {}
for key in ids:
    eid, flag = (key.split("|") + [""])[:2]; base = A / "raw_v2" / ("_rejected_sanity" if flag else "") / eid
    sess, ep = eid.rsplit("_", 1); ev = json.load(open(RAWS / sess / f"episode_{ep}" / "events.json"))
    if not any(e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "prep_start" for e in ev): why[eid] = "old protocol"; continue
    zu = np.load(base / "raw_episode.npz"); To, w = origin_pose(zu, ev)
    if To is None: why[eid] = w; continue
    t = zu["right_t_ns"].astype(np.int64); gc = int([e for e in ev if e["kind"] == "go_cue"][0]["t_ns"]); m = (t < gc) & (t > t[0] + 300_000_000)
    if np.median(np.linalg.norm((np.linalg.inv(To)[None] @ pose_to_T(zu["right_position"], zu["right_quaternion"])[m])[:, :3, 3], axis=1)) * 1000 > 10: why[eid] = "origin hold moved"; continue
    o = OR.get(eid)
    if o is None or not o["yaw_measured"]: why[eid] = "yaw not measured"; continue
    rc = A / "raw_v2_robotcam" / eid
    if (rc / "raw_episode.npz").exists():
        z = np.load(rc / "raw_episode.npz", allow_pickle=True); Tr = pose_to_T(z["right_position"], z["right_quaternion"]); v = z["right_valid"].astype(bool); tt = z["right_t_ns"].astype(np.int64)
    else:
        Tr = pose_to_T(zu["right_position"], zu["right_quaternion"]) @ np.linalg.inv(M); v = zu["right_valid"].astype(bool); tt = t
    if flag: v = v & (tt >= int([e for e in ev if e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "go"][0]["t_ns"]))
    TGo = np.eye(4); TGo[:3, :3] = np.array(o["R_G_umitcp"]); S[eid] = (F @ TGo @ np.linalg.inv(To) @ Tr[np.flatnonzero(v)[0]])
from collections import Counter
print("episodes", len(ids), "usable starts", len(S), "excluded", dict(Counter(why.values())))
P = np.array([T[:3, 3] for T in S.values()]); R = Rot.from_matrix(np.array([T[:3, :3] for T in S.values()]))
new = np.array(["20261007" in k for k in S]); print("new-session starts", int(new.sum()))
def med(P, R):
    T = np.eye(4); T[:3, 3] = np.median(P, 0); T[:3, :3] = R.mean().as_matrix(); return T
Tm = med(P, R); Told = med(P[~new], R[~new]); Tn = med(P[new], R[new])
D = np.linalg.norm(P[:, None] - P[None], axis=2) * 1000 + 3 * np.degrees((R[:, None] if False else None) or 0) if False else None
pd = np.linalg.norm(P[:, None] - P[None], axis=2) * 1000
rd = np.array([[np.degrees((R[i].inv() * R[j]).magnitude()) for j in range(len(S))] for i in range(len(S))])
k = int(np.argmin((pd + 3 * rd).sum(1))); Tmed = S[list(S)[k]]
for nm, T in (("old-only (pre-27) median", Told), ("new-27 median", Tn), ("ALL median", Tm), (f"ALL medoid {list(S)[k]}", Tmed)):
    print(f"{nm:45s} {np.round(T[:3,3]*1000,1)}")
spread = np.linalg.norm(P - Tm[:3, 3], axis=1) * 1000; print("start spread to ALL median p50/p90 mm", np.percentile(spread, [50, 90]).round(0))
for nm, T, note in (("04_median_all.json", Tm, f"median of {len(S)} usable HRA_A100 human starts (incl. 1007 session, measured origin orientation)"),
                    ("05_medoid_all.json", Tmed, f"medoid episode {list(S)[k]} of {len(S)} usable human starts")):
    q, _, _ = PK.solve(np.zeros(12), [PK.fk(np.zeros(12))[0], T], iters=600); f = PK.fk(q)[1]
    res = np.linalg.norm(f[:3, 3] - T[:3, 3]) * 1000; mg = np.degrees(np.minimum(q - lo, hi - q))[6:].min(); clr = (tipz(T[None])[0] - TABLE) * 1000
    json.dump(dict(t="2026-10-07", note=note + f"; IK resid {res:.1f} mm, margin {mg:.1f} deg, tip clearance {clr:.0f} mm", T_base_tcp=T.tolist()), open(H / f"c8/hra_red/start_poses/{nm}", "w"), indent=1)
    print(nm, np.round(T[:3, 3] * 1000, 1), f"IK resid {res:.1f} mm margin {mg:.1f} deg tip {clr:.0f} mm")
json.dump({k_: T.tolist() for k_, T in S.items()}, open(A / "human_starts_all.json", "w"))
