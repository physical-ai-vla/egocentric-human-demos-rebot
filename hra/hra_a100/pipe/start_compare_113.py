"""[2026-10-07 user "8번이 가장 마음에 들어" + pasted J(T_s)] compare start poses 02/04/05/08 over ALL usable HRA_A100 episodes, split
old (1006) / new (1007): start correction |dp|, rotation correction, table lift (max), IK (sequential, ok ratio + min joint margin),
robot C922 camera vs human GO camera (median / p90 distance + axis angle). Same CT5 warp + table QP + IK replay as table_aware_C5."""
import json, pathlib
import numpy as np
H = pathlib.Path.home(); A = H / "c8/hra_a100"
exec(open(A / "pipe/table_aware_C5.py").read().split("ids = sorted(")[0])
Xr = np.array(json.load(open(H / "c8/hra_red/handeye/robot_right_wrist_handeye.json"))["X_tcp_cam"]["park"])
M = np.array(json.load(open(A / "raw_v2_robotcam/ROBOTCAM.json"))["M_robot_tcp_to_umi_tcp"]); sst = lambda x: 3 * np.clip(x, 0, 1) ** 2 - 2 * np.clip(x, 0, 1) ** 3
ids = sorted(p.name for p in (A / "raw_v2").iterdir() if (p / "raw_episode.npz").exists()) + [f"{p.name}|sanity" for p in (A / "raw_v2/_rejected_sanity").iterdir() if (p / "raw_episode.npz").exists()]
TR = {}
for key in ids:
    eid, flag = (key.split("|") + [""])[:2]; base = A / "raw_v2" / ("_rejected_sanity" if flag else "") / eid
    sess, ep = eid.rsplit("_", 1); ev = json.load(open(RAWS / sess / f"episode_{ep}" / "events.json"))
    if not any(e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "prep_start" for e in ev): continue
    zu = np.load(base / "raw_episode.npz"); To, _ = origin_pose(zu, ev)
    if To is None: continue
    t = zu["right_t_ns"].astype(np.int64); gc = int([e for e in ev if e["kind"] == "go_cue"][0]["t_ns"]); m = (t < gc) & (t > t[0] + 300_000_000)
    if np.median(np.linalg.norm((np.linalg.inv(To)[None] @ pose_to_T(zu["right_position"], zu["right_quaternion"])[m])[:, :3, 3], axis=1)) * 1000 > 10: continue
    o = OR.get(eid)
    if o is None or not o["yaw_measured"]: continue
    rc = A / "raw_v2_robotcam" / eid
    if (rc / "raw_episode.npz").exists():
        z = np.load(rc / "raw_episode.npz", allow_pickle=True); Tr = pose_to_T(z["right_position"], z["right_quaternion"]); v = z["right_valid"].astype(bool); tt = z["right_t_ns"].astype(np.int64)
    else:
        Tr = pose_to_T(zu["right_position"], zu["right_quaternion"]) @ np.linalg.inv(M); v = zu["right_valid"].astype(bool); tt = t
    if flag: v = v & (tt >= int([e for e in ev if e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "go"][0]["t_ns"]))
    TGo = np.eye(4); TGo[:3, :3] = np.array(o["R_G_umitcp"]); TR[eid] = (F[None] @ TGo[None] @ np.linalg.inv(To)[None] @ Tr)[np.flatnonzero(v)]
print(len(TR), "usable episodes", flush=True)
grp = {"old": [k for k in TR if "20261006" in k], "new": [k for k in TR if "20261007" in k]}
cam = {k: (T[0] @ Xr) for k, T in TR.items()}
SP = H / "c8/hra_red/start_poses"; out = {}
for nm in ("02_CT6_back6_down4", "04_median_all", "05_medoid_all", "08_medoid_back6_right6_down4"):
    TS_ = np.array(json.load(open(SP / f"{nm}.json"))["T_base_tcp"])
    qs, _, _ = PK.solve(np.zeros(12), [PK.fk(np.zeros(12))[0], TS_], iters=600); globals()["q_start"] = qs; globals()["LEFT"] = PK.fk(qs)[0]
    smg = np.degrees(np.minimum(qs - lo, hi - qs))[6:].min(); C = TS_ @ Xr; r = {}
    for g, ks in grp.items():
        dps, drs, lifts, mg, ok, rej, dc, da = [], [], [], [], [], 0, [], []
        for k in ks:
            Tb = TR[k]; dp = TS_[:3, 3] - Tb[0, :3, 3]; rv = Rot.from_matrix(TS_[:3, :3] @ Tb[0, :3, :3].T).as_rotvec(); s = np.linspace(0, 1, len(Tb))
            wxy = 1 - sst((s - 0.2) / 0.6); wz = 1 - sst(s / 0.3); wr = 1 - sst((s - 0.1) / 0.5); Tn = Tb.copy()
            for i in range(len(Tb)):
                Tn[i, :3, 3] = Tb[i, :3, 3] + np.array([wxy[i] * dp[0], wxy[i] * dp[1], wz[i] * dp[2]]); Tn[i, :3, :3] = Rot.from_rotvec(wr[i] * rv).as_matrix() @ Tb[i, :3, :3]
            dps.append(np.linalg.norm(dp) * 1000); drs.append(np.degrees(np.linalg.norm(rv)))
            dc.append(np.linalg.norm(C[:3, 3] - cam[k][:3, 3]) * 1000); da.append(np.degrees(np.arccos(np.clip(C[:3, 2] @ cam[k][:3, 2], -1, 1))))
            c = qp_lift(tipz(Tn))
            if c is None: rej += 1; continue
            Tc = Tn.copy(); Tc[:, 2, 3] += c; lifts.append(c.max() * 1000); rr = ik_replay(Tc); mg.append(rr["min_joint_margin_deg"]); ok.append(rr["ik_ok_ratio"])
        P = lambda x: f"{np.percentile(x,50):.0f}/{np.percentile(x,90):.0f}"
        r[g] = dict(n=len(ks), start_dp_mm=P(dps), start_rot_deg=P(drs), cam_dist_mm=P(dc), cam_axis_deg=P(da), lift_mm=P(lifts), lift_gt30=int((np.array(lifts) > 30).sum()),
                    endpoint_reject=rej, ik_fail_eps=int((np.array(ok) < 0.98).sum()), joint_margin_min_p10=f"{np.min(mg):.1f}/{np.percentile(mg,10):.1f}")
        print(nm, g, r[g], flush=True)
    out[nm] = dict(start_joint_margin_deg=round(float(smg), 1), **r)
json.dump(out, open(A / "start_compare_113.json", "w"), indent=1)
