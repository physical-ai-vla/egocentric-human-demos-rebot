"""[2026-10-07 user paste "시작 자세를 뒤로 2/4/6 cm"] offline comparison of robot start candidates = current start (umi_start_pose.json) moved
BACK horizontally (away from the mean human endpoint = retreat along the approach), orientation kept. Per candidate the exact CT5 pipeline
(table_aware_C5.py: measured origin orientation, per-axis decay, table-aware QP, Pink IK replay) over the CT5-86 episodes, plus the robot
wrist-camera pose at the start vs the human camera at GO (robot-TCP label @ hand-eye X_tcp_cam, base frame). No files written except the log."""
import json, pathlib
import numpy as np
H = pathlib.Path.home(); A = H / "c8/hra_a100"
exec(open(A / "pipe/table_aware_C5.py").read().split("ids = sorted(")[0])
Xr = np.array(json.load(open(H / "c8/hra_red/handeye/robot_right_wrist_handeye.json"))["X_tcp_cam"]["park"])
rep5 = json.load(open(A / "table_aware_C5_report.json")); IDS = [k for k, v in rep5.items() if v["status"] in ("PASS", "PASS-CORRECTED")]
sst = lambda x: 3 * np.clip(x, 0, 1) ** 2 - 2 * np.clip(x, 0, 1) ** 3; M = np.array(json.load(open(A / "raw_v2_robotcam/ROBOTCAM.json"))["M_robot_tcp_to_umi_tcp"])
trajs = {}
for eid in IDS:
    sess, ep = eid.rsplit("_", 1); ev = json.load(open(RAWS / sess / f"episode_{ep}" / "events.json"))
    flag = not (A / "raw_v2" / eid / "raw_episode.npz").exists(); base = A / "raw_v2" / ("_rejected_sanity" if flag else "") / eid
    zu = np.load(base / "raw_episode.npz"); To, _ = origin_pose(zu, ev); rc = A / "raw_v2_robotcam" / eid
    if (rc / "raw_episode.npz").exists(): z = np.load(rc / "raw_episode.npz", allow_pickle=True); Tr = pose_to_T(z["right_position"], z["right_quaternion"]); v = z["right_valid"].astype(bool); t = z["right_t_ns"].astype(np.int64)
    else: Tr = pose_to_T(zu["right_position"], zu["right_quaternion"]) @ np.linalg.inv(M); v = zu["right_valid"].astype(bool); t = zu["right_t_ns"].astype(np.int64)
    if flag: v = v & (t >= int([e for e in ev if e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "go"][0]["t_ns"]))
    TGo = np.eye(4); TGo[:3, :3] = np.array(OR[eid]["R_G_umitcp"]); trajs[eid] = (F[None] @ TGo[None] @ np.linalg.inv(To)[None] @ Tr)[np.flatnonzero(v)]
print(len(trajs), "episodes (CT5-86)")
TS0 = TS.copy(); endm = np.median([T[-1, :3, 3] for T in trajs.values()], 0); bdir = TS0[:3, 3] - endm; bdir[2] = 0; bdir /= np.linalg.norm(bdir)
gx = F[:3, 0]; print(f"back direction (base xy) {np.round(bdir[:2],3)}  = {np.degrees(np.arctan2(bdir[1],bdir[0])):.1f} deg; angle to G -x: {np.degrees(np.arccos(np.clip(-gx[:2]@bdir[:2]/np.linalg.norm(gx[:2]),-1,1))):.1f} deg")
hc = np.array([(T[0] @ Xr)[:3, 3] for T in trajs.values()]); ha = np.array([(T[0] @ Xr)[:3, 2] for T in trajs.values()])
hcm = np.median(hc, 0); ham = ha.mean(0); ham /= np.linalg.norm(ham); hs = np.linalg.norm(hc - hcm, axis=1) * 1000
hang = np.degrees(np.arccos(np.clip(ha @ ham, -1, 1)))
print(f"human camera at GO: median {np.round(hcm*1000)} mm (z above table {(hcm[2]-TABLE)*1000:.0f}); spread to median p50/p90 {np.percentile(hs,50):.0f}/{np.percentile(hs,90):.0f} mm, axis p50/p90 {np.percentile(hang,50):.1f}/{np.percentile(hang,90):.1f} deg")
for back in (0.0, 0.02, 0.04, 0.06):
    TSc = TS0.copy(); TSc[:3, 3] += back * bdir
    qs, _, _ = PK.solve(np.zeros(12), [PK.fk(np.zeros(12))[0], TSc], iters=600); q_start = qs; LEFT = PK.fk(qs)[0]
    fe = PK.fk(qs)[1]; sres = np.linalg.norm(fe[:3, 3] - TSc[:3, 3]) * 1000; smg = np.degrees(np.minimum(qs - lo, hi - qs))[6:]
    L, rej, mg, ok, nstart = [], 0, [], [], []
    for eid, Tb in trajs.items():
        dp = TSc[:3, 3] - Tb[0, :3, 3]; rv = Rot.from_matrix(TSc[:3, :3] @ Tb[0, :3, :3].T).as_rotvec(); s = np.linspace(0, 1, len(Tb))
        wxy = 1 - sst((s - 0.2) / 0.6); wz = 1 - sst(s / 0.3); wr = 1 - sst((s - 0.1) / 0.5); Tn = Tb.copy()
        for i in range(len(Tb)):
            Tn[i, :3, 3] = Tb[i, :3, 3] + np.array([wxy[i] * dp[0], wxy[i] * dp[1], wz[i] * dp[2]]); Tn[i, :3, :3] = Rot.from_rotvec(wr[i] * rv).as_matrix() @ Tb[i, :3, :3]
        nstart.append(np.linalg.norm(dp) * 1000); c = qp_lift(tipz(Tn))
        if c is None: rej += 1; continue
        Tc = Tn.copy(); Tc[:, 2, 3] += c; L.append(c.max() * 1000); r = ik_replay(Tc); mg.append(r["min_joint_margin_deg"]); ok.append(r["ik_ok_ratio"])
    L, mg, ok = np.array(L), np.array(mg), np.array(ok)
    cam = TSc @ Xr; dc = np.linalg.norm(cam[:3, 3] - hcm) * 1000; da = np.degrees(np.arccos(np.clip(cam[:3, 2] @ ham, -1, 1)))
    pct = (hs < dc).mean() * 100
    print(f"\n== back {back*100:.0f} cm  start TCP {np.round(TSc[:3,3]*1000)} mm  IK start resid {sres:.1f} mm, start joint margin min {smg.min():.1f} deg (joint {int(np.argmin(smg))})\n"
          f"   camera: {np.round(cam[:3,3]*1000)} mm, {(cam[2,3]-TABLE)*1000:.0f} above table; to human-GO median {dc:.0f} mm (human p{pct:.0f}), axis {da:.1f} deg\n"
          f"   start offset human->robot p50/p90 {np.percentile(nstart,50):.0f}/{np.percentile(nstart,90):.0f} mm\n"
          f"   IK: eps with ok<0.98 {int((ok<0.98).sum())}, ok min {ok.min():.2f}; joint margin min p10/min {np.percentile(mg,10):.1f}/{mg.min():.1f} deg\n"
          f"   table: lifted {int((L>0.5).sum())}/{len(L)}, max lift p90/max {np.percentile(L,90):.1f}/{L.max():.1f} mm, 20-30 {int(((L>20)&(L<=30)).sum())}, >30 {int((L>30).sum())}, endpoint reject {rej}", flush=True)
