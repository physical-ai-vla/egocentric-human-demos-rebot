# [2026-10-07 user "B ... 결과가 좋으면 CT5를 새 start pose로 다시 생성"] CT6 = CT5 with the start pose start_pose_CT6.json (6 cm back, 4 cm down).
# [2026-10-07 user "실측 방향으로 다시"] CT5 = CT4 with the MEASURED per-episode origin orientation (origin_orient.json) and the state in G. CT4 = table_aware_C.py (CT81, v3b) with per-axis start-offset decay (see [CT4] block). Outputs raw_v2_CT4, *_C4_report.
"""[2026-10-07 user "107개로 한번 다 확인 / gripper 높이는 urdf"] C_TABLEAWARE over ALL HRA_A100 exports.
Per episode (raw_v2 robot-TCP labels, metric):
 1 origin = the LONGEST still run (< 1.5 mm/row for >= 12 rows) inside the origin hold (before go_cue) of the HandUMI fingertip (raw_v2 UMI TCP)
 2 absolute robot-TCP trajectory in the base: T_b = F @ inv(T_umi_origin) @ T_robot      (F = B_anchor_F.json, X from URDF fingertip contact)
 3 nominal C warp of the trained segment (after the AUTO go): start -> robot start pose T_S, weight 1 (s<0.2) -> smooth -> 0 (s>=0.8), endpoint kept
 4 physical clearance = lowest corner of the URDF fingertip box (TCP x -70/-80 mm, z +-19.6 mm) above the table (G_frame_calib table_z)
 5 table-aware z correction c_z (base z only, x/y/rot untouched): QP min sum w_p c^2 + w_v dc^2 + w_a ddc^2, c[0] = c[-1] = 0,
   tip_z + c >= table + MARGIN; infeasible (endpoint itself below z_min) -> REJECT
 6 sequential Pink IK of the corrected trajectory from the robot start (prev-q seed): residuals, joint margin, joint step, TCP jump
 7 class: PASS / REVIEW (max c > 40 mm) / REJECT (max c > 80 mm, infeasible, IK fail, no origin, no prep (old protocol))
Writes raw_v2_CT/<eid>/raw_episode.npz (corrected, in F; PASS+REVIEW), table_aware_C_report.json/.csv, before/after plots on the Desktop."""
import csv, json, pathlib, shutil, sys
import numpy as np, quadprog
from scipy.spatial.transform import Rotation as Rot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = pathlib.Path.home(); sys.path[:0] = [str(H / "holobrain-mac-model"), str(H / "ego_cart20")]
import eef_kin, pink_ik
from ego_cart20.geometry.transforms import pose_to_T
from ego_cart20.geometry.rotation6d import matrix_to_quaternion
A = H / "c8/hra_a100"; RAWS = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"; OUT = A / "raw_v2_CT6"; PLOTS = H / "Desktop/hra_a100_CT6_tableaware"
MARGIN = 0.005; PREF = 0.012; W_P, W_V, W_A, W_S = 1.0, 50.0, 2000.0, 60.0   # [v3] hard floor +5 mm, soft preferred clearance +12 mm
F = np.array(json.load(open(A / "G_frame_calib.json"))["T_base_G"]); Fi = np.linalg.inv(F);   # [CT5] common frame = G (X, tape x, up z) for every episode and the robot
OR = json.load(open(A / "origin_orient.json"))["episodes"]; TABLE = json.load(open(A / "G_frame_calib.json"))["table_z_m"]; ZMIN = TABLE + MARGIN
TS = np.array(json.load(open(A / "start_pose_CT6.json"))["T_base_tcp"]); TIPS = np.load(A / "urdf_gripper/tip_box_tcp.npy")
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
PK = pink_ik.PinkIK(names); lo, hi = PK.model.lowerPositionLimit, PK.model.upperPositionLimit
q_start, _, _ = PK.solve(np.zeros(12), [PK.fk(np.zeros(12))[0], TS], iters=600); LEFT = PK.fk(q_start)[0]
tipz = lambda T: (T @ TIPS.T)[..., 2, :].min(-1)                     # (n,4,4) -> lowest fingertip-box corner z


def origin_pose(zu, ev):
    t = zu["right_t_ns"].astype(np.int64); Tu = pose_to_T(zu["right_position"], zu["right_quaternion"]); gc = [e for e in ev if e["kind"] == "go_cue"]
    m = (t < int(gc[0]["t_ns"])) & np.isfinite(zu["right_position"]).all(1); idx = np.flatnonzero(m)
    if len(idx) < 12: return None, "origin hold too short"
    sp = np.r_[np.inf, np.linalg.norm(np.diff(Tu[idx, :3, 3], axis=0), axis=1)]; best, cur = (0, 0), [idx[0], 0]
    for k in range(len(idx)):
        if sp[k] < 0.0015 and k > 0: cur[1] += 1
        else: cur = [k, 1]
        if cur[1] > best[1]: best = tuple(cur)
    if best[1] < 12: return None, f"no still run >= 12 rows (longest {best[1]})"
    w = idx[best[0]:best[0] + best[1]]; To = np.eye(4); To[:3, :3] = Rot.from_matrix(Tu[w, :3, :3]).mean().as_matrix(); To[:3, 3] = np.median(Tu[w, :3, 3], 0)
    return To, f"still run {best[1]} rows, spread {np.linalg.norm(Tu[w, :3, 3] - To[:3, 3], axis=1).max()*1000:.1f} mm"


END_MAX = 0.030   # [v2] endpoint z relaxation: lift the endpoint by exactly what it needs, up to 30 mm (calibration / scale / tilt error)


def qp_lift(zt):
    n = len(zt); need = ZMIN - zt
    if need[0] > 0 or need[-1] > END_MAX: return None
    if need.max() <= -PREF + MARGIN: return np.zeros(n)          # [v3] already >= preferred clearance everywhere
    if False: pass
    pref_end = float(need[-1]) - MARGIN + PREF                       # [v3b] endpoint to the PREFERRED clearance if that fits the 30 mm budget,
    c_end = max(0.0, pref_end) if pref_end <= END_MAX else max(0.0, float(need[-1]))   # else to the hard floor (no end kink)
    D1 = np.diff(np.eye(n), axis=0); D2 = np.diff(np.eye(n), 2, axis=0)
    # [v3] soft target: frames whose fingertip is below table + PREF are pulled toward table + PREF (with the smoothness terms this makes a
    # shallow arc instead of riding the +5 mm hard floor); the hard floor stays a constraint
    g = (ZMIN - MARGIN + PREF) - zt; msk = (g > 0).astype(float); g = np.where(g > 0, g, 0.0)
    Gm = 2 * (W_P * np.eye(n) + W_V * D1.T @ D1 + W_A * D2.T @ D2 + W_S * np.diag(msk)) + 1e-9 * np.eye(n)
    E = np.zeros((n, 2)); E[0, 0] = 1; E[-1, 1] = 1
    Cm = np.hstack([E, np.eye(n)]); b = np.r_[0, c_end, need]
    return quadprog.solve_qp(Gm, 2 * W_S * msk * g, Cm, b, meq=2)[0]


def ik_replay(Tc):
    q = q_start.copy(); pe, re, mg, js = [], [], [], []; qp = q.copy()
    for Tt in Tc[::2]:
        q, _, _ = PK.solve(q, [LEFT, Tt], iters=60); f = PK.fk(q)[1]
        pe.append(np.linalg.norm(f[:3, 3] - Tt[:3, 3]) * 1000); re.append(np.degrees(np.linalg.norm(Rot.from_matrix(f[:3, :3].T @ Tt[:3, :3]).as_rotvec())))
        mg.append(np.degrees(np.minimum(q - lo, hi - q))[6:].min()); js.append(np.degrees(np.abs(q - qp))[6:].max()); qp = q.copy()
    pe, re = np.array(pe), np.array(re)
    return dict(ik_ok_ratio=float(np.mean((pe < 5) & (re < 3) & (np.array(mg) > 2))), ik_pos_p95_mm=float(np.percentile(pe, 95)), ik_rot_p95_deg=float(np.percentile(re, 95)),
                min_joint_margin_deg=float(min(mg)), max_joint_step_deg=float(max(js[1:]) if len(js) > 1 else 0))


ids = sorted(p.name for p in (A / "raw_v2").iterdir() if (p / "raw_episode.npz").exists()) + [f"{p.name}|sanity" for p in (A / "raw_v2/_rejected_sanity").iterdir() if (p / "raw_episode.npz").exists()]
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(); PLOTS.mkdir(parents=True, exist_ok=True); rep = {}
for key in ids:
    eid, flag = (key.split("|") + [""])[:2]; base = A / "raw_v2" / ("_rejected_sanity" if flag else "") / eid
    sess, ep = eid.rsplit("_", 1); ev = json.load(open(RAWS / sess / f"episode_{ep}" / "events.json"))
    r = dict(episode=eid, sanity_rejected=bool(flag))
    if not any(e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "prep_start" for e in ev):
        rep[eid] = dict(r, status="HARD REJECT", reason="old protocol (no origin hold)"); continue
    zu = np.load(base / "raw_episode.npz"); To, why = origin_pose(zu, ev); r["origin"] = why
    if To is None: rep[eid] = dict(r, status="HARD REJECT", reason=why); continue
    # [CT5b] the origin hold must actually sit at the still-run origin: median distance of all hold frames (0.3 s .. go_cue) <= 10 mm
    _t = zu["right_t_ns"].astype(np.int64); _gc = int([e for e in ev if e["kind"] == "go_cue"][0]["t_ns"]); _m = (_t < _gc) & (_t > _t[0] + 300_000_000)
    _hd = float(np.median(np.linalg.norm((np.linalg.inv(To)[None] @ pose_to_T(zu["right_position"], zu["right_quaternion"])[_m])[:, :3, 3], axis=1)) * 1000); r["origin_hold_median_mm"] = round(_hd, 2)
    if _hd > 10: rep[eid] = dict(r, status="HARD REJECT", reason=f"origin hold moved (median {_hd:.0f} mm from the still-run origin)"); continue
    rc = A / ("raw_v2_robotcam") / eid
    if not (rc / "raw_episode.npz").exists():                                              # sanity-rejected: robot-TCP labels from the UMI raw on the fly
        M = np.array(json.load(open(A / "raw_v2_robotcam/ROBOTCAM.json"))["M_robot_tcp_to_umi_tcp"]); Tr_all = pose_to_T(zu["right_position"], zu["right_quaternion"]) @ np.linalg.inv(M)
        z = dict(zu)
    else:
        z = dict(np.load(rc / "raw_episode.npz", allow_pickle=True)); Tr_all = pose_to_T(z["right_position"], z["right_quaternion"])
    v = z["right_valid"].astype(bool); idx = np.flatnonzero(v)
    if flag:   # sanity-rejected raw kept all rows valid: trim at the AUTO go here
        go = int([e for e in ev if e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "go"][0]["t_ns"]); v = v & (z["right_t_ns"].astype(np.int64) >= go); idx = np.flatnonzero(v); z["right_valid"] = v
    # [CT5] the MEASURED origin orientation of this episode in G (IMU tilt + image-matched yaw); fingertip at the X (= G origin)
    o = OR.get(eid)
    if o is None or not o["yaw_measured"]:
        rep[eid] = dict(r, status="HARD REJECT", reason="origin yaw not measured (image match failed)"); continue
    TGo = np.eye(4); TGo[:3, :3] = np.array(o["R_G_umitcp"]); r.update(origin_yaw_deg=o["yaw_deg"], origin_pitch_deg=o["tcp_pitch_deg"], origin_roll_deg=o["tcp_roll_deg"])
    Tb = F[None] @ TGo[None] @ np.linalg.inv(To)[None] @ Tr_all
    # [CT4] start alignment separated from shape preservation: position offset dp and orientation offset (rv, applied in place, NOT about the
    # base origin) decay per axis: x/y 1 until s=0.2 then smooth to 0 at 0.8; z smooth 1 -> 0 over s in [0, 0.3] (fast return to the human
    # height); rotation 1 until 0.1 then smooth to 0 at 0.6.  Start = T_S exactly, endpoint = human exactly.
    sst = lambda x: 3 * np.clip(x, 0, 1) ** 2 - 2 * np.clip(x, 0, 1) ** 3
    dp = TS[:3, 3] - Tb[idx[0], :3, 3]; rv = Rot.from_matrix(TS[:3, :3] @ Tb[idx[0], :3, :3].T).as_rotvec(); tE = dp
    s = np.clip((np.arange(len(Tb)) - idx[0]) / max(idx[-1] - idx[0], 1), 0, 1)
    wxy = 1 - sst((s - 0.2) / 0.6); wz = 1 - sst(s / 0.3); wr = 1 - sst((s - 0.1) / 0.5)
    Tn = Tb.copy()
    for i in idx:
        Tn[i, :3, 3] = Tb[i, :3, 3] + np.array([wxy[i] * dp[0], wxy[i] * dp[1], wz[i] * dp[2]])
        Tn[i, :3, :3] = Rot.from_rotvec(wr[i] * rv).as_matrix() @ Tb[i, :3, :3]
    r["start_z_diff_mm"] = round(float(dp[2]) * 1000, 1); adz = abs(r["start_z_diff_mm"])
    r["start_z_group"] = ">=90" if adz >= 90 else ">=70" if adz >= 70 else ">=50" if adz >= 50 else "<50"
    zt = tipz(Tn[idx]); c = qp_lift(zt)
    r.update(start_shift_mm=round(float(np.linalg.norm(tE)) * 1000, 1), nominal_min_clear_mm=round(float((zt.min() - TABLE) * 1000), 1),
             endpoint_clear_mm=round(float((zt[-1] - TABLE) * 1000), 1))
    if c is None:
        rep[eid] = dict(r, status="REJECT", reason=f"endpoint needs > {END_MAX*1000:.0f} mm lift (fingertip {r['endpoint_clear_mm']} mm vs table)"); continue
    Tc = Tn.copy(); Tc[idx, 2, 3] += c; zc = tipz(Tc[idx])
    jump = float(np.linalg.norm(np.diff(Tc[idx, :3, 3], axis=0), axis=1).max() * 1000)
    r.update(corrected_min_clear_mm=round(float((zc.min() - TABLE) * 1000), 2), max_corr_mm=round(float(c.max() * 1000), 1), mean_corr_mm=round(float(c[c > 1e-5].mean() * 1000) if (c > 1e-5).any() else 0.0, 1),
             corrected_frames=int((c > 1e-4).sum()), endpoint_err_mm=round(float(np.linalg.norm(Tc[idx[-1], :3, 3] - Tb[idx[-1], :3, 3]) * 1000), 4),
             start_err_mm=round(float(np.linalg.norm(Tc[idx[0], :3, 3] - TS[:3, 3]) * 1000), 4), max_tcp_jump_mm=round(jump, 1))
    r.update({k: round(v_, 3) for k, v_ in ik_replay(Tc[idx]).items()})
    r["endpoint_z_correction_mm"] = round(float(c[-1] * 1000), 2)
    r["floor_plateau_frames"] = int((zc - ZMIN < 0.001).sum())                    # [v3] frames riding the hard floor (within 1 mm)
    mc = r["max_corr_mm"]
    if r["ik_ok_ratio"] < 0.98: st, why = "HARD REJECT", f"IK ok ratio {r['ik_ok_ratio']}"
    elif mc > 80: st, why = "HARD REJECT", f"z correction {mc} mm > 80"
    elif mc > 40: st, why = "EXCLUDED", f"z correction {mc} mm > 40 (not in main C)"
    elif mc > 30: st, why = "REVIEW", f"z correction {mc} mm in 30-40"
    elif mc > 20: st, why = "PASS-CORRECTED", f"z correction {mc} mm in 20-30"
    else: st, why = "PASS", ""
    rep[eid] = dict(r, status=st, reason=why)
    if st in ("PASS", "PASS-CORRECTED"):
        (OUT / eid).mkdir(); [shutil.copy(p_, OUT / eid / p_.name) for p_ in base.iterdir() if p_.is_file() and p_.name != "raw_episode.npz"]
        json.dump(dict(endpoint_z_correction_mm=r["endpoint_z_correction_mm"], max_z_correction_mm=r["max_corr_mm"], status=st, start_z_diff_mm=r["start_z_diff_mm"], start_z_group=r["start_z_group"], version="CT6", origin_yaw_deg=r.get("origin_yaw_deg"), origin_pitch_deg=r.get("origin_pitch_deg"), origin_roll_deg=r.get("origin_roll_deg"), state_frame="G"), open(OUT / eid / "C_TABLEAWARE.json", "w"))
        TF = Fi[None] @ Tc; z["right_position"] = TF[:, :3, 3]; z["right_quaternion"] = matrix_to_quaternion(TF[:, :3, :3]); np.savez(OUT / eid / "raw_episode.npz", **z)
    if (c is not None and c.max() > 1e-4) or r["nominal_min_clear_mm"] < 5:
        tt = np.arange(len(idx)) / 30.0; fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(tt, Tn[idx, 2, 3] * 1000, "C0--", label="nominal TCP z"); ax.plot(tt, Tc[idx, 2, 3] * 1000, "C0-", label="corrected TCP z")
        ax.plot(tt, zt * 1000, "C3--", label="nominal fingertip (lowest, URDF)"); ax.plot(tt, zc * 1000, "C3-", label="corrected fingertip")
        ax.axhline(TABLE * 1000, color="k", lw=1.5, label=f"table {TABLE*1000:.1f} mm"); ax.axhline(ZMIN * 1000, color="k", ls=":", label="table + 5 mm (hard)"); ax.axhline((ZMIN - MARGIN + PREF) * 1000, color="g", ls=":", label="table + 12 mm (preferred)")
        ax.set_xlabel("time after GO [s]"); ax.set_ylabel("base z [mm]"); ax.set_title(f"{eid}  {st}  max lift {r['max_corr_mm']} mm  nominal min clearance {r['nominal_min_clear_mm']} mm", fontsize=9)
        ax.legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(PLOTS / f"{eid}_{st}.png", dpi=110); plt.close(fig)
    print(eid[-13:], st, why, {k: r.get(k) for k in ("nominal_min_clear_mm", "max_corr_mm", "corrected_min_clear_mm", "ik_ok_ratio")}, flush=True)
json.dump(rep, open(A / "table_aware_C6_report.json", "w"), indent=1)
keys = sorted({k for r in rep.values() for k in r})
with open(A / "table_aware_C6_report.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=keys); wtr.writeheader(); [wtr.writerow(r) for r in rep.values()]
cnt = {s: sum(r["status"] == s for r in rep.values()) for s in ("PASS", "PASS-CORRECTED", "REVIEW", "EXCLUDED", "REJECT", "HARD REJECT")}
print("TOTAL", len(rep), cnt); print("REJECT reasons:", sorted({r.get("reason")[:60] for r in rep.values() if "REJECT" in r["status"]}))
