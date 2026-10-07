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
A = H / "c8/hra_a100"; RAWS = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"; OUT = A / "raw_v2_CT"; PLOTS = H / "Desktop/hra_a100_C_tableaware"
MARGIN = 0.005; W_P, W_V, W_A = 1.0, 50.0, 2000.0
F = np.array(json.load(open(A / "B_anchor_F.json"))["T_base_F"]); Fi = np.linalg.inv(F); TABLE = json.load(open(A / "G_frame_calib.json"))["table_z_m"]; ZMIN = TABLE + MARGIN
TS = np.array(json.load(open(H / "c8/hra_red/umi_start_pose.json"))["T_base_tcp"]); TIPS = np.load(A / "urdf_gripper/tip_box_tcp.npy")
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


def qp_lift(zt):
    n = len(zt); need = ZMIN - zt
    if need[0] > 0 or need[-1] > 0: return None
    if need.max() <= 0: return np.zeros(n)
    D1 = np.diff(np.eye(n), axis=0); D2 = np.diff(np.eye(n), 2, axis=0)
    Gm = 2 * (W_P * np.eye(n) + W_V * D1.T @ D1 + W_A * D2.T @ D2) + 1e-9 * np.eye(n)
    E = np.zeros((n, 2)); E[0, 0] = 1; E[-1, 1] = 1
    Cm = np.hstack([E, np.eye(n)]); b = np.r_[0, 0, need]
    return quadprog.solve_qp(Gm, np.zeros(n), Cm, b, meq=2)[0]


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
        rep[eid] = dict(r, status="REJECT", reason="old protocol (no origin hold)"); continue
    zu = np.load(base / "raw_episode.npz"); To, why = origin_pose(zu, ev); r["origin"] = why
    if To is None: rep[eid] = dict(r, status="REJECT", reason=why); continue
    rc = A / ("raw_v2_robotcam") / eid
    if not (rc / "raw_episode.npz").exists():                                              # sanity-rejected: robot-TCP labels from the UMI raw on the fly
        M = np.array(json.load(open(A / "raw_v2_robotcam/ROBOTCAM.json"))["M_robot_tcp_to_umi_tcp"]); Tr_all = pose_to_T(zu["right_position"], zu["right_quaternion"]) @ np.linalg.inv(M)
        z = dict(zu)
    else:
        z = dict(np.load(rc / "raw_episode.npz", allow_pickle=True)); Tr_all = pose_to_T(z["right_position"], z["right_quaternion"])
    v = z["right_valid"].astype(bool); idx = np.flatnonzero(v)
    if flag:   # sanity-rejected raw kept all rows valid: trim at the AUTO go here
        go = int([e for e in ev if e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "go"][0]["t_ns"]); v = v & (z["right_t_ns"].astype(np.int64) >= go); idx = np.flatnonzero(v); z["right_valid"] = v
    Tb = F[None] @ np.linalg.inv(To)[None] @ Tr_all
    E = TS @ np.linalg.inv(Tb[idx[0]]); rv = Rot.from_matrix(E[:3, :3]).as_rotvec(); tE = E[:3, 3]
    s = np.clip((np.arange(len(Tb)) - idx[0]) / max(idx[-1] - idx[0], 1), 0, 1); u = np.clip((s - 0.2) / 0.6, 0, 1); w = 1 - (3 * u ** 2 - 2 * u ** 3)
    Tn = Tb.copy()
    for i in idx:
        Ew = np.eye(4); Ew[:3, :3] = Rot.from_rotvec(w[i] * rv).as_matrix(); Ew[:3, 3] = w[i] * tE; Tn[i] = Ew @ Tb[i]
    zt = tipz(Tn[idx]); c = qp_lift(zt)
    r.update(start_shift_mm=round(float(np.linalg.norm(tE)) * 1000, 1), nominal_min_clear_mm=round(float((zt.min() - TABLE) * 1000), 1),
             endpoint_clear_mm=round(float((zt[-1] - TABLE) * 1000), 1))
    if c is None:
        rep[eid] = dict(r, status="REJECT", reason="endpoint (or start) below table+margin: cannot lift with fixed ends"); continue
    Tc = Tn.copy(); Tc[idx, 2, 3] += c; zc = tipz(Tc[idx])
    jump = float(np.linalg.norm(np.diff(Tc[idx, :3, 3], axis=0), axis=1).max() * 1000)
    r.update(corrected_min_clear_mm=round(float((zc.min() - TABLE) * 1000), 2), max_corr_mm=round(float(c.max() * 1000), 1), mean_corr_mm=round(float(c[c > 1e-5].mean() * 1000) if (c > 1e-5).any() else 0.0, 1),
             corrected_frames=int((c > 1e-4).sum()), endpoint_err_mm=round(float(np.linalg.norm(Tc[idx[-1], :3, 3] - Tb[idx[-1], :3, 3]) * 1000), 4),
             start_err_mm=round(float(np.linalg.norm(Tc[idx[0], :3, 3] - TS[:3, 3]) * 1000), 4), max_tcp_jump_mm=round(jump, 1))
    r.update({k: round(v_, 3) for k, v_ in ik_replay(Tc[idx]).items()})
    st, why = "PASS", ""
    if r["ik_ok_ratio"] < 0.98: st, why = "REJECT", f"IK ok ratio {r['ik_ok_ratio']}"
    elif r["max_corr_mm"] > 80: st, why = "REJECT", f"z correction {r['max_corr_mm']} mm > 80"
    elif r["max_corr_mm"] > 40: st, why = "REVIEW", f"z correction {r['max_corr_mm']} mm > 40"
    rep[eid] = dict(r, status=st, reason=why)
    if st in ("PASS", "REVIEW"):
        (OUT / eid).mkdir(); [shutil.copy(p_, OUT / eid / p_.name) for p_ in base.iterdir() if p_.is_file() and p_.name != "raw_episode.npz"]
        TF = Fi[None] @ Tc; z["right_position"] = TF[:, :3, 3]; z["right_quaternion"] = matrix_to_quaternion(TF[:, :3, :3]); np.savez(OUT / eid / "raw_episode.npz", **z)
    if (c is not None and c.max() > 1e-4) or r["nominal_min_clear_mm"] < 5:
        tt = np.arange(len(idx)) / 30.0; fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(tt, Tn[idx, 2, 3] * 1000, "C0--", label="nominal TCP z"); ax.plot(tt, Tc[idx, 2, 3] * 1000, "C0-", label="corrected TCP z")
        ax.plot(tt, zt * 1000, "C3--", label="nominal fingertip (lowest, URDF)"); ax.plot(tt, zc * 1000, "C3-", label="corrected fingertip")
        ax.axhline(TABLE * 1000, color="k", lw=1.5, label=f"table {TABLE*1000:.1f} mm"); ax.axhline(ZMIN * 1000, color="k", ls=":", label="table + 5 mm")
        ax.set_xlabel("time after GO [s]"); ax.set_ylabel("base z [mm]"); ax.set_title(f"{eid}  {st}  max lift {r['max_corr_mm']} mm  nominal min clearance {r['nominal_min_clear_mm']} mm", fontsize=9)
        ax.legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(PLOTS / f"{eid}_{st}.png", dpi=110); plt.close(fig)
    print(eid[-13:], st, why, {k: r.get(k) for k in ("nominal_min_clear_mm", "max_corr_mm", "corrected_min_clear_mm", "ik_ok_ratio")}, flush=True)
json.dump(rep, open(A / "table_aware_C_report.json", "w"), indent=1)
keys = sorted({k for r in rep.values() for k in r})
with open(A / "table_aware_C_report.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=keys); wtr.writeheader(); [wtr.writerow(r) for r in rep.values()]
cnt = {s: sum(r["status"] == s for r in rep.values()) for s in ("PASS", "REVIEW", "REJECT")}
print("TOTAL", len(rep), cnt); print("REJECT reasons:", sorted({r.get("reason") for r in rep.values() if r["status"] == "REJECT"}))
