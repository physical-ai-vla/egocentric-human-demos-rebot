"""[2026-10-07 user "변환한것도 시각화"] C_TABLEAWARE in the ROBOT BASE frame, per episode video: grey = human (B, unwarped, via the X / F
calibration), blue dashed = nominal C (start -> robot start pose, endpoint kept), red = final C_TABLEAWARE (table-aware z lift) with the
TCP triad; table plane + robot start; bottom: fingertip (URDF lowest corner) height vs table for human / nominal / final.
Left: head C922, right-wrist fisheye, right-wrist robot-C922 view (the training image). usage: viz_ct.py <eid> ..."""
import json, pathlib, sys
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = pathlib.Path.home(); A = H / "c8/hra_a100"
src = open(A / "pipe/table_aware_C.py").read(); exec(src.split("ids = sorted(")[0])          # F, TS, TIPS, TABLE, ZMIN, origin_pose, qp_lift, tipz, ...
OUTV = H / "Desktop/hra_a100_CT_viz"; REP = json.load(open(A / "table_aware_C_report.json"))


def frames(p):
    c = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out


def render(eid):
    sess, ep = eid.rsplit("_", 1); epd = RAWS / sess / f"episode_{ep}"; ev = json.load(open(epd / "events.json"))
    zu = np.load(A / "raw_v2" / eid / "raw_episode.npz"); To, _ = origin_pose(zu, ev)
    zr = np.load(A / "raw_v2_robotcam" / eid / "raw_episode.npz"); Tr = pose_to_T(zr["right_position"], zr["right_quaternion"])
    v = zr["right_valid"].astype(bool); idx = np.flatnonzero(v); Tb = F[None] @ np.linalg.inv(To)[None] @ Tr
    zc_ = np.load(A / "raw_v2_CT" / eid / "raw_episode.npz"); Tc = F[None] @ pose_to_T(zc_["right_position"], zc_["right_quaternion"])
    E = TS @ np.linalg.inv(Tb[idx[0]]); rv = Rot.from_matrix(E[:3, :3]).as_rotvec(); tE = E[:3, 3]
    s = np.clip((np.arange(len(Tb)) - idx[0]) / max(idx[-1] - idx[0], 1), 0, 1); u = np.clip((s - 0.2) / 0.6, 0, 1); w = 1 - (3 * u ** 2 - 2 * u ** 3); Tn = Tb.copy()
    for i in idx:
        Ew = np.eye(4); Ew[:3, :3] = Rot.from_rotvec(w[i] * rv).as_matrix(); Ew[:3, 3] = w[i] * tE; Tn[i] = Ew @ Tb[i]
    Pb, Pn, Pc = (X[idx, :3, 3] * 1000 for X in (Tb, Tn, Tc)); zb, zn, zc = ((tipz(X[idx]) - TABLE) * 1000 for X in (Tb, Tn, Tc))
    vf = zr["right_video_frame"].astype(int)[idx]; t = zr["right_t_ns"].astype(np.int64)[idx]; hcam = zr["cam_head_t_ns"].astype(np.int64); hfr = zr["cam_head_frame"].astype(int)
    wr = frames(epd / "right_wrist.mp4"); wc = frames(epd / "right_wrist_c922.mp4"); hd = frames(epd / "head.mp4")
    r = REP[eid]; title = f"HRA_A100 {eid}  C_TABLEAWARE {r['status']}  max lift {r.get('max_corr_mm')} mm  endpoint lift {r.get('endpoint_z_correction_mm')} mm"
    allp = np.vstack([Pb, Pn, Pc, TS[None, :3, 3] * 1000]); lo, hi = allp.min(0), allp.max(0); ctr = (lo + hi) / 2; half = max((hi - lo).max() / 2, 60) * 1.1
    out = OUTV / f"{eid}_CT.mp4"; vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720)); fig = plt.figure(figsize=(8.0, 7.2), dpi=100)
    xx, yy = np.meshgrid([ctr[0] - half, ctr[0] + half], [ctr[1] - half, ctr[1] + half]); tt = (t - t[0]) / 1e9
    for i in range(len(idx)):
        fig.clf(); ax = fig.add_axes([0.02, 0.30, 0.96, 0.63], projection="3d")
        ax.plot_surface(xx, yy, np.full_like(xx, TABLE * 1000), color="#cccccc", alpha=0.25)
        ax.plot(*Pb.T, color="#999999", lw=1.2, label="human (B, unwarped)"); ax.plot(*Pn.T, "--", color="#1f77b4", lw=1, label="nominal C (start warp)")
        ax.plot(*Pc.T, color="#f2b6b6", lw=1); ax.plot(*Pc[:i + 1].T, color="#d62728", lw=2.2, label="C_TABLEAWARE (trained)")
        for cc, col in zip(Tc[idx[i]][:3, :3].T, ("r", "g", "b")): ax.plot(*np.c_[Pc[i], Pc[i] + cc * half * 0.25], color=col, lw=2)
        ax.scatter(*(TS[:3, 3] * 1000), color="k", s=25); ax.text(*(TS[:3, 3] * 1000), " robot start", fontsize=7)
        ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half); ax.set_zlim(max(ctr[2] - half, TABLE * 1000 - 20), ctr[2] + half)
        ax.set_xlabel("base x [mm]", fontsize=7); ax.set_ylabel("base y [mm]", fontsize=7); ax.set_zlabel("base z [mm]", fontsize=7); ax.view_init(20, -55); ax.legend(loc="upper left", fontsize=7)
        fig.text(0.01, 0.965, title + f"   t={tt[i]:.1f}s", fontsize=8)
        b = fig.add_axes([0.10, 0.05, 0.85, 0.20]); b.plot(tt, zb, color="#999999", label="human fingertip"); b.plot(tt, zn, "--", color="#1f77b4", label="nominal C fingertip")
        b.plot(tt, zc, color="#d62728", label="C_TABLEAWARE fingertip"); b.axhline(0, color="k", lw=1.2); b.axhline(5, color="k", ls=":", lw=1); b.axhline(12, color="g", ls=":", lw=1)
        b.axvline(tt[i], color="k", lw=0.8); b.set_ylabel("above table [mm]", fontsize=7); b.set_xlabel("time after GO [s]", fontsize=7); b.legend(fontsize=6, loc="upper right")
        b.set_ylim(min(-40, float(np.min([zb.min(), zn.min()]))) - 5, float(max(zb.max(), zc.max())) + 10)
        fig.canvas.draw(); plot = cv2.resize(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR), (800, 720))
        j = min(vf[i], len(wr) - 1); left = np.zeros((720, 480, 3), np.uint8); hk = hfr[int(np.argmin(np.abs(hcam - t[i])))]
        left[0:240, 80:400] = cv2.resize(hd[min(hk, len(hd) - 1)], (320, 240)); left[240:510] = cv2.resize(wr[j], (480, 270)); left[510:720, 100:380] = cv2.resize(wc[min(j, len(wc) - 1)], (280, 210))
        for txt, y in (("head (C922)", 20), ("right wrist (fisheye, raw)", 262), ("robot C922 view = training image", 530)):
            cv2.putText(left, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        vw.write(np.hstack([left, plot]))
    vw.release(); plt.close(fig); print(out, len(idx), "frames", flush=True)


if __name__ == "__main__":
    for e in sys.argv[1:]: render(e)
