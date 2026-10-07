"""[2026-10-07 user] HRA_A100: the SLAM trajectory of one episode in the two training state frames, side by side.
  A (start-anchored): pose relative to the robot-TCP pose at the AUTO "go" (= what the A93START model sees as state, origin = go)
  B (origin-anchored): pose in F = the HandUMI fingertip-on-X frame (= the A93ORIGIN state, origin = the X)
Poses = the training labels (raw_v2_robotcam / raw_v2_origin: robot-TCP convention, metric). Grey = origin hold + prep (not trained),
red = approach after go (trained). Left: head C922, right-wrist fisheye, right-wrist robot-C922 view.  usage: viz_ab.py <episode_id> ..."""
import json, pathlib, sys
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20.geometry.transforms import pose_to_T
A = H / "c8/hra_a100"; RAWS = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"; OUT = H / "Desktop/hra_a100_AB_viz"


def frames(p):
    c = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out


def render(eid):
    sess, ep = eid.rsplit("_", 1); epd = RAWS / sess / f"episode_{ep}"
    zr = np.load(A / "raw_v2_robotcam" / eid / "raw_episode.npz"); zo = np.load(A / "raw_v2_origin" / eid / "raw_episode.npz")
    Tr = pose_to_T(zr["right_position"], zr["right_quaternion"]); To = pose_to_T(zo["right_position"], zo["right_quaternion"])
    t = zr["right_t_ns"].astype(np.int64); vf = zr["right_video_frame"].astype(int); trained = zr["right_valid"].astype(bool)
    ev = json.load(open(epd / "events.json")); go = [e for e in ev if e["kind"] == "auto_loop" and e.get("detail", {}).get("cue") == "go"][0]
    ig = int(np.argmin(np.abs(t - int(go["t_ns"]))))
    TA = np.linalg.inv(Tr[ig])[None] @ Tr; PA = TA[:, :3, 3]; PB = To[:, :3, 3]
    meta = json.load(open(A / "raw_v2" / "_scale_qc" / f"{eid}.json")) if (A / "raw_v2/_scale_qc" / f"{eid}.json").exists() else {}
    fb = {json.loads(l)["tag"][5:]: json.loads(l) for l in open(A / "raw_v2/scale_fallback.jsonl")} if (A / "raw_v2/scale_fallback.jsonl").exists() else {}
    tag = f"{sess[9:]}_{ep}_right"; scale_src = "origin plane" if tag in fb else "cube PnP"
    wr = frames(epd / "right_wrist.mp4"); wc = frames(epd / "right_wrist_c922.mp4"); hd = frames(epd / "head.mp4")
    hcam = zr["cam_head_t_ns"].astype(np.int64); hfr = zr["cam_head_frame"].astype(int)
    out = OUT / f"{eid}_A_vs_B.mp4"; W, Hh = 1280, 720; vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, Hh))
    fig = plt.figure(figsize=(8.0, 7.2), dpi=100)
    def lim(P):
        lo, hi = P.min(0), P.max(0); c = (lo + hi) / 2; h = max(float((hi - lo).max()) / 2, 0.05) * 1.1; return c, h
    cA, hA = lim(PA); cB, hB = lim(np.vstack([PB, np.zeros((1, 3))]))
    for i in range(len(t)):
        fig.clf()
        for k, (P, T_, c, h, ttl, zero_lbl) in enumerate(((PA, TA, cA, hA, "A  start-anchored (0,0,0 = pose at GO)", "GO"), (PB, To, cB, hB, "B  origin-anchored (0,0,0 = X)", "X"))):
            ax = fig.add_axes([0.0 + 0.5 * k, 0.30, 0.5, 0.62], projection="3d")
            ax.plot(*P[~trained].T, color="#bbbbbb", lw=1, label="origin hold + prep (not trained)")
            ax.plot(*P[trained].T, color="#e8b4b4", lw=1)
            m = trained[:i + 1]; ax.plot(*P[:i + 1][m].T, color="#d62728", lw=2.2, label="approach after GO (trained)")
            ax.plot(*P[:i + 1][~m].T, color="#555555", lw=1.8)
            for cc, col in zip(T_[i][:3, :3].T, ("r", "g", "b")): ax.plot(*np.c_[P[i], P[i] + cc * h * 0.25], color=col, lw=2)
            ax.scatter(0, 0, 0, color="k", s=30); ax.text(0, 0, 0, f" {zero_lbl}", fontsize=8)
            ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
            ax.set_xlabel("x [m]", fontsize=7); ax.set_ylabel("y [m]", fontsize=7); ax.set_zlabel("z [m]", fontsize=7); ax.view_init(22, -60); ax.set_title(ttl, fontsize=9)
            if k == 0: ax.legend(loc="upper left", fontsize=6)
        fig.text(0.01, 0.965, f"HRA_A100 {eid}   robot-TCP labels, metric scale = {scale_src}   t = {(t[i] - t[0]) / 1e9:.1f} s   "
                              f"{'TRAINED' if trained[i] else 'origin hold / prep'}", fontsize=8)
        b = fig.add_axes([0.10, 0.05, 0.85, 0.20]); tt = (t - t[0]) / 1e9
        b.plot(tt, np.linalg.norm(PA, axis=1), label="|state A| distance from GO pose [m]"); b.plot(tt, np.linalg.norm(PB, axis=1), label="|state B| distance from X [m]")
        b.axvline(tt[ig], color="g", ls="--", lw=1); b.axvline(tt[i], color="k", lw=1); b.legend(fontsize=7, loc="upper left"); b.set_xlabel("time [s]")
        fig.canvas.draw(); plot = cv2.resize(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR), (800, 720))
        j = min(vf[i], len(wr) - 1); left = np.zeros((720, 480, 3), np.uint8)
        hk = hfr[int(np.argmin(np.abs(hcam - t[i])))] if len(hcam) else 0
        left[0:240, 80:400] = cv2.resize(hd[min(hk, len(hd) - 1)], (320, 240)); left[240:510] = cv2.resize(wr[j], (480, 270))
        left[510:720, 100:380] = cv2.resize(wc[min(j, len(wc) - 1)], (280, 210))
        for txt, y in (("head (C922)", 20), ("right wrist (fisheye, raw)", 262), ("robot C922 view", 530)):
            cv2.putText(left, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vw.write(np.hstack([left, plot]))
    vw.release(); plt.close(fig); print(out, "frames", len(t), "scale", scale_src, flush=True)


if __name__ == "__main__":
    for e in sys.argv[1:]: render(e)
