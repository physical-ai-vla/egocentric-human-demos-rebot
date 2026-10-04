"""[2026-10-02] MASt3R-SLAM wrist-pose visualisation of an ego HandUMI episode (ego_cart20_v2 raw export).
Left: head / left-wrist / right-wrist video. Right: 3D trajectory of both HandUMI tool frames (MASt3R-SLAM pose stack
c8_phase3.side, per-episode IMU-VI metric scale) with the current pose + tool axes (x red, y green, z blue), gripper below.
SLAM-lost samples are gaps. usage: viz_mast3r_episode.py <raw episode dir> <out.mp4>"""
import json, sys
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ep, out = sys.argv[1], sys.argv[2]
J = json.load(open(f"{ep}/raw_episode.json")); Z = np.load(f"{ep}/raw_episode.npz")


def quat_to_R(q):  # wxyz (ego_cart20.geometry.rotation6d.matrix_to_quaternion)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# [gravity-aligned] each wrist has its OWN MASt3R-SLAM world (first-keyframe camera frame, not gravity aligned). Rotate it so
# +z = physical up (IMU-VI gravity, /tmp/claude_gravity_check.json), x = horizontal projection of the SLAM x axis; origin = the
# arm's first valid pose. Yaw between the two arms is NOT known (separate SLAM maps), so the overlay is per-arm relative.
UP = json.load(open("/tmp/claude_gravity_check.json"))


def align_R(up):
    z = up / np.linalg.norm(up); x = np.array([1.0, 0, 0]) - z[0] * z; x /= np.linalg.norm(x); y = np.cross(z, x)
    return np.stack([x, y, z])          # rows: new axes in SLAM coords -> p_new = A @ p


arms = {}
for a in ("left", "right"):
    p = Z[f"{a}_position"].copy(); v = Z[f"{a}_valid"]; p[~v] = np.nan
    A = align_R(np.array(UP[f"{J['episode_id']}_{a}"]["up_in_slam"])); p0 = p[np.flatnonzero(v)[0]]
    p = (p - p0) @ A.T
    arms[a] = dict(t=Z[f"{a}_t_ns"], p=p, q=Z[f"{a}_quaternion"], v=v, g=Z[f"{a}_gripper"], A=A)
allp = np.concatenate([arms[a]["p"] for a in arms]); lo, hi = np.nanmin(allp, 0), np.nanmax(allp, 0)
c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.1
cov = J["provenance"]["arms"]
th = Z["cam_head_t_ns"]; t0 = th[0]
keys = ("head", "left_wrist", "right_wrist")
caps = {k: cv2.VideoCapture(J["videos"][k]) for k in keys}
fidx = {"head": Z["cam_head_frame"], "left_wrist": Z["cam_left_wrist_frame"], "right_wrist": Z["cam_right_wrist_frame"]}
tcam = {"head": th, "left_wrist": Z["cam_left_wrist_t_ns"], "right_wrist": Z["cam_right_wrist_t_ns"]}
frames = {k: None for k in keys}; fi = {k: -1 for k in keys}
W, H = 1280, 720
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
fig = plt.figure(figsize=(8.0, 7.2), dpi=100)
col = {"left": "tab:blue", "right": "tab:red"}
for i, tn in enumerate(th):
    tiles = []
    for k in keys:
        j = int(np.clip(np.searchsorted(tcam[k], tn), 0, len(tcam[k]) - 1)); want = int(fidx[k][j])
        while fi[k] < want:
            ok, f = caps[k].read(); fi[k] += 1
            if not ok: break
            frames[k] = f
        f = frames[k] if frames[k] is not None else np.zeros((240, 480, 3), np.uint8)
        f = cv2.resize(f, (480, 240)); cv2.putText(f, k, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(f)
    fig.clf(); ax = fig.add_axes([0.0, 0.25, 1.0, 0.72], projection="3d"); axg = fig.add_axes([0.1, 0.05, 0.85, 0.15])
    for a, d in arms.items():
        k = int(np.clip(np.searchsorted(d["t"], tn), 0, len(d["t"]) - 1)); P = d["p"]
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col[a], alpha=0.15, lw=1)
        ax.plot(P[:k + 1, 0], P[:k + 1, 1], P[:k + 1, 2], color=col[a], lw=2, label=f"{a} wrist (SLAM coverage {cov[a]['coverage']:.2f})")
        if d["v"][k]:
            o = P[k]; R = d["A"] @ quat_to_R(d["q"][k])
            for e, cc in zip(R.T, ("r", "g", "b")):
                ax.plot(*np.stack([o, o + r * 0.18 * e]).T, color=cc, lw=2)
            ax.scatter(*o, color=col[a], s=40)
        else:
            ax.text2D(0.02 if a == "left" else 0.55, 0.02, f"{a}: SLAM lost", transform=ax.transAxes, color=col[a])
        axg.plot((d["t"] - t0) / 1e9, d["g"], color=col[a], lw=1)
    axg.axvline((tn - t0) / 1e9, color="k", lw=1); axg.set_ylim(-0.05, 1.05); axg.set_ylabel("gripper\n1=open"); axg.set_xlabel("time [s]")
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("UP (gravity) [m]"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"MASt3R-SLAM wrist poses, gravity-aligned (z = up, per-arm origin = start)  {J['episode_id']}  order {J['stack_order']}  t={(tn - t0) / 1e9:5.1f}s", fontsize=9)
    ax.view_init(elev=25, azim=-60 + 0.15 * i)
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1]
    vw.write(np.hstack([np.vstack(tiles), cv2.resize(img, (W - 480, H))]))
vw.release(); print("wrote", out, len(th), "frames")
