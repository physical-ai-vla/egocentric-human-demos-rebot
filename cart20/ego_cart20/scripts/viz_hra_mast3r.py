#!/usr/bin/env python3
"""HRA_red right-wrist MASt3R-SLAM trajectory video, in the style of ~/Desktop/mast3r_slam_viz/*_gravity_up.mp4.

Left: head camera + right wrist fisheye (left wrist: none on this rig).  Right: the right wrist path in METRES -- the SLAM
trajectory scaled by the cube-PnP metric scale (the IMU-VI scale is shown for comparison only) -- gravity-aligned (z = up,
from the accelerometer during the start still window, camera<-IMU from the export's IMU.T_b_c1), origin = start, with a
camera triad at the current frame.  Bottom: wrist-to-cube distance from PnP (metric, independent of SLAM) and wrist speed.

usage: viz_hra_mast3r.py <episode number e.g. 000150> [...] [--out ~/Desktop/mast3r_slam_viz] [--runs DIR]
"""
import argparse, json, pathlib, re, sys
import numpy as np

H = pathlib.Path.home(); sys.path[:0] = [str(H / "ego_cart20"), str(H / "ego_collector")]
SESS = H / "ego_collector/datasets/human_handumi_raw/HRA_red/HRA_red_20261003_152502"
EXP = H / "c8/robotlike/export"


def quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def align_up(u):
    """rotation taking unit vector u to +z"""
    z = np.array([0, 0, 1.0]); v = np.cross(u, z); c = float(u @ z)
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def render(ep, runs, out_dir):
    import cv2, pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ego_cart20 import cube_pnp as CP
    from handumi_collector.pose.episode_io import RawEpisode
    tag = f"red_20261003_152502_{ep}_right"; ex = EXP / tag; epd = SESS / f"episode_{ep}"
    ss = pd.read_csv(runs / tag / "ss1.csv"); ok = ss.is_lost.astype(str).str.lower() == "false"
    sc = json.load(open(H / "c8/hra_red/raw/_scale_qc" / f"HRA_red_20261003_152502_{ep}.json")) if (H / "c8/hra_red/raw/_scale_qc" / f"HRA_red_20261003_152502_{ep}.json").exists() else None
    if sc is None:
        sc = CP.episode_pnp_scale(ex / "raw_video.mp4", ex / "orbslam_setting.yaml", runs / tag / "ss1.csv")
    s_pnp = sc.get("s_pnp"); s_imu = sc.get("s_imu")
    if not s_pnp: raise SystemExit(f"{ep}: no valid cube-PnP scale ({sc.get('reason')})")
    # gravity: accelerometer mean over the first 1.5 s (operator holds still 3 s), body -> camera(frame 0) = SLAM world
    t = (ex / "orbslam_setting.yaml").read_text()
    T_b_c = np.array([float(x) for x in re.search(r"IMU.T_b_c1:.*?data: \[([^\]]+)\]", t, re.S).group(1).split(",")]).reshape(4, 4)
    acc = json.load(open(ex / "imu_data.json"))["1"]["streams"]["ACCL"]["samples"]
    a0 = np.mean([s["value"] for s in acc if s["cts"] < 1500], axis=0); up = T_b_c[:3, :3].T @ a0; up /= np.linalg.norm(up)
    G = align_up(up)
    P = ss[["x", "y", "z"]].values * s_pnp; P = (P - P[0]) @ G.T
    R = np.array([G @ quat_R(q) for q in ss[["q_x", "q_y", "q_z", "q_w"]].values])
    n = len(ss); fps = 30.0; tt = ss.frame_idx.values / fps
    v = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1) * fps]; v[~ok.values] = np.nan
    # cube distance per frame (PnP, metric) + head frames aligned by capture time
    K, D, size = CP.read_setting(ex / "orbslam_setting.yaml"); Pk, und = CP.undistorter(K, D, size)
    re_ = RawEpisode.load(epd); nv = len(re_.frames["right_wrist"]); off = nv - n
    cap_r = re_.frames["right_wrist"].capture_ns; hf = re_.frames["head"]
    head_idx = [int(hf.video_frame[np.argmin(np.abs(hf.capture_ns - cap_r[min(off + i, nv - 1)]))]) for i in range(n)]
    head_frames = {}
    for vf, fi, c, img in re_.iter_frames("head"):
        head_frames[vf] = img
    cap = cv2.VideoCapture(str(ex / "raw_video.mp4")); wrist = []; dist = np.full(n, np.nan)
    for i in range(n):
        okf, img = cap.read()
        if not okf: break
        wrist.append(img)
        m = CP.cube_blob(und(img))
        if m is not None:
            hx, sol = CP.hexagon(m)
            if hx is not None and sol >= CP.MIN_SOLIDITY:
                r = CP.pnp_cube(hx, Pk)
                if r and r[2] <= CP.MAX_REPROJ_PX: dist[i] = float(np.linalg.norm(r[0]))
    cap.release()
    out = out_dir / f"hra_{ep}_right_gravity_up.mp4"; W, Hh = 1280, 720
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
    fig = plt.figure(figsize=(8.0, 7.2), dpi=100)
    lo, hi = np.nanmin(P[ok.values], 0), np.nanmax(P[ok.values], 0); ctr = (lo + hi) / 2; half = max(float(np.max(hi - lo)) / 2, 0.05) * 1.1
    imu_txt = f" (IMU-VI {s_imu:.3f}, x{s_pnp / s_imu:.1f})" if s_imu else " (IMU-VI: see scale QC)"
    title = (f"HRA_red right wrist, MASt3R-SLAM, gravity-aligned (z = up), origin = start   ep {ep}\n"
             f"metric scale = cube PnP {s_pnp:.3f}{imu_txt}   scale QC: {'PASS' if sc.get('valid') else 'FAIL - ' + str(sc.get('reason'))}")
    for i in range(min(n, len(wrist))):
        fig.clf(); ax = fig.add_axes([0.02, 0.30, 0.96, 0.62], projection="3d")
        ax.plot(*P[ok.values].T, color="#e8b4b4", lw=1)
        ax.plot(*P[:i + 1][ok.values[:i + 1]].T, color="#d62728", lw=2.2, label=f"right wrist (SLAM coverage {ok.mean():.2f})")
        if ok.values[i]:
            for c, col in zip(R[i].T, ("r", "g", "b")):
                ax.plot(*np.c_[P[i], P[i] + c * half * 0.25], color=col, lw=2)
        ax.scatter(*P[0], color="k", s=20)
        ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half); ax.set_zlim(ctr[2] - half, ctr[2] + half)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("UP (gravity) [m]"); ax.view_init(22, -60); ax.legend(loc="upper left", fontsize=8)
        fig.text(0.01, 0.955, title + f"   t={tt[i]:.1f}s", fontsize=8)
        b = fig.add_axes([0.10, 0.05, 0.85, 0.20])
        b.plot(tt, dist, color="#d62728", label="wrist->cube centre (PnP) [m]"); b.plot(tt, v, color="#1f77b4", alpha=0.7, label="wrist speed [m/s]")
        b.axvline(tt[i], color="k", lw=1); b.set_xlabel("time [s]"); b.legend(fontsize=7, loc="upper right"); b.set_ylim(0, max(0.8, np.nanmax(dist) * 1.1 if np.isfinite(dist).any() else 0.8))
        fig.canvas.draw(); plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
        plot = cv2.resize(plot, (800, 720))
        # aspect preserved: head C922 640x480 (4:3, pinhole, NOT undistorted) -> 320x240; wrist fisheye 960x540 (16:9) -> 480x270
        hv = cv2.resize(head_frames.get(head_idx[i], np.zeros((480, 640, 3), np.uint8)), (320, 240), interpolation=cv2.INTER_AREA)
        wv = cv2.resize(wrist[i], (480, 270), interpolation=cv2.INTER_AREA); left = np.zeros((720, 480, 3), np.uint8)
        left[0:240, 80:400] = hv; left[240:510] = wv
        for txt, y in (("head (C922, raw)", 20), ("right_wrist (fisheye, raw)", 262), ("left_wrist: none (right-only rig)", 540)):
            cv2.putText(left, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if np.isfinite(dist[i]): cv2.putText(left, f"cube PnP {dist[i]:.3f} m", (8, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
        vw.write(np.hstack([left, plot]))
    vw.release(); plt.close(fig)
    print(f"{out}  frames {min(n, len(wrist))}  s_pnp {s_pnp:.3f}  s_imu {s_imu}  net {np.linalg.norm(P[ok.values][-1]):.3f} m  up_cam {np.round(up, 3)}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("episodes", nargs="+")
    ap.add_argument("--out", type=pathlib.Path, default=H / "Desktop/mast3r_slam_viz"); ap.add_argument("--runs", type=pathlib.Path, default=H / "c8/robotlike/runs_viz")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for e in a.episodes: render(e, a.runs, a.out)
