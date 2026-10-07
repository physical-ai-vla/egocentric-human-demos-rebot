"""Offline inspection of an RGB-D rigid-body tracking run (§27): mesh overlay video + trajectory plots + the QA report.

    python -m handumi_collector.tools.inspect_depth_pose --episode <episode> --side left [--backend icp]

Writes into <episode>/derived/depth_pose_<backend>/:
    <side>_overlay.mp4      RGB with the tracked CAD projected on it, the body axes, and the per-frame validity/confidence
    <side>_trajectory.png   xyz, orientation, confidence/depth-residual and the validity strip over time
    <side>_gt_error.png     only with ground truth (synthetic pilots): absolute pose error against it

The overlay is the check that the QA numbers cannot make for you: an auto-detected "static" window and a low residual
both look healthy when a tracker is frozen on the wrong object. Look at the video before trusting a verdict."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..pose.depth_config import load_depth_pose_cfg
from ..pose.depth_run import derived_depth_dir, load_run, poses_from_table
from ..pose.rgbd_io import RgbdEpisode, project
from ..pose.se3 import poses7_to_T
from ..pose.tracking_mesh import load_tracking_mesh

AXES = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
AXIS_BGR = ((0, 0, 255), (0, 255, 0), (255, 0, 0))          # x red, y green, z blue


def draw_overlay(img, T, K, model_pts, *, valid: bool, text: str):
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    if valid and T is not None:
        P = (T[:3, :3] @ model_pts.T + T[:3, 3:4]).T
        uv = project(P, K)
        ok = np.isfinite(uv).all(axis=1)
        u = np.round(uv[ok, 0]).astype(int); v = np.round(uv[ok, 1]).astype(int)
        m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        layer = out.copy()
        layer[v[m], u[m]] = (0, 255, 255)
        out = cv2.addWeighted(layer, 0.55, out, 0.45, 0)
        A = (T[:3, :3] @ AXES.T + T[:3, 3:4]).T
        a = project(A, K)
        if np.isfinite(a).all():
            o = tuple(np.round(a[0]).astype(int))
            for i, c in enumerate(AXIS_BGR):
                cv2.arrowedLine(out, o, tuple(np.round(a[i + 1]).astype(int)), c, 2, tipLength=0.25)
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if valid else (0, 0, 255), 1, cv2.LINE_AA)
    return out


def plot_trajectory(out_png: Path, t_s, Ts, valid, df, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rv = np.full((len(Ts), 3), np.nan)
    rv[valid] = Rotation.from_matrix(Ts[valid][:, :3, :3]).as_rotvec(degrees=True)
    p = np.full((len(Ts), 3), np.nan)
    p[valid] = Ts[valid][:, :3, 3] * 1e3
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True, height_ratios=[3, 3, 2, 1])
    for i, (lab, c) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
        ax[0].plot(t_s, p[:, i], c, lw=1.1, label=lab)
        ax[1].plot(t_s, rv[:, i], c, lw=1.1, label=f"r{lab}")
    ax[0].set_ylabel("position [mm]"); ax[0].legend(ncol=3, fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_ylabel("rotation vector [deg]"); ax[1].legend(ncol=3, fontsize=8); ax[1].grid(alpha=.3)
    if "confidence" in df:
        ax[2].plot(t_s, df["confidence"].to_numpy(float), "tab:purple", lw=1.0, label="confidence")
    if "depth_residual_mm" in df:
        ax2 = ax[2].twinx()
        ax2.plot(t_s, df["depth_residual_mm"].to_numpy(float), "tab:orange", lw=1.0, label="depth residual [mm]")
        ax2.set_ylabel("residual [mm]", color="tab:orange")
    ax[2].set_ylabel("confidence"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8, loc="lower left")
    ax[3].imshow(valid.reshape(1, -1), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                 extent=[t_s[0], t_s[-1], 0, 1])
    ax[3].set_yticks([]); ax[3].set_xlabel("t [s]"); ax[3].set_ylabel("valid", rotation=0, labelpad=22)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def gt_error(Ts, valid, G) -> dict:
    E = np.einsum("nij,njk->nik", np.linalg.inv(Ts[valid]), G[valid])
    dt = np.linalg.norm(E[:, :3, 3], axis=1) * 1e3
    dr = np.degrees(Rotation.from_matrix(E[:, :3, :3]).magnitude())
    return dict(n=int(valid.sum()), translation_mm=dt, rotation_deg=dr,
                translation_mm_median=float(np.median(dt)), translation_mm_p95=float(np.percentile(dt, 95)),
                translation_mm_max=float(dt.max()), rotation_deg_median=float(np.median(dr)),
                rotation_deg_p95=float(np.percentile(dr, 95)), rotation_deg_max=float(dr.max()))


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--backend")
    ap.add_argument("--mesh")
    ap.add_argument("--stream")
    ap.add_argument("--config")
    ap.add_argument("--ground-truth", help="ground_truth.json (default: the episode's, when present)")
    ap.add_argument("--overlay-points", type=int, default=4000)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=float, default=15.0, help="overlay video playback fps")
    a = ap.parse_args(argv)

    cfg = load_depth_pose_cfg(a.config)
    backend = a.backend or cfg.backend
    ep_path = Path(a.episode)
    df, prov = load_run(ep_path, a.side, backend)
    t_ns, Ts, valid = poses_from_table(df)
    out = derived_depth_dir(ep_path, backend)
    t_s = (t_ns - t_ns[0]) / 1e9

    rp = out / f"{a.side}_report.txt"
    if rp.exists():
        print(rp.read_text())

    plot_trajectory(out / f"{a.side}_trajectory.png", t_s, Ts, valid, df,
                    f"{ep_path.name} [{a.side}] {backend}")
    print(f"wrote {out/f'{a.side}_trajectory.png'}")

    gt_path = a.ground_truth or (ep_path / "ground_truth.json" if (ep_path / "ground_truth.json").exists() else None)
    if gt_path:
        gt = json.loads(Path(gt_path).read_text())
        G = poses7_to_T(np.array(gt["poses7_T_depthcam_body"]))[: len(Ts)]
        e = gt_error(Ts, valid, G)
        print(f"\nGround truth ({Path(gt_path).name}), {e['n']} valid frames")
        print(f"  absolute translation error  median {e['translation_mm_median']:.2f}  p95 {e['translation_mm_p95']:.2f}  max {e['translation_mm_max']:.2f}  mm")
        print(f"  absolute rotation error     median {e['rotation_deg_median']:.3f}  p95 {e['rotation_deg_p95']:.3f}  max {e['rotation_deg_max']:.3f}  deg")
        for s in gt.get("segments", []):
            sel = valid.copy(); sel[: s["i0"]] = False; sel[s["i1"] + 1:] = False
            if sel.sum() < 2:
                print(f"    {s['name']:20s} no valid frames")
                continue
            se = gt_error(Ts, sel, G)
            print(f"    {s['name']:20s} n={se['n']:3d}  trans p95 {se['translation_mm_p95']:6.2f} mm   rot p95 {se['rotation_deg_p95']:5.2f} deg")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        dt = np.full(len(Ts), np.nan); drot = np.full(len(Ts), np.nan)
        dt[valid] = e["translation_mm"]; drot[valid] = e["rotation_deg"]
        ax[0].plot(t_s, dt, "tab:red", lw=1.1); ax[0].set_ylabel("|Δt| [mm]"); ax[0].grid(alpha=.3)
        ax[1].plot(t_s, drot, "tab:blue", lw=1.1); ax[1].set_ylabel("|Δr| [deg]"); ax[1].set_xlabel("t [s]"); ax[1].grid(alpha=.3)
        for s in gt.get("segments", []):
            for x in (t_s[s["i0"]],):
                ax[0].axvline(x, color="k", lw=.5, alpha=.4); ax[1].axvline(x, color="k", lw=.5, alpha=.4)
            ax[0].text(t_s[s["i0"]], ax[0].get_ylim()[1], s["name"], fontsize=6, rotation=90, va="top")
        fig.suptitle(f"{ep_path.name} [{a.side}] absolute error vs ground truth")
        fig.tight_layout(); fig.savefig(out / f"{a.side}_gt_error.png", dpi=120); plt.close(fig)
        print(f"wrote {out/f'{a.side}_gt_error.png'}")

    if a.no_video:
        return 0
    mesh_src = a.mesh or (prov.get("mesh") or {}).get("path") or cfg.mesh_path(a.side)
    if not mesh_src or not Path(mesh_src).exists():
        print(f"\n(no mesh available at {mesh_src!r} — skipping the overlay video)")
        return 0
    tm = load_tracking_mesh(mesh_src, side=a.side)
    model_pts, _n = tm.sample_points(a.overlay_points, seed=1)
    ep = RgbdEpisode.load(ep_path, stream=a.stream or cfg.stream, fps_fallback=cfg.fps)
    fr = prov.get("frames", {})
    vid = out / f"{a.side}_overlay.mp4"
    writer = None
    by_index = {int(r): i for i, r in enumerate(df["frame_index"].to_numpy())}
    for f in ep.iter_frames(int(fr.get("start", 0)), fr.get("stop"), int(fr.get("step", 1))):
        i = by_index.get(f.index)
        if i is None:
            continue
        conf = df["confidence"].to_numpy(float)[i]
        res = df["depth_residual_mm"].to_numpy(float)[i]
        state = str(df["tracking_state"].iloc[i])
        txt = (f"{a.side} f{f.index:05d} {state:11s} conf={conf:.3f} residual={res:.2f}mm"
               if np.isfinite(conf) else f"{a.side} f{f.index:05d} {state}")
        img = draw_overlay(f.rgb, Ts[i] if valid[i] else None, ep.intrinsics.K, model_pts, valid=bool(valid[i]), text=txt)
        if writer is None:
            writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()
        print(f"wrote {vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
