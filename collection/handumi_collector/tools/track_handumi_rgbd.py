"""Offline RGB-D rigid-body tracking pilot for one HandUMI (Stage A / M2A).

    python -m handumi_collector.tools.track_handumi_rgbd --episode <episode> --side left \
        --mesh assets/handumi/left_tracking_body.obj --pick

Writes <episode>/derived/depth_pose_<backend>/ and prints the §37 pilot report. Raw is never modified.
`--export-packet DIR` instead writes a portable flat RGB-D directory (+mesh +init mask) for a CUDA-only backend on a
GPU host; run this same tool there with `--episode DIR --backend foundationpose` and rsync `derived/` back."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from ..pose.depth_config import load_depth_pose_cfg
from ..pose.depth_hand_tracker import available_backends, backend_info
from ..pose.depth_run import export_packet, run_side
from ..pose.rgbd_io import RgbdEpisode


def parse_bbox(s: str) -> tuple[int, int, int, int]:
    parts = [int(round(float(x))) for x in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox expects X,Y,W,H")
    return tuple(parts)


def load_mask(path: Path, shape) -> np.ndarray:
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"cannot read mask {path}")
    if m.shape[:2] != tuple(shape):
        raise SystemExit(f"mask {path} is {m.shape[:2]}, depth is {tuple(shape)}")
    return m > 0


def pick_bbox(rgb, depth_m) -> tuple[int, int, int, int]:
    """Manual first-frame ROI (§9): drag a box round the HandUMI rigid body, ENTER to accept."""
    import cv2
    d = depth_m.copy()
    d[~np.isfinite(d)] = 0
    vis = np.hstack([rgb, cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255.0 / max(d.max(), 1e-6)), cv2.COLORMAP_TURBO)])
    win = "select the HandUMI RIGID BODY (not the fingers) - ENTER to accept, c to cancel"
    box = cv2.selectROI(win, vis, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0:
        raise SystemExit("no ROI selected")
    if x >= rgb.shape[1]:                                  # picked on the depth half -> map back to image coords
        x -= rgb.shape[1]
    return x, y, w, h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="raw episode directory, or a flat color/+depth/ packet")
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--mesh", help="tracking mesh (.obj/.stl/.ply) or an assembly .yaml; default from depth_pose.yaml")
    ap.add_argument("--backend", help=f"default from depth_pose.yaml; known: {available_backends()}")
    ap.add_argument("--stream", help="depth stream name (recorder episodes with several *_depth/)")
    ap.add_argument("--config", help="path to depth_pose.yaml")
    ap.add_argument("--bbox", type=parse_bbox, help="initial ROI X,Y,W,H on the first frame")
    ap.add_argument("--mask", type=Path, help="initial mask PNG (non-zero = object) on the first frame")
    ap.add_argument("--pick", action="store_true", help="select the initial ROI interactively")
    ap.add_argument("--frames", default="0::1", help="START:STOP:STEP slice of the episode (default 0::1)")
    ap.add_argument("--export-packet", type=Path, help="write a portable flat RGB-D packet here instead of tracking")
    ap.add_argument("--no-write", action="store_true", help="do not write derived/ (report only)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_depth_pose_cfg(a.config)
    backend = a.backend or cfg.backend
    sl = (a.frames.split(":") + ["", "", ""])[:3]
    start = int(sl[0] or 0)
    stop = int(sl[1]) if sl[1] else None
    step = int(sl[2] or 1)

    ep = RgbdEpisode.load(a.episode, stream=a.stream or cfg.stream, fps_fallback=cfg.fps)
    if not a.quiet:
        print(f"episode  {ep.path}  layout={ep.layout} stream={ep.stream} frames={ep.n_frames} fps={ep.fps:.2f}")
        print(f"depth    unit={ep.intrinsics.depth_unit_m*1e3:.3f} mm/count  aligned_to_rgb={ep.intrinsics.aligned_to_rgb}  "
              f"K=[fx {ep.intrinsics.fx:.1f} fy {ep.intrinsics.fy:.1f} cx {ep.intrinsics.cx:.1f} cy {ep.intrinsics.cy:.1f}]")

    bbox, mask = a.bbox, None
    first = next(ep.iter_frames(start, start + 1, 1), None)
    if first is None:
        raise SystemExit(f"{a.episode}: no frame at index {start}")
    if a.mask:
        mask = load_mask(a.mask, first.depth_m.shape)
    elif a.pick:
        bbox = pick_bbox(first.rgb, first.depth_m)
        print(f"picked bbox {bbox}")
    elif bbox is None and (ep.path / "init_mask.png").exists():
        mask = load_mask(ep.path / "init_mask.png", first.depth_m.shape)
        if not a.quiet:
            print(f"using packet init mask {ep.path/'init_mask.png'}")
    elif bbox is None and ep.meta.get("init_bbox"):
        bbox = tuple(int(v) for v in ep.meta["init_bbox"])
        if not a.quiet:
            print(f"using packet init bbox {bbox}")

    mesh = a.mesh or (ep.path / ep.meta["mesh"] if ep.meta.get("mesh") and (ep.path / ep.meta["mesh"]).exists() else None)

    if a.export_packet:
        out = export_packet(a.episode, a.export_packet, stream=a.stream or cfg.stream, start=start, stop=stop, step=step,
                            mesh=mesh, initial_bbox=bbox, initial_mask=mask, side=a.side)
        print(f"packet written: {out}")
        print(f"  next: rsync -a {out} <gpu-host>:/data/ && ssh <gpu-host> python -m handumi_collector.tools."
              f"track_handumi_rgbd --episode /data/{out.name} --side {a.side} --backend foundationpose")
        return 0

    info = backend_info(backend)
    if not a.quiet:
        print(f"backend  {info.name} (runs_on={info.runs_on}, reacquire={info.can_reacquire})")

    def progress(n, est):
        if not a.quiet and n % 25 == 0:
            print(f"  frame {n:5d}  {est.tracking_state.value:12s} conf={est.confidence if est.confidence is None else round(est.confidence,3)}", flush=True)

    res = run_side(a.episode, a.side, cfg=cfg, mesh=mesh, backend=backend, initial_bbox=bbox, initial_mask=mask,
                   stream=a.stream or cfg.stream, start=start, stop=stop, step=step, write=not a.no_write,
                   progress=None if a.quiet else progress)
    print()
    print(res.report)
    if res.out_dir:
        print(f"\nwritten: {res.out_dir}")
    if res.error:
        return 2
    return 0 if res.qa and res.qa.verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
