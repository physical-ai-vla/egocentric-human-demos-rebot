"""Render C922-like virtual wrist videos from the raw Arducam fisheye streams of an episode (or session). RAW FISHEYE IS NEVER
DELETED: these renders are derived and can be regenerated with different virtual-camera parameters at any time.
<episode>/derived/virtual_wrist/{left,right}_wrist_c922like.mp4 + virtual_wrist.json (target K, size, rotation, calibration versions).
Raw is untouched; the exporter (M5) uses these as the human wrist policy observation.

    python -m handumi_collector.tools.render_virtual_wrist PATH [--side left] [--cal-dir DIR] [--preview out.png]"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from ..collector.video_writer import VideoStreamWriter
from ..pose.calibration import SideCalibration
from ..pose.episode_io import RawEpisode, write_json
from ..pose.virtual_view import VirtualPinholeView, load_virtual_wrist
from .pose_process import episodes_under


def render_episode(ep_path: Path, *, sides=("left", "right"), cal_dir: Path | None = None, codec: str = "h264_videotoolbox", preview: Path | None = None,
                   mount_revision: str = "", notes: str = "") -> dict:
    ep = RawEpisode.load(ep_path); out = ep.path / "derived" / "virtual_wrist"; out.mkdir(parents=True, exist_ok=True)
    import time
    vver, vw = load_virtual_wrist(cal_dir=cal_dir)
    # provenance: enough to re-render (or re-render differently) any H120 episode from the raw fisheye later
    meta = dict(schema="handumi_virtual_wrist_render/v2", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                target_virtual_wrist=vver or "DEFAULT_ESTIMATE (no virtual_wrist_vNNN.yaml)", mount_revision=mount_revision, notes=notes,
                K_target={k: vw["K_target"][k].tolist() for k in vw["K_target"]},
                D_target={k: (None if vw["D_target"][k] is None else vw["D_target"][k].tolist()) for k in vw["D_target"]},
                size=list(vw["size"]), rpy_deg=vw["rpy_deg"], source=vw["source"], sides={})
    tiles = []
    for side in sides:
        stream = f"{side}_wrist"
        if stream not in ep.frames: meta["sides"][side] = dict(error=f"no {stream} stream"); continue
        cal = SideCalibration(side, cal_dir=cal_dir)
        if cal.intrinsics is None: meta["sides"][side] = dict(error="fisheye intrinsics missing (tools.calibrate_fisheye)"); continue
        view = VirtualPinholeView(cal.intrinsics, vw["K_target"][side], vw["size"], vw["rpy_deg"][side], D_target=vw["D_target"][side])
        fps = int(ep.meta.get("streams", {}).get(stream, {}).get("fps_measured") or 30) or 30
        w = VideoStreamWriter(out / f"{stream}_c922like.mp4", width=vw["size"][0], height=vw["size"][1], fps=fps, codec=codec, bitrate_kbps=4000); n = 0
        for vf, fi, t, img in ep.iter_frames(stream):
            r = view.render(img); w.put(r); n += 1
            if preview is not None and n == max(1, len(ep.frames[stream]) // 2): tiles.append(np.concatenate([cv2.resize(img, (640, 360)), cv2.resize(r, (640, 360))], axis=0))
        w.close()
        meta["sides"][side] = dict(frames=n, source_fisheye_intrinsics=cal.versions["fisheye"], rpy_deg=vw["rpy_deg"][side],
                                   hfov_deg=round(view.hfov_deg(), 1), video=str((out / f"{stream}_c922like.mp4").relative_to(ep.path)))
    if preview is not None and tiles: cv2.imwrite(str(preview), np.concatenate(tiles, axis=1))
    write_json(meta, out / "virtual_wrist.json"); return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--side", action="append", default=None); ap.add_argument("--cal-dir", default=None)
    ap.add_argument("--preview", default=None); ap.add_argument("--codec", default="h264_videotoolbox")
    ap.add_argument("--mount-revision", default="", help="e.g. handumi_left_mount_v2 — recorded in virtual_wrist.json")
    a = ap.parse_args(argv)
    for ep in episodes_under(Path(a.path)):
        m = render_episode(ep, sides=a.side or ("left", "right"), cal_dir=Path(a.cal_dir) if a.cal_dir else None, codec=a.codec,
                           preview=Path(a.preview) if a.preview else None, mount_revision=a.mount_revision)
        print(ep.name, {k: (v.get("frames"), v.get("hfov_deg"), v.get("error")) for k, v in m["sides"].items()})
    return 0


if __name__ == "__main__": raise SystemExit(main())
