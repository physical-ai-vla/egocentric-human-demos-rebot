"""Pre- and post-flight check for an RGB-D pose pilot. Run BEFORE recording, and again right after, before taking the
rig off — everything it reports is cheap to fix while the hardware is still on your head and expensive afterwards.

    sudo .venv/bin/python -m handumi_collector.tools.check_rgbd_ready --live          # before: does the Orbbec deliver?
    .venv/bin/python -m handumi_collector.tools.check_rgbd_ready --episode EP [--mesh M]   # after: is it processable?

Why --live matters: the recorder writes <stream>_depth/intrinsics.json only when the camera reported intrinsics at
open() (collector/recorder.py). If it did not, the episode still records and still looks complete — but the tracker
refuses it, because a guessed focal length is a silent metric-scale error in every pose downstream. That is a thing to
discover in ten seconds, not after a twenty-second take."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "


def _p(ok: bool, msg: str, *, warn: bool = False) -> bool:
    """warn=True: report it, but do not fail the check (recoverable / next-time advice)."""
    print(f"[{OK if ok else (WARN if warn else BAD)}] {msg}", flush=True)
    return True if warn else ok


def _step(msg: str) -> None:
    """Progress line, flushed immediately. Opening the Orbbec blocks for up to ~17 s (pipeline start plus a 30-frame
    warm-up) and stdout is block-buffered when piped, so without these the tool looks like it hung."""
    print(f"  ... {msg}", flush=True)


def check_live(frames: int) -> int:
    import os
    print(f"Orbbec live check   (uid={os.geteuid()}{'' if os.geteuid() == 0 else '  <- NOT root: the SDK cannot claim the UVC interface, re-run with sudo'})",
          flush=True)
    _step("importing pyorbbecsdk")
    from ego_collector.camera.capture import CaptureConfig
    from ego_collector.camera.orbbec_capture import OrbbecCapture
    _step("enumerating devices")
    try:
        from pyorbbecsdk import Context
        n = Context().query_devices().get_count()
        if not _p(n > 0, f"{n} Orbbec device(s) on the bus"):
            return 1
    except Exception as exc:
        _p(False, f"device enumeration failed: {exc}  (needs sudo on macOS)")
        return 1
    cap = OrbbecCapture(CaptureConfig(index=0, width=848, height=480, fps=30))
    _step("starting colour+depth streams and warming up — this blocks for up to ~17 s, be patient")
    try:
        cap.open()
    except Exception as exc:
        _p(False, f"Orbbec open failed: {exc}  (macOS needs sudo — UVCAssistant claims the interface otherwise)")
        return 1
    _p(True, "streams started")
    ok = True
    try:
        intr = getattr(cap, "intrinsics", None)
        ok &= _p(bool(intr), f"intrinsics reported by the SDK: {intr}  <- without these the episode is unprocessable")
        ds = getattr(cap, "depth_scale", None)
        ok &= _p(ds is not None, f"depth_scale: {ds} mm/count  (metres = raw * depth_scale / 1000)")
        _step(f"grabbing {frames} depth frames")
        got, cov, zs = 0, [], []
        for _ in range(frames * 3):
            f = cap.poll()
            if f is None or getattr(f, "depth", None) is None:
                continue
            d = np.asarray(f.depth)
            got += 1
            cov.append(float((d > 0).mean()))
            v = d[d > 0]
            if len(v):
                zs.append(float(np.median(v)) * (ds or 1.0) / 1000.0)
            if got >= frames:
                break
        ok &= _p(got >= frames, f"depth frames grabbed: {got}/{frames}")
        if cov:
            ok &= _p(float(np.mean(cov)) > 0.5, f"valid-depth coverage: {np.mean(cov):.1%} of pixels "
                                                f"(median distance {np.median(zs):.2f} m)")
            print(f"         point the camera at the HandUMI at working distance and re-run if coverage looks low")
    finally:
        cap.close()
    print("\nREADY" if ok else "\nNOT READY — fix the above before recording", flush=True)
    return 0 if ok else 1


def check_episode(ep_path: Path, mesh: str | None, stream: str | None) -> int:
    from ..pose.depth_config import load_depth_pose_cfg
    from ..pose.rgbd_io import RgbdEpisode
    cfg = load_depth_pose_cfg()
    ok = True
    try:
        ep = RgbdEpisode.load(ep_path, stream=stream or cfg.stream, fps_fallback=cfg.fps)
    except Exception as exc:
        _p(False, f"episode not readable: {exc}")
        return 1
    k = ep.intrinsics
    _p(True, f"layout {ep.layout}  stream {ep.stream}  frames {ep.n_frames}  fps {ep.fps:.2f}")
    ok &= _p(bool(k.fx and k.fy), f"intrinsics fx {k.fx:.1f} fy {k.fy:.1f} cx {k.cx:.1f} cy {k.cy:.1f} "
                                  f"({k.width}x{k.height}, depth unit {k.depth_unit_m*1e3:.3f} mm, "
                                  f"aligned_to_rgb={k.aligned_to_rgb})")
    dur = (ep.t_ns[-1] - ep.t_ns[0]) / 1e9 if ep.n_frames > 1 else 0.0
    ok &= _p(dur >= 5.0, f"duration {dur:.1f} s")
    ok &= _p(20.0 <= ep.fps <= 40.0, f"frame rate {ep.fps:.2f} Hz")
    cov, zmed = [], []
    for f in ep.iter_frames(0, min(ep.n_frames, 30), max(ep.n_frames // 30, 1)):
        d = f.depth_m
        cov.append(float((d > 0).mean()))
        v = d[d > 0]
        if len(v):
            zmed.append(float(np.median(v)))
    if cov:
        ok &= _p(float(np.mean(cov)) > 0.3, f"valid-depth coverage {np.mean(cov):.1%}, median distance "
                                            f"{np.median(zmed):.2f} m")
    # Spatial agreement, not just presence: a depth map shifted a few pixels still looks flat, still covers the frame
    # and still sits at the right distance. At a metre, four pixels is a couple of centimetres of cube.
    from ..pose.rgbd_align import edge_alignment, summarise
    per = [edge_alignment(f.rgb, f.depth_m) for f in ep.iter_frames(0, min(ep.n_frames, 12), max(ep.n_frames // 12, 1))]
    al = summarise(per)
    if al.get("median_offset_px") is not None:
        _p(not al["problems"], f"colour/depth edge offset: median {al['median_offset_px']:.1f} px, "
                               f"p95 {al['p95_offset_px']:.1f} px, overlap {al['overlap']:.0%}, "
                               f"invalid depth near edges {al['invalid_depth_near_edges']:.1%}"
                               if al["invalid_depth_near_edges"] is not None else
                               f"colour/depth edge offset: median {al['median_offset_px']:.1f} px",
           warn=bool(al["warnings"]) and not al["problems"])
    ok &= not al["problems"]
    for w in al["problems"] + al["warnings"]:
        print(f"         {w}")

    ev = ep.path / "events.json"
    marks = []
    if ev.exists():
        try:
            marks = [e.get("kind") for e in json.loads(ev.read_text())]
        except Exception:
            pass
    home = ("home_leave" in marks and "home_return" in marks) or ("static_begin" in marks and "static_end" in marks)
    _p(home, "HOME marks present (home_leave + home_return) -> trustworthy static jitter and return drift"
             if home else "no HOME marks — static windows will be AUTO-DETECTED FROM THE POSE (a frozen tracker reads "
                          "as static). Usable, but mark them next time", warn=not home)
    bad = [k2 for k2 in ("camera_drop", "timestamp_jump", "device_error", "recorder_error") if k2 in marks]
    _p(not bad, "no hardware events during the take" if not bad else f"hardware events recorded: {sorted(set(bad))}",
       warn=bool(bad))
    if mesh:
        from ..pose.tracking_mesh import load_tracking_mesh
        try:
            d = load_tracking_mesh(mesh).diagnostics()
            ok &= _p(not d["problems"], f"mesh {Path(mesh).name}: {d['n_triangles']} tris, extent "
                                        f"{d['extent_m']} m, origin_inside_bbox={d['origin_inside_bbox']}")
            for pr in d["problems"]:
                print(f"         {pr}")
        except Exception as exc:
            ok &= _p(False, f"mesh unusable: {exc}")
    print("\nPROCESSABLE — run tools.track_handumi_rgbd next" if ok else "\nNOT PROCESSABLE — see the failures above", flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true", help="open the Orbbec now and check it delivers what the tracker needs")
    g.add_argument("--episode", type=Path, help="check a recorded episode is processable")
    ap.add_argument("--mesh", help="also validate a tracking mesh")
    ap.add_argument("--stream")
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args(argv)
    return check_live(a.frames) if a.live else check_episode(a.episode, a.mesh, a.stream)


if __name__ == "__main__":
    raise SystemExit(main())
