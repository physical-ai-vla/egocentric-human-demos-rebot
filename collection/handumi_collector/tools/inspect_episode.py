"""Raw episode inspector (works without any pose processing; adds trajectories when derived/pose_<backend> exists).
Shows: head video frame strip, per-stream frame timing (fps, gaps), L/R grip raw+normalized, events on a timeline, integrity
result; with derived poses: L/R TCP XYZ + tracking state (3-D view lives in tools.pose_inspector).

    python -m handumi_collector.tools.inspect_episode --episode EP [--backend opencv_vo] [--save out.png] [--no-show] [--validate]"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ..collector.integrity import validate_episode
from ..pose.episode_io import RawEpisode, derived_dir, read_table
from ..pose.rgbd_align import edge_alignment


def head_strip(ep: RawEpisode, n: int = 8):
    import av, cv2
    fm = ep.frames.get("head")
    if fm is None or not ep.video_path("head").exists(): return None
    want = set(np.linspace(0, len(fm) - 1, n).astype(int).tolist()); tiles = []
    with av.open(str(ep.video_path("head"))) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                img = cv2.resize(fr.to_ndarray(format="rgb24"), (160, 120)); cv2.putText(img, f"{(fm.capture_ns[i]-ep.t_start_ns)/1e9:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1); tiles.append(img)
            if i > max(want): break
    return np.concatenate(tiles, axis=1) if tiles else None


def figure(ep: RawEpisode, *, backend: str | None = None, validation: dict | None = None):
    import matplotlib.pyplot as plt
    t0 = ep.t_start_ns; dur = (ep.t_stop_ns - t0) / 1e9
    fig = plt.figure(figsize=(16, 11)); fig.suptitle(f"{ep.path.parent.name}/{ep.path.name}  order {ep.meta.get('order')}  {ep.meta.get('duration_s')}s  quality {ep.meta.get('quality')}")
    gs = fig.add_gridspec(5, 1, height_ratios=[1.3, 1, 1, 1, 0.5])
    ax0 = fig.add_subplot(gs[0]); strip = head_strip(ep)
    if strip is not None: ax0.imshow(strip); ax0.set_title("HEAD C922 (policy observation)"); ax0.axis("off")
    ax1 = fig.add_subplot(gs[1])
    for st, fm in ep.frames.items():
        t = (fm.capture_ns - t0) / 1e9
        if len(t) > 1: ax1.plot(t[1:], np.diff(fm.capture_ns) / 1e6, lw=0.8, label=f"{st} ({len(t)} f, {((len(t)-1)/(t[-1]-t[0])):.1f} fps)")
    ax1.axhline(1000 / 30, color="k", ls=":", lw=0.6); ax1.set_ylabel("frame Δt (ms)"); ax1.legend(fontsize=7, ncol=4); ax1.set_xlim(0, dur); ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    for side, g in ep.grip.items():
        t = (g.t_ns - t0) / 1e9; ax2.plot(t, g.normalized, label=f"{side} grip norm ({len(t)} @ {((len(t)-1)/max(t[-1]-t[0],1e-6)):.0f} Hz)")
    ax2.set_ylabel("grip 0=closed 1=open"); ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    if not ep.grip: ax2.text(0.5, 0.5, "no gripper samples", ha="center", transform=ax2.transAxes)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    if backend:
        for side, col in (("left", "tab:red"), ("right", "tab:blue")):
            try: tcp = read_table(derived_dir(ep.path, backend) / f"{side}_tcp_pose")
            except FileNotFoundError: continue
            t = (tcp["t_ns"].values - t0) / 1e9; v = tcp["valid"].values.astype(bool)
            for k, nm in enumerate("xyz"): ax3.plot(t, np.where(v, tcp[nm].values, np.nan), color=col, alpha=0.4 + 0.3 * k, lw=0.9, label=f"{side} {nm}")
        ax3.set_ylabel(f"TCP xyz [{backend}]"); ax3.legend(fontsize=7, ncol=6)
    else:
        for side, im in ep.imu.items():
            t = (im.host_ns - t0) / 1e9; ax3.plot(t, np.degrees(np.linalg.norm(im.gyro, axis=1)), lw=0.6, label=f"{side} |gyro| dps")
        ax3.set_ylabel("IMU |gyro| (dps)" if ep.imu else "no IMU / no derived poses"); ax3.legend(fontsize=7) if ep.imu else None
    ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(gs[4], sharex=ax1); ax4.set_yticks([]); ax4.set_xlabel("t (s)")
    for e in ep.events:
        x = (e["t_ns"] - t0) / 1e9; ax4.axvline(x, color="tab:orange" if e["kind"].startswith("home") else "tab:red", lw=1); ax4.text(x, 0.5, e["kind"], rotation=90, fontsize=6, va="center")
    if validation:
        txt = ("INTEGRITY OK" if validation["ok"] else "INTEGRITY PROBLEMS: " + "; ".join(validation["problems"])[:200])
        fig.text(0.01, 0.005, txt, fontsize=8, color="green" if validation["ok"] else "red", family="monospace")
    fig.tight_layout(); return fig


MASTER_PREFERENCE = ("head", "head_depth")
KEY_HELP = "SPACE play/pause   <- -> step   , . x10   D depth   Q quit"


def _master_stream(ep: RawEpisode) -> str:
    for n in MASTER_PREFERENCE:
        if n in ep.frames and len(ep.frames[n]): return n
    return next(n for n, f in ep.frames.items() if len(f))


class _Reader:
    """Random access to one stream's video, by capture time. Keeps the last decoded image so scrubbing inside a frame
    costs nothing, and reports how far the frame it returned is from the time that was asked for -- the whole point of
    looking at several streams at once is that the answer is not zero."""

    def __init__(self, ep: RawEpisode, stream: str) -> None:
        import cv2
        self.stream = stream
        self.fm = ep.frames[stream]
        self.cap = cv2.VideoCapture(str(ep.video_path(stream)))
        self.depth_dir = ep.path / f"{stream}_depth"
        self.has_depth = self.depth_dir.is_dir()
        self._vf = None
        self._img = None

    def at(self, t_ns: int):
        """(image, delta_ms, video_frame) for the frame nearest `t_ns`, or (None, nan, -1)."""
        import cv2
        cap_ns = self.fm.capture_ns
        if not len(cap_ns): return None, float("nan"), -1
        i = int(np.clip(np.searchsorted(cap_ns, t_ns), 1, len(cap_ns) - 1))
        i = i if abs(cap_ns[i] - t_ns) < abs(cap_ns[i - 1] - t_ns) else i - 1
        vf = int(self.fm.video_frame[i])
        delta_ms = float(cap_ns[i] - t_ns) / 1e6
        if vf != self._vf:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ok, img = self.cap.read()
            self._vf, self._img = vf, (img if ok else None)
        return self._img, delta_ms, vf

    def depth_at(self, vf: int):
        import cv2
        if not self.has_depth or vf < 0: return None
        p = self.depth_dir / f"{vf:06d}.png"
        if not p.exists(): return None
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return None if d is None else d.astype(np.float32) * 1e-3

    def release(self):
        self.cap.release()


def _trace(imu, t_ns: int, window_s: float, size: tuple[int, int], label: str, colour):
    """Gyro magnitude either side of now, with the sample the viewer would actually use marked on it."""
    import cv2
    h, w = size
    canvas = np.full((h, w, 3), 24, np.uint8)
    cv2.putText(canvas, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    if imu is None or not len(imu.host_ns): return canvas, float("nan")
    lo, hi = t_ns - window_s * 1e9, t_ns + window_s * 1e9
    m = (imu.host_ns >= lo) & (imu.host_ns <= hi)
    if m.sum() < 2: return canvas, float("nan")
    t = imu.host_ns[m]; g = np.degrees(np.linalg.norm(imu.gyro[m], axis=1))
    top = max(float(g.max()), 5.0)
    xs = ((t - lo) / (hi - lo) * (w - 1)).astype(int)
    ys = (h - 4 - g / top * (h - 20)).astype(int)
    cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, colour, 1, cv2.LINE_AA)
    cv2.line(canvas, (w // 2, 16), (w // 2, h - 2), (90, 90, 90), 1)          # now
    j = int(np.argmin(np.abs(imu.host_ns - t_ns)))
    nearest_ms = float(imu.host_ns[j] - t_ns) / 1e6
    if lo <= imu.host_ns[j] <= hi:
        x = int((imu.host_ns[j] - lo) / (hi - lo) * (w - 1))
        y = int(h - 4 - np.degrees(np.linalg.norm(imu.gyro[j])) / top * (h - 20))
        cv2.drawMarker(canvas, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 9, 1)
    cv2.putText(canvas, f"{top:.0f} dps full scale", (6, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas, nearest_ms


def _tile(img, size, title, sub, ok=True):
    import cv2
    h, w = size
    t = np.full((h, w, 3), 18, np.uint8)
    if img is not None:
        sc = min(w / img.shape[1], (h - 26) / img.shape[0])
        r = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
        y0, x0 = 26 + ((h - 26) - r.shape[0]) // 2, (w - r.shape[1]) // 2
        t[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    else:
        cv2.putText(t, "no frame", (w // 2 - 34, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(t, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(t, sub, (w - 8 - 7 * len(sub), 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (120, 255, 120) if ok else (90, 90, 255), 1, cv2.LINE_AA)
    return t


def compose(ep: RawEpisode, readers: dict, t_ns: int, *, master: str, sides: list, serials: dict,
            align_cache: dict, imu_window_s: float, show_depth: bool, playing: bool, step_mult: int,
            pos_ms: int, t0: int, t1: int, validation: dict | None, tile: tuple[int, int] = (264, 424)):
    """Build the whole frame. Separated from the event loop so it can be rendered and checked without a window --
    a layout assembled by stacking arrays is exactly the kind of code that breaks on an episode with a different
    number of streams, and finding that out interactively is the slow way."""
    import cv2
    TH, TW = tile
    meta = ep.meta
    tiles, master_vf = [], -1
    for name, r in readers.items():
        img, d_ms, vf = r.at(t_ns)
        if show_depth and r.has_depth:
            dm = r.depth_at(vf)
            if dm is not None:
                vis = cv2.applyColorMap(np.clip((dm - 0.2) / 1.3 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                vis[dm <= 0] = (40, 40, 40)
                img = vis if img is None else cv2.addWeighted(img, 0.35, cv2.resize(vis, (img.shape[1], img.shape[0])), 0.65, 0)
        tiles.append(_tile(img, (TH, TW), name, f"{d_ms:+.1f} ms", abs(d_ms) < 20))
        if name == master and img is not None and r.has_depth and vf not in align_cache:
            dm = r.depth_at(vf)
            if dm is not None: align_cache[vf] = edge_alignment(img, dm)
        if name == master: master_vf = vf
    while len(tiles) % 3: tiles.append(np.full((TH, TW, 3), 12, np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    canvas = np.vstack(rows)
    W = canvas.shape[1]

    for side, colour in zip(sides, ((255, 190, 120), (120, 190, 255))):
        tr, near_ms = _trace(ep.imu.get(side), t_ns, imu_window_s, (90, W),
                             f"imu {side}  serial {serials.get(side, '?')}", colour)
        cv2.putText(tr, f"nearest {near_ms:+.1f} ms", (W - 150, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 255, 120) if abs(near_ms) < 10 else (90, 90, 255), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, tr])
    if not sides:
        strip = np.full((40, W, 3), 24, np.uint8)
        cv2.putText(strip, "no IMU in this episode", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, strip])

    al = align_cache.get(master_vf, {})
    info = np.full((150, W, 3), 16, np.uint8)
    lines = [
        (f"t {pos_ms/1000:7.3f} / {(t1-t0)/1e9:.3f} s      master stream: {master}      {'PLAY' if playing else 'PAUSE'}"
         f"      step x{step_mult}", (235, 235, 235)),
        (f"{meta.get('order','?')}   {str(meta.get('instruction',''))[:118]}", (200, 200, 255)),
        (f"status {meta.get('status','?')}   quality {meta.get('quality','?')}"
         + (f"   QA {'ok' if validation.get('ok') else 'PROBLEMS: ' + '; '.join(validation['problems'][:2])}" if validation else ""),
         (120, 255, 120) if (not validation or validation.get("ok")) else (90, 90, 255)),
        (f"rgb/depth  shift {tuple(al.get('shift_px', ('?','?')))} px   overlap "
         f"{al.get('overlap', float('nan')):.0%}" if al else "rgb/depth  (no depth on the master stream)",
         (120, 255, 120) if al.get("shift_magnitude_px", 0) <= 3 else (80, 200, 255)),
        (KEY_HELP, (150, 150, 150)),
    ]
    y = 22
    for text, col in lines:
        cv2.putText(info, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA); y += 26
    for e in ep.events:
        x = int((int(e.get("t_ns", t0)) - t0) / max(t1 - t0, 1) * (W - 1))
        cv2.line(info, (x, 140), (x, 149), (90, 200, 255), 1)
    cv2.line(info, (int(pos_ms * 1e6 / max(t1 - t0, 1) * (W - 1)), 136), (int(pos_ms * 1e6 / max(t1 - t0, 1) * (W - 1)), 149), (255, 255, 255), 2)
    canvas = np.vstack([canvas, info])


    return canvas


def replay(ep: RawEpisode, *, imu_window_s: float = 0.75, validation: dict | None = None) -> int:
    """Every stream at one instant, with the cost of putting them there shown next to each.

    The streams do not share a clock tick: cameras land where they land and the IMU runs at its own rate, so a viewer
    has to pick a nearest frame for each. Printing the delta it had to accept is the difference between a tool that
    shows synchronisation and one that assumes it."""
    import cv2
    master = _master_stream(ep)
    t0, t1 = int(ep.frames[master].capture_ns[0]), int(ep.frames[master].capture_ns[-1])
    order = [n for n in MASTER_PREFERENCE if n in ep.frames] + sorted(n for n in ep.frames if n not in MASTER_PREFERENCE)
    readers = {n: _Reader(ep, n) for n in order if len(ep.frames[n])}
    sides = sorted(ep.imu)
    serials = (ep.meta.get("imu_serials") or {})
    meta = ep.meta
    align_cache: dict[int, dict] = {}

    TW, TH = 424, 264
    win = f"replay {ep.path.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TW * min(len(readers), 3), TH + 150 + 90 * max(len(sides), 1))
    cv2.createTrackbar("t (ms)", win, 0, max(int((t1 - t0) / 1e6), 1), lambda _v: None)

    playing, show_depth, pos_ms, step_mult = False, False, 0, 1
    try:
        while True:
            if playing: pos_ms = (pos_ms + 33) % max(int((t1 - t0) / 1e6), 1)
            cv2.setTrackbarPos("t (ms)", win, int(pos_ms))
            pos_ms = cv2.getTrackbarPos("t (ms)", win)
            t_ns = t0 + int(pos_ms * 1e6)

            canvas = compose(ep, readers, t_ns, master=master, sides=sides, serials=serials,
                             align_cache=align_cache, imu_window_s=imu_window_s, show_depth=show_depth,
                             playing=playing, step_mult=step_mult, pos_ms=pos_ms, t0=t0, t1=t1,
                             validation=validation, tile=(TH, TW))

            cv2.imshow(win, canvas)
            k = cv2.waitKey(1 if playing else 20) & 0xFF
            if k == ord("q"): break
            if k == ord(" "): playing = not playing
            if k == ord("d"): show_depth = not show_depth
            if k == ord(","): step_mult = 1
            if k == ord("."): step_mult = 10
            if k in (81, ord("a")): pos_ms = max(0, pos_ms - 33 * step_mult)
            if k in (83, ord("s")): pos_ms = min(int((t1 - t0) / 1e6), pos_ms + 33 * step_mult)
    finally:
        for r in readers.values(): r.release()
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--episode", required=True); ap.add_argument("--backend", default=None)
    ap.add_argument("--save", default=None); ap.add_argument("--no-show", action="store_true"); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--replay", action="store_true", help="scrub every stream on one timeline instead of drawing the overview")
    ap.add_argument("--imu-window-s", type=float, default=0.75, help="--replay: IMU history drawn either side of now")
    a = ap.parse_args(argv)
    if a.no_show:
        import matplotlib; matplotlib.use("Agg")
    ep = RawEpisode.load(a.episode)
    # Replay always validates: the point of looking at an episode frame by frame is to decide whether to keep it, and
    # the verdict belongs on screen next to the pictures rather than in a separate run someone may not have made.
    val = (validate_episode(Path(a.episode), expect_grippers=list(ep.grip), expect_imus=list(ep.imu))
           if (a.validate or a.replay) else None)
    if a.replay:
        return replay(ep, imu_window_s=a.imu_window_s, validation=val)
    if val: print(json.dumps(val, indent=1))
    fig = figure(ep, backend=a.backend, validation=val)
    if a.save: fig.savefig(a.save, dpi=110); print("saved", a.save)
    if not a.no_show:
        import matplotlib.pyplot as plt; plt.show()
    return 0


if __name__ == "__main__": raise SystemExit(main())
