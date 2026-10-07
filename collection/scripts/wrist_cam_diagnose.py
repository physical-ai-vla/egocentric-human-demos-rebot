#!/usr/bin/env python3
"""[2026-09-18] Why the wrist cameras run at 23 fps, and which control actually changes their picture.

Both of the findings that forced this script turned out to be wrong, and the corrections are the point of it now.

The 23 fps is not real: all four 1080p indices hold 30.0 fps open together (gap p95 36 ms), and all 385 recorded
episodes report 29.97-30.04 fps with zero skipped capture indices. Nothing was ever contending.

`exposure-time-abs` is not dead either. FisheyeCamLeft sits in auto-exposure-mode 8 by DEVICE DEFAULT, and in auto mode
a UVC camera stores the exposure you write and ignores it -- the exact "reads back faithfully, picture does not move"
symptom. Driven to mode 1 and verified by read-back, both units sweep cleanly (exposure 1 -> mean 184 / sharpness 381,
exposure 330 -> mean 243 / sharpness 5). Two further controls were quietly fighting the setting: `auto-exposure-priority`
(1 = the sensor may drop frames to lengthen exposure) and `backlight-compensation` (default 1, brightens behind your back).

What remains true: measure, do not assume. `brightness` is a LUMA PEDESTAL -- it darkens without touching chroma, so
every step of it multiplies saturation (0 -> sat 72, -64 -> sat 110) and turns a faint lighting tint into a strong cast.
It is still the right control once exposure is at its floor, but it must be paired with a matching `saturation` cut.

    --fps       open cameras alone, in pairs, and all together; report the fps each one actually achieves
    --controls  sweep every control that could darken the picture; report mean level AND temporal noise

Temporal noise is reported because the current workaround (brightness at its -64 floor) is a digital pedestal, and the
operator sees the result as static in the preview. A control that darkens the picture while raising the per-pixel
standard deviation is not a fix; it is the same problem written differently.
    sudo .venv/bin/python scripts/wrist_cam_diagnose.py --fps
"""
from __future__ import annotations
import argparse, itertools, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
UVC = str(Path.home() / "robot-cockpit/bin/uvc-util")
NAMES = {"left": "FisheyeCamLeft", "right": "Arducam 1080P Low Light"}


def uvc(name, *args):
    cp = subprocess.run([UVC, f"--select-by-name={name}", *args], capture_output=True, text=True, timeout=15)
    return cp.stdout.strip()


def opencv_indices(max_index=6):
    """every index that opens at 1080p, with the frame size it reports"""
    out = []
    for i in range(max_index):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ok, f = c.read()
            if ok and f is not None: out.append((i, f.shape[1], f.shape[0]))
        c.release()
    return out


def open_cam(i):
    c = cv2.VideoCapture(i)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(5): c.read()
    return c


def run_fps(indices, seconds):
    """One THREAD per camera, because that is how the collector runs them.

    Reading every camera in a single loop would serialise the MJPEG decodes and report contention even when the
    cameras are perfectly independent -- it would measure this script, not the hardware. With a thread each, a drop
    when several stream at once is a real drop."""
    import threading
    caps = {i: open_cam(i) for i in indices}
    gaps = {i: [] for i in indices}; n = {i: 0 for i in indices}
    stop = threading.Event()

    def pump(i):
        c = caps[i]; last = time.monotonic()
        while not stop.is_set():
            ok, f = c.read()
            if not ok: continue
            t = time.monotonic(); gaps[i].append(t - last); last = t; n[i] += 1

    threads = [threading.Thread(target=pump, args=(i,), daemon=True) for i in indices]
    t0 = time.monotonic()
    for t in threads: t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads: t.join(2.0)
    dur = time.monotonic() - t0
    for c in caps.values(): c.release()
    return {i: dict(fps=n[i] / dur, p50=float(np.median(gaps[i])) * 1000 if gaps[i] else float("nan"),
                    p95=float(np.percentile(gaps[i], 95)) * 1000 if gaps[i] else float("nan")) for i in indices}


def measure(cap, n=20, patch=200):
    """mean level, clipping, and how much a static patch wanders frame to frame (the 'static' the operator sees)"""
    for _ in range(6): cap.read()
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if ok: frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if len(frames) < 5: return None
    S = np.stack(frames).astype(np.float32)
    h, w = S.shape[1:]
    P = S[:, h // 2 - patch // 2:h // 2 + patch // 2, w // 2 - patch // 2:w // 2 + patch // 2]
    return dict(mean=float(S.mean()), clipped=float((S >= 250).mean() * 100), black=float((S <= 5).mean() * 100),
                noise=float(np.median(P.std(axis=0))),            # per-pixel temporal std: the static
                flicker=float(S.reshape(len(S), -1).mean(1).std()),  # frame-to-frame level swing: the flicker
                sharp=float(cv2.Laplacian(frames[-1], cv2.CV_64F).var()))


SWEEPS = [("exposure-time-abs", [1, 20, 100, 330]), ("gain", [0, 25, 50]), ("brightness", [64, 0, -32, -64]),
          ("gamma", [72, 100, 200, 300]), ("contrast", [16, 32, 48]), ("backlight-compensation", [0, 1])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", action="store_true"); ap.add_argument("--controls", action="store_true")
    ap.add_argument("--grid", action="store_true", help="exposure x brightness grid; the one that picks the value to freeze")
    ap.add_argument("--index", type=int, default=None, help="camera index for --controls")
    ap.add_argument("--name", default="left", choices=list(NAMES), help="which camera --controls addresses by name")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    found = opencv_indices()
    print("1080p-capable OpenCV indices: " + ", ".join(f"{i} ({w}x{h})" for i, w, h in found) + "\n")
    idx = [i for i, _, _ in found]

    if a.fps:
        combos = [(i,) for i in idx] + list(itertools.combinations(idx, 2))
        if len(idx) >= 3: combos.append(tuple(idx))
        print(f"{'cameras open':>16s} {'index':>6s} {'fps':>7s} {'gap p50':>9s} {'gap p95':>9s}")
        for combo in combos:
            r = run_fps(list(combo), a.seconds)
            for i in combo:
                print(f"{str(combo):>16s} {i:6d} {r[i]['fps']:7.1f} {r[i]['p50']:8.1f}ms {r[i]['p95']:8.1f}ms")
            print()
        print("reading: an index that holds 30 alone and drops in company is contending. The wrist cameras sit on\n"
              "SEPARATE USB controllers (left 03:1.1, right 00:1.4, both USB 2.0), so a drop when they stream together\n"
              "cannot be bus bandwidth and points at host-side MJPEG decode.")
        return 0

    if a.grid:
        if a.index is None: print("--grid needs --index"); return 1
        name = NAMES[a.name]
        for ctl, v in (("auto-exposure-mode", 1), ("auto-white-balance-temp", "false"),
                       ("white-balance-temp", 4600), ("gain", 0), ("gamma", 100), ("power-line-frequency", 2)):
            uvc(name, f"--set={ctl}={v}")
        time.sleep(1.0)
        got = {c: uvc(name, f"--get-value={c}") for c in ("auto-exposure-mode", "gain", "auto-white-balance-temp")}
        print("frozen: " + ", ".join(f"{k}={v}" for k, v in got.items()))
        if got.get("auto-exposure-mode") != "1":
            print("REFUSING: the camera did not go to manual exposure, so nothing measured here would be reproducible")
            return 1
        cap = open_cam(a.index); time.sleep(1.5)
        print(f"\n{'exposure':>9s} {'bright':>7s} {'mean':>7s} {'clip %':>7s} {'black %':>8s} {'noise':>7s} {'sharp':>8s}")
        rows = []
        for b in (0, -16, -32, -48, -64):
            uvc(name, f"--set=brightness={b}")
            for e in (1, 5, 10, 20, 40, 80):
                uvc(name, f"--set=exposure-time-abs={e}")
                time.sleep(0.7); m = measure(cap)
                if m is None: continue
                rows.append((e, b, m))
                print(f"{e:9d} {b:7d} {m['mean']:7.0f} {m['clipped']:7.2f} {m['black']:8.2f} {m['noise']:7.2f} {m['sharp']:8.0f}", flush=True)
            print(flush=True)
        cap.release()
        ok = [(e, b, m) for e, b, m in rows if m["clipped"] < 0.5 and m["black"] < 2.0 and 90 <= m["mean"] <= 140]
        if not ok:
            print("nothing met clipped < 0.5 %, black < 2 % and mean 90-140 -- the lighting itself is out of range")
            return 1
        # Shortest shutter first so a fast hand does not smear, then the least negative brightness. The pedestal is not
        # contrast thrown away -- measured here it RAISES detail (sharpness 431 -> 533 from 0 to -48) by pulling the
        # histogram off the clipping ceiling. What it does cost is colour: chroma is untouched, so saturation rises with
        # every step and has to be cut to match (see the module docstring).
        e, b, m = min(ok, key=lambda r: (r[0], -r[1]))
        print(f"\nproposed: exposure-time-abs={e}, brightness={b}  (mean {m['mean']:.0f}, clipped {m['clipped']:.2f} %, "
              f"black {m['black']:.2f} %, noise {m['noise']:.2f}, sharpness {m['sharp']:.0f})")
        print("chosen as the SHORTEST exposure that behaves, with brightness left as close to 0 as that allows")
        return 0

    if a.controls:
        if a.index is None: print("--controls needs --index"); return 1
        name = NAMES[a.name]
        # Freeze the camera BEFORE measuring, and prove it froze by read-back. Without this the sweep runs against a
        # live auto-exposure loop that is quietly undoing each step -- which is exactly how a baseline of 187 came to
        # disagree with the 143 the same camera gave at the same brightness minutes later.
        for ctl, v in (("auto-exposure-mode", 1), ("auto-white-balance-temp", "false"), ("gain", 0)):
            uvc(name, f"--set={ctl}={v}")
        time.sleep(1.0)
        frozen = {c: uvc(name, f"--get-value={c}") for c in ("auto-exposure-mode", "auto-white-balance-temp", "gain")}
        print("frozen before sweep: " + ", ".join(f"{k}={v}" for k, v in frozen.items()))
        cap = open_cam(a.index)
        time.sleep(1.5)                      # the picture keeps settling for about a second after the stream opens
        base = measure(cap)
        print(f"baseline: mean {base['mean']:.0f}, clipped {base['clipped']:.2f}%, noise {base['noise']:.2f}, "
              f"flicker {base['flicker']:.2f}, sharpness {base['sharp']:.0f}\n")
        print(f"{'control':>24s} {'value':>7s} {'mean':>7s} {'clip %':>7s} {'noise':>7s} {'flicker':>8s} {'sharp':>8s}")
        for ctl, values in SWEEPS:
            prev = uvc(name, f"--get-value={ctl}")
            for v in values:
                uvc(name, f"--set={ctl}={v}")
                time.sleep(0.6); m = measure(cap)
                if m is None: continue
                print(f"{ctl:>24s} {v:7g} {m['mean']:7.0f} {m['clipped']:7.2f} {m['noise']:7.2f} "
                      f"{m['flicker']:8.2f} {m['sharp']:8.0f}", flush=True)
            uvc(name, f"--set={ctl}={prev}")       # leave the camera as it was found
            print()
        cap.release()
        print("a control is only a fix if it lowers `mean` into 90-140 WITHOUT raising `noise` above the baseline.")
        return 0

    ap.print_help(); return 0


if __name__ == "__main__":
    sys.exit(main())
