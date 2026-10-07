#!/usr/bin/env python3
"""[2026-09-18] Why MediaPipe found no hand in 605/605 frames of every HRGBD_qa10 episode.

A detector that is merely struggling produces a low hit rate, not a clean zero. A clean zero across 16 independent
tracks points at the input, not at the model, so this audits the input path before anything is tuned or replaced:

  1  what actually reaches MediaPipe   resolution, dtype, channel order, orientation, and the pixel size of the hand
  2  standalone vs integrated          the same frames through raw MediaPipe (IMAGE mode, explicit BGR->RGB) and
                                       through the collector's HandLandmarker (VIDEO mode, GPU/SRGBA)
  3  scale                             full frame vs a workspace ROI vs ROI upscaled 2x
  4  threshold                         min_detection 0.3 -- if 0.3 is still zero the threshold was never the problem

Frames are written out so a human can look at exactly what the model was given. Nothing in the depth, camera_to_tcp
or IK path is touched or needed here.

    .venv/bin/python scripts/hand_detector_audit.py --episode <episode_dir> [--frames 20]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path("/tmp/hand_audit")


def sample_frames(ep: Path, stream: str, n: int) -> list[tuple[int, int, np.ndarray]]:
    """The frames MediaPipe would see, taken the same way the tracker takes them: decode <stream>.mp4 in order and
    keep the ones the depth track actually joins to. Any difference between this and what MediaPipe got would itself
    be the bug, so it is deliberately the same reader."""
    from handumi_collector.pose.rgbd_io import RgbdEpisode
    e = RgbdEpisode.load(ep, stream=stream)
    step = max(1, e.n_frames // n)
    out = []
    for f in e.iter_frames(0, e.n_frames, step):
        out.append((f.index, int(f.t_ns), f.rgb.copy()))
        if len(out) >= n:
            break
    return out


def describe(frames) -> None:
    i, t, img = frames[len(frames) // 2]
    b, g, r = (float(img[..., k].mean()) for k in range(3))
    print(f"  frames sampled      {len(frames)}  (indices {frames[0][0]}..{frames[-1][0]})")
    print(f"  shape / dtype       {img.shape} {img.dtype}   contiguous={img.flags['C_CONTIGUOUS']}")
    print(f"  channel means       ch0 {b:.1f}  ch1 {g:.1f}  ch2 {r:.1f}")
    print(f"                      (decoded as bgr24, so ch0=B ch1=G ch2=R; skin reads R>G>B, a swap reads B>G>R)")
    print(f"  value range         {img.min()}..{img.max()}")
    print(f"  timestamps          {frames[0][1]} .. {frames[-1][1]} ns, strictly increasing="
          f"{all(frames[k][1] < frames[k+1][1] for k in range(len(frames)-1))}")


def raw_landmarker(*, num_hands: int, min_detection: float):
    """Raw MediaPipe, IMAGE mode -- no VIDEO-mode tracking state, no wrapper, no timestamp bookkeeping."""
    from mediapipe.tasks.python import BaseOptions, vision
    from ego_collector.hands.mediapipe_tracker import ensure_model
    import sys as _s
    delegate = BaseOptions.Delegate.GPU if _s.platform == "darwin" else BaseOptions.Delegate.CPU
    base = BaseOptions(model_asset_path=str(ensure_model()), delegate=delegate)
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.IMAGE, num_hands=num_hands,
        min_hand_detection_confidence=min_detection, min_hand_presence_confidence=min_detection,
        min_tracking_confidence=min_detection))


def mp_image(img_bgr: np.ndarray):
    import mediapipe as mp
    import sys as _s
    if _s.platform == "darwin":       # the macOS GPU graph only accepts SRGBA
        return mp.Image(image_format=mp.ImageFormat.SRGBA,
                        data=np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)))
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))


def run_standalone(frames, *, min_detection: float, transform=None, tag: str = "") -> dict:
    lm = raw_landmarker(num_hands=2, min_detection=min_detection)
    hits, labels, boxes = 0, [], []
    for idx, _t, img in frames:
        im = transform(img) if transform else img
        res = lm.detect(mp_image(im))
        if res.hand_landmarks:
            hits += 1
            h, w = im.shape[:2]
            for hd, lms in zip(res.handedness, res.hand_landmarks):
                labels.append(f"{hd[0].category_name}:{hd[0].score:.2f}")
                xs = [p.x * w for p in lms]; ys = [p.y * h for p in lms]
                boxes.append((max(xs) - min(xs), max(ys) - min(ys)))
            if len(boxes) <= 3:
                cv2.imwrite(str(OUT / f"hit_{tag}_{idx:06d}.jpg"), im)
    lm.close()
    return dict(hits=hits, n=len(frames), labels=labels[:6],
                bbox_px=[(round(w), round(h)) for w, h in boxes[:6]])


def run_integrated(frames) -> dict:
    """The collector's own path: VIDEO mode, the wrapper's colour conversion, the wrapper's timestamps."""
    from ego_collector.hands.mediapipe_tracker import HandLandmarker
    lm = HandLandmarker(num_hands=2, min_detection=0.5, min_presence=0.5, min_tracking=0.5)
    hits, labels = 0, []
    for idx, t_ns, img in frames:
        dets = lm.detect_all(img, int(t_ns // 1_000_000))
        if dets:
            hits += 1
            labels += [f"{d.label}:{d.label_confidence:.2f}" for d in dets]
    return dict(hits=hits, n=len(frames), labels=labels[:6])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--stream", default="head_depth")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="workspace ROI; default is the lower-central two thirds, where the hands work")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    frames = sample_frames(a.episode, a.stream, a.frames)
    if not frames:
        print("no frames"); return 1
    H, W = frames[0][2].shape[:2]
    x0, y0, x1, y1 = a.roi or (int(W * 0.10), int(H * 0.20), int(W * 0.95), H)

    print(f"\n[1] what reaches MediaPipe   ({a.episode.name}/{a.stream})")
    describe(frames)
    for i, (idx, _t, img) in enumerate(frames):
        cv2.imwrite(str(OUT / f"frame_{idx:06d}.jpg"), img)
    print(f"  frames written      {OUT}/frame_*.jpg   ({len(frames)} files -- open one and check it looks normal)")
    print(f"  ROI used            ({x0},{y0})-({x1},{y1})  = {x1-x0}x{y1-y0} px of {W}x{H}")

    crop = lambda im: im[y0:y1, x0:x1]
    crop2 = lambda im: cv2.resize(im[y0:y1, x0:x1], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    print("\n[2][3][4] detection rate by condition")
    rows = [
        ("integrated (VIDEO, wrapper, 0.5)", run_integrated(frames)),
        ("standalone (IMAGE, raw, 0.5)", run_standalone(frames, min_detection=0.5, tag="full50")),
        ("standalone (IMAGE, raw, 0.3)", run_standalone(frames, min_detection=0.3, tag="full30")),
        ("standalone + ROI crop (0.3)", run_standalone(frames, min_detection=0.3, transform=crop, tag="crop30")),
        ("standalone + ROI crop 2x (0.3)", run_standalone(frames, min_detection=0.3, transform=crop2, tag="crop2x30")),
    ]
    print(f"  {'condition':>34s} {'hits':>10s}  labels / hand bbox px")
    for name, r in rows:
        extra = ", ".join(r.get("labels", [])) or "-"
        bb = r.get("bbox_px")
        if bb:
            extra += "   bbox " + ", ".join(f"{w}x{h}" for w, h in bb)
        print(f"  {name:>34s} {r['hits']:4d}/{r['n']:<4d}  {extra}")

    print("\nreading")
    integrated, standalone = rows[0][1]["hits"], rows[1][1]["hits"]
    if standalone > 0 and integrated == 0:
        print("  standalone detects and the integrated path does not -> a code/format bug in the wrapper or its caller,")
        print("  NOT the model. Compare the two colour conversions and the VIDEO-mode timestamps.")
    elif standalone == 0 and rows[4][1]["hits"] > 0:
        print("  only the upscaled crop detects -> the hand is too small at this resolution. The head camera is fixed,")
        print("  so a constant ROI + 2x upscale is a cheap fix; no new model is needed.")
    elif all(r["hits"] == 0 for _n, r in rows):
        print("  nothing detects under any condition, including threshold 0.3 and a 2x crop -> this is not threshold,")
        print("  not scale and not the wrapper. The hand itself is not presentable to MediaPipe (gloved and wrapped")
        print("  around the device). A hand-landmark model is the wrong detector for this rig.")
    else:
        print("  mixed -- see the per-condition rates above.")
    print(f"\n  hit crops written to {OUT}/hit_*.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
