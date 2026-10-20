#!/usr/bin/env python3
"""[2026-10-13] WiLoR (teacher) -> YOLO-pose dataset for the realtime fingertip student used by finger_grip_teleop --detector student.
Frames are fisheye-undistorted exactly like the teleop (balance 0.6), resized to 640 wide. Label = the LARGEST WiLoR hand:
box + 4 keypoints [wrist 0, thumb tip 4, index MCP 5, index tip 8]. Frames with no hand -> empty label (negatives);
low-confidence hands -> frame dropped. Temporal split: last 20 % of each video = val.
usage: finger_distill_label.py --out ~/finger_distill/ds video.mp4 [...] [--stride 2] [--calib fisheye_right_v003.yaml]"""
import argparse, json, pathlib, sys, time
import cv2, numpy as np, yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
KP = [0, 4, 5, 8]; W_OUT = 640


def undistort_map(calib, w, h):
    y = yaml.safe_load(open(pathlib.Path.home() / "ego_collector/configs/calibration" / calib))
    K = np.asarray(y["K"], float); D = np.asarray(y["D"], float).reshape(4, 1); cw, ch = y["image_size"]; K[0] *= w / cw; K[1] *= h / ch
    Kp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.6)
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kp, (w, h), cv2.CV_16SC2), Kp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--calib", default="fisheye_right_v003.yaml"); ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args(); out = pathlib.Path(a.out).expanduser()
    from ego_collector.hands3d.wilor import WilorTracker
    stats = dict(frames=0, pos=0, neg=0, dropped=0); tr = None
    for vp in a.videos:
        vp = pathlib.Path(vp).expanduser(); cap = cv2.VideoCapture(str(vp)); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); i = 0; t0 = time.time()
        while True:
            ok = cap.grab()
            if not ok: break
            if i % a.stride: i += 1; continue
            _, img = cap.retrieve(); h, w = img.shape[:2]
            if tr is None:
                (m1, m2), Kp = undistort_map(a.calib, w, h)
                tr = WilorTracker(focal_px=float(Kp[0, 0]), image_size=(w, h), hand_conf=0.2)
            u = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR); r = tr(u)
            seen = [x for x in r.values() if x.visible and x.kp2d is not None]
            hf = max(seen, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), default=None)
            stats["frames"] += 1
            if hf is not None and np.isfinite(hf.confidence) and hf.confidence < a.min_conf: stats["dropped"] += 1; i += 1; continue
            split = "val" if i >= 0.8 * N else "train"; stem = f"{vp.stem}_{i:06d}"
            (out / "images" / split).mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            s = W_OUT / w; hh = int(round(h * s)); cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), cv2.resize(u, (W_OUT, hh), interpolation=cv2.INTER_AREA))
            lab = ""
            if hf is not None:
                kp = hf.kp2d[KP]; pts = np.vstack([kp, hf.bbox.reshape(2, 2)])
                x0, y0 = np.clip(pts.min(0) - 10, 0, [w - 1, h - 1]); x1, y1 = np.clip(pts.max(0) + 10, 0, [w - 1, h - 1])
                vis = [2 if (0 <= p[0] < w and 0 <= p[1] < h) else 0 for p in kp]
                lab = f"0 {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} {(x1 - x0) / w:.5f} {(y1 - y0) / h:.5f} " + " ".join(
                    f"{np.clip(p[0] / w, 0, 1):.5f} {np.clip(p[1] / h, 0, 1):.5f} {v}" for p, v in zip(kp, vis))
                stats["pos"] += 1
            else: stats["neg"] += 1
            (out / "labels" / split / f"{stem}.txt").write_text(lab + ("\n" if lab else ""))
            if stats["frames"] % 100 == 0: print(f"{vp.name} {i}/{N}  {stats}  {(time.time() - t0) / max(i, 1) * (N - i):.0f}s left", flush=True)
            i += 1
    (out / "data.yaml").write_text(yaml.safe_dump(dict(path=str(out), train="images/train", val="images/val", names={0: "hand"},
                                                       kpt_shape=[4, 3], flip_idx=[0, 1, 2, 3])))
    (out / "label_stats.json").write_text(json.dumps(stats)); print("done", stats)


if __name__ == "__main__":
    main()
