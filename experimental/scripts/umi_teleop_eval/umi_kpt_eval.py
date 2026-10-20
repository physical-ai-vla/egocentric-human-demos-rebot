#!/usr/bin/env python3
"""[2026-10-15] Offline evaluation of the markerless UMI jaw-tip model on a recording (no robot, no hub):
detection rate per frame, dropout runs, and -- where the teacher has a label (umi_pseudo_labels source == 1) -- pixel error.
Runs the ultralytics checkpoint (finger_distill venv) or, with --onnx, the exact cv2.dnn path the hub uses (ego_collector venv).
usage: umi_kpt_eval.py <rec_dir> [--model ~/finger_distill/umi_kpt.pt | --onnx ~/finger_distill/umi_kpt.onnx] [--conf 0.5] [--stride 1]"""
import argparse, json, pathlib, sys
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--model", default="~/finger_distill/umi_kpt.pt"); ap.add_argument("--onnx", default=None)
    ap.add_argument("--conf", type=float, default=0.5); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--from-frac", type=float, default=0.0)
    a = ap.parse_args(); rec = pathlib.Path(a.rec).expanduser()
    t_vid = np.load(rec / "frames" / "rgbd_color_t_ns.npy"); lab = {}
    if (rec / "umi_pseudo_labels.npz").exists():
        L = np.load(rec / "umi_pseudo_labels.npz"); lab = {int(t): uv for t, uv, s in zip(L["t_ns"], L["tape_uv"], L["source"]) if s == 1}
    if a.onnx:
        sys.path.insert(0, str(pathlib.Path.home() / "ego_collector")); from visual_teleop.vision.umi_kpt import UmiKptTracker
        tr = UmiKptTracker(a.onnx, np.eye(4), conf=a.conf); det = lambda im: [(u, v, c) for c, u, v in tr.infer(im)][:1]
    else:
        from ultralytics import YOLO
        m = YOLO(str(pathlib.Path(a.model).expanduser()))
        def det(im):
            r = m.predict(im, imgsz=640, conf=a.conf, verbose=False, device="mps")[0]
            if r.keypoints is None or not len(r.boxes): return []
            j = int(np.argmax(r.boxes.conf.cpu().numpy())); k = r.keypoints.xy.cpu().numpy()[j][0]; return [(float(k[0]), float(k[1]), float(r.boxes.conf[j]))]
    cap = cv2.VideoCapture(str(rec / "frames" / "rgbd_color.mp4")); start = int(a.from_frac * len(t_vid)); hit, err, ts = [], [], []
    for i, t in enumerate(t_vid):
        ok, im = cap.read()
        if not ok: break
        if i < start or i % a.stride: continue
        d = det(im); hit.append(bool(d)); ts.append(t / 1e9)
        if d and int(t) in lab and np.all(np.isfinite(lab[int(t)])): err.append(float(np.hypot(d[0][0] - lab[int(t)][0], d[0][1] - lab[int(t)][1])))
    hit = np.array(hit); ts = np.array(ts) - ts[0]; drops, s0 = [], None
    for i, h in enumerate(hit):
        if not h and s0 is None: s0 = i
        if s0 is not None and (h or i == len(hit) - 1): drops.append(float(ts[i] - ts[s0])); s0 = None
    out = dict(frames=int(len(hit)), detection_pct=round(100 * float(hit.mean()), 1), dropout_runs=len(drops),
               dropout_median_s=round(float(np.median(drops)), 2) if drops else 0.0, dropout_max_s=round(max(drops), 2) if drops else 0.0,
               labelled=len(err), px_err_median=round(float(np.median(err)), 1) if err else None, px_err_p90=round(float(np.percentile(err, 90)), 1) if err else None)
    print(f"[모델 평가] {rec.name}: 프레임 {out['frames']}, 검출 {out['detection_pct']}%, 끊김 {out['dropout_runs']}회 (중앙값 {out['dropout_median_s']} s, 최장 {out['dropout_max_s']} s)"
          + (f", 라벨 {out['labelled']}장 픽셀 오차 중앙값 {out['px_err_median']} / p90 {out['px_err_p90']}" if err else ", 라벨 없음(테이프 없는 녹화)"))
    (rec / "umi_kpt_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
