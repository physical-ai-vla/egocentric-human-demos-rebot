"""Fingertip detector interface for the wrist RGB cameras (2026-10-14, spec 8). Only thumb + index tips are assumed visible;
nothing downstream depends on MediaPipe (it found the hand in 17 % of wrist-cam frames, memory finger-grip-teleop-distill).

    detect(image_bgr) -> FingertipResult(thumb_tip, index_tip, confidence, pinch_metric)

`pinch_metric` is the backend's scale-free thumb-index distance (smaller = pinched); thresholds live in the YAML (gripper.*)
and are calibrated per backend. Backends:
    student   WiLoR-distilled YOLOv8n-pose, 4 kpts [wrist, thumb tip, index MCP, index tip]; metric = |thumb-index| / |wrist-index MCP|
              in % of hand. ~130 fps MPS. Runs in ~/finger_distill/.venv (ultralytics 8.4; the ego_collector venv's 8.1 crashes on MPS).
    scripted  replay / tests.
Future: HaMeR-derived, RTMPose/custom 2-keypoint -- implement the same Protocol, register in BACKENDS."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
import numpy as np


@dataclass
class FingertipResult:
    thumb_tip: np.ndarray | None          # (2,) px in the (undistorted) wrist image
    index_tip: np.ndarray | None
    confidence: float                     # 0 => not detected
    pinch_metric: float | None            # backend units (student: % of hand), None if not measurable
    backend: str = ""

    @property
    def detected(self) -> bool: return self.confidence > 0 and self.pinch_metric is not None and np.isfinite(self.pinch_metric)


class FingertipDetector(Protocol):
    name: str
    def detect(self, image_bgr: np.ndarray) -> FingertipResult: ...


class StudentYoloDetector:
    """WiLoR-distilled YOLO-pose student (scripts/finger_distill_*.py). Picks the LARGEST hand (the wearer's; the far hand is small)."""
    name = "student"

    def __init__(self, model: str = "~/finger_distill/student.pt", conf: float = 0.4, imgsz: int = 640, device: str = "mps",
                 undistort: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        from ultralytics import YOLO      # lazy: only in the finger_distill venv
        self.model = YOLO(str(Path(model).expanduser())); self.conf, self.imgsz, self.device, self.undistort = conf, imgsz, device, undistort

    def detect(self, image_bgr: np.ndarray) -> FingertipResult:
        img = self.undistort(image_bgr) if self.undistort else image_bgr
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)[0]
        if res.keypoints is None or not len(res.boxes): return FingertipResult(None, None, 0.0, None, self.name)
        b = res.boxes.xyxy.cpu().numpy(); j = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
        k4 = res.keypoints.xy.cpu().numpy()[j]; ref = float(np.linalg.norm(k4[0] - k4[2])); c = float(res.boxes.conf.cpu().numpy()[j])
        if ref <= 5.0: return FingertipResult(k4[1], k4[3], 0.0, None, self.name)
        return FingertipResult(k4[1].copy(), k4[3].copy(), c, 100.0 * float(np.linalg.norm(k4[1] - k4[3])) / ref, self.name)


class StudentTwoKptDetector(StudentYoloDetector):
    """Thumb/index-ONLY student (scripts/finger_distill_2kpt.py): 2 kpts [thumb tip, index tip], box around the tips only, so no
    wrist/palm needs to be in view. metric = tip distance in % of image WIDTH (camera rigid on the wrist); thresholds per mount."""
    name = "student2"

    def __init__(self, model: str = "~/finger_distill/student2kpt.pt", **kw) -> None: super().__init__(model, **kw)

    def detect(self, image_bgr: np.ndarray) -> FingertipResult:
        img = self.undistort(image_bgr) if self.undistort else image_bgr
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)[0]
        if res.keypoints is None or not len(res.boxes): return FingertipResult(None, None, 0.0, None, self.name)
        j = int(np.argmax(res.boxes.conf.cpu().numpy())); k2 = res.keypoints.xy.cpu().numpy()[j]; c = float(res.boxes.conf.cpu().numpy()[j])
        return FingertipResult(k2[0].copy(), k2[1].copy(), c, 100.0 * float(np.linalg.norm(k2[0] - k2[1])) / img.shape[1], self.name)


class JawTapeDetector:
    """HandUMI jaw opening from the orange-red tape on the two jaw tips (user 10-15: drive the robot gripper from the UMI, not
    the fingers). Reuses scripts/umi_fusion/jaw_track.parts (10-14 v2: red mask, components split at the centre column) on the
    RAW wrist frame; live selection per half = the part nearest the previous pick (continuity), else the largest part, and the
    other half must sit within BAND_PX of the same height (rejects the other unit's tape ~430 px above, desk specks).
    metric = tip distance in px at 1920 width (thresholds 10-14: closed <= 200, open >= 260). confidence 1 = both tips found."""
    name = "jaw_tape"
    BAND_PX = 320.0
    JUMP_PX = 400.0

    def __init__(self, undistort=None, **_ignored) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "umi_fusion"))
        self.prev = {0: None, 1: None}                   # red_mask = scripts/umi_fusion/jaw_redx (10-13 v3), imported in detect

    def detect(self, image_bgr: np.ndarray) -> FingertipResult:
        """10-15: split around the JAW TIPS' own centre, not the image centre (jaw_track.parts splits at the image centre column;
        the UMI camera shifted relative to the jaws and the closed tips sat entirely left of centre -> one half only -> None)."""
        import cv2
        from jaw_redx import red_mask
        h, w = image_bgr.shape[:2]; s = 1920.0 / w; y0 = int(0.10 * h)
        m = red_mask(image_bgr[y0:]); n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        comps = [k for k in range(1, n) if st[k, 4] >= 1e-4 * h * w]
        if not comps: return FingertipResult(None, None, 0.0, None, self.name)
        if self.prev[0] is not None:                       # continuity: the component nearest the previous jaw centre first
            pc = np.array([(self.prev[0][0] + self.prev[1][0]) / 2, (self.prev[0][1] + self.prev[1][1]) / 2]) / s - [0, y0]
            comps.sort(key=lambda k: np.hypot(*(cen[k] - pc)))
            if np.hypot(*(cen[comps[0]] - pc)) * s > self.JUMP_PX: comps.sort(key=lambda k: -st[k, 4])
        else: comps.sort(key=lambda k: -st[k, 4])
        a = comps[0]
        # second tip = another sizeable component at about the same height (open jaws); else split the one blob (closed)
        mates = [k for k in comps[1:] if st[k, 4] >= 0.25 * st[a, 4] and abs(cen[k][1] - cen[a][1]) * s <= self.BAND_PX]
        if mates:
            b = min(mates, key=lambda k: np.hypot(*(cen[k] - cen[a]))); P = [cen[a], cen[b]]
        else:
            ys, xs = np.nonzero(lab == a); mid = np.median(xs); L, R = xs < mid, xs >= mid
            if L.sum() < 5 or R.sum() < 5: return FingertipResult(None, None, 0.0, None, self.name)
            P = [np.array([xs[L].mean(), ys[L].mean()]), np.array([xs[R].mean(), ys[R].mean()])]
        A, B = sorted(P, key=lambda q: q[0]); A = (A + [0, y0]) * s; B = (B + [0, y0]) * s
        self.prev = {0: (A[0], A[1]), 1: (B[0], B[1])}
        return FingertipResult(A, B, 1.0, float(np.linalg.norm(A - B)), self.name)


class ScriptedDetector:
    """Replay / tests: a function t -> (pinch_metric | None, confidence)."""
    name = "scripted"

    def __init__(self, fn: Callable[[], tuple[float | None, float]]) -> None: self.fn = fn

    def detect(self, image_bgr=None) -> FingertipResult:
        m, c = self.fn()
        return FingertipResult(None, None, float(c) if m is not None else 0.0, m, self.name)


def fisheye_undistorter(calib_yaml: str | Path, image_wh: tuple[int, int], balance: float = 0.6) -> Callable[[np.ndarray], np.ndarray]:
    """Same undistortion the student's WiLoR labels were made with (balance 0.6)."""
    import cv2, yaml
    y = yaml.safe_load(Path(calib_yaml).expanduser().read_text())
    K = np.asarray(y["K"], float); D = np.asarray(y["D"], float).reshape(4, 1)
    Kp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, tuple(image_wh), np.eye(3), balance=balance)
    m = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kp, tuple(image_wh), cv2.CV_16SC2)
    return lambda im: cv2.remap(im, *m, interpolation=cv2.INTER_LINEAR)


BACKENDS = {"student": StudentYoloDetector, "student2": StudentTwoKptDetector, "jaw_tape": JawTapeDetector, "scripted": ScriptedDetector}


def make_detector(cfg: dict, **kw) -> FingertipDetector:
    d = dict(cfg); b = d.pop("backend")
    if b not in BACKENDS: raise ValueError(f"unknown fingertip backend {b!r}; have {sorted(BACKENDS)}")
    return BACKENDS[b](**{**d, **kw})
