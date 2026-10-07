"""[2026-10-06 user] live/recorded RIGHT-wrist fisheye -> the MEASURED reBot right-wrist C922 view (shared by the UI preview and the
recorder's right_wrist_c922.mp4). Robot camera model = configs/calibration/robot_right_wrist_c922_v001.json (hand-eye 2026-10-06,
640x480, f 651.4, c (318.3, 224.9), k1 0.051 k2 -0.129); R = I (robot camera placed exactly at the HandUMI camera)."""
from __future__ import annotations
import numpy as np, cv2


class C922View:
    """[2026-10-06 user] live RIGHT-wrist fisheye -> the MEASURED reBot right-wrist C922 (configs/calibration/robot_right_wrist_c922_v001.json,
    hand-eye 2026-10-06: f 651.4, c (318.3, 224.9), k1 0.051 k2 -0.129), R = I (the robot camera placed exactly where the HandUMI camera is),
    with the robot's pose-A wrist frame blended in (robot_A_right_view_v001.jpg) so the operator can match the start view. Preview only."""
    def __init__(self) -> None:
        import json, yaml
        from pathlib import Path
        c = Path(__file__).resolve().parents[2] / "configs/calibration"
        y = yaml.safe_load(open(c / "fisheye_right_v002.yaml")); self.K = np.array(y["K"], np.float64); self.D = np.array(y["D"], np.float64).reshape(4, 1)
        self.size = tuple(int(x) for x in y["image_size"])
        r = json.load(open(c / "robot_right_wrist_c922_v001.json")); self.Kr = np.array(r["K"], np.float64); self.Dr = np.array(r["dist"], np.float64)
        ref = cv2.imread(str(c / "robot_A_right_view_v001.jpg")); self.ref = None if ref is None else cv2.resize(ref, (640, 480))
        self.alpha = 0.35; self._maps = {}

    def _map(self, w: int, h: int):
        if (w, h) not in self._maps:
            K = self.K.copy(); sx, sy = w / self.size[0], h / self.size[1]; K[0] *= sx; K[1] *= sy
            u, v = np.meshgrid(np.arange(640, dtype=np.float64), np.arange(480, dtype=np.float64))
            n = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2), self.Kr, self.Dr).reshape(-1, 2)
            pts = np.concatenate([n, np.ones((len(n), 1))], 1).reshape(-1, 1, 3)
            uv = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), K, self.D)[0].reshape(480, 640, 2).astype(np.float32)
            self._maps[(w, h)] = (uv[..., 0], uv[..., 1])
        return self._maps[(w, h)]

    def render(self, img: np.ndarray, overlay: bool = True) -> np.ndarray:
        mx, my = self._map(img.shape[1], img.shape[0])
        out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if overlay and self.ref is not None: out = cv2.addWeighted(out, 1.0 - self.alpha, self.ref, self.alpha, 0)
        return out
