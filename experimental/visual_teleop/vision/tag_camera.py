"""Fixed RGB camera for the UMI AprilTag (2026-10-15, user: C922 + AprilTag/PnP first). Opens the camera, locks autofocus (a
moving focus changes the focal length, i.e. the PnP scale), undistorts with the plumb_bob intrinsics, and returns frames with the same
fields the tag tracker reads from an RgbdFrame (color_bgr, depth_m=None, calib.color, timestamp_ns). No depth: metric scale comes from
the printed tag size only -- measure it."""
from __future__ import annotations
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Intr:
    fx: float; fy: float; cx: float; cy: float


@dataclass
class _Calib:
    color: _Intr


@dataclass
class TagCamFrame:
    timestamp_ns: int
    color_bgr: np.ndarray
    calib: _Calib
    depth_m: None = None


class TagCamera:
    def __init__(self, cfg: dict) -> None:
        import cv2
        self.cfg = cfg; y = yaml.safe_load((ROOT / cfg["intrinsics"]).read_text())
        K = np.asarray(y["K"], float).reshape(3, 3); D = np.asarray(y["D"], float).ravel(); cw, ch = y["image_size"]
        w, h = int(cfg["width"]), int(cfg["height"])
        if (w, h) != (cw, ch): K[0] *= w / cw; K[1] *= h / ch
        self.K, self.D = K, D
        self.maps = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
        self.calib = _Calib(_Intr(K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
        idx = cfg.get("cv_index")
        if idx is None:
            from ..tools.fingertip_node import find_camera
            idx = find_camera(cfg["match_name"])
        self.index = int(idx); self.cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); self.cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
        self.focus = self._lock_focus(cfg.get("focus", 0))
        for _ in range(60):
            ok, f = self.cap.read()
            if ok and f.shape[1] == w: break
            time.sleep(0.05)
        else: raise RuntimeError(f"tag camera {cfg['match_name']} (OpenCV {self.index}): no {w}x{h} frames")

    def _lock_focus(self, focus) -> str:
        """Autofocus off + fixed focus via robot-cockpit/bin/uvc-util (the same tool calibrate_c922.sh uses)."""
        u = Path.home() / "robot-cockpit/bin/uvc-util"
        if not u.exists(): return "uvc-util missing: autofocus NOT locked"
        try:
            lst = subprocess.run([str(u), "-d"], capture_output=True, text=True, timeout=5).stdout
            loc = self.cfg.get("uvc_location")       # 10-15: two C922s connected (robot wrist + tripod) -> pick by USB LocationID
            line = next((l for l in lst.splitlines() if self.cfg["match_name"].split()[0] in l and (loc is None or loc.lower() in l.lower())), None)
            if line is None: return "uvc-util: camera not listed, autofocus NOT locked"
            i = line.split()[0].strip("[]:")
            run = lambda *a: subprocess.run([str(u), "-I", i, *a], capture_output=True, text=True, timeout=5).stdout
            if focus in (None, "auto"):          # let AF settle on the scene, then freeze it there (as calibrate_c922.sh does)
                run("-s", "auto-focus=true"); time.sleep(2.0)
                got = run("-g", "focus-abs").strip(); import re as _re; m = _re.findall(r"-?\d+", got)
                if not m: return f"uvc-util: could not read focus ({got!r}); autofocus left ON"
                focus = int(m[-1])
            run("-s", "auto-focus=false"); run("-s", f"focus-abs={int(focus)}")
            return f"autofocus off, focus={int(focus)} (uvc-util #{i})"
        except Exception as e: return f"uvc-util failed ({e}): autofocus NOT locked"

    def read(self) -> TagCamFrame | None:
        import cv2
        ok, f = self.cap.read(); t = time.monotonic_ns()
        if not ok: return None
        return TagCamFrame(t, cv2.remap(f, *self.maps, interpolation=cv2.INTER_LINEAR), self.calib)

    def close(self) -> None: self.cap.release()
