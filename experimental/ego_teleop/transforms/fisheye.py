"""Wrist-fisheye (Kannala-Brandt) -> pinhole rectification, in ONE place.

The wrist Arducam is calibrated as Kannala-Brandt (`configs/calibration/fisheye_<side>_vNNN.yaml`). MASt3R-Fusion's
`Intrinsics.from_calib` knows `pinhole` (+ radtan) and `mei` (omnidir) and nothing else, so anything that feeds it
has to undistort first.

Two callers do that, and they MUST do it identically, or the live tracker and the offline bake-off are looking at
different cameras and their numbers cannot be compared:

    ego_teleop/tools/export_euroc.py            offline: episode -> EuRoC dir -> MASt3R-Fusion main.py
    ego_teleop/tracking/backends/mast3r_live.py live:    Arducam frame -> socket -> MASt3R-Fusion server

`balance` is the knob that decides how much of the fisheye's periphery survives: 0 crops to the fully-valid centre
(the default, and what the bake-off used), 1 keeps everything and fills the corners with stretched nothing. It is
recorded in the export metadata and sent in the live handshake precisely because it changes the effective field of
view, and therefore the feature count, and therefore the tracking.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DEFAULT_BALANCE = 0.0          # crop to the valid centre — the convention the EuRoC export and the bake-off used


@dataclass(frozen=True)
class Rectifier:
    """Precomputed KB -> pinhole remap for one camera at one downscale."""
    K_new: np.ndarray                  # 3x3 pinhole intrinsics of the RECTIFIED image
    size: tuple[int, int]              # (width, height) of the rectified image
    balance: float
    _mapx: np.ndarray
    _mapy: np.ndarray

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.remap(img, self._mapx, self._mapy, cv2.INTER_LINEAR)

    def intrinsics(self) -> dict:
        """Pinhole intrinsics of the rectified image, in the shape the pose backends take."""
        return dict(model="pinhole", width=int(self.size[0]), height=int(self.size[1]),
                    fx=float(self.K_new[0, 0]), fy=float(self.K_new[1, 1]),
                    cx=float(self.K_new[0, 2]), cy=float(self.K_new[1, 2]),
                    distortion=[0.0, 0.0, 0.0, 0.0])


def make_rectifier(K: np.ndarray, D: np.ndarray, image_size, *, downscale: int = 1,
                   balance: float = DEFAULT_BALANCE) -> Rectifier:
    """The exact call `export_euroc.py` makes, so live and replay see the same rectified camera."""
    import cv2
    W0, H0 = int(image_size[0]), int(image_size[1])
    d = max(int(downscale), 1)
    W, H = W0 // d, H0 // d
    Ks = np.asarray(K, np.float64).copy(); Ks[:2] /= d
    Dv = np.asarray(D, np.float64).reshape(4, 1)
    Kn = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(Ks, Dv, (W, H), np.eye(3), balance=float(balance),
                                                                new_size=(W, H))
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(Ks, Dv, np.eye(3), Kn, (W, H), cv2.CV_32FC1)
    return Rectifier(K_new=Kn, size=(W, H), balance=float(balance), _mapx=mapx, _mapy=mapy)


def load_wrist_fisheye(side: str, *, cal_dir: Path | None = None) -> dict:
    """The versioned `fisheye_<side>` calibration, or a RuntimeError naming what is missing. Never a guess: an
    assumed intrinsic is a systematic error the tracker cannot see and the operator cannot feel."""
    from handumi_collector.pose.calibration import load_versioned
    v, d = load_versioned(f"fisheye_{side}", cal_dir=cal_dir) if cal_dir else load_versioned(f"fisheye_{side}")
    if not d or "K" not in d or "D" not in d:
        raise RuntimeError(f"no fisheye_{side} intrinsics in configs/calibration/ — the wrist camera model is "
                           f"required before any live visual tracking (spec section 9)")
    model = str(d.get("model", "kannala_brandt"))
    if model not in ("kannala_brandt", "kb", "equidistant", "fisheye"):
        raise RuntimeError(f"fisheye_{side} says model={model!r}; this rectifier implements Kannala-Brandt only")
    return dict(version=v, K=np.asarray(d["K"], float), D=np.asarray(d["D"], float),
                image_size=tuple(int(x) for x in d.get("image_size", (1920, 1080))), model=model)


def wrist_rectifier(side: str, *, downscale: int = 1, balance: float = DEFAULT_BALANCE,
                    cal_dir: Path | None = None) -> tuple[Rectifier, dict]:
    c = load_wrist_fisheye(side, cal_dir=cal_dir)
    return make_rectifier(c["K"], c["D"], c["image_size"], downscale=downscale, balance=balance), c
