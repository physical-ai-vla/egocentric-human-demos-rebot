"""DPVO adapter (visual-only deep patch VO; monocular ⇒ non-metric scale). Requires the `dpvo` package + network weights
(pose.yaml backend_options.dpvo.{network, config}). Fisheye frames are undistorted to a pinhole view first because DPVO
takes pinhole intrinsics. Absence of dpvo is reported as BackendUnavailable."""
from __future__ import annotations
import numpy as np
from ..estimator import BackendInfo, BackendUnavailable, PoseEstimate, PoseEstimator, TrackingState
from ..se3 import pose7_to_T
from .opencv_vo import FisheyeRectifier


class DpvoBackend(PoseEstimator):
    INFO = BackendInfo("dpvo", uses_imu=False, metric_scale=False, online_capable=True, provides=(),
                       notes="deep patch VO; scale arbitrary; IMU used for independent QA only")
    info = INFO

    def __init__(self, *, network: str | None = None, config: str | None = None, rect_fov_deg: float = 90.0, rect_size: tuple[int, int] = (640, 480),
                 device: str = "cuda", **_ignored) -> None:
        self.network, self.config, self.rect_fov, self.rect_size, self.device = network, config, rect_fov_deg, tuple(rect_size), device
        self.reset()

    def reset(self) -> None:
        self._slam = None; self._rect = None; self._frames: list[tuple[int, int]] = []; self._last = None

    def initialize(self, *, intrinsics=None, T_camera_imu=None, imu_noise=None, image_size=None) -> None:
        try:
            import torch                                    # noqa: F401
            from dpvo.dpvo import DPVO                      # type: ignore
            from dpvo.config import cfg as dpvo_cfg         # type: ignore
        except Exception as exc:
            raise BackendUnavailable(f"dpvo: package not importable ({exc}); install DPVO (github.com/princeton-vl/DPVO) into this venv") from exc
        if not self.network: raise BackendUnavailable("dpvo: network weights path required (backend_options.dpvo.network)")
        if intrinsics is None: raise BackendUnavailable("dpvo: fisheye intrinsics required to rectify to pinhole")
        self._rect = FisheyeRectifier(intrinsics, out_size=self.rect_size, fov_deg=self.rect_fov)
        if self.config: dpvo_cfg.merge_from_file(self.config)
        self._DPVO, self._cfg, self._torch = DPVO, dpvo_cfg, torch

    def push_image(self, t_ns: int, frame_index: int, image_bgr) -> PoseEstimate:
        img = self._rect.rectify(image_bgr)
        fx, fy, cx, cy = self._rect.pinhole()
        if self._slam is None:
            self._slam = self._DPVO(self._cfg, self.network, ht=img.shape[0], wd=img.shape[1], viz=False)
        t_img = self._torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).to(self.device)
        intr = self._torch.tensor([fx, fy, cx, cy], dtype=self._torch.float32).to(self.device)
        self._slam(len(self._frames), t_img, intr)
        self._frames.append((int(t_ns), int(frame_index)))
        est = PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING); self._last = est
        return est

    def finish(self) -> list[PoseEstimate]:
        if self._slam is None: return []
        poses, tstamps = self._slam.terminate()          # (N,7) camera-in-world [x y z qx qy qz qw], frame ids
        by_id = {int(i): np.asarray(p, np.float64) for p, i in zip(poses, tstamps)}
        out = []
        for i, (t, fi) in enumerate(self._frames):
            p = by_id.get(i)
            out.append(PoseEstimate(t, fi, pose7_to_T(p), TrackingState.TRACKING) if p is not None else PoseEstimate(t, fi, None, TrackingState.LOST))
        return out
