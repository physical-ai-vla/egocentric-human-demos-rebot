"""`foundationpose` — NVLabs FoundationPose adapter (model-based 6D pose + tracking from RGB-D and a CAD mesh).

    initialize()  ->  est.register(K, rgb, depth_m, ob_mask, iteration=refine_iter_init)
    track()       ->  est.track_one(rgb, depth_m, K, iteration=refine_iter_track)

REQUIREMENTS — this backend does not run on macOS. FoundationPose needs an NVIDIA GPU (torch+CUDA, nvdiffrast,
the FoundationPose repo on PYTHONPATH and its weights). On a machine without them the constructor raises
TrackerUnavailable with the exact missing piece, and the pipeline stops rather than silently using another backend.

Offline-pilot workflow across machines (the Mac records, a GPU box tracks):

    mac$   python -m handumi_collector.tools.track_handumi_rgbd --episode <ep> --side left \\
               --mesh assets/handumi/left_tracking_body.obj --bbox X,Y,W,H --export-packet /tmp/pkt_left
    mac$   rsync -a /tmp/pkt_left  gpu:/data/
    gpu$   python -m handumi_collector.tools.track_handumi_rgbd --episode /data/pkt_left --side left \\
               --mesh /data/pkt_left/mesh.obj --backend foundationpose --mask /data/pkt_left/init_mask.png
    mac$   rsync -a gpu:/data/pkt_left/derived/  <ep>/derived/

FoundationPose convention check (asserted at import time against its own docs): `depth` is float metres, `K` is the 3x3
colour intrinsics, `ob_mask` is a bool/uint8 object mask on the FIRST frame, and the returned 4x4 is T_camera_object —
the same T_D_B this interface defines. RGB is expected in **RGB** order, so BGR frames are converted here."""
from __future__ import annotations
import numpy as np
from ..depth_hand_tracker import DepthBackendInfo, DepthHandPoseTracker, RigidPoseEstimate, TrackerUnavailable, TrackingState
from ..tracking_mesh import TrackingMesh, load_tracking_mesh


class FoundationPoseTracker(DepthHandPoseTracker):
    INFO = DepthBackendInfo(
        name="foundationpose", needs_mesh=True, needs_init_roi=True, metric_scale=True, can_reacquire=False, runs_on="cuda",
        provides=("confidence", "depth_residual_mm"),
        notes="NVLabs FoundationPose. CUDA only. register() on the first frame from a mask, track_one() afterwards; "
              "a re-registration is triggered from the last ROI after `reinit_after_lost_frames` bad frames.")

    def __init__(self, *, mesh: TrackingMesh | str | None = None, refine_iter_init: int = 5, refine_iter_track: int = 2,
                 score_min: float = 0.0, reinit_after_lost_frames: int = 0, debug: int = 0, debug_dir: str | None = None,
                 **_ignored) -> None:
        super().__init__()
        if mesh is None:
            raise TrackerUnavailable("`foundationpose` backend needs the HandUMI rigid tracking mesh (mesh=<path|TrackingMesh>)")
        missing = []
        try:
            import torch
            if not torch.cuda.is_available():
                missing.append("a CUDA device (torch.cuda.is_available() is False)")
        except Exception as exc:
            missing.append(f"torch ({exc})")
        for mod, why in (("nvdiffrast.torch", "nvdiffrast"), ("estimater", "the FoundationPose repo on PYTHONPATH")):
            try:
                __import__(mod)
            except Exception as exc:
                missing.append(f"{why} ({exc})")
        if missing:
            raise TrackerUnavailable("foundationpose is not runnable here: missing " + "; ".join(missing) +
                                     ". This backend needs an NVIDIA GPU host — see the module docstring for the "
                                     "record-on-Mac / track-on-GPU packet workflow.")
        import trimesh
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr

        self.mesh = mesh if isinstance(mesh, TrackingMesh) else load_tracking_mesh(mesh)
        tm = trimesh.Trimesh(vertices=self.mesh.vertices, faces=self.mesh.triangles, process=False)
        self._tm = tm
        self.opt = dict(refine_iter_init=int(refine_iter_init), refine_iter_track=int(refine_iter_track),
                        score_min=float(score_min), reinit_after_lost_frames=int(reinit_after_lost_frames))
        self.options = dict(self.opt)
        self._est = FoundationPose(model_pts=tm.vertices, model_normals=tm.vertex_normals, mesh=tm,
                                   scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                                   glctx=dr.RasterizeCudaContext(), debug=int(debug), debug_dir=debug_dir)
        self.reset()

    def reset(self) -> None:
        self._last = None
        self._mask = None
        self._lost_run = 0
        self._n_frames = 0
        self._n_valid = 0

    def get_status(self) -> dict:
        d = super().get_status()
        d.update(frames=self._n_frames, valid=self._n_valid, lost_run=self._lost_run)
        return d

    @staticmethod
    def _rgb(img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(img)[..., ::-1])          # BGR (OpenCV / our episodes) -> RGB

    def _emit(self, T, t_ns, frame_index, score=None) -> RigidPoseEstimate:
        self._n_frames += 1
        ok = T is not None and np.all(np.isfinite(T)) and (score is None or score >= self.opt["score_min"])
        if not ok:
            self._lost_run += 1
            est = RigidPoseEstimate(timestamp_ns=int(t_ns), side=self.side or "", frame_index=int(frame_index),
                                    tracking_valid=False, tracking_state=TrackingState.LOST, confidence=score)
        else:
            state = TrackingState.REACQUIRED if self._lost_run else TrackingState.TRACKING
            self._lost_run = 0
            self._n_valid += 1
            est = RigidPoseEstimate.from_T(np.asarray(T, np.float64), timestamp_ns=t_ns, side=self.side or "",
                                           frame_index=frame_index, state=state, confidence=score)
        self._last = est
        return est

    def initialize(self, rgb, depth, camera_intrinsics, side, initial_mask=None, initial_bbox=None, *,
                   timestamp_ns: int = 0, frame_index: int = 0) -> RigidPoseEstimate:
        from .icp import _as_mask
        self.side = side
        self.intrinsics = camera_intrinsics
        self.reset()
        self.side = side
        depth_m = np.asarray(depth, np.float32)
        self._mask = _as_mask(depth_m.shape, initial_mask, initial_bbox)
        T = self._est.register(K=camera_intrinsics.K.astype(np.float64), rgb=self._rgb(rgb), depth=depth_m,
                               ob_mask=self._mask.astype(np.uint8), iteration=self.opt["refine_iter_init"])
        return self._emit(T, timestamp_ns, frame_index, score=_score(self._est))

    def track(self, rgb, depth, timestamp_ns: int, *, frame_index: int = -1) -> RigidPoseEstimate:
        if self.intrinsics is None:
            raise RuntimeError("track() before initialize()")
        depth_m = np.asarray(depth, np.float32)
        n = self.opt["reinit_after_lost_frames"]
        if n and self._lost_run >= n and self._mask is not None:
            T = self._est.register(K=self.intrinsics.K.astype(np.float64), rgb=self._rgb(rgb), depth=depth_m,
                                   ob_mask=self._mask.astype(np.uint8), iteration=self.opt["refine_iter_init"])
        else:
            T = self._est.track_one(rgb=self._rgb(rgb), depth=depth_m, K=self.intrinsics.K.astype(np.float64),
                                    iteration=self.opt["refine_iter_track"])
        return self._emit(T, timestamp_ns, frame_index, score=_score(self._est))


def _score(est) -> float | None:
    """FoundationPose exposes its last selection score under different names across revisions; report it when present."""
    for attr in ("pose_last_score", "best_score", "score"):
        v = getattr(est, attr, None)
        if v is None:
            continue
        try:
            return float(np.asarray(v).reshape(-1)[0])
        except Exception:
            return None
    return None
