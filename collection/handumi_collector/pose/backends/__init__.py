"""Backend registry. Each backend implements handumi_collector.pose.estimator.PoseEstimator. Names are config values
(pose.yaml `backend:`), never hard-coded elsewhere. Backends whose software is missing raise BackendUnavailable on
construction/initialize — the pipeline records that and moves on; it never substitutes fake poses."""
from __future__ import annotations
from ..estimator import BackendInfo, BackendUnavailable, PoseEstimator

_REGISTRY: dict[str, str] = {
    "mock": "handumi_collector.pose.backends.mock:MockBackend",
    "opencv_vo": "handumi_collector.pose.backends.opencv_vo:OpenCvVoBackend",
    "orbslam3": "handumi_collector.pose.backends.orbslam3:OrbSlam3Backend",
    "dpvo": "handumi_collector.pose.backends.dpvo:DpvoBackend",
    "trajectory_replay": "handumi_collector.pose.backends.trajectory_replay:TrajectoryReplayBackend",
    "mast3r": "handumi_collector.pose.backends.trajectory_replay:TrajectoryReplayBackend",       # MASt3R-Fusion result.txt -> derived/pose_mast3r
    "mast3r_filtered": "handumi_collector.pose.backends.trajectory_replay:MASt3RFilteredBackend",   # same + isolated-spike post-filter -> derived/pose_mast3r_filtered
}


def available_backends() -> list[str]: return sorted(_REGISTRY)


def make_backend(name: str, **options) -> PoseEstimator:
    if name not in _REGISTRY: raise KeyError(f"unknown pose backend {name!r}; known: {available_backends()}")
    mod, cls = _REGISTRY[name].split(":")
    import importlib
    return getattr(importlib.import_module(mod), cls)(**options)


def backend_info(name: str) -> BackendInfo:
    mod, cls = _REGISTRY[name].split(":")
    import importlib
    return getattr(importlib.import_module(mod), cls).INFO
