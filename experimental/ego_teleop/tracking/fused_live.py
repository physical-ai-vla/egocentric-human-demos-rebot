"""Live assembly of the three sensors behind one `FusedWristPoseProvider` (multi-sensor spec sections 11, 15).

    chest RGB-D  OrbbecHeadCamera ──► thread ──► RgbdHandWristPoseProvider ──► push_rgbd ─┐
                                                                                          ├─► FusedWristPoseProvider
    wrist camera  UvcCamera  ─┐                                                           │        │ get_pose(now)
                              ├─► LiveVioFeeder ──► EstimatorWristPoseProvider ──► push_vi┘        ▼
    wrist IMU     imu tap  ───┘    (its own thread, full IMU rate)                         control loop @ rate_hz

Three clocks, three rates, no shared pacing (section 15): each sensor thread pushes at its own rate into the fused
provider, and the control loop pulls the latest state. Nothing queues frames — the feeder takes `camera.latest()` and
the RGB-D thread takes the newest frame, so a slow consumer loses frames instead of accumulating latency.

Device split: the wrist camera and the wrist IMU come from the collector's `DeviceManager` (which already knows how
to identify them by USB serial), while the chest camera is opened directly as `OrbbecHeadCamera` because the fusion
needs an `RgbdFrame` (colour + ALIGNED depth + intrinsics), not the collector's raw `CameraFrame`. The hardware
profile passed here must therefore NOT also list the Orbbec as a camera — one process cannot open it twice.

STATUS: written against the existing device APIs; it has not been run on the three-sensor rig (the wrist IMU unit is
the missing hardware). P0 (`tools/f0_sensors.py`) is the thing to run first, and it uses the collector's own device
path rather than this one, so it can prove the sensors before this file is trusted."""
from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from handumi_collector.config import DEFAULT_CONFIG_DIR, load_hardware
from handumi_collector.devices.base import now_ns
from handumi_collector.devices.manager import DeviceManager
from .fused_wrist import FusedWristPoseProvider
from .imu_tap import LiveVioFeeder, install_imu_tap

log = logging.getLogger("ego_teleop.fused_live")


@dataclass
class FusedLiveRig:
    """Opens the three sensors, wires them into one fused provider, and hands the control loop a `WristPoseProvider`."""
    cfg: "FusedWristCfg"                      # ego_teleop.config.FusedWristCfg
    teleop: "TeleopCfg"
    hardware: str = "handumi_v1"              # collector profile for the WRIST camera + IMU (must not list the Orbbec)
    wrist_camera: str | None = None           # default: f"{side}_wrist"
    downscale: int = 1
    estimator: object | None = None           # a PoseEstimator; default: built from cfg.vi_backend
    provider: FusedWristPoseProvider | None = None
    _threads: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    stats: dict = field(default_factory=lambda: dict(rgbd=0, rgbd_errors=0, last_error=""))
    # Latest frame of each wrist/chest stream, for an observer (f5_teleop_hud). Written by the sensor threads and
    # read by whoever asks: a HUD that misses a frame must not slow the thread down, so there is no lock and no
    # queue here — the reader gets whatever the last complete assignment was.
    last_rgbd_frame: object | None = None
    last_wrist_frame: object | None = None

    def __post_init__(self) -> None:
        self.side = self.cfg.side
        self.wrist_camera = self.wrist_camera or f"{self.side}_wrist"
        self.dm = None; self.head = None; self.feeder = None; self.vi = None; self.rgbd = None

    # ---- lifecycle -------------------------------------------------------------------------------------
    def open(self, *, config_dir: Path = DEFAULT_CONFIG_DIR) -> FusedWristPoseProvider:
        from ..hand3d.head_camera import HeadRgbdCalibration, OrbbecHeadCamera
        from ..transforms.calibration import load_teleop_calibration

        cal = load_teleop_calibration(self.side)
        cal.require("wrist_camera")              # T_H_C: without it every wrist rotation fabricates translation
        offset_ns = int((cal.camera_imu_time_offset_ms or 0.0) * 1e6)

        hwp = Path(self.hardware) if Path(self.hardware).exists() else config_dir / f"hardware_{self.hardware}.yaml"
        hw = load_hardware(hwp)
        if any(c.backend == "orbbec" for c in hw.cameras):
            raise RuntimeError(f"{hwp.name} lists the Orbbec as a collector camera; the fusion opens it itself as an "
                               f"RGB-D source and one process cannot open it twice. Use a profile without head_depth.")
        self.dm = DeviceManager(hw); self.dm.build(); self.dm.connect_all()
        if self.wrist_camera not in self.dm.cameras: raise RuntimeError(f"no camera {self.wrist_camera!r} in {hwp.name}")
        if self.side not in self.dm.imus: raise RuntimeError(f"no {self.side} IMU in {hwp.name}")

        self.head = OrbbecHeadCamera(calib=HeadRgbdCalibration.load()).open()
        self.rgbd = self.cfg.build_rgbd_provider(head_rgbd=self.teleop.head_rgbd)
        # initialize=True: a LIVE backend must be handed this camera's model before the first frame (see
        # FusedWristCfg.initialize_estimator). The replay stages pass a ready estimator and skip it.
        self.vi = self.cfg.build_vi_provider(estimator=self.estimator, T_H_C=cal.T_H_C, camera_imu_offset_ns=offset_ns,
                                             initialize=True, downscale=self.downscale)
        self.provider = FusedWristPoseProvider(self.cfg.provider_config())
        tap = install_imu_tap(self.dm.imus[self.side])
        self.feeder = LiveVioFeeder(_PushTo(self.vi, self.provider, self), imu_tap=tap,
                                    camera=self.dm.cameras[self.wrist_camera], downscale=self.downscale)
        return self.provider

    def start(self) -> None:
        self._stop.clear()
        self.feeder.start()
        t = threading.Thread(target=self._rgbd_loop, name="fused-rgbd", daemon=True); t.start()
        self._threads.append(t)

    def _rgbd_loop(self) -> None:
        idx = 0
        while not self._stop.is_set():
            try:
                f = self.head.read(timeout_ms=200)
                if f is None: continue
                self.provider.push_rgbd(self.rgbd.push_image(int(f.timestamp_ns), idx, f))
                self.last_rgbd_frame = f
                idx += 1; self.stats["rgbd"] = idx
            except Exception as exc:                     # a chest-camera failure must degrade the anchor, not the arm
                self.stats["rgbd_errors"] += 1; self.stats["last_error"] = repr(exc)
                log.exception("chest RGB-D step failed; the wrist branch keeps running")
                self._stop.wait(0.2)

    def close(self) -> None:
        self._stop.set()
        if self.feeder is not None: self.feeder.stop()
        for t in self._threads: t.join(2.0)
        self._threads.clear()
        if self.head is not None:
            try: self.head.close()
            except Exception: log.exception("closing the chest camera")
        if self.dm is not None: self.dm.close_all()

    def health(self) -> dict:
        return dict(rgbd_frames=self.stats["rgbd"], rgbd_errors=self.stats["rgbd_errors"],
                    last_error=self.stats["last_error"],
                    vi=self.feeder.stats if self.feeder is not None else {},
                    fusion=self.provider.stats() if self.provider is not None else {})

    def __enter__(self): self.open(); self.start(); return self
    def __exit__(self, *exc): self.close()


class _PushTo:
    """Adapter: `LiveVioFeeder` pushes IMU and images into a provider; every produced VI pose also goes to the fusion."""

    def __init__(self, vi, fused, rig=None) -> None: self.vi, self.fused, self.rig = vi, fused, rig

    def push_imu(self, t_ns: int, gyro, accel) -> None: self.vi.push_imu(t_ns, gyro, accel)

    def push_image(self, t_ns: int, frame_index: int, image):
        wp = self.vi.push_image(t_ns, frame_index, image)
        self.fused.push_vi(wp)
        if self.rig is not None: self.rig.last_wrist_frame = image
        return wp
