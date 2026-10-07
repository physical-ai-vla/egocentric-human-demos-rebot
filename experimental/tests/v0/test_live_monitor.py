"""Live monitors + preview composition on synthetic frames (no window needed)."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ego_collector.camera.capture import CaptureConfig, Frame
from ego_collector.recording.live_monitor import LiveTagMonitor, VisibilityCounter
from ego_collector.recording.recorder import Recorder
from ego_collector.tracking.detector import OpenCVTagDetector
from synth import INTR, head_pose, render, scene_faces, world_map, wrist_pose


class FakeCam:
    frames_dropped = 0

    def __init__(self, image):
        self._img = image
        self.i = 0

    def latest(self):
        self.i += 1
        return Frame(index=self.i, timestamp_ns=time.monotonic_ns(), image=self._img)


def _scene(with_left=True):
    wm = world_map()
    wr = {20: (0.035, wrist_pose(0.18, 0.33, 0.14))}
    if with_left:
        wr[10] = (0.035, wrist_pose(-0.15, 0.30, 0.10))
    return render(scene_faces(wm, head_pose(), wr), INTR)


def test_tag_monitor_counts_visibility_and_roles():
    cam = FakeCam(_scene())
    mon = LiveTagMonitor(cam, OpenCVTagDetector("tag36h11"), left_id=10, right_id=20, world_ids=(100, 101, 102, 103), wrist_size_m=0.035, focal_px=INTR.fx, rate_hz=200)
    mon.start()
    mon.start_counting()
    time.sleep(0.6)
    res = mon.latest()
    assert res is not None and res.left_visible and res.right_visible and res.world_visible >= 2
    roles = {res.roles.get(d.tag_id) for d in res.detections}
    assert {"left", "right", "world"} <= roles
    left = next(d for d in res.detections if d.tag_id == 10)
    dist = mon.distance_m(left, 0.035)
    assert dist is not None and 0.3 < dist < 1.0  # camera ~0.6-0.7 m from the hand
    ratios = mon.stop_counting()
    mon.stop()
    assert ratios["frames"] >= 3 and ratios["all_required"] == 1.0 and ratios["left"] == 1.0


def test_tag_monitor_all_required_drops_when_a_wrist_is_missing():
    mon = LiveTagMonitor(FakeCam(_scene(with_left=False)), OpenCVTagDetector("tag36h11"), left_id=10, right_id=20, world_ids=(), rate_hz=200)
    mon.start()
    mon.start_counting()
    time.sleep(0.5)
    r = mon.stop_counting()
    mon.stop()
    assert r["left"] == 0.0 and r["right"] == 1.0 and r["all_required"] == 0.0
    c = VisibilityCounter()
    c.add(left=True, right=True, world=False, need_world=False)
    c.add(left=True, right=True, world=False, need_world=True)
    assert c.ratios()["all_required"] == 0.5


def test_compose_preview_renders_panel_and_overlays(tmp_path):
    img = _scene()
    cam = FakeCam(img)
    mon = LiveTagMonitor(cam, OpenCVTagDetector("tag36h11"), left_id=10, right_id=20, world_ids=(100, 101, 102, 103), wrist_size_m=0.035, focal_px=INTR.fx, rate_hz=200)
    mon.start()
    time.sleep(0.4)
    rec = Recorder(raw_root=tmp_path, capture=CaptureConfig(), task="tracking_motion", instruction="move hands left/right", tag_monitor=mon, preview_scale=0.5)
    frame = Frame(index=cam.i, timestamp_ns=0, image=img)
    out = rec.compose_preview(frame, 29.7, cam)
    mon.stop()
    assert out.shape == (540, 960, 3)
    cv2.imwrite(str(tmp_path / "preview.png"), out)
    # overlay drew coloured polylines: the frame is no longer grayscale-only
    b, g, r = out[..., 0].astype(int), out[..., 1].astype(int), out[..., 2].astype(int)
    assert (np.abs(b - g) > 40).sum() > 500


def test_tag_monitor_live_wrist_pose_and_aperture():
    from ego_collector.tracking.transforms import T_from_xyz_rpy
    from ego_collector.tracking.wrist_pose import WristExtrinsics

    wm = world_map()
    ap = 0.06
    faces = {
        20: (0.05, wrist_pose(0.18, 0.33, 0.14)),
        21: (0.016, T_from_xyz_rpy([0.05, 0.30, 0.12], [np.radians(50), 0, 0])),
        22: (0.016, T_from_xyz_rpy([0.05 + ap, 0.30, 0.12], [np.radians(50), 0, 0])),
    }
    img = render(scene_faces(wm, head_pose(), faces), INTR)
    wr = WristExtrinsics.default("tag36h11", with_fingers=True)
    mon = LiveTagMonitor(
        FakeCam(img), OpenCVTagDetector("tag36h11"), left_id=10, right_id=20,
        world_ids=(100, 101, 102, 103), rate_hz=200, extrinsics=wr, intr=INTR, world_map=wm,
    )
    mon.start()
    time.sleep(0.6)
    res = mon.latest()
    mon.stop()
    assert res is not None and "right" in res.hands
    h = res.hands["right"]
    assert h["frame"] == "world" and h["thumb"] and h["index"]
    assert abs(h["aperture_mm"] - ap * 1000) < 5.0
    np.testing.assert_allclose(h["xyz"], [0.18, 0.33, 0.14], atol=0.02)
    assert {"right_thumb", "right_index"} <= set(res.roles.values())
