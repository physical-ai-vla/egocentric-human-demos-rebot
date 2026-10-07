"""Camera identity, fail closed.

Two failures made every camera result today untrustworthy, and both are silent:

  * the AVFoundation enumeration ORDER is not stable (observed changing between calls on 2026-09-11), so an ordinal
    does not name a device; and
  * two cameras of the same model share a product name AND a USB serial out of the box (both Arducams report UC684),
    so no amount of care with the ordinal can tell them apart.

AVFoundation's uniqueID does not help: it is literally the USB locationID plus VID/PID, so it carries no per-unit
identity and changes with the port. The durable fix is to flash distinct product names and serials onto the units; the
job of the code here is to REFUSE to record whenever it cannot prove which physical camera it opened."""
import pytest

# ---------------------------------------------------------------------------------- camera identity (fail closed)
def test_usb_serials_parsed_from_ioreg(monkeypatch):
    import subprocess
    from handumi_collector.devices import camera as cam
    fake = """
    +-o AppleUSB20Hub@01100000  <class AppleUSB20Hub>
        "USB Product Name" = "USB2.0 Hub"
        "USB Serial Number" = "HUB1"
    +-o Arducam 1080P Low Light@03131000  <class IOUSBHostDevice>
        "USB Product Name" = "Arducam 1080P Low Light"
        "USB Serial Number" = "UC684"
    +-o Arducam 1080P Low Light@00133000  <class IOUSBHostDevice>
        "USB Serial Number" = "UC684"
        "USB Product Name" = "Arducam 1080P Low Light"
    """
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"stdout": fake})())
    s = cam.usb_serials_by_product()
    assert s["Arducam 1080P Low Light"] == ["UC684", "UC684"]      # order of the two keys inside a block must not matter
    assert s["USB2.0 Hub"] == ["HUB1"]


def test_identity_refuses_an_ambiguous_ordinal():
    """Two cameras of the same model share a product name AND a USB serial, and the AVFoundation enumeration order is
    not stable — so an ordinal does not name a physical camera. Strict mode must refuse rather than guess."""
    from handumi_collector.config import CameraCfg
    from handumi_collector.devices.camera import verify_identity
    listing = ["Arducam 1080P Low Light", "Orbbec Gemini 336 RGB Camera", "Arducam 1080P Low Light"]
    lax = CameraCfg("left_wrist", "pose_estimation", match_name="Arducam", ordinal=0)
    assert verify_identity(lax, listing)["matches"] == 2          # unchanged behaviour when strict_identity is off
    strict = CameraCfg("left_wrist", "pose_estimation", match_name="Arducam", ordinal=0, strict_identity=True)
    with pytest.raises(RuntimeError, match="cannot name a physical camera"):
        verify_identity(strict, listing)


def test_identity_accepts_a_unique_name_and_checks_the_serial(monkeypatch):
    from handumi_collector.config import CameraCfg
    from handumi_collector.devices import camera as cam
    listing = ["HandUMI Left Wrist", "HandUMI Right Wrist", "Orbbec Gemini 336 RGB Camera"]
    cfg = CameraCfg("left_wrist", "pose_estimation", match_name="HandUMI Left Wrist", ordinal=0, strict_identity=True)
    assert cam.verify_identity(cfg, listing)["matches"] == 1      # unique product name: the ordinal is irrelevant
    monkeypatch.setattr(cam, "usb_serials_by_product", lambda: {"HandUMI Left Wrist": ["HUMLEFT001"]})
    cfg.serial = "HUMLEFT001"
    assert cam.verify_identity(cfg, listing)["serial_ok"] is True
    cfg.serial = "HUMRIGHT001"                                    # the two units swapped on the bench
    with pytest.raises(RuntimeError, match="refusing to record"):
        cam.verify_identity(cfg, listing)


def test_strict_format_refuses_a_substituted_mode():
    """A UVC camera that quietly delivers a different size is how a take ends up attributed to the wrong intrinsics."""
    from ego_collector.camera.capture import CaptureConfig
    assert CaptureConfig().strict_format is False                 # default off: several cameras substitute a near mode
    assert CaptureConfig(strict_format=True).strict_format is True


# ---------------------------------------------------------------------------------- UVC image-control freeze
def test_uvc_preset_verdict_is_the_readback_not_the_exit_status(monkeypatch):
    """A tool that exits 0 has not proven anything. The verdict is what the camera reads back, so a preset that did
    not take is recorded as not applied — never as a success."""
    from handumi_collector.devices import uvc_controls as uc
    state = {"auto-exposure-mode": "8", "exposure-time-abs": "156"}
    def fake_run(argv, **kw):
        ctl = argv[-1]
        if ctl.startswith("--set="):
            k, v = ctl[len("--set="):].split("=", 1)
            state[k] = v if k != "exposure-time-abs" else "999"     # the camera clamps this one
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        k = ctl[len("--get="):]
        return type("R", (), {"returncode": 0, "stdout": f"{k} = {state[k]}\n", "stderr": ""})()
    monkeypatch.setattr(uc.subprocess, "run", fake_run)
    recs = uc.apply_preset("FisheyeCamLeft", {"auto-exposure-mode": 1, "exposure-time-abs": 156}, binary="/x/uvc-util")
    by = {r["control"]: r for r in recs}
    assert by["auto-exposure-mode"]["applied"] is True
    assert by["exposure-time-abs"]["applied"] is False and "999" in by["exposure-time-abs"]["reason"]


def test_uvc_preset_compares_booleans_and_numbers_without_guessing_types(monkeypatch):
    from handumi_collector.devices import uvc_controls as uc
    monkeypatch.setattr(uc.subprocess, "run",
                        lambda argv, **kw: type("R", (), {"returncode": 0, "stderr": "",
                                                          "stdout": "auto-white-balance-temp = false\n"})())
    recs = uc.apply_preset("X", {"auto-white-balance-temp": False}, binary="/x/uvc-util")
    assert recs[0]["applied"] is True
    recs = uc.apply_preset("X", {"auto-white-balance-temp": True}, binary="/x/uvc-util")
    assert recs[0]["applied"] is False


def test_missing_uvc_util_is_reported_not_ignored(monkeypatch):
    """If the profile asks for a frozen preset and the tool is absent, the camera must be refused — recording with
    automatic exposure while the config claims otherwise is the failure this whole path exists to prevent."""
    from handumi_collector.devices import uvc_controls as uc
    monkeypatch.setattr(uc.shutil, "which", lambda _n: None)
    monkeypatch.setattr(uc.os.path, "isfile", lambda _p: False)
    monkeypatch.setattr(uc.Path, "is_file", lambda _self: False)
    monkeypatch.delenv("UVC_UTIL", raising=False)
    with pytest.raises(uc.UvcUtilUnavailable, match="AVFoundation refuses"):
        uc.find_uvc_util()


IOREG_ONE_DOCK = """
  +-o Root  <class IORegistryEntry, id 0x100000100, retain 35>
  +-o AppleT8142USBXHCI@00000000  <class AppleT8142USBXHCI, id 0x100000471, registered>
  | +-o USB2.0 Hub@00100000  <class IOUSBHostDevice, id 0x100000b89, registered>
  | | |   "Device Speed" = 2
  | | +-o Arducam 1080P Low Light@00140000  <class IOUSBHostDevice, id 0x100006ea3, registered>
  | |       "Device Speed" = 2
  | +-o USB3.1 Hub@00200000  <class IOUSBHostDevice, id 0x100000b8f, registered>
  |   |   "Device Speed" = 4
  |   +-o Orbbec Gemini 336@00220000  <class IOUSBHostDevice, id 0x1000073c0, registered>
  |         "Device Speed" = 3
  +-o AppleT8142USBXHCI@01000000  <class AppleT8142USBXHCI, id 0x100000476, registered>
  | +-o USB2.0 Hub@01100000  <class IOUSBHostDevice, id 0x100000cf0, registered>
  |   |   "Device Speed" = 2
  |   +-o FisheyeCamLeft@01110000  <class IOUSBHostDevice, id 0x100006df8, registered>
  |         "Device Speed" = 2
"""

# The wiring as it stands after the 2026-09-15 replug: Orbbec alone on controller 01, both wrists on 00.
IOREG_FIXED = (IOREG_ONE_DOCK.replace("Orbbec Gemini 336@00220000", "Orbbec Gemini 336@01200000")
                             .replace("FisheyeCamLeft@01110000", "FisheyeCamLeft@00111000"))


def _ioreg(monkeypatch, text):
    from handumi_collector.devices import camera as cam
    monkeypatch.setattr(cam.subprocess, "run",
                        lambda argv, **kw: type("R", (), {"returncode": 0, "stderr": "", "stdout": text})())
    return cam


def test_usb_location_reads_controller_port_chain_and_speed(monkeypatch):
    cam = _ioreg(monkeypatch, IOREG_ONE_DOCK)
    assert cam.usb_locations_by_product()["Arducam 1080P Low Light"] == ["00:1.4"]
    assert cam.usb_locations_by_product()["Orbbec Gemini 336"] == ["00:2.2"]
    assert cam.usb_locations_by_product()["FisheyeCamLeft"] == ["01:1.1"]
    speeds = {n: sp for n, _loc, sp in cam._usb_nodes()}
    assert speeds["Orbbec Gemini 336"] == 3 and speeds["Arducam 1080P Low Light"] == 2   # attached to the right node


def test_a_superspeed_camera_sharing_a_dock_uplink_is_named(monkeypatch):
    """The 2026-09-15 failure: right_wrist (00:1.4) and head_depth (00:2.2) recorded 0 frames for a whole 30 s take
    while left_wrist (01:1.1) recorded a flawless 900. A USB-C dock presents a USB2 hub on one root port and a USB3
    hub on another, so the two look unrelated and share one cable. Group by controller, not by port."""
    cam = _ioreg(monkeypatch, IOREG_ONE_DOCK)
    warns = cam.shared_controller_warnings(["FisheyeCamLeft", "Arducam 1080P Low Light", "Orbbec"])
    assert len(warns) == 1
    assert "Orbbec" in warns[0] and "Arducam 1080P Low Light" in warns[0]
    assert "FisheyeCamLeft" not in warns[0]           # alone on controller 01, not implicated


def test_two_high_speed_wrist_cameras_on_one_controller_do_not_warn(monkeypatch):
    """Warn on the pairing that is known to fail, or the warning gets trained away. With the Orbbec moved to its own
    controller, both wrist cameras sit on 00 and all three measured 29.8-30.3 Hz at 1920x1080 for 20 s (2026-09-15).
    Two high-speed UVC cameras coexist; it is the SuperSpeed RGB-D neighbour that starves them."""
    cam = _ioreg(monkeypatch, IOREG_FIXED)
    assert cam.usb_locations_by_product()["Orbbec Gemini 336"] == ["01:2"]
    assert cam.usb_locations_by_product()["FisheyeCamLeft"] == ["00:1.1.1"]
    assert not cam.shared_controller_warnings(["FisheyeCamLeft", "Arducam 1080P Low Light", "Orbbec"])


def test_a_camera_whose_poll_blocks_is_reported_as_stalled():
    """`_run` only errors after 60 consecutive None returns. When the USB transfer cannot be scheduled the read BLOCKS
    instead, the counter never advances, and the device reports connected/error=None while delivering nothing — which
    is exactly how 30 s of empty right_wrist was recorded and kept. Age, not return values, is the evidence."""
    from handumi_collector.devices.base import now_ns
    from handumi_collector.devices.camera import UvcCamera
    from handumi_collector.config import CameraCfg
    c = UvcCamera(CameraCfg("right_wrist", "pose_estimation", backend="uvc", width=1920, height=1080, fps=30))
    c._status.connected = True
    assert c.status().error is None                               # not running yet: nothing to say
    c._status.running = True
    c._status.running_since_ns = now_ns() - int(0.5e9)
    assert c.status().error is None                               # inside the settle window: still nothing to say
    c._status.running_since_ns = now_ns() - int(30e9)
    assert "no frame in" in (c.status().error or "")              # 30 s of silence is a failure, whatever poll() did
    c._status.error = None
    c._status.last_sample_ns = now_ns()
    assert c.status().error is None                               # delivering again -> healthy


class _FlakyCam:
    """A camera that loses the bandwidth lottery `fails` times, then wins it."""

    def __init__(self, fails: int, name: str = "right_wrist") -> None:
        from handumi_collector.config import CameraCfg
        from handumi_collector.devices.base import DeviceStatus
        self.cfg = CameraCfg(name, "pose_estimation", backend="uvc", width=1920, height=1080, fps=30)
        self.name, self.required, self.fails = name, True, fails
        self.opens, self.closes = 0, 0
        self._st = DeviceStatus(name)

    def open(self) -> None:
        self.opens += 1
        self._st.connected = True

    def start(self) -> None:
        self._st.running = True
        self._st.rate_hz = 0.8 if self.opens <= self.fails else 30.0     # the measured stuck rate

    def status(self):
        return self._st

    def close(self) -> None:
        self.closes += 1
        self._st.connected = self._st.running = False


def _fast_retries(monkeypatch):
    from handumi_collector.devices.manager import DeviceManager
    monkeypatch.setattr(DeviceManager, "CAMERA_SETTLE_S", 0.05)
    monkeypatch.setattr(DeviceManager, "CAMERA_RELEASE_S", 0.0)
    return DeviceManager


def test_a_camera_that_loses_the_bandwidth_lottery_is_reopened(monkeypatch):
    """Measured 2026-09-15: three identical open/close cycles gave (right_wrist 0.8 Hz + head_depth no frame at all),
    (left_wrist no frame at all), then (all three nominal). The loser never recovers — right_wrist held exactly
    0.8 Hz for a full 20 s — so the reservation has to be handed back and requested again."""
    from handumi_collector.config import HardwareCfg
    DeviceManager = _fast_retries(monkeypatch)
    m = DeviceManager(HardwareCfg())
    cam = _FlakyCam(fails=2)
    ok, why = m._bring_up_camera(cam)
    assert ok and why == ""
    assert cam.opens == 3 and cam.closes == 2          # released the reservation between attempts
    assert cam.status().detail["open_attempts"] == 3   # provenance: this take needed three draws


def test_one_frame_is_not_proof_a_camera_is_delivering(monkeypatch):
    """`CameraCapture.open()` only proves ONE frame arrived, which the 0.8 Hz camera managed — so the old check let
    the failure straight through and 30 s were recorded with an empty stream. Sustained rate is the evidence."""
    from handumi_collector.config import HardwareCfg
    DeviceManager = _fast_retries(monkeypatch)
    m = DeviceManager(HardwareCfg())
    cam = _FlakyCam(fails=99)
    ok, why = m._bring_up_camera(cam)
    assert not ok
    assert "0.8 Hz" in why and "3 attempts" in why
    assert cam.opens == 3 and cam.closes == 3          # gave the bandwidth back rather than holding a dead handle


def test_the_orbbec_is_opened_before_the_uvc_cameras():
    """Opening the Orbbec kills UVC cameras that are already streaming. Measured 2026-09-15 across three 30 s takes
    with each camera on its own host controller: the two runs where the Orbbec needed a second open attempt each lost
    a wrist camera at the exact moment it opened (last frame ~0.5 s after), and the one run where the Orbbec came up
    first try recorded 901/901/902. Both wrists had already passed their own sustained-rate check. Ordering is the
    fix, not more retries."""
    from handumi_collector.config import CameraCfg, HardwareCfg
    from handumi_collector.devices.manager import DeviceManager
    hw = HardwareCfg(cameras=[CameraCfg("left_wrist", "pose_estimation", backend="uvc", match_name="L", fps=30),
                              CameraCfg("right_wrist", "pose_estimation", backend="uvc", match_name="R", fps=30),
                              CameraCfg("head_depth", "aux_depth", backend="orbbec", fps=30)])
    m = DeviceManager(hw); m.build()
    order = [d.name for d in m.bring_up_order()]
    assert order[0] == "head_depth", order
    assert order.index("head_depth") < order.index("left_wrist") < order.index("right_wrist")
    assert sorted(order) == sorted(d.name for d in m.all_devices())    # ordering only, nothing dropped or doubled


def test_a_listing_position_is_not_an_opencv_index(monkeypatch):
    """Measured 2026-09-15: ffmpeg listed the three cameras in exactly the REVERSE of the order OpenCV opens them,
    and the listing order changed again minutes later. `list_video_devices()` claims its order IS the OpenCV index
    order; it is not. Opening `match_name: FisheyeCamLeft` at its listing position opened the centre C922, and a
    30 s take was recorded and kept with the wrong picture in left_wrist.mp4.

    probe_index_map darkens each camera BY NAME through uvc-util and takes the index whose picture follows, so the
    listing's order stops mattering."""
    import numpy as np
    from handumi_collector.devices import camera as cam

    real = {"C922 Pro Stream Webcam": 0, "Arducam 1080P Low Light": 1, "FisheyeCamLeft": 2}
    state = {"dark": None}

    class _Cap:
        def __init__(self, i): self.i = i
        def isOpened(self): return self.i < 3
        def set(self, *a): return True
        def read(self):
            lit = state["dark"] is None or real.get(state["dark"]) != self.i
            return True, np.full((4, 4, 3), 200 if lit else 20, np.uint8)
        def release(self): pass

    import sys
    monkeypatch.setitem(sys.modules, "cv2", type("m", (), {"VideoCapture": _Cap,
                                                          "CAP_PROP_FRAME_WIDTH": 3, "CAP_PROP_FRAME_HEIGHT": 4}))
    fake_uc = type("uc", (), {
        "find_uvc_util": staticmethod(lambda: "/x/uvc-util"),
        "get_control": staticmethod(lambda n, c, binary=None: "8" if "mode" in c else "400"),
        "apply_preset": staticmethod(lambda n, ctl, binary=None: state.__setitem__(
            "dark", n if str(ctl.get("exposure-time-abs")) == "1" else None) or []),
    })
    import handumi_collector.devices as _devs
    monkeypatch.setattr(_devs, "uvc_controls", fake_uc, raising=False)
    monkeypatch.setitem(sys.modules, "handumi_collector.devices.uvc_controls", fake_uc)
    monkeypatch.setattr(cam.time, "sleep", lambda *_a: None)

    got = cam.probe_index_map(list(real), max_index=6)
    assert got == real, got                       # the measured map, not the listing order
