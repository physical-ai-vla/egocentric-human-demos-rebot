"""Builds every configured device, connects them, and reports connectivity for the UI's recording gate."""
from __future__ import annotations
import itertools
import logging
import time
from dataclasses import dataclass, field
from .base import DeviceStatus
from .camera import OrbbecCamera, UvcCamera, list_video_devices, probe_index_map, shared_controller_warnings
from .feetech_gripper import FeetechGripper
from .mock import MockCamera, MockGripper, MockImu
from .teensy_imu import TeensyImu
from ..config import HardwareCfg
from ..calibration import load_gripper_calibration

log = logging.getLogger("handumi.devices")


@dataclass
class DeviceManager:
    hw: HardwareCfg
    cameras: dict = field(default_factory=dict)     # name -> camera device
    imus: dict = field(default_factory=dict)        # side -> imu device
    grippers: dict = field(default_factory=dict)    # side -> gripper device
    video_listing: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    usb_warnings: list[str] = field(default_factory=list)   # cameras sharing one host controller
    index_map: dict = field(default_factory=dict)          # product name -> measured OpenCV index
    gripper_calibration_version: str | None = None

    def apply_gripper_calibration(self, which: str | None = None) -> str | None:
        ver, sides = load_gripper_calibration(which or self.hw.gripper_calibration)
        self.gripper_calibration_version = ver
        for side, g in self.grippers.items():
            c = sides.get(side)
            if c:                      # no versioned file yet -> keep whatever the hardware yaml / test config provided
                g.set_calibration(c["ticks_closed"], c["ticks_open"])
        return ver

    def identity(self) -> dict[str, str]:
        """What the UI must show before START: which physical device is LEFT / RIGHT."""
        out = {}
        for side, d in self.imus.items():
            out[f"{side.upper()} IMU"] = f"{getattr(d, 'serial_number', None) or '?'} @ {getattr(d, 'port', None) or '?'}"
        for side, d in self.grippers.items():
            out[f"{side.upper()} GRIP"] = f"{d.cfg.port} id {d.cfg.servo_id}" if d.cfg.backend != "mock" else "mock"
        for n, c in self.cameras.items():
            out[n] = f"index {c.index}" if c.index is not None else "not open"
        return out

    def build(self) -> None:
        for c in self.hw.cameras:
            self.cameras[c.name] = MockCamera(c) if c.backend == "mock" else OrbbecCamera(c) if c.backend == "orbbec" else UvcCamera(c)
        for i in self.hw.imus:
            self.imus[i.side] = MockImu(i) if i.backend == "mock" else TeensyImu(i)
        for g in self.hw.grippers:
            self.grippers[g.side] = MockGripper(g) if g.backend == "mock" else FeetechGripper(g)
        self.apply_gripper_calibration()

    def all_devices(self) -> list:
        return [*self.cameras.values(), *self.imus.values(), *self.grippers.values()]

    def connect_all(self) -> None:
        """Open + start every device; failures are recorded per device (required ones block recording in the UI)."""
        self._check_imu_identity()
        self._warn_shared_usb_controllers()
        if any(c.backend == "uvc" for c in self.hw.cameras):
            self.video_listing = list_video_devices()
            # Measure where each camera actually is before opening anything. The listing's order is not OpenCV's --
            # on 2026-09-15 it was exactly reversed -- so its positions cannot be used as indices.
            wanted = []
            for c in self.hw.cameras:
                if c.backend != "uvc" or not c.match_name: continue
                hit = next((n for n in self.video_listing if c.match_name.lower() in n.lower()), None)
                if hit and hit not in wanted: wanted.append(hit)
            self.index_map = probe_index_map(wanted) if wanted else {}
        self.errors.clear()
        used_ports: list[str] = []
        for d in self.bring_up_order():
            try:
                if d.name in self.cameras and getattr(d.cfg, "backend", "") != "mock":
                    ok, why = self._bring_up_camera(d)
                    if not ok:
                        raise RuntimeError(why)
                    continue
                if isinstance(d, TeensyImu):
                    d.open(exclude=tuple(used_ports)); used_ports.append(d.port)
                else:
                    d.open()
                d.start()
            except Exception as exc:
                self.errors[d.name] = str(exc)
                log.warning("device %s failed: %s", d.name, exc)

    def bring_up_order(self) -> list:
        """Devices in the order they must be opened: the Orbbec first, then everything else.

        Opening the Orbbec DISTURBS UVC cameras that are already streaming. Measured 2026-09-15, three 30 s takes,
        each camera alone on its own host controller:

          run 1  Orbbec open 12:48:23.4, retried once -> right_wrist's last frame ~12:48:24, ends at 2.9 Hz
          run 3  Orbbec open 12:49:31.3, retried once -> left_wrist's last frame ~12:49:31, ends at 13.7 Hz
          run 2  Orbbec up first try, no retry           -> all three at nominal, 901/901/902 frames

        The wrist cameras were opened first and each PASSED its own sustained-rate check; they died afterwards, as
        the Orbbec opened. Perfect correlation across every take today, including the two lost this morning. So the
        repair is ordering, not more retries: let the disruptive open happen -- and retry if it needs to -- while no
        UVC camera is streaming yet, and bring the wrists up against a bus that has stopped moving."""
        disruptive = [c for n, c in self.cameras.items() if getattr(c.cfg, "backend", "") == "orbbec"]
        rest = [d for d in self.all_devices() if d not in disruptive]
        return [*disruptive, *rest]

    def _warn_shared_usb_controllers(self) -> None:
        """Say out loud when two cameras are on one host controller. Advisory only -- it may well work."""
        names = [c.match_name or ("Orbbec" if c.backend == "orbbec" else c.name)
                 for c in self.hw.cameras if c.backend != "mock"]
        for w in shared_controller_warnings([n for n in names if n]):
            log.warning("%s", w)
            self.usb_warnings.append(w)

    CAMERA_ATTEMPTS = 3            # draws at the bandwidth lottery before a camera is called dead
    CAMERA_SETTLE_S = 3.0          # a camera that wins its reservation is at nominal well inside this
    CAMERA_RELEASE_S = 1.5         # let the controller actually reclaim the reservation before asking again

    def _bring_up_camera(self, d) -> tuple[bool, str]:
        """Open one camera and prove it is DELIVERING, and if it is not, hand the bandwidth back and ask again.

        Cameras race for USB isochronous bandwidth at open(): the winner keeps its reservation and the loser sticks
        at a few frames a second FOREVER. Measured 2026-09-15 -- three open/close cycles, identical wiring, no
        recording -- gave (right_wrist 0.8 Hz + head_depth never a single frame), then (left_wrist never a frame),
        then (all three at nominal). The loser never recovers on its own: right_wrist sat at exactly 0.8 Hz for a
        full 20 s, so a longer settle window cannot fix this, and which camera loses is random.

        `CameraCapture.open()` only proves ONE frame arrived, which the 0.8 Hz camera managed, so the failure walks
        straight past it. Judge by sustained rate instead, and on failure close the device -- releasing the
        reservation -- before the next attempt, which then renegotiates against the load the already-running cameras
        are really placing on the bus."""
        nominal = getattr(d.cfg, "fps", 0) or 30
        last = ""
        for attempt in range(1, self.CAMERA_ATTEMPTS + 1):
            try:
                if isinstance(d, UvcCamera) and not isinstance(d, OrbbecCamera):
                    d.open(self.video_listing, self.index_map)
                else:
                    d.open()
                d.start()
            except Exception as exc:
                last = str(exc)
            else:
                t0 = time.monotonic()
                while time.monotonic() - t0 < self.CAMERA_SETTLE_S:
                    if d.status().rate_hz >= nominal * 0.8:
                        d.status().detail["open_attempts"] = attempt
                        if attempt > 1:
                            log.info("camera %s came up on attempt %d/%d", d.name, attempt, self.CAMERA_ATTEMPTS)
                        return True, ""
                    time.sleep(0.1)
                last = f"opened but reached only {d.status().rate_hz:.1f} Hz of {nominal:g}"
            log.warning("camera %s attempt %d/%d: %s", d.name, attempt, self.CAMERA_ATTEMPTS, last)
            try: d.close()
            except Exception: pass
            time.sleep(self.CAMERA_RELEASE_S)
        return False, f"{last} after {self.CAMERA_ATTEMPTS} attempts"

    def _check_imu_identity(self) -> None:
        """Two sides may not name the same board. Serial matching is by substring and ignores the taken-port list, so a
        duplicated or truncated serial would have both sides open one port and read half a stream each, under both names."""
        cfgs = [i for i in self.hw.imus if i.backend == "teensy" and (i.serial_number or "").strip()]
        for a, b in itertools.combinations(cfgs, 2):
            x, y = a.serial_number.strip(), b.serial_number.strip()
            if x in y or y in x:
                raise RuntimeError(f"imu {a.side} and imu {b.side} both claim serial {x!r}/{y!r} — LEFT/RIGHT identity would be "
                                   f"undefined; give each side the full, distinct USB serial (`imu_logger --list`)")

    def statuses(self) -> dict[str, DeviceStatus]:
        out = {}
        for d in self.all_devices():
            st = d.status()
            if d.name in self.errors and not st.connected:
                st.error = self.errors[d.name]
            out[d.name] = st
        return out

    def required_ok(self, *, min_fraction_of_nominal: float = 0.5, settle_s: float = 2.0) -> tuple[bool, list[str]]:
        """Ready means delivering, not merely opened.

        Three cameras once opened cleanly and reported connected while two of them were producing 0.8 and 0.0 frames a
        second, and thirty seconds were recorded and kept before anything noticed. A device that answered open() and
        then went quiet is not a device you can record with."""
        missing = []
        for d in self.all_devices():
            if not d.required: continue
            st = d.status()
            if not st.connected:
                missing.append(d.name); continue
            nominal = getattr(getattr(d, "cfg", None), "fps", None) or getattr(getattr(d, "cfg", None), "rate_hz", None)
            if nominal and st.running_for_s >= settle_s and st.rate_hz < nominal * min_fraction_of_nominal:
                missing.append(f"{d.name} (open but {st.rate_hz:.1f} Hz, expected ~{nominal})")
        return (not missing, missing)

    def close_all(self) -> None:
        for d in self.all_devices():
            try: d.close()
            except Exception: pass
