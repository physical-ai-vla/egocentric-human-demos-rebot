"""Camera devices. UvcCamera wraps ego_collector.camera.capture.CameraCapture (threaded grab + monotonic stamps);
OrbbecCamera wraps ego_collector.camera.orbbec_capture.OrbbecCapture (aux RGB-D, main-thread poll pumped by a thread here).
Device identity: AVFoundation listing (ffmpeg) by product name + ordinal → OpenCV index (same enumeration order on macOS)."""
from __future__ import annotations
import logging
import re
import subprocess
import threading
import time
from .base import CameraFrame, DeviceStatus, RateMeter, SampleBuffer, now_ns
from ..config import CameraCfg

log = logging.getLogger("handumi.camera")


def list_video_devices() -> list[str]:
    """Names of video devices in OpenCV/AVFoundation index order (macOS via ffmpeg). Empty list if unavailable."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                             capture_output=True, text=True, timeout=10).stderr
    except Exception:
        return []
    names: dict[int, str] = {}
    sec = False
    for line in out.splitlines():
        if "video devices" in line: sec = True; continue
        if "audio devices" in line: break
        m = re.search(r"\[(\d+)\]\s+(.*)$", line) if sec else None
        if m: names[int(m.group(1))] = m.group(2).strip()
    return [names[i] for i in sorted(names)]


def usb_serials_by_product() -> dict[str, list[str]]:
    """{USB product name: [serial, ...]} from ioreg. Empty when ioreg is unavailable.

    Needed because neither OpenCV nor this ffmpeg build can select or report a capture device by serial: AVFoundation's
    uniqueID is literally the USB locationID plus VID/PID (verified 2026-09-11 — it carries no per-unit identity and
    changes with the port), and the avfoundation demuxer here offers only `-video_device_index`. So the serial is read
    out of band and used to VERIFY what the name+ordinal lookup selected, never to select it."""
    import json  # noqa: F401  (kept local; this helper is only called at open())
    try:
        out = subprocess.run(["ioreg", "-p", "IOUSB", "-w0", "-l"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return {}
    serials: dict[str, list[str]] = {}
    cur: dict[str, str] = {}
    for line in out.splitlines():
        if "+-o " in line:
            cur = {}
            continue
        m = re.search(r'"(USB Product Name|USB Serial Number)" = "([^"]*)"', line)
        if not m:
            continue
        cur[m.group(1)] = m.group(2)
        if "USB Product Name" in cur and "USB Serial Number" in cur:
            serials.setdefault(cur["USB Product Name"], []).append(cur["USB Serial Number"])
            cur = {}
    return serials


def _usb_nodes() -> list[tuple[str, str, int]]:
    """[(product name, "00:1.4", device speed), ...] from ioreg.

    Apple's locationID carries the controller in its top byte and one nibble per hub port below it, and ioreg prints
    it as the @suffix on the node name, so `Arducam 1080P Low Light@00140000` reads as controller 00, root port 1,
    port 4. Speed is AppleUSB's code: 2 = high (480M), 3 = super (5G), 4 = super+ (10G). A node's properties follow
    its own `+-o` line and precede any child's, so the most recent name is the right one to attach them to."""
    try:
        out = subprocess.run(["ioreg", "-p", "IOUSB", "-w0", "-l"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    nodes: list[tuple[str, str, int]] = []
    cur: list | None = None
    for line in out.splitlines():
        m = re.search(r"\+-o (.+?)@([0-9a-fA-F]{8})\s", line)
        if m:
            loc = m.group(2).lower()
            ports = [c for c in loc[2:] if c != "0"]
            cur = [m.group(1).strip(), loc[:2] + (":" + ".".join(ports) if ports else ""), -1]
            nodes.append(cur)
            continue
        sp = re.search(r'"Device Speed" = (\d+)', line)
        if sp and cur is not None and cur[2] < 0:
            cur[2] = int(sp.group(1))
    return [tuple(n) for n in nodes]


def _mean_brightness(cap, n: int = 4) -> float:
    """Mean grey level of the next few frames of an open capture, for prove_binding."""
    import numpy as np
    vals = []
    for _ in range(n):
        try: f = cap.poll()
        except Exception: f = None
        if f is not None and getattr(f, "image", None) is not None:
            vals.append(float(np.asarray(f.image).mean()))
        time.sleep(0.05)
    return float(np.median(vals)) if vals else 0.0


def probe_index_map(names: list[str], *, max_index: int = 6, probe_size=(640, 480), settle_s: float = 0.8) -> dict:
    """Measure which OpenCV index each product NAME actually opens on, by darkening the name and watching the pixels.

    `list_video_devices()` claims to return names "in OpenCV/AVFoundation index order" and that claim is false on this
    machine: ffmpeg reported [0] FisheyeCamLeft while OpenCV index 0 was the C922, and the ffmpeg order itself changed
    between two calls minutes apart. Opening the index a name resolved to therefore opens whatever happens to sit
    there, which is how a take was recorded on 2026-09-15 with the centre camera's picture inside left_wrist.mp4.

    uvc-util addresses a camera BY NAME, so this drives each name's picture dark in turn and watches every open index:
    the one that actually darkens IS that camera. Decided by largest relative drop rather than a fixed threshold -- a
    real hit measured only 235->151 (36%) because a darkened camera is not black -- and the winner must be clearly
    ahead of the runner-up or the name is left unmapped.

    [2026-09-18] The darkening signal is BRIGHTNESS swung max->min, not exposure. Brightness is kept because it is
    the one control every camera here implements with a real range and it darkens from whatever exposure state the
    camera happens to be in; an exposure probe silently proves nothing while the camera sits in auto mode.

    The earlier note here said exposure-time-abs was dead on the wrist units (0c45:0261). It is not. FisheyeCamLeft was
    in auto-exposure-mode 8 (its device DEFAULT), where the AE loop holds the level and writes to exposure-time-abs are
    stored and ignored -- which is exactly the "reads back correctly, picture does not move" symptom. Driven to mode 1
    and verified by read-back, BOTH units sweep cleanly: exposure 1 -> mean 184/sharpness 381, exposure 330 -> mean
    243/sharpness 5. Controls also survive cv2.VideoCapture.open(), so applying them before open() is safe.

    Probes at low resolution so three simultaneous captures stay cheap. Returns {name: index} for what it could prove."""
    import cv2
    import numpy as np
    from . import uvc_controls as uc
    try:
        binary = uc.find_uvc_util()
    except Exception as exc:
        log.warning("cannot probe camera identity: %s", exc)
        return {}
    caps: dict[int, object] = {}
    restore: dict[str, dict] = {}
    try:
        # Freeze exposure on EVERY camera under test first. Otherwise the others' auto-exposure drifts during the
        # second it takes to darken one of them, and that drift shows up as a response: a real hit at 27 % was once
        # rejected because an unrelated index happened to wander 14 % in the same window.
        #
        # [2026-09-19] This used to write a hardcoded `exposure-time-abs: 400` and then, in the finally below, hand
        # every camera back to `auto-exposure-mode: 8` + auto white balance. Both were wrong in the same way: an
        # identity probe has no business choosing the imaging settings. A camera whose profile carries no
        # `uvc_controls` was left in an auto state nobody asked for, and one that does carry them only recovered
        # because `_apply_uvc_preset` happened to run afterwards. Now the probe records what it found, imposes
        # nothing but the mode it needs to hold the level still, and puts back exactly what was there.
        for n in names:
            try:
                before = {c: uc.get_control(n, c, binary=binary).strip()
                          for c in ("auto-exposure-mode", "auto-white-balance-temp")}
                # mode 1 only: manual exposure holds the level without deciding what that level should be.
                uc.apply_preset(n, {"auto-exposure-mode": 1}, binary=binary)
                restore[n] = before
            except Exception:
                pass
        time.sleep(settle_s)
        for i in range(max_index):
            c = cv2.VideoCapture(i)
            if not c.isOpened():
                c.release(); break
            c.set(cv2.CAP_PROP_FRAME_WIDTH, probe_size[0]); c.set(cv2.CAP_PROP_FRAME_HEIGHT, probe_size[1])
            for _ in range(4): c.read()
            caps[i] = c

        def snap() -> dict:
            out = {}
            for i, c in caps.items():
                v = []
                for _ in range(3):
                    ok, f = c.read()
                    if ok: v.append(float(np.asarray(f).mean()))
                out[i] = float(np.median(v)) if v else 0.0
            return out

        found: dict[str, int] = {}
        for name in names:
            try:
                mode = uc.get_control(name, "auto-exposure-mode", binary=binary).strip()
                prev_b = uc.get_control(name, "brightness", binary=binary).strip()
                lo, hi = uc.control_range(name, "brightness", binary=binary)
            except Exception as exc:
                log.warning("camera %r: cannot be addressed by name (%s); leaving it unmapped", name, str(exc)[:80])
                continue
            if lo is None or hi is None:
                log.warning("camera %r: reports no brightness range, so it cannot be identified this way", name)
                continue
            try:
                uc.apply_preset(name, {"auto-exposure-mode": 1, "brightness": hi}, binary=binary)
                time.sleep(settle_s); base = snap()
                uc.apply_preset(name, {"brightness": lo}, binary=binary)
                time.sleep(settle_s); dark = snap()
            finally:
                try: uc.apply_preset(name, {"brightness": prev_b, "auto-exposure-mode": mode}, binary=binary)
                except Exception: pass
            drops = {i: (base[i] - dark[i]) / base[i] if base[i] > 5 else 0.0 for i in caps}
            ranked = sorted(drops.items(), key=lambda kv: kv[1], reverse=True)
            best, second = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0.0))
            # Absolute margin, not a ratio: at small drops a ratio test is brittle (27 % lost to a 14 % neighbour
            # because 2x14 > 27), while the margin says the same thing without exploding as the numbers shrink.
            if best[1] >= 0.15 and (best[1] - second[1]) >= 0.08:
                found[name] = best[0]
                log.info("camera %r is OpenCV index %d (drop %.0f%%, next %.0f%%)", name, best[0], 100 * best[1], 100 * second[1])
            else:
                log.warning("camera %r: no index responded clearly (%s)", name,
                            ", ".join(f"{i}:{100 * d:.0f}%" for i, d in ranked))
        return found
    finally:
        for c in caps.values():
            try: c.release()
            except Exception: pass
        for n, before in restore.items():     # put back exactly what was there; impose nothing
            try: uc.apply_preset(n, before, binary=binary)
            except Exception: pass


def prove_binding(product_name: str, sample, *, drop: float = 0.15, settle_s: float = 0.9) -> dict:
    """Prove an OPEN capture really is the camera whose product name is `product_name`.

    The bug this closes: `resolve_camera_index` looks a product name up in the ffmpeg/AVFoundation listing and returns
    an INDEX, and OpenCV is then opened on that index. That is only correct while the enumeration order has not moved
    since the listing was taken, and the order does move -- this module's own docstring says so. With three cameras
    attached on 2026-09-15 the streams bound to the wrong devices and nothing noticed: a take was recorded with the
    centre camera's picture inside left_wrist.mp4.

    uvc-util addresses a camera BY NAME. So drive that name's BRIGHTNESS to its minimum and look at these pixels: if
    this capture is that camera it goes dark, and if it does not, the index belongs to someone else. Returns a record
    with `proved` True/False, or None when the test could not run (no uvc-util, no brightness range, picture already
    dark), in which case the caller keeps the camera rather than refusing to record.

    [2026-09-18] Brightness, not exposure: a camera parked in auto-exposure mode stores exposure-time-abs and ignores
    it, so an exposure-based proof silently proves nothing. Brightness darkens from any AE state. See `probe_index_map`."""
    from . import uvc_controls as uc
    rec = dict(product_name=product_name, proved=None, reason="", baseline=None, darkened=None)
    try:
        binary = uc.find_uvc_util()
        prev_mode = uc.get_control(product_name, "auto-exposure-mode", binary=binary).strip()
        prev_b = uc.get_control(product_name, "brightness", binary=binary).strip()
        lo, hi = uc.control_range(product_name, "brightness", binary=binary)
    except Exception as exc:
        rec["reason"] = f"cannot address {product_name!r} by name: {str(exc)[:100]}"
        return rec
    if lo is None or hi is None:
        rec["reason"] = f"{product_name!r} reports no brightness range"
        return rec
    try:
        uc.apply_preset(product_name, {"auto-exposure-mode": 1, "brightness": hi}, binary=binary)
        time.sleep(settle_s)
        rec["baseline"] = base = sample()
        if not base or base <= 5:
            rec["reason"] = f"baseline brightness {base} too low to test"
            return rec
        uc.apply_preset(product_name, {"brightness": lo}, binary=binary)
        time.sleep(settle_s)
        rec["darkened"] = dark = sample()
        rec["proved"] = bool(dark < base * (1.0 - drop))
        if not rec["proved"]:
            rec["reason"] = (f"darkening {product_name!r} moved this stream only {base:.0f}->{dark:.0f}: "
                             f"the index it opened on is a different camera")
    finally:
        try: uc.apply_preset(product_name, {"brightness": prev_b, "auto-exposure-mode": prev_mode}, binary=binary)
        except Exception: pass
    return rec


def usb_locations_by_product() -> dict[str, list[str]]:
    """{USB product name: ["00:1.4", ...]} -- which controller and hub-port chain each device hangs off."""
    found: dict[str, list[str]] = {}
    for name, loc, _speed in _usb_nodes():
        found.setdefault(name, []).append(loc)
    return found


SUPERSPEED = 3


def shared_controller_warnings(names: list[str]) -> list[str]:
    """Name cameras that must contend for one host controller WITH a SuperSpeed camera, before a take records zeros.

    Warn on the pairing that is known to fail, not on every pairing, or the warning gets trained away. Measured on
    this rig 2026-09-15, both at 1920x1080:

      * right_wrist + the Orbbec on controller 00 through one USB-C dock -> BOTH recorded 0 frames for 30 s, while
        left_wrist on controller 01 recorded a flawless 900 at 30.0 fps.
      * the two wrist cameras alone on controller 00, Orbbec moved to 01 -> all three deliver 29.8-30.3 Hz for 20 s.

    So two high-speed UVC cameras coexist on one controller and a SuperSpeed RGB-D camera does not coexist with
    anything: its transfers are what starve the neighbour. A dock also hides the sharing, presenting a USB2 hub on one
    root port and a USB3 hub on another, so group by controller and never by port."""
    nodes = _usb_nodes()
    by_ctrl: dict[str, list[tuple[str, str, int]]] = {}
    for want in names:
        hit = next((n for n in nodes if want and want.lower() in n[0].lower()), None)
        if hit:
            by_ctrl.setdefault(hit[1].split(":")[0], []).append((want, hit[1], hit[2]))
    out = []
    for ctrl, cams in by_ctrl.items():
        fast = [c for c in cams if c[2] >= SUPERSPEED]
        if len(cams) > 1 and fast:
            out.append(f"camera {fast[0][0]} ({fast[0][1]}) is SuperSpeed and shares USB controller {ctrl} with "
                       f"{', '.join(c[0] for c in cams if c is not fast[0])} -- one uplink, and this pairing has "
                       f"recorded zero frames before; move it to a port on another controller")
    return out


def _log_identity(cfg: CameraCfg, info: dict) -> None:
    """Print EXPECTED vs OBSERVED before anything opens, so a swapped pair is visible in the log even on
    the runs where strict_identity is off and the collector would otherwise record silently."""
    exp = [f"logical_side: {cfg.name}", f"product: {cfg.match_name!r}", f"ordinal: {cfg.ordinal}"]
    if cfg.serial:
        exp.append(f"serial: {cfg.serial!r}")
    obs = [f"product: {info.get('listing_name')!r}", f"matches: {info.get('matches')}",
           f"serials: {info.get('observed_serials') or '-'}", f"usb_paths: {info.get('usb_paths') or '-'}"]
    ok = info.get("listing_name") is not None and info.get("matches") == 1
    if cfg.serial and info.get("serial_ok") is False:
        ok = False
    log.info("[%s]\n  EXPECTED:\n    %s\n  OBSERVED:\n    %s\n  IDENTITY: %s",
             cfg.name.upper(), "\n    ".join(exp), "\n    ".join(obs),
             "PASS" if ok else ("FAIL — recording blocked" if cfg.strict_identity else
                                "AMBIGUOUS — strict_identity is off, recording NOT blocked"))


def verify_identity(cfg: CameraCfg, listing: list[str]) -> dict:
    """Fail closed before opening: prove the name+ordinal lookup can only mean one physical camera.

    Two independent things go wrong otherwise, and both are silent. The AVFoundation enumeration ORDER is not stable
    (observed changing between calls on 2026-09-11), so an ordinal does not name a device; and two cameras of the same
    model share a product name AND a USB serial out of the box, so no amount of care in the ordinal fixes it. With
    `strict_identity` the collector refuses rather than record LEFT data into the RIGHT stream."""
    hits = [n for n in listing if cfg.match_name and cfg.match_name.lower() in n.lower()]
    chosen = hits[cfg.ordinal] if len(hits) > cfg.ordinal else None
    info = dict(match_name=cfg.match_name, matches=len(hits), listing_name=chosen,
                expected_serial=cfg.serial, serial_ok=None, strict=bool(cfg.strict_identity))
    # The serial goes into the episode as PROVENANCE whether or not it is verified. Both HandUMI Arducams report the
    # factory serial "UC684" — a model string, not a unit id — so it can never be the L/R discriminator; but recording
    # what the hardware actually said is what lets a future reader tell which physical unit produced a take.
    if chosen:
        info["observed_serials"] = usb_serials_by_product().get(chosen, [])
        # The USB port path is the only key here that separates two units of the same model: the factory
        # serial is a model string ("UC684" on both HandUMI Arducams) and AVFoundation's uniqueID is that
        # same locationID with VID/PID glued on. It identifies the PORT, not the camera -- which is exactly
        # what makes it useful once each unit has a dedicated port, and exactly why the port must then be
        # part of the contract rather than left to chance.
        info["usb_paths"] = [pth for nm, pth, _ in _usb_nodes() if nm == chosen]
    _log_identity(cfg, info)
    if not cfg.strict_identity:
        return info
    if len(hits) != 1:
        raise RuntimeError(f"camera {cfg.name}: {len(hits)} devices match {cfg.match_name!r} — an ordinal cannot name a "
                           f"physical camera (AVFoundation order is not stable). Give each unit a unique product name.")
    if cfg.serial:
        serials = info.get("observed_serials") or usb_serials_by_product().get(hits[0], [])
        info["serial_ok"] = cfg.serial in serials
        if not info["serial_ok"]:
            raise RuntimeError(f"camera {cfg.name}: expected serial {cfg.serial!r} for {hits[0]!r}, USB reports "
                               f"{serials or 'nothing'} — refusing to record with an unverified camera identity")
    return info


def resolve_camera_index(cfg: CameraCfg, listing: list[str] | None = None) -> int:
    if cfg.index is not None:
        return int(cfg.index)
    listing = list_video_devices() if listing is None else listing
    hits = [i for i, n in enumerate(listing) if cfg.match_name and cfg.match_name.lower() in n.lower()]
    if len(hits) <= cfg.ordinal:
        raise RuntimeError(f"camera {cfg.name}: no device #{cfg.ordinal} matching {cfg.match_name!r} in {listing}")
    return hits[cfg.ordinal]


class _BaseCamera:
    def __init__(self, cfg: CameraCfg, *, buffer_s: float = 4.0) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.required = cfg.required
        self.buffer: SampleBuffer[CameraFrame] = SampleBuffer(int(cfg.fps * buffer_s))
        self._status = DeviceStatus(cfg.name)
        self._rate = RateMeter(60)
        self._latest: CameraFrame | None = None
        self._lock = threading.Lock()
        self.index: int | None = None

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    def _publish(self, frame: CameraFrame) -> None:
        with self._lock:
            self._latest = frame
        self.buffer.append(frame)
        self._rate.tick(frame.capture_ns)
        self._status.last_sample_ns = frame.capture_ns

    STALL_S = 1.5

    def status(self) -> DeviceStatus:
        """Judge a camera by frame age, not by what poll() returned.

        `_run` only raises an error after 60 consecutive `None`s, which assumes poll() returns. When the USB transfer
        cannot be scheduled, OpenCV's AVFoundation read BLOCKS instead, so the counter never advances and the device
        reports `connected, error=None` while delivering nothing: right_wrist recorded 0 frames across a whole 30 s
        take that way on 2026-09-15 and no gate saw it. Silence for longer than the settle window is the failure."""
        self._status.rate_hz = self._rate.hz()
        if self._status.running and self._status.error is None:
            waited, quiet_ms = self._status.running_for_s, self._status.age_ms
            quiet = waited if quiet_ms is None else quiet_ms / 1e3
            if waited > self.STALL_S and quiet > self.STALL_S:
                self._status.error = f"no frame in {quiet:.1f}s"
        return self._status


class UvcCamera(_BaseCamera):
    def __init__(self, cfg: CameraCfg, **kw) -> None:
        super().__init__(cfg, **kw)
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def open(self, listing: list[str] | None = None, index_map: dict | None = None) -> None:
        from ego_collector.camera.capture import CameraCapture, CaptureConfig
        listing = list_video_devices() if listing is None else listing
        identity = verify_identity(self.cfg, listing)
        uvc = self._apply_uvc_preset(listing)
        want = identity.get("listing_name") or self.cfg.match_name or ""
        listed = resolve_camera_index(self.cfg, listing)
        # The listing's position is NOT the OpenCV index -- measured 2026-09-15, ffmpeg reported the three cameras in
        # exactly the reverse of the order OpenCV opens them, so `match_name: FisheyeCamLeft` opened the centre C922
        # and a whole take was recorded with the wrong picture in left_wrist.mp4. Use the measured map when the
        # manager has one, and fall back to the listing position when it has not.
        self.index = int(index_map[want]) if (index_map and want in index_map) else listed
        cap = CameraCapture(CaptureConfig(index=self.index, width=self.cfg.width, height=self.cfg.height, fps=self.cfg.fps,
                                          fourcc=self.cfg.fourcc, threaded=False,
                                          strict_format=self.cfg.strict_format,
                                          exposure_auto=self.cfg.exposure_auto, exposure=self.cfg.exposure,
                                          gain=self.cfg.gain, white_balance_auto=self.cfg.white_balance_auto,
                                          white_balance=self.cfg.white_balance))
        cap.open()
        proof = prove_binding(want, lambda: _mean_brightness(cap)) if want else dict(proved=None, reason="no match_name")
        if proof.get("proved") is False:
            try: cap.close()
            except Exception: pass
            raise RuntimeError(f"camera {self.name}: index {self.index} is not {want!r} — {proof.get('reason','')}")
        if self.index != listed:
            log.info("camera %s: %r is OpenCV index %d, not the %d its position in the listing suggested",
                     self.name, want, self.index, listed)
        identity["binding"] = dict(listing_position=listed, opened_index=self.index, proved=proof.get("proved"),
                                   reason=proof.get("reason", ""))
        self._cap = cap
        self._status.connected = True; self._status.error = None
        self._status.detail.update(index=self.index, actual=dict(cap.actual), identity=identity)
        if uvc is not None:
            self._status.detail["uvc_controls"] = uvc
        if cap.controls:                      # what the camera ACTUALLY honoured, straight into the episode provenance
            self._status.detail["controls"] = cap.controls
            bad = [c["control"] for c in cap.controls if not c["applied"]]
            if bad: log.warning("camera %s: controls not applied: %s", self.name, bad)

    def _apply_uvc_preset(self, listing: list[str] | None = None) -> list[dict] | None:
        """Freeze the camera's image controls BEFORE OpenCV opens it, and refuse the camera if any did not take.

        Fail closed, because the whole point is that the calibration take and the data takes share one frozen setting:
        a preset that exists only in a config file is worse than no preset, since it looks deliberate.

        uvc-util selects by the camera's FULL product name; `match_name` is a substring meant for the device listing.
        They coincide for the wrist cameras, so a head `match_name: "C922"` was the first to hit it -- and, because
        this path fails closed, it refused the camera outright rather than recording with the preset off."""
        if not self.cfg.uvc_controls:
            return None
        from .uvc_controls import apply_preset, find_uvc_util
        want = self.cfg.match_name or self.name
        listing = list_video_devices() if listing is None else listing
        name = next((n for n in listing if want and want.lower() in n.lower()), want)
        binary = find_uvc_util(self.cfg.uvc_util)          # raises UvcUtilUnavailable -> camera refused
        recs = apply_preset(name, dict(self.cfg.uvc_controls), binary=binary)
        bad = [r for r in recs if not r["applied"]]
        if bad:
            raise RuntimeError(f"camera {self.name}: UVC preset not applied: " +
                               "; ".join(f"{r['control']} ({r['reason']})" for r in bad))
        log.info("camera %s: UVC preset applied and verified (%s)", self.name,
                 ", ".join(f"{r['control']}={r['readback']}" for r in recs))
        return recs

    def start(self) -> None:
        if self._thread: return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True); self._thread.start()
        self._status.running = True; self._status.running_since_ns = now_ns()

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                f = self._cap.poll()
            except Exception as exc:
                self._status.error = str(exc); f = None
            if f is None:
                failures += 1
                if failures >= 60: self._status.error = self._status.error or "camera returns no frames"
                time.sleep(0.002); continue
            failures = 0; self._status.error = None
            self._publish(CameraFrame(self.name, f.index, f.timestamp_ns, f.image))

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(2.0); self._thread = None
        self._status.running = False

    def close(self) -> None:
        self.stop()
        if self._cap is not None:
            try: self._cap.close()
            except Exception: pass
        self._cap = None; self._status.connected = False


class OrbbecCamera(UvcCamera):
    """Aux RGB-D. Same threading as UvcCamera; frames carry depth. Needs sudo on macOS (pyorbbecsdk UVC access)."""

    def open(self, listing: list[str] | None = None) -> None:
        from ego_collector.camera.capture import CaptureConfig
        from ego_collector.camera.orbbec_capture import OrbbecCapture
        cap = OrbbecCapture(CaptureConfig(index=0, width=self.cfg.width, height=self.cfg.height, fps=self.cfg.fps))
        cap.open()
        self._cap = cap
        self.intrinsics = getattr(cap, "intrinsics", None); self.depth_scale = getattr(cap, "depth_scale", None)
        self._status.connected = True; self._status.error = None
        self._status.detail.update(actual=dict(cap.actual), intrinsics=self.intrinsics, depth_scale=self.depth_scale)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                f = self._cap.poll()
            except Exception as exc:
                self._status.error = str(exc); f = None
            if f is None:
                time.sleep(0.002); continue
            self._status.error = None
            self._publish(CameraFrame(self.name, f.index, f.timestamp_ns, f.image, getattr(f, "depth", None)))
