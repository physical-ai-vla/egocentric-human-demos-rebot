"""Freeze a UVC camera's image controls before recording, by talking to the camera instead of to AVFoundation.

macOS's AVFoundation path does not expose exposure/gain/white-balance: every `cv2.VideoCapture.set()` for them is
refused (measured — set() returns False, read-back 0). That is a limitation of the PATH, not of the hardware: the same
Arducam advertises the controls over UVC and honours them.

    uvc-util --select-by-name=<product name> --set=<control>=<value>   then --get to verify
              -> then OpenCV opens the device and records

Verified on the real camera: writes apply, read-backs match, and the settings survive and persist unchanged while
OpenCV holds the stream open (mode/exposure identical before, during and after a 9 s capture, with stable brightness).

Selection is BY PRODUCT NAME. The two HandUMI Arducams share VID/PID (0x0c45:0x0261) and the factory serial, so a
vendor/product selector would be ambiguous; the unique product names are the only safe key, and location-id binding is
deliberately not used.

This is a thin wrapper around one external tool. It is not a capture backend and it never opens the camera."""
from __future__ import annotations
import os
import re
import shutil
import subprocess
from pathlib import Path

# uvc-util has no Homebrew formula; build it from https://github.com/jtfrey/uvc-util:
#   gcc -o uvc-util -framework IOKit -framework Foundation uvc-util.m UVCController.m UVCType.m UVCValue.m
SEARCH = ("uvc-util",)
FALLBACK_PATHS = ("/usr/local/bin/uvc-util", "/opt/homebrew/bin/uvc-util", str(Path.home() / "bin" / "uvc-util"))


class UvcUtilUnavailable(RuntimeError):
    """The binary is not installed. Callers report it; they never pretend the controls were applied."""


def find_uvc_util(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("UVC_UTIL")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    for name in SEARCH:
        p = shutil.which(name)
        if p:
            return p
    for p in FALLBACK_PATHS:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    raise UvcUtilUnavailable(
        "uvc-util not found. Build it from https://github.com/jtfrey/uvc-util and put it on PATH (or set UVC_UTIL / the "
        "profile's `uvc_util`). It is the only way to set exposure/gain/white balance on macOS — AVFoundation refuses.")


def _run(binary: str, name: str, *args, timeout: float = 15.0) -> str:
    cp = subprocess.run([binary, f"--select-by-name={name}", *args], capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        raise RuntimeError(f"uvc-util {' '.join(args)} on {name!r} failed: {(cp.stderr or cp.stdout).strip()[:200]}")
    return cp.stdout


def list_controls(name: str, *, binary: str | None = None) -> list[str]:
    out = _run(binary or find_uvc_util(), name, "--list-controls")
    return [ln.strip() for ln in out.splitlines() if ln.startswith("  ") and ln.strip()]


def get_control(name: str, control: str, *, binary: str | None = None) -> str:
    out = _run(binary or find_uvc_util(), name, f"--get={control}")
    m = re.search(rf"{re.escape(control)}\s*=\s*(.+)", out)
    return m.group(1).strip() if m else out.strip()


def control_range(name: str, control: str, *, binary: str | None = None) -> tuple[float | None, float | None]:
    """The (minimum, maximum) a control admits, as the camera itself reports them.

    Needed because the identity probe drives a control to its extremes, and the extremes differ per device: brightness
    is -64..64 on the wrist units and 0..255 on the C922. Returns (None, None) when the device reports no range."""
    try:
        out = _run(binary or find_uvc_util(), name, f"--show-control={control}")
    except Exception:
        return None, None
    def pick(word):
        m = re.search(rf"{word}:\s*(-?[0-9.]+)", out)
        return float(m.group(1)) if m else None
    return pick("minimum"), pick("maximum")


def _same(want, got: str) -> bool:
    """Compare a requested value to the read-back string without guessing at types."""
    g = got.strip()
    if isinstance(want, bool):
        return g.lower() == ("true" if want else "false")
    try:
        return abs(float(g) - float(want)) < 1e-9
    except ValueError:
        return g == str(want)


def apply_preset(name: str, controls: dict, *, binary: str | None = None) -> list[dict]:
    """Apply `controls` (uvc-util control name -> value) in order and verify each by read-back.

    Order matters and is the caller's: an auto-mode flag must be cleared before the manual value it governs, or the
    camera overwrites it immediately. Returns one record per control; `applied` is the READ-BACK verdict, never the
    exit status, and a failure is recorded rather than raised so the caller decides whether to refuse the episode."""
    b = binary or find_uvc_util()
    out: list[dict] = []
    for ctl, want in controls.items():
        rec = dict(control=ctl, requested=want, readback=None, applied=False, reason="")
        v = "true" if want is True else "false" if want is False else str(want)
        try:
            _run(b, name, f"--set={ctl}={v}")
            rec["readback"] = get_control(name, ctl, binary=b)
            rec["applied"] = _same(want, rec["readback"])
            if not rec["applied"]:
                rec["reason"] = f"read back {rec['readback']!r}, asked for {want!r}"
        except Exception as exc:
            rec["reason"] = str(exc)[:200]
        out.append(rec)
    return out
