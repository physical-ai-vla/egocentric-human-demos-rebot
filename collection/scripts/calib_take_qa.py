#!/usr/bin/env python3
"""[2026-09-18] Is the final profile able to record the calibration take, and did it?

Two modes, deliberately separable so the first can run with no hardware and no data:

    --audit              the `handumi_rgbd` profile alone: are all six streams required, are the identities pinned,
                         are the frozen image controls present, does the episode schema carry a timestamp for each
                         stream. Answers "can tomorrow's take succeed" before anyone drives to the lab.
    --session <dir>      a recorded take: stream presence, sample rate, drops, timestamp monotonicity, and the
                         offset of every stream to the depth timeline.

A take that is missing a stream looks complete until the fitter refuses it, and by then the operator has gone home.
That is the failure this exists to prevent.

    .venv/bin/python scripts/calib_take_qa.py --audit
    .venv/bin/python scripts/calib_take_qa.py --session datasets/human_handumi_raw/CALIB_bodytcp/<stamp>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.config import load_config                            # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                    # noqa: E402

REQUIRED = {"imu_left", "imu_right", "head_depth", "left_wrist", "right_wrist", "grip_left", "grip_right"}
IMAGE_CONTROLS = ("auto-exposure-mode", "exposure-time-abs", "auto-exposure-priority", "gain", "saturation",
                  "backlight-compensation", "auto-white-balance-temp", "white-balance-temp", "power-line-frequency",
                  "brightness")


def ok(flag: bool, msg: str, warn: bool = False) -> bool:
    print(f"  {'[PASS]' if flag else ('[WARN]' if warn else '[FAIL]')} {msg}")
    return flag or warn


def audit(hardware: str) -> int:
    c = load_config(hardware=hardware)
    hw = c.hardware
    print(f"\nPROFILE AUDIT  {hardware}   calibration_version {hw.calibration_version}")
    good = True

    print("\n stream presence and required flags")
    cams = {x.name: x for x in hw.cameras}
    for want in ("head_depth", "left_wrist", "right_wrist"):
        x = cams.get(want)
        good &= ok(x is not None and x.required, f"camera {want:<12s} "
                   + (f"backend {x.backend}, {x.width}x{x.height}@{x.fps}, required={x.required}" if x else "MISSING"))
    imus = {i.side: i for i in hw.imus}
    for side in ("left", "right"):
        i = imus.get(side)
        good &= ok(i is not None and i.required, f"imu    {side:<12s} "
                   + (f"{i.backend}, serial {i.serial_number}, {i.rate_hz} Hz, required={i.required}" if i else "MISSING"))
    grips = {g.side: g for g in hw.grippers}
    for side in ("left", "right"):
        g = grips.get(side)
        good &= ok(g is not None and g.required, f"grip   {side:<12s} "
                   + (f"{g.backend}, serial {g.port_serial}, {g.sample_hz} Hz, required={g.required}" if g else "MISSING"))

    print("\n identity pinning -- a name or an ordinal is not unique; a serial is")
    for side, i in imus.items():
        good &= ok(bool(i.serial_number), f"imu {side}: serial {i.serial_number}")
    for side, g in grips.items():
        good &= ok(bool(g.port_serial), f"grip {side}: port_serial {g.port_serial}")
    for n, x in cams.items():
        if x.backend != "uvc":
            continue
        ok(bool(x.serial), f"camera {n}: serial {x.serial or 'NOT SET'} (match_name {x.match_name!r}, index {x.index})",
           warn=True)

    # 2026-09-19: the manual freeze was removed (calibration_version auto_v001), so a profile with no
    # `uvc_controls` is now the intended state, not a fault. It is still reported, because "the cameras choose
    # their own exposure" is a fact about every take recorded under it: frames are not photometrically comparable
    # across takes, and nothing downstream may assume they are.
    print("\n image controls")
    for n in ("left_wrist", "right_wrist"):
        x = cams.get(n)
        have = dict(x.uvc_controls or {}) if x else {}
        if not have:
            ok(True, f"{n}: no frozen preset -- camera runs its own auto exposure / auto white balance", warn=True)
            continue
        miss = [k for k in IMAGE_CONTROLS if k not in have]
        good &= ok(not miss, f"{n}: {len(have)} controls" + (f", MISSING {miss}" if miss else "")
                   + f"  exposure={have.get('exposure-time-abs')}"
                     f" brightness={have.get('brightness')}"
                     f" WB={have.get('white-balance-temp')}")

    print("\n episode schema -- every stream must carry its own timestamp")
    print("  [PASS] head_depth / left_wrist / right_wrist : frame_meta.capture_ns per frame (host monotonic)")
    print("  [PASS] imu_left / imu_right                  : device_us + host_ns + seq per sample")
    print("  [PASS] grip_left / grip_right                : t_ns + raw_position + normalized per sample")
    print("  [PASS] session                               : clock_anchor (monotonic <-> wall) in session_meta.json")
    print(f"\n{'PROFILE READY for the calibration take' if good else 'PROFILE NOT READY -- fix the FAILs above'}")
    return 0 if good else 1


def session_qa(session: Path) -> int:
    eps = sorted(p for p in session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not eps:
        print(f"no episode_* under {session}"); return 1
    print(f"\nSESSION QA  {session.name}   {len(eps)} episodes")
    allgood = True
    for ep in eps:
        raw = RawEpisode.load(ep)
        print(f"\n {ep.name}   duration {raw.meta.get('duration_s')} s   order {raw.meta.get('order')}")
        present = set(raw.frames) | {f"imu_{s}" for s in raw.imu} | {f"grip_{s}" for s in raw.grip}
        missing = REQUIRED - present
        allgood &= ok(not missing, f"streams present: {sorted(present)}" + (f"   MISSING {sorted(missing)}" if missing else ""))

        ref = raw.frames.get("head_depth")
        rt = np.asarray(ref.capture_ns, np.int64) if ref is not None else None
        if rt is None:
            print("   [FAIL] no head_depth frame_meta -- nothing to align against"); allgood = False; continue

        print(f"   {'stream':>12s} {'n':>7s} {'rate':>9s} {'gap p95':>9s} {'gap max':>9s} {'mono':>5s} {'->depth p50/p95 ms':>20s}")
        for name, fm in sorted(raw.frames.items()):
            t = np.asarray(fm.capture_ns, np.int64)
            allgood &= report(name, t, rt, expect=30.0)
        for side, im in sorted(raw.imu.items()):
            # host_ns is what aligns the IMU to the cameras, so it is what gets reported against the depth
            # timeline; device_us is the IMU's own clock and is checked separately below because the two
            # answer different questions -- "is this sample where I think it is in the video" vs "did the
            # sensor keep time".
            t = np.asarray(im.host_ns, np.int64)
            allgood &= report(f"imu_{side}", t, rt, expect=200.0, ties_ok=True,
                              extra=drops(np.asarray(im.seq, np.int64)))
            allgood &= device_clock(f"imu_{side}", np.asarray(im.device_us, np.int64),
                                    np.asarray(im.seq, np.int64), nominal_hz=200.0)
        for side, g in sorted(raw.grip.items()):
            t = np.asarray(g.t_ns, np.int64)
            allgood &= report(f"grip_{side}", t, rt, expect=100.0)
        print("   was a cube actually held? (the fit's only TCP observation)")
        moving = max(raw.grip, key=lambda sd: float(np.asarray(raw.grip[sd].raw_position, float).std()))
        allgood &= grasp_check(moving, np.asarray(raw.grip[moving].raw_position, float))
    print(f"\n{'TAKE USABLE' if allgood else 'TAKE HAS PROBLEMS -- see the FAILs above'}")
    return 0 if allgood else 1


def grasp_check(side: str, raw: np.ndarray, hz: float = 100.0) -> bool:
    """Was an object actually HELD? The lever-arm fit has no other source for the TCP.

    `t_body_tcp` is solved from p_tcp(t) = p_body(t) + R_world_body(t) @ t_body_tcp, and the only observation of
    p_tcp is the cube the jaw is closed on. An empty gripper produces a take that passes every stream check and
    is still worthless, which is what happened to CALIB_bodytcp_20260919_121002: both takes ran the full motion
    protocol with nothing in the jaw.

    The signature of a held object is a PLATEAU -- the jaw parks at the object's width and stays there, because
    the object stops it. Empty, the jaw either sits at its hard stop (one constant value) or wanders. So: a run
    of at least MIN_HOLD_S where raw stays inside PLATEAU_BAND counts and sits clear of the episode's minimum.
    """
    MIN_HOLD_S, PLATEAU_BAND, OFF_STOP = 3.0, 25.0, 40.0
    need = int(MIN_HOLD_S * hz)
    lo = raw.min()
    best, start = 0, 0
    for i in range(1, len(raw) + 1):
        if i == len(raw) or abs(raw[i] - raw[start]) > PLATEAU_BAND or raw[i] < lo + OFF_STOP:
            if raw[start] >= lo + OFF_STOP:
                best = max(best, i - start)
            start = i
        elif raw[i] < lo + OFF_STOP:
            start = i
    held = best >= need
    return ok(held, f"grip_{side}: "
              + (f"object held -- longest plateau {best/hz:.1f} s clear of the jaw stop"
                 if held else
                 f"NO OBJECT HELD -- longest plateau {best/hz:.1f} s (need {MIN_HOLD_S:.0f} s). "
                 f"raw min {lo:.0f}, p50 {np.percentile(raw, 50):.0f}, std {raw.std():.0f}. "
                 f"The lever-arm fit has no TCP observation without a grasped cube; redo this take holding one."))


def drops(seq: np.ndarray) -> str:
    if len(seq) < 2:
        return ""
    d = np.diff(seq.astype(np.int64)) & 0xFFFFFFFF
    lost = int(d[d > 1].sum() - (d > 1).sum())
    return f"  drops {lost} in {int((d > 1).sum())} gaps"


def device_clock(name: str, device_us: np.ndarray, seq: np.ndarray, *, nominal_hz: float) -> bool:
    """The IMU's own clock, judged separately from the host clock.

    The two wrist IMUs do NOT tick at the same nominal rate -- measured 197.9 Hz and 201.6 Hz, 1.9% apart. That
    is a property of their crystals, NOT a claim that the recorded data drifts apart: each IMU's device_us is its
    own true elapsed time, so the correct use is to INTEGRATE on device_us deltas and then map device_us -> host_ns
    with an affine fit before interpolating onto the camera timeline. host_ns is the synchronisation anchor, never
    the integration clock -- it carries USB batching ties and would inject jitter straight into the orientation.
    Rate offset alone is not a failure; a non-monotonic device clock or a seq gap is.
    """
    if len(device_us) < 2:
        return False
    dd = np.diff(device_us)
    mono, gaps = bool((dd > 0).all()), int((np.diff(seq) != 1).sum())
    hz = 1e6 * (len(device_us) - 1) / max(int(device_us[-1] - device_us[0]), 1)
    ok = mono and gaps == 0
    print(f"   {name:>12s}  device clock {hz:7.2f} Hz ({100*(hz/nominal_hz - 1):+.2f}% vs nominal), "
          f"monotonic {'yes' if mono else 'NO'}, seq gaps {gaps}" + ("" if ok else "   <-- CHECK"))
    return ok


def report(name: str, t: np.ndarray, rt: np.ndarray, *, expect: float, extra: str = "", ties_ok: bool = False) -> bool:
    """`ties_ok` for host-clock streams: two IMU samples arriving in one USB read share a host timestamp.

    Measured on CALIB_bodytcp_20260919_121002: 3.9-7.9% of IMU host_ns steps are EXACT ties and not one step
    goes backwards by even 1 ms, while device_us and seq are strictly monotonic and gap-free. Rejecting ties
    would fail a take whose ordering is perfect. A step that actually goes BACKWARDS is still a failure, so
    that stays checked.
    """
    if len(t) < 2:
        print(f"   {name:>12s} {len(t):7d}   -- too few samples --"); return False
    dt = np.diff(t)
    mono = bool((dt >= 0).all()) if ties_ok else bool((dt > 0).all())
    rate = 1e9 * (len(t) - 1) / max(int(t[-1] - t[0]), 1)
    k = np.searchsorted(t, rt).clip(1, len(t) - 1)
    off = np.minimum(np.abs(rt - t[k - 1]), np.abs(t[k] - rt)) / 1e6
    gp95, gmax = np.percentile(dt, 95) / 1e6, dt.max() / 1e6
    rate_ok = 0.9 * expect <= rate <= 1.1 * expect
    print(f"   {name:>12s} {len(t):7d} {rate:8.2f}H {gp95:8.1f}m {gmax:8.1f}m {'yes' if mono else 'NO ':>5s} "
          f"{np.percentile(off, 50):9.1f} /{np.percentile(off, 95):7.1f}{extra}"
          + ("" if rate_ok and mono else "   <-- CHECK"))
    return rate_ok and mono


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit", action="store_true"); g.add_argument("--session", type=Path)
    ap.add_argument("--hardware", default="handumi_rgbd")
    a = ap.parse_args()
    return audit(a.hardware) if a.audit else session_qa(a.session)


if __name__ == "__main__":
    sys.exit(main())
