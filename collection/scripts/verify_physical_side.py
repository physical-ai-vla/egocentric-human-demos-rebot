#!/usr/bin/env python3
"""Confirm WHICH PHYSICAL WRIST each named wrist camera is mounted on, by covering one lens and measuring which stream
goes dark.

Both units report USB serial UC684, so the serial cannot tell them apart; identity rests entirely on the product names
(`FisheyeCamLeft` vs `Arducam 1080P Low Light`), which do differ and which the config matches on. That guarantees the
calibration file and the camera always stay paired -- but NOT that the camera named `left_wrist` is the one strapped to
the left wrist. If they are swapped, every downstream artifact is consistently and invisibly mislabelled: the pose is
right, the hand it is attributed to is wrong, and no residual anywhere will show it.

This is deliberately a measurement and not a question. "Did the left preview go dark?" asks the operator to judge the
thing being tested while looking at a window labelled with the answer. Covering a lens and asking which stream's
luminance collapsed does not.

Writes `physical_side_verified` into both fisheye calibration files on PASS.

    python scripts/verify_physical_side.py --cover left
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DROP_FRACTION = 0.45      # a covered lens loses far more than this; a shadow or a passing hand does not
MAX_OTHER_DROP = 0.15     # the uncovered stream must stay put, or the two are not independent
SETTLE_S, MEASURE_S = 1.5, 2.0


def _open(side: str, hardware: str):
    from handumi_collector.config import load_config
    from handumi_collector.devices.camera import list_video_devices, resolve_camera_index, verify_identity

    stream = f"{side}_wrist"
    cfg = next((c for c in load_config(hardware=hardware).hardware.cameras if c.name == stream), None)
    if cfg is None:
        raise SystemExit(f"profile {hardware!r} has no camera {stream!r}")
    listing = list_video_devices()
    identity = verify_identity(cfg, listing)          # fails closed on an ambiguous name
    cap = cv2.VideoCapture(resolve_camera_index(cfg, listing), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(cfg.fourcc or "MJPG")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    return cap, cfg, identity


def _luma(cap, seconds: float) -> float:
    vals, t0 = [], time.time()
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if ok and f is not None:
            vals.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
    if not vals:
        raise SystemExit("camera returned no frames")
    return float(np.median(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", choices=["left", "right"], default="left",
                    help="which wrist you will physically cover (default: left)")
    ap.add_argument("--hardware", default="handumi_rgbd")
    ap.add_argument("--no-write", action="store_true", help="measure only; do not touch the calibration files")
    a = ap.parse_args()
    other = "right" if a.cover == "left" else "left"

    caps = {}
    for s in ("left", "right"):
        cap, cfg, ident = _open(s, a.hardware)
        caps[s] = dict(cap=cap, cfg=cfg, name=ident.get("listing_name"))
        print(f"  {s}_wrist -> {ident.get('listing_name')!r} (match_name {cfg.match_name!r})")

    try:
        print(f"\nUNCOVER both lenses. Measuring baseline in {SETTLE_S:.0f} s...")
        time.sleep(SETTLE_S)
        base = {s: _luma(caps[s]["cap"], MEASURE_S) for s in ("left", "right")}
        print(f"  baseline luminance  left {base['left']:.1f}   right {base['right']:.1f}")
        if min(base.values()) < 8:
            raise SystemExit("a stream is already almost black — uncover both lenses and re-run")

        print(f"\nNow COMPLETELY COVER the lens of the unit you believe is the {a.cover.upper()} wrist.")
        input("  press ENTER once it is covered and held still... ")
        cov = {s: _luma(caps[s]["cap"], MEASURE_S) for s in ("left", "right")}
        print(f"  covered luminance   left {cov['left']:.1f}   right {cov['right']:.1f}")
    finally:
        for c in caps.values():
            c["cap"].release()

    drop = {s: (base[s] - cov[s]) / max(base[s], 1e-6) for s in ("left", "right")}
    print(f"\n  luminance drop      left {drop['left']:+.1%}   right {drop['right']:+.1%}")

    if drop[a.cover] >= DROP_FRACTION and drop[other] <= MAX_OTHER_DROP:
        verdict, ok = f"PASS — the camera configured as {a.cover}_wrist is physically on the {a.cover} wrist", True
    elif drop[other] >= DROP_FRACTION and drop[a.cover] <= MAX_OTHER_DROP:
        verdict, ok = (f"FAIL — THE WRISTS ARE SWAPPED. You covered the {a.cover} unit and {other}_wrist went dark. "
                       f"Either swap the physical mounts or swap match_name between left_wrist and right_wrist in the "
                       f"hardware profile. Do NOT record data until this is resolved."), False
    else:
        verdict, ok = (f"INCONCLUSIVE — left {drop['left']:+.1%}, right {drop['right']:+.1%}; need one stream to drop "
                       f">={DROP_FRACTION:.0%} while the other stays <={MAX_OTHER_DROP:.0%}. Cover the lens completely "
                       f"and keep the scene still, then re-run."), False
    print("\n" + verdict)

    if not ok or a.no_write:
        sys.exit(0 if ok else 2)

    import yaml
    stamp = dict(physical_side_verified=True,
                 physical_side_verified_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                 physical_side_verified_by="scripts/verify_physical_side.py --cover " + a.cover,
                 physical_side_evidence=dict(covered=a.cover,
                                             baseline_luma={k: round(v, 2) for k, v in base.items()},
                                             covered_luma={k: round(v, 2) for k, v in cov.items()},
                                             drop_fraction={k: round(v, 4) for k, v in drop.items()},
                                             product_names={s: caps[s]["name"] for s in caps}))
    for s in ("left", "right"):
        p = Path(f"configs/calibration/fisheye_{s}_v001.yaml")
        if not p.exists():
            print(f"  (no {p}; skipped)")
            continue
        d = yaml.safe_load(p.read_text()); d.update(stamp)
        p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
        print(f"  stamped {p}")


if __name__ == "__main__":
    main()
