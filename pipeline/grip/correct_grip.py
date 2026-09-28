#!/usr/bin/env python3
"""Corrected HandUMI gripper channel: raw ticks -> physical jaw width in metres.

The stored `normalized` cannot be used directly. Its `ticks_open` sits beyond the mechanical stop in
almost every session (right reaches only ~70% of the calibrated span; left's direction even flips between
sessions and 140419-left's metadata is simply wrong), so the same physical opening maps to different
normalized values across sessions. Everything is therefore re-derived from raw:

    raw tick
      -> direction from session_meta's sign(ticks_open - ticks_closed)   (140419-left: regression-recovered)
      -> endpoints re-anchored to what the hardware actually travelled
      -> physical open fraction in [0,1]
      -> width_m = fraction * APERTURE_M

Anchors are the 90th percentile of the per-episode extremes rather than the absolute extreme: the jaws
hit a hard stop, so per-episode extremes cluster there, and one odd episode should not define the scale.
That single odd episode is exactly what broke 135136-left (its absolute 'closed' frame shows a visible
gap on video).

APERTURE_M is a locked physical constant, video-validated: closed / 50 mm cube grasp / fully-open frames
were read off the wrist camera for all 5 sessions x 2 arms, and the median over the 8 combos with valid
anchors is 67.5 mm. Uncertainty ~±5 mm; it is a single global scale on this one channel, so a later
caliper measurement can be applied by changing this number alone.
"""
import argparse, glob, json, pathlib, collections
import numpy as np

APERTURE_M = 0.0675
APERTURE_UNCERTAINTY_M = 0.005
RAW = pathlib.Path.home() / "ego_collector/datasets/human_handumi_raw/Hpilot"
# 140419-left: session_meta says (6943, 5389) but regressing stored normalized on raw returns this pair
# with R^2 = 1.0, so the metadata is wrong and the recovered pair is what the recording actually used.
RECOVERED = {("20260916_140419", "left"): (2847, 1293)}
EXCLUDED_FROM_APERTURE = {
    ("20260917_135136", "left"): "absolute closed anchor was an outlier episode; video shows a gap",
    ("20260916_140419", "left"): "no closing plateau detected (7 episodes only)",
}


def session_meta_ticks():
    out = {}
    for sd in sorted(RAW.glob("Hpilot_*")):
        p = sd / "session_meta.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text()).get("devices", {})
        out[sd.name.replace("Hpilot_", "")] = {
            s: (d.get(f"gripper_{s}", {}).get("ticks_closed"), d.get(f"gripper_{s}", {}).get("ticks_open"))
            for s in ("left", "right")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", default=str(pathlib.Path.home() / ".claude/jobs/e649aef3/tmp/handumi_grip"))
    ap.add_argument("--out", default=str(pathlib.Path.home() / ".claude/jobs/e649aef3/tmp/handumi_grip_corrected"))
    a = ap.parse_args()
    src, out = pathlib.Path(a.grip), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = session_meta_ticks()

    by = collections.defaultdict(list)
    for f in sorted(glob.glob(str(src / "*.npz"))):
        by["_".join(pathlib.Path(f).stem.split("_")[:2])].append(f)

    anchors, table = {}, []
    for sess in sorted(by):
        for side in ("left", "right"):
            c, o = RECOVERED.get((sess, side), meta[sess][side])
            sign = 1 if o > c else -1
            ex_c, ex_o = [], []
            for f in by[sess]:
                r = np.load(f)[f"{side}_raw"].astype(float)
                ex_c.append(r.min() if sign > 0 else r.max())
                ex_o.append(r.max() if sign > 0 else r.min())
            # The CLOSED anchor is trustworthy -- the jaws touching is easy to reach and was confirmed
            # on video for every session. The OPEN anchor is not: in 140419/170333/100511 the left jaw
            # travels only ~530 ticks against ~1100 in the sessions that do reach the stop, i.e. the
            # operator never opened it fully there. The mechanical span is a property of the DEVICE, so
            # it is taken from the sessions that reach the stop and applied from each session's own
            # closed anchor, instead of trusting a short session's widest moment as "fully open".
            closed = np.percentile(ex_c, 10 if sign > 0 else 90)
            opened = np.percentile(ex_o, 90 if sign > 0 else 10)
            anchors[(sess, side)] = (float(closed), float(opened), sign)
            table.append(dict(session=sess, side=side, direction=int(sign),
                              closed_raw=round(float(closed), 1), open_raw=round(float(opened), 1),
                              span=round(abs(float(opened) - float(closed)), 1),
                              ticks_source="regression-recovered" if (sess, side) in RECOVERED else "session_meta",
                              meta_ticks=[c, o],
                              excluded_from_aperture=EXCLUDED_FROM_APERTURE.get((sess, side))))

    # device span per side, from the sessions whose travel reaches the stop (top half of observed spans)
    # left: take the span from 150545, the one left session whose closed anchor is video-validated AND
    # whose travel reaches the stop. 135136-left also looks full but its closed anchor is the flagged
    # outlier, so its span would drag the constant low.
    # right: every session reaches the stop, so the median across them is the device span.
    span = {}
    sp_r = [abs(anchors[(k, "right")][1] - anchors[(k, "right")][0]) for k in by]
    span["right"] = float(np.median(sp_r))
    ref = "20260917_150545"
    span["left"] = float(abs(anchors[(ref, "left")][1] - anchors[(ref, "left")][0]))
    for (sess, side), (cl, op, sign) in list(anchors.items()):
        full = cl + sign * span[side]
        anchors[(sess, side)] = (cl, full, sign)
        for t in table:
            if t["session"] == sess and t["side"] == side:
                t["open_raw_observed"] = t["open_raw"]
                t["open_raw"] = round(full, 1)
                t["span"] = round(span[side], 1)
                t["span_source"] = "device span (sessions reaching the stop)"
                t["travel_fraction"] = round(abs(op - cl) / span[side], 3)
    print(f"   device span: left {span['left']:.0f}  right {span['right']:.0f} ticks")

    n, clipped = 0, 0
    for sess in sorted(by):
        for f in by[sess]:
            z = dict(np.load(f))
            for side in ("left", "right"):
                cl, op, _ = anchors[(sess, side)]
                fr = (z[f"{side}_raw"].astype(np.float64) - cl) / (op - cl)
                clipped += int(((fr < 0) | (fr > 1)).sum())
                fr = np.clip(fr, 0.0, 1.0)
                z[f"{side}_open_fraction"] = fr.astype(np.float32)
                z[f"{side}_width_m"] = (fr * APERTURE_M).astype(np.float32)
            np.savez_compressed(out / pathlib.Path(f).name, **z)
            n += 1

    prov = dict(schema="handumi_gripper_contract/v1", created="2026-09-22",
                semantic="0 m = fully closed, larger = more open. NO inversion; same direction as reBot.",
                aperture_m=APERTURE_M, aperture_uncertainty_m=APERTURE_UNCERTAINTY_M,
                aperture_method="video-validated closed / 50 mm cube grasp / fully-open frames, "
                                "5 sessions x 2 arms; median over the 8 combos with valid anchors",
                aperture_todo="measure the assembled device's full aperture with a caliper; only this "
                              "scalar changes, no SLAM or video reprocessing",
                anchor_rule="closed = 90th-percentile of per-episode closed extremes (reliable, video-confirmed); open = closed + device span. Device span: left from 20260917_150545 (only left session with a video-validated closed anchor AND full travel), right from the median across sessions (all reach the stop). Three left sessions travel only ~50% of the span -- the operator never opened the left jaw fully there, and their width_m correctly saturates below the full aperture.",
                conversion="width_m = clip((raw - closed_raw)/(open_raw - closed_raw), 0, 1) * aperture_m",
                state="width_m[t]", action="width_m[t+1]",
                anchors=table)
    (out / "gripper_contract.json").write_text(json.dumps(prov, indent=1))

    print(f"=== corrected gripper — {n} episodes ===")
    print(f"   aperture_m = {APERTURE_M} (±{APERTURE_UNCERTAINTY_M})   clipped samples {clipped}")
    print(f"\n   {'session':<18} {'side':<6} {'dir':>4} {'closed':>8} {'open':>8} {'span':>6} {'travel':>6}  source")
    for t in table:
        note = f"   [{t['excluded_from_aperture']}]" if t["excluded_from_aperture"] else ""
        tf = t.get("travel_fraction", 1.0)
        warn = "  <-- never fully opened" if tf < 0.85 else ""
        print(f"   {t['session']:<18} {t['side']:<6} {t['direction']:>4} {t['closed_raw']:>8.0f} "
              f"{t['open_raw']:>8.0f} {t['span']:>6.0f} {tf:>6.2f}  {t['ticks_source']}{note}{warn}")
    w = np.concatenate([np.load(p)[f"{s}_width_m"] for p in out.glob("2026*.npz") for s in ("left", "right")])
    print(f"\n   width_m: min {w.min():.4f}  p50 {np.percentile(w,50):.4f}  p95 {np.percentile(w,95):.4f}  max {w.max():.4f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
