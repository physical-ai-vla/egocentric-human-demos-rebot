#!/usr/bin/env python3
"""[2026-10-13] UMI100 raw-data QA (the 20-episode checkpoint): NOT pose accuracy, only "was everything recorded right".
Per KEPT episode: 3 cameras (frames, fps, skipped, encoder errors, wrist/head frame-count agreement), Orbbec depth (PNG count,
valid-depth ratio on 5 samples), IMUs (rate, max gap), timestamps (monotonic, max inter-frame gap, cross-stream start skew),
grippers (rate, raw range moved = grasps happened, rest reads closed), 3 s init hold (go_cue held_still both sides + clean),
metadata (umi_tags garment / initial_state / outcome).  -> <session>/raw_qa.json + a table.  Exit code 1 if any FAIL.
usage: raw_qa.py <session dir> [<session dir> ...]
"""
import json, pathlib, sys

import numpy as np

TH = dict(fps_lo=28.5, fps_hi=31.5, max_frame_gap_ms=100.0, wrist_count_diff=3, head_count_rel=0.01, depth_valid_min=0.5,
          imu_hz_min=180.0, imu_max_gap_ms=30.0, start_skew_ms=150.0, grip_hz_min=80.0, grip_moved_ticks=300, hold_s_min=2.0)


def episode(ep):
    from mcap.reader import make_reader
    m = json.loads((ep / "episode_meta.json").read_text()); ev = json.loads((ep / "events.json").read_text())
    issues, warns = [], []; st = m.get("streams", {}); dur = float(m.get("duration_s") or 0)
    if dur < 5.0:                                                    # [2026-10-13] empty take auto-kept on close
        return dict(episode=f"{ep.parent.name}/{ep.name}", duration_s=dur, quality=m.get("quality"), status=m.get("status"),
                    verdict="FAIL", fails=[f"take only {dur:.2f} s (empty / auto-kept on close)"], warns=[])
    for cam in ("head_depth", "left_wrist", "right_wrist"):
        s = st.get(cam)
        if not s: issues.append(f"{cam}: no stream"); continue
        if not (TH["fps_lo"] <= s["fps_measured"] <= TH["fps_hi"]): issues.append(f"{cam}: fps {s['fps_measured']}")
        if s.get("skipped_capture_indices"): warns.append(f"{cam}: skipped {s['skipped_capture_indices']}")
        if s.get("encoder_error"): issues.append(f"{cam}: encoder error {s['encoder_error']}")
        if not (ep / f"{cam}.mp4").exists(): issues.append(f"{cam}: mp4 missing")
    if st.get("left_wrist") and st.get("right_wrist") and abs(st["left_wrist"]["frames"] - st["right_wrist"]["frames"]) > TH["wrist_count_diff"]:
        warns.append(f"wrist frame counts differ {st['left_wrist']['frames']} vs {st['right_wrist']['frames']}")
    # depth
    dd = ep / "head_depth_depth"; pngs = sorted(dd.glob("*.png")) if dd.exists() else []
    if st.get("head_depth") and abs(len(pngs) - st["head_depth"]["frames"]) > 2: issues.append(f"depth PNGs {len(pngs)} vs colour frames {st['head_depth']['frames']}")
    if pngs:
        import cv2
        vr = [float((cv2.imread(str(pngs[int(k)]), cv2.IMREAD_UNCHANGED) > 0).mean()) for k in np.linspace(0, len(pngs) - 1, 5)]
        if min(vr) < TH["depth_valid_min"]: warns.append(f"depth valid ratio min {min(vr):.2f}")
    else: issues.append("no depth frames")
    # mcap streams
    fm, imu, grip = {}, {"left": [], "right": []}, {"left": [], "right": []}
    with open(ep / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            t = ch.topic
            if t.endswith("/frame_meta"): fm.setdefault(t.split("/")[1], []).append(json.loads(msg.data)["capture_ns"])
            elif t.endswith("/imu"): imu[t.split("/")[1]].append(json.loads(msg.data)["host_receive_ns"])
            elif t.endswith("/gripper"):
                d = json.loads(msg.data); grip[d["side"]].append((d["sample_ns"], d["raw_position_mod"], d.get("normalized")))
    starts = []
    for cam, ts in fm.items():
        ts = np.array(ts, np.int64); g = np.diff(ts) / 1e6; starts.append(ts[0])
        if (g <= 0).any(): issues.append(f"{cam}: non-monotonic capture_ns")
        if g.max() > TH["max_frame_gap_ms"]: warns.append(f"{cam}: frame gap {g.max():.0f} ms")
    if starts and (max(starts) - min(starts)) / 1e6 > TH["start_skew_ms"]: warns.append(f"stream start skew {(max(starts) - min(starts)) / 1e6:.0f} ms")
    rec = dict(episode=f"{ep.parent.name}/{ep.name}", duration_s=dur, quality=m.get("quality"), status=m.get("status"))
    for side in ("left", "right"):
        ts = np.sort(np.array(imu[side], np.int64))
        if len(ts) < 10: issues.append(f"imu_{side}: no data"); continue
        hz = len(ts) / ((ts[-1] - ts[0]) / 1e9); gap = np.diff(ts).max() / 1e6; rec[f"imu_{side}_hz"] = round(hz, 1)
        if hz < TH["imu_hz_min"]: issues.append(f"imu_{side}: {hz:.0f} Hz")
        if gap > TH["imu_max_gap_ms"]: warns.append(f"imu_{side}: gap {gap:.0f} ms")
        gs = np.array(grip[side], float)
        if len(gs) < 10:
            # [2026-10-13] servos removed: the jaw opening comes from the red X marks in the wrist video
            import cv2, sys as _s; _s.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from jaw_redx import jaw_redx
            cap = cv2.VideoCapture(str(ep / f"{side}_wrist.mp4")); i = 0; d = []
            while True:
                ok = cap.grab()
                if not ok: break
                if i % 10 == 0: _, img = cap.retrieve(); d.append(jaw_redx(img)[0])
                i += 1
            d = np.array(d); det = float(np.isfinite(d).mean()) if len(d) else 0.0; rec[f"redx_{side}_detect"] = round(det, 3)
            if det < 0.5: issues.append(f"jaw_{side}: no servo and red X seen in only {100 * det:.0f}% of frames")
            elif det < 0.85: warns.append(f"jaw_{side}: red X seen in {100 * det:.0f}% of frames")
            elif np.nanpercentile(d, 95) - np.nanpercentile(d, 5) < 150: warns.append(f"jaw_{side}: red X distance barely changed (jaw not used?)")
            continue
        ghz = len(gs) / ((gs[-1, 0] - gs[0, 0]) / 1e9); rng = np.percentile(gs[:, 1], 98) - np.percentile(gs[:, 1], 2)
        rec[f"grip_{side}_range_ticks"] = int(rng); rec[f"grip_{side}_hz"] = round(ghz, 1)
        if ghz < TH["grip_hz_min"]: issues.append(f"gripper_{side}: {ghz:.0f} Hz")
        if rng < TH["grip_moved_ticks"]: warns.append(f"gripper_{side}: barely moved ({rng:.0f} ticks)")
        n0 = gs[(gs[:, 0] - gs[0, 0]) < 2e9, 2]
        if len(n0) and np.nanmedian(n0.astype(float)) > 0.35: warns.append(f"gripper_{side}: reads OPEN in the start hold (calibration?)")
    go = [e for e in ev if e["kind"] == "go_cue"]
    if not go: issues.append("no go_cue (init hold missing)")
    else:
        hs = go[0]["detail"].get("held_still", {}); rec["hold_s"] = hs
        if not go[0]["detail"].get("clean", False): warns.append("init hold broken (moved before GO)")
        if any(v < TH["hold_s_min"] for v in hs.values()): warns.append(f"init hold short {hs}")
    tg = m.get("umi_tags") or {}
    if not tg.get("garment_id") or not tg.get("initial_state"): warns.append("umi_tags garment/initial_state missing")
    if not tg.get("outcome"): warns.append("outcome not set")
    rec.update(verdict="FAIL" if issues else ("WARN" if warns else "PASS"), fails=issues, warns=warns)
    return rec


def main():
    allrec = []; bad = False
    for s in sys.argv[1:]:
        S = pathlib.Path(s); recs = [episode(ep) for ep in sorted(S.glob("episode_*")) if (ep / ".complete").exists()]
        (S / "raw_qa.json").write_text(json.dumps(dict(thresholds=TH, episodes=recs), indent=1, default=float)); allrec += recs
    for r in allrec:
        bad |= r["verdict"] == "FAIL"
        print(f"{r['verdict']:4s} {r['episode']:45s} {r['duration_s']:6.1f}s  " + "; ".join(r["fails"] + r["warns"]))
    v = [r["verdict"] for r in allrec]; print(f"\n{len(v)} episodes: PASS {v.count('PASS')}  WARN {v.count('WARN')}  FAIL {v.count('FAIL')}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
