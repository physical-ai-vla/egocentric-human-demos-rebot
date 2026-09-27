#!/usr/bin/env python3
"""Per-frame grip01 for the A_bimanual episodes, aligned from sensors.mcap.

The gripper stream already carries `normalized` = (raw - ticks_closed) / (ticks_open - ticks_closed),
verified against session_meta on both sides, so 0 = closed and 1 = open -- the SAME direction as the
reBot stored width (video-confirmed 2026-09-22: raw 0 = jaws touching, -118 = holding a 50 mm cube,
-270 = wide open). No inversion is applied anywhere.

Gripper runs at ~95 Hz and video at 30, so each frame takes the nearest sample; the alignment error is
reported rather than assumed.
"""
import argparse, json, pathlib, sys
import numpy as np
from mcap.reader import make_reader

RAW = pathlib.Path.home() / "ego_collector/datasets/human_handumi_raw/Hpilot"
SIDE_STREAM = {"left": "left_wrist", "right": "right_wrist"}


def read_episode(session, episode):
    p = RAW / f"Hpilot_{session}" / f"episode_{episode}" / "sensors.mcap"
    if not p.is_file():
        return None
    frames = {s: [] for s in SIDE_STREAM.values()}
    grip = {s: [] for s in SIDE_STREAM}
    with open(p, "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            t = ch.topic
            if t.endswith("/frame_meta"):
                st = t.split("/")[1]
                if st in frames:
                    d = json.loads(msg.data)
                    frames[st].append((int(d["video_frame"]), int(d["capture_ns"])))
            elif t.endswith("/gripper"):
                sd = t.split("/")[1]
                if sd in grip:
                    d = json.loads(msg.data)
                    tl = d.get("telemetry") or {}
                    grip[sd].append((int(d["sample_ns"]), float(d["normalized"]), int(d["raw_position"]),
                                     float(tl.get("load_percent") or 0.0)))
    return frames, grip


def align(frames, grip):
    """nearest-sample lookup; returns grip01 per video_frame plus the time error in ms"""
    fr = sorted(frames)
    if not fr or not grip:
        return None
    g = np.array(sorted(grip), dtype=np.float64)
    gt, gv, graw, gld = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    vf = np.array([x[0] for x in fr], int)
    ft = np.array([x[1] for x in fr], np.float64)
    j = np.clip(np.searchsorted(gt, ft), 1, len(gt) - 1)
    pick = np.where(np.abs(gt[j] - ft) < np.abs(gt[j - 1] - ft), j, j - 1)
    return dict(video_frame=vf, grip01=gv[pick].astype(np.float32),
                raw=graw[pick].astype(np.int32), load=gld[pick].astype(np.float32),
                dt_ms=np.abs(gt[pick] - ft) / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(pathlib.Path.home() /
                    ".claude/jobs/e649aef3/tmp/handumi_vla_manifest.json"))
    ap.add_argument("--out", default=str(pathlib.Path.home() / ".claude/jobs/e649aef3/tmp/handumi_grip"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = json.loads(pathlib.Path(a.manifest).read_text())["items"]
    if a.limit:
        items = items[: a.limit]

    dts, viol, jumps, done, fail = [], 0, [], 0, []
    for it in items:
        s, e = it["session_id"], it["episode"]
        r = read_episode(s, e)
        if r is None:
            fail.append((it["episode_id"], "no mcap")); continue
        frames, grip = r
        rec = {}
        for side, stream in SIDE_STREAM.items():
            al = align(frames[stream], grip[side])
            if al is None:
                fail.append((it["episode_id"], f"{side}: empty")); rec = None; break
            rec[f"{side}_video_frame"] = al["video_frame"]
            rec[f"{side}_grip01"] = al["grip01"]
            rec[f"{side}_raw"] = al["raw"]
            rec[f"{side}_load"] = al["load"]
            rec[f"{side}_dt_ms"] = al["dt_ms"].astype(np.float32)
            dts.append(al["dt_ms"])
            viol += int(((al["grip01"] < -1e-6) | (al["grip01"] > 1 + 1e-6)).sum())
            d = np.abs(np.diff(al["grip01"]))
            if len(d):
                jumps.append(d.max())
        if rec:
            np.savez_compressed(out / f"{it['episode_id']}.npz", **rec)
            done += 1

    D = np.concatenate(dts) if dts else np.array([0.0])
    J = np.array(jumps) if jumps else np.array([0.0])
    print(f"=== gripper 추출 — {done}/{len(items)} episodes ===")
    if fail:
        print(f"  실패 {len(fail)}: {fail[:5]}")
    print(f"\n[QA 1] pose-gripper 시간차 (ms)   p50 {np.percentile(D,50):.2f}  "
          f"p95 {np.percentile(D,95):.2f}  max {D.max():.2f}   "
          f"{'PASS' if D.max() < 1000/30 else 'WARN >1 frame'}")
    print(f"[QA 2] grip01 in [0,1] 위반        {viol}   {'PASS' if viol == 0 else 'FAIL'}")
    print(f"[QA 3] 프레임 간 최대 변화          p50 {np.percentile(J,50):.4f}  "
          f"p95 {np.percentile(J,95):.4f}  max {J.max():.4f}   (1.0 = 한 프레임에 완전 개폐)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
