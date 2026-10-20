"""Quick install check of the GLOBAL Orbbec for the UMI AprilTag anchors (2026-10-17): ~10 s of colour + depth, tag36h11
detection of the right-UMI tags 200-202 and the left-UMI tags 210-212, per tag: frames seen, pixel side, PnP distance (tag size
0.05 m), depth at the tag centre. Saves 3 annotated snapshots for a visual check. Needs sudo on macOS (pyorbbecsdk).
    sudo .venv/bin/python -m visual_teleop.tools.orbbec_tag_check [--secs 10]   -> datasets/tag_rig/check_<ts>/"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RIGHT, LEFT = (200, 201, 202), (210, 211, 212)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=10.0); ap.add_argument("--tag-size", type=float, default=0.048); a = ap.parse_args()
    from ego_collector.camera.capture import CaptureConfig
    from ego_collector.camera.orbbec_capture import OrbbecCapture
    cap = OrbbecCapture(CaptureConfig(index=0, width=1280, height=720, fps=30)); cap.open()
    I = cap.intrinsics; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]])
    s = a.tag_size / 2; obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
    out = ROOT / "datasets/tag_rig" / ("check_" + time.strftime("%Y%m%d_%H%M%S")); out.mkdir(parents=True, exist_ok=True)
    stats = {i: dict(n=0, side=[], dist=[], depth=[]) for i in RIGHT + LEFT}; n = 0; snaps = 0; t0 = time.monotonic(); last = 0.0
    print("[점검] 두 UMI를 평소 작업 위치에 들고, 태그가 카메라 쪽을 보게 천천히 움직이세요", flush=True)
    while time.monotonic() - t0 < a.secs:
        f = cap.get(timeout=0.5)
        if f is None:
            if n == 0 and time.monotonic() - t0 > 6: cap.close(); raise SystemExit("[점검] 6초 동안 Orbbec 프레임 0장 -- USB 재연결 후 다시")
            continue
        n += 1; img = f.image; gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); dep = np.asarray(f.depth, np.float64) * float(cap.depth_scale or 1.0)
        if dep.max() > 100: dep = dep / 1000.0
        cs, ids, _ = det.detectMarkers(gray); seen = []
        for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
            i = int(i)
            if i not in stats: continue
            px = c.reshape(4, 2).astype(np.float64); ok, rv, tv = cv2.solvePnP(obj, px, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            side = float(np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1))); u, v = px.mean(0).astype(int)
            win = dep[max(0, v - 3):v + 4, max(0, u - 3):u + 4]; win = win[win > 0.05]
            st = stats[i]; st["n"] += 1; st["side"].append(side); st["dist"].append(float(np.linalg.norm(tv)) if ok else np.nan); st["depth"].append(float(np.median(win)) if win.size else np.nan)
            seen.append(i); cv2.polylines(img, [px.astype(np.int32)], True, (0, 255, 0) if i in RIGHT else (255, 0, 255), 2)
            cv2.putText(img, f"{i} {side:.0f}px", tuple(px[0].astype(int)), 0, 0.6, (0, 0, 255), 2)
        if time.monotonic() - last > 1.0:
            last = time.monotonic(); print(f"[점검] {last - t0:4.1f}s 오른손 {[i for i in seen if i in RIGHT]} 왼손 {[i for i in seen if i in LEFT]}", flush=True)
        if snaps < 3 and time.monotonic() - t0 > 2 + snaps * 3: cv2.imwrite(str(out / f"snap_{snaps}.jpg"), img); snaps += 1
    cap.close()
    rep = {"frames": n, "size": [int(img.shape[1]), int(img.shape[0])], "intrinsics": I}
    print(f"[점검] 프레임 {n} ({img.shape[1]}x{img.shape[0]}), 결과 (한 변 px 30 이상 권장):", flush=True)
    for i, st in stats.items():
        if st["n"]: rep[str(i)] = dict(seen=st["n"], side_px_p50=float(np.median(st["side"])), dist_m_p50=float(np.nanmedian(st["dist"])), depth_m_p50=float(np.nanmedian(st["depth"])) if np.isfinite(st["depth"]).any() else None)
        print(f"   {'오른손' if i in RIGHT else '왼손'} {i}: " + (f"보임 {st['n']}/{n}, 한 변 {np.median(st['side']):.0f}px, PnP 거리 {np.nanmedian(st['dist']):.2f} m, depth {np.nanmedian(st['depth']) if np.isfinite(st['depth']).any() else float('nan'):.2f} m" if st["n"] else "안 보임"), flush=True)
    (out / "check.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        for p in [out, *out.iterdir()]: os.chown(p, int(uid), int(gid))
    print(f"[점검] 저장 {out}", flush=True)


if __name__ == "__main__":
    main()
