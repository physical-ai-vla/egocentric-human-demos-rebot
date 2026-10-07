"""[2026-10-06 user "slam도 c922로"] C922-view MASt3R inputs from the fisheye exports: export_c922/<tag>/
  raw_video.mp4      = the recorded right_wrist_c922.mp4 frames that the fisheye export kept (same order / count; leading
                       frames dropped by the exporter's IMU preroll are dropped here too) -> identical timestamps (CORI)
  imu_data.json      = copied unchanged (same frame times)
  orbslam_setting.yaml = Camera.type PinHole, measured robot C922 K + k1 k2 p1 p2 k3, 640x480 (runner: run_perframe_resident_pinhole.py)
Frame alignment is VERIFIED per episode: export frame 0 rendered through the C922 model must match the chosen c922 frame best."""
import json, pathlib, re, sys, cv2, numpy as np
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_collector"))
from handumi_collector.robotlike.c922_view import C922View
V = C922View(); R = json.load(open(H / "ego_collector/configs/calibration/robot_right_wrist_c922_v001.json")); K = np.array(R["K"]); D = np.array(R["dist"])
RAW = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"; EX = H / "c8/hra_a100/export"; OUT = H / "c8/hra_a100/export_c922"
def frames(p):
    c = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out
rep = {}
for d in sorted(EX.iterdir()):
    tag = d.name; o = OUT / tag
    if (o / "raw_video.mp4").exists() or not (d / "raw_video.mp4").exists(): continue
    s, e, _ = tag.rsplit("_", 2)[0], tag.rsplit("_", 2)[1], None
    ep = RAW / f"HRA_A100_{tag[:15]}" / f"episode_{tag.split('_')[2]}"
    log = (d / "export.log").read_text(); m = re.search(r"dropping (\d+) leading frames", log); drop = int(m.group(1)) if m else 0
    fe = frames(d / "raw_video.mp4"); fc = frames(ep / "right_wrist_c922.mp4")
    n = len(fe)
    # verify alignment on frame 0 and a middle frame: candidate offsets drop-2..drop+2
    def score(off, i):
        a = cv2.cvtColor(V.render(fe[i], overlay=False), cv2.COLOR_BGR2GRAY).astype(np.float32); b = cv2.cvtColor(fc[i + off], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.abs(a - b).mean())
    # the exporter writes EVERY frame to raw_video.mp4 (its IMU-preroll "drop" only trims CORI), so the c922 video is 1:1
    if len(fc) != n: print(f"{tag}: SKIP frame count c922 {len(fc)} != export {n}"); continue
    sc = {k: score(k, 0) + score(k, n // 2) for k in (0, 1) if k + n // 2 < len(fc) and k < len(fc)}; best = 0
    if min(sc, key=sc.get) != 0: print(f"{tag}: WARNING offset 1 matches better ({sc})")
    o.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(o / "raw_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for f in fc[best:best + n]: vw.write(f)
    vw.release()
    (o / "imu_data.json").write_bytes((d / "imu_data.json").read_bytes())
    t = (d / "orbslam_setting.yaml").read_text()
    t = re.sub(r"^Camera\.type:.*$", 'Camera.type: "PinHole"', t, flags=re.M)
    for k, v in (("Camera1.fx", K[0, 0]), ("Camera1.fy", K[1, 1]), ("Camera1.cx", K[0, 2]), ("Camera1.cy", K[1, 2]), ("Camera.width", 640), ("Camera.height", 480)):
        t = re.sub(rf"^{re.escape(k)}:.*$", f"{k}: {v}", t, flags=re.M)
    for k in ("k1", "k2", "k3", "k4", "p1", "p2"): t = re.sub(rf"^Camera1\.{k}:.*\n", "", t, flags=re.M)
    t = t.rstrip() + "\n" + "".join(f"Camera1.{k}: {v}\n" for k, v in zip(("k1", "k2", "p1", "p2", "k3"), D))
    (o / "orbslam_setting.yaml").write_text(t)
    rep[tag] = dict(frames=n, offset=best, exporter_drop=drop, mae0=round(sc[best], 2))
    print(tag, rep[tag], flush=True)
json.dump(rep, open(H / "c8/hra_a100/export_c922_report.json", "w"), indent=1)
