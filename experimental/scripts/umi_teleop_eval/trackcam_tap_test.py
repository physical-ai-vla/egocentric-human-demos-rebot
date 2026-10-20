"""Tap test: per-event camera<->IMU latency and its JITTER for the UMI tracking fisheye (2026-10-15).
The UMI lies on the desk; it is tapped ~10 times (spoken cue every 2.5 s). Each tap = a sharp accelerometer spike (IMU device clock ->
host) and a burst of image change (frame difference). latency_i = t_image_event - t_imu_spike; mean ~ the camera-IMU offset,
STD = how much the per-frame host timestamps jitter (the suspected reason both VIOs ignore vision). Also logs the driver's own frame
timestamp (CAP_PROP_POS_MSEC) next to the host receive time, to see whether a steadier clock exists.
    .venv/bin/python scripts/trackcam_tap_test.py [--index 4] [--location 0x00140000] [--taps 10]"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from visual_teleop import config as vcfg                                   # noqa: E402
from visual_teleop.tools.calib_right import open_imu                        # noqa: E402
from visual_teleop.tools.wrist_frontend_check import lock_exposure          # noqa: E402
from handumi_collector.pose.timing import fit_device_to_host                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--index", type=int, default=4); ap.add_argument("--location", default="0x00140000"); ap.add_argument("--taps", type=int, default=10)
    a = ap.parse_args(); c = vcfg.load(); out = ROOT / "datasets" / "trackcam_tap" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); cap.set(cv2.CAP_PROP_FPS, 30)
    print(lock_exposure(a.location, 15)); [cap.read() for _ in range(10)]
    dev = open_imu(c, "right"); dev.buffer.drain()
    subprocess.run(["say", "-v", "Yuna", "유미를 책상에 놓고, 소리가 날 때마다 손가락으로 톡 치세요"]); dev.buffer.drain()
    S, th, tp, diffs, prev = [], [], [], [], None; t0 = time.monotonic(); dur = 2.0 + 2.5 * a.taps; nxt = 2.0
    while time.monotonic() - t0 < dur:
        if time.monotonic() - t0 >= nxt: subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"]); nxt += 2.5
        ok, f = cap.read(); t = time.monotonic_ns(); pm = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ok:
            g = cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2GRAY).astype(np.int16)
            diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g; th.append(t); tp.append(pm)
        S += dev.buffer.drain()
    S += dev.buffer.drain(); dev.stop(); dev.close(); cap.release(); subprocess.Popen(["say", "-v", "Yuna", "끝났습니다"])
    du = np.array([s.device_timestamp_us for s in S]); hn = np.array([s.host_receive_ns for s in S]); acc = np.array([[s.ax, s.ay, s.az] for s in S])
    ti = fit_device_to_host(du, hn).to_host_ns(du).astype(float); th = np.array(th, float); tp = np.array(tp); diffs = np.array(diffs)
    np.savez(out / "tap.npz", t_imu=ti, acc=acc, t_cam_host=th, t_cam_driver_ms=tp, diff=diffs)
    an = np.linalg.norm(acc - np.median(acc, 0), axis=1); thr_i = max(1.5, 6 * np.median(an))
    spikes = []
    for i in np.flatnonzero(an > thr_i):
        if not spikes or ti[i] - spikes[-1] > 0.8e9: spikes.append(ti[i])
    d_base = np.median(diffs); thr_c = d_base + 6 * np.median(np.abs(diffs - d_base)) + 0.5; events = []
    for i in np.flatnonzero(diffs > thr_c):
        if not events or th[i] - events[-1] > 0.8e9: events.append(th[i])
    lat = []
    for s in spikes:
        e = [x for x in events if -0.1e9 < x - s < 0.4e9]
        if e: lat.append((min(e) - s) / 1e6)
    dt = np.diff(th) / 1e6; dpm = np.diff(tp)
    rep = dict(imu_spikes=len(spikes), image_events=len(events), matched=len(lat), latency_ms=[round(x, 1) for x in lat],
               latency_mean_ms=round(float(np.mean(lat)), 1) if lat else None, latency_std_ms=round(float(np.std(lat)), 1) if lat else None,
               host_frame_dt_ms=dict(p5=round(float(np.percentile(dt, 5)), 1), p50=round(float(np.median(dt)), 1), p95=round(float(np.percentile(dt, 95)), 1), max=round(float(dt.max()), 1)),
               driver_pos_msec_dt=dict(p5=round(float(np.percentile(dpm, 5)), 2), p50=round(float(np.median(dpm)), 2), p95=round(float(np.percentile(dpm, 95)), 2)) if np.any(tp > 0) else "unavailable",
               note="latency = first image change after the IMU spike, quantised to frames (~33 ms) -> std below ~10 ms means steady stamps")
    (out / "tap.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
