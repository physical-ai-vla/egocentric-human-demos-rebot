#!/usr/bin/env python3
"""[2026-09-28] Replay recorded episodes through the robot_like_v1 LIVE monitor (handumi_collector/robotlike/monitor.py).

Feeds each episode's sensors.mcap /<side>/imu and /<side>/gripper samples into the same RobotLikeMonitor class the
collector runs, on a 15 Hz clock driven by the recorded host timestamps, so episodes recorded BEFORE the protocol existed
(the 259 C8 / C-old source episodes) get exactly the verdict and statistics a robot_like_v1 episode gets live.
Parity check: replaying a robot_like_v1 episode must reproduce its derived/robot_like/live_summary.json.

Read-only on the raw data; writes one JSON per episode into --out.
usage: robotlike_replay_live.py --out <dir> (--episodes-json <json with "episodes": [<C8 id>]> | --episode <dir> ...)
"""
import argparse, json, pathlib, sys
from types import SimpleNamespace
from mcap.reader import make_reader
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_collector"))
from handumi_collector.devices.base import ImuSample, SampleBuffer          # noqa: E402
from handumi_collector.robotlike.monitor import RobotLikeMonitor, load_protocol  # noqa: E402
RAW = H / "ego_collector/datasets/human_handumi_raw"


class _Imu:
    def __init__(self): self.buffer = SampleBuffer(1 << 20); self.cfg = SimpleNamespace(rate_hz=400)


class _Grip:
    def __init__(self): self.norm = float("nan")
    def quality(self): return SimpleNamespace(normalized=self.norm)


def replay(ep: pathlib.Path, proto: dict) -> dict:
    imu, grip = {"left": [], "right": []}, {"left": [], "right": []}
    with open(ep / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=["/left/imu", "/right/imu", "/left/gripper", "/right/gripper"]):
            d = json.loads(msg.data); side, kind = ch.topic.split("/")[1:3]
            if kind == "imu": imu[side].append(ImuSample(side, d["seq"], d["device_timestamp_us"], d["host_receive_ns"], d["ax"], d["ay"], d["az"], d["gx"], d["gy"], d["gz"], d.get("temperature_c", 0.0)))
            else: grip[side].append((d["sample_ns"], d.get("normalized")))
    dev = SimpleNamespace(imus={s: _Imu() for s in imu if imu[s]}, grippers={s: _Grip() for s in grip if grip[s]})
    mon = RobotLikeMonitor(proto, dev)
    t_all = [s.host_receive_ns for v in imu.values() for s in v]
    if not t_all: return dict(episode=str(ep), error="no IMU samples")
    t0, t1 = min(t_all), max(t_all); step = int(1e9 / mon.fps)
    ii = {s: 0 for s in imu}; gi = {s: 0 for s in grip}
    mon.begin_episode(pathlib.Path("/nonexistent")); mon.episode_dir = None      # never write into the raw episode
    t = t0 + step
    while t <= t1 + step:
        for s, v in imu.items():
            while ii[s] < len(v) and v[ii[s]].host_receive_ns <= t: dev.imus[s].buffer.append(v[ii[s]]); ii[s] += 1
        for s, v in grip.items():
            while gi[s] < len(v) and v[gi[s]][0] <= t:
                n = v[gi[s]][1]; dev.grippers[s].norm = float("nan") if n is None else float(n); gi[s] += 1
        mon.tick(t); t += step
    summ = mon.end_episode(); summ["duration_s"] = round((t1 - t0) / 1e9, 2); summ["episode"] = str(ep); summ["replayed"] = True
    return summ


def c8_dir(eid: str) -> pathlib.Path:
    sess, n = eid.rsplit("_", 1); return RAW / "Hpilot" / f"Hpilot_{sess}" / f"episode_{n}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--episodes-json"); ap.add_argument("--episode", action="append", default=[])
    ap.add_argument("--protocol", default="robot_like_v1"); a = ap.parse_args()
    proto = load_protocol(a.protocol); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = [pathlib.Path(e) for e in a.episode] + ([c8_dir(e) for e in json.load(open(a.episodes_json))["episodes"]] if a.episodes_json else [])
    for ep in eps:
        try: s = replay(ep, proto)
        except Exception as exc: s = dict(episode=str(ep), error=f"{type(exc).__name__}: {exc}")
        name = f"{ep.parent.name}__{ep.name}.json"; (out / name).write_text(json.dumps(s, indent=1, default=float))
        print(f"{ep.parent.name}/{ep.name}: {s.get('verdict', 'ERROR')}  grasps {s.get('grasps_total')}  " + ("; ".join(s.get("reasons", [])) or s.get("error", "")), flush=True)
