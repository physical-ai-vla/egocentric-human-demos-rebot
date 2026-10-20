"""[2026-10-12] Orbbec + chosen wrist cameras through the collector's own DeviceManager (no IMU/gripper/UI); fps every 5 s.
sudo .venv/bin/python tools/orbbec_combo_test.py both|left|right|none [seconds]"""
import sys, time, logging, dataclasses
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
from handumi_collector.config import load_config
from handumi_collector.devices.manager import DeviceManager
which = sys.argv[1] if len(sys.argv) > 1 else "both"; dur = float(sys.argv[2]) if len(sys.argv) > 2 else 180
extra = set(sys.argv[3:])          # optional: imu | imuL | imuR, grip  (add the IMUs / grippers too)
keep = {"both": ("head_depth", "left_wrist", "right_wrist"), "left": ("head_depth", "left_wrist"),
        "right": ("head_depth", "right_wrist"), "none": ("head_depth",)}[which]
cfg = load_config(hardware="handumi_rgbd")
hw = dataclasses.replace(cfg.hardware, cameras=[c for c in cfg.hardware.cameras if c.name in keep],
                         imus=[i for i in cfg.hardware.imus if "imu" in extra or f"imu{i.side[0].upper()}" in extra], grippers=cfg.hardware.grippers if "grip" in extra else [])
dm = DeviceManager(hw); dm.build(); dm.connect_all()
print("errors:", dm.errors, flush=True)
t0 = time.time(); worst = {}
try:
    while time.time() - t0 < dur:
        time.sleep(5)
        line = []
        for c in dm.all_devices():
            n = c.name
            st = c.status(); worst[n] = min(worst.get(n, 99), st.rate_hz or 0)
            line.append(f"{n} {st.rate_hz or 0:5.1f} (age {st.age_ms or 0:5.0f} ms)")
        print(f"{time.time() - t0:5.0f} s  " + " | ".join(line), flush=True)
finally:
    dm.close_all()
print("RESULT", which, sorted(extra), "worst fps:", {k: round(v, 1) for k, v in worst.items()})
