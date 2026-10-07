#!/usr/bin/env python3
"""[2026-08-30] Map Damiao arm controllers (USB CDC; both boards report serial 00000000050C, so /dev names follow plug order)
to their physical USB port (locationID) and tty path, using the IOSerialBSDClient ancestry tree.
usage: python usb_arm_ports.py   |   from usb_arm_ports import arm_ports -> {locationID(int): "/dev/cu.usbmodemXXXX"}"""
import re, subprocess
def arm_ports():
    out = subprocess.run(["ioreg", "-r", "-c", "IOSerialBSDClient", "-t", "-l", "-w0"], capture_output=True, text=True).stdout
    res, loc, armed = {}, None, False
    for line in out.splitlines():
        if "+-o CDC Device@" in line:
            loc, armed = None, True; continue
        if armed and loc is None:
            m = re.search(r'"locationID" = (\d+)', line)
            if m: loc = int(m.group(1)); continue
        if armed and loc is not None:
            m = re.search(r'"IOCalloutDevice" = "([^"]+)"', line)
            if m: res[loc] = m.group(1); loc, armed = None, False
    return res
if __name__ == "__main__":
    m = arm_ports()
    for l, p in sorted(m.items()): print(f"loc={l} (0x{l:08x})  {p}")
    if not m: print("no arm controllers found")
