#!/usr/bin/env python3
"""Extract training-loss curves from LeRobot `lerobot_train` console logs (Ray job logs) into CSV.

LeRobot prints one `step:...` line every `log_freq` (200 here) steps and rounds the step to "60K" above 1k, so the exact
step is taken as (line index + 1) * LOG_FREQ, not from the printed value. Logs overwrite lines with '\r'.
The loss printed is the running mean since the previous log line (training loss, not a held-out metric).

usage: loss_curves_from_logs.py <out_dir> <name>=<ray_log> [<name>=<ray_log> ...]
writes <out_dir>/<name>.csv with columns step,loss,lr,grad_norm,epoch
"""
import csv, pathlib, re, sys

LOG_FREQ = 200
PAT = re.compile(r"step:(\S+) .*?epch:([0-9.]+) loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.e+-]+)")


def parse(path):
    rows = []
    for line in pathlib.Path(path).read_text(errors="replace").replace("\r", "\n").splitlines():
        m = PAT.search(line)
        if m:
            rows.append({"step": (len(rows) + 1) * LOG_FREQ, "loss": float(m.group(3)), "lr": float(m.group(5)),
                         "grad_norm": float(m.group(4)), "epoch": float(m.group(2))})
    return rows


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        rows = parse(path)
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "grad_norm", "epoch"]); w.writeheader(); w.writerows(rows)
        print(f"{name}: {len(rows)} points, last step {rows[-1]['step'] if rows else '-'}")
