"""Deterministic Teensy test-stream generator (firmware-free host testing): writes a byte stream with configurable
faults. python -m handumi_collector.tools.gen_test_stream out.bin --samples 4000 [--garbage-every 500] [--corrupt-every 700]
[--drop-every 900] [--seq-start 4294967000] [--t0-us 18446744073709000000]. Also importable: build_stream(...)."""
from __future__ import annotations
import argparse
import math
import random
from ..devices.teensy_imu import G0, encode_imu, encode_status


def build_stream(samples: int, *, rate_hz: int = 400, seq_start: int = 0, t0_us: int = 1_000_000, garbage_every: int = 0,
                 corrupt_every: int = 0, drop_every: int = 0, seed: int = 0) -> tuple[bytes, dict]:
    rng = random.Random(seed); out = bytearray(); truth = dict(emitted=0, dropped=0, corrupted=0, garbage=0, statuses=0)
    dt = int(1e6 / rate_hz)
    for i in range(samples):
        seq = (seq_start + i) % (1 << 32); t_us = (t0_us + i * dt) % (1 << 64)
        w = 2 * math.pi * 0.5 * i / rate_hz
        pk = encode_imu(seq, t_us, (0.1 * math.sin(w), 0.0, G0), (0.0, 0.2 * math.cos(w), 0.0), 30.0)
        if i % rate_hz == rate_hz - 1:                    # 1 Hz status, emitted regardless of faults on this sample
            out += encode_status(t_us, rate_hz, i + 1, truth["dropped"]); truth["statuses"] += 1
        if drop_every and i % drop_every == drop_every - 1:
            truth["dropped"] += 1; continue
        if corrupt_every and i % corrupt_every == corrupt_every - 1:
            b = bytearray(pk); b[12] ^= 0x55; pk = bytes(b); truth["corrupted"] += 1
        else:
            truth["emitted"] += 1
        if garbage_every and i % garbage_every == garbage_every - 1:
            out += bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40))); truth["garbage"] += 1
        out += pk
    return bytes(out), truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--samples", type=int, default=4000)
    for k in ("garbage-every", "corrupt-every", "drop-every"): ap.add_argument(f"--{k}", type=int, default=0)
    ap.add_argument("--seq-start", type=int, default=0); ap.add_argument("--t0-us", type=int, default=1_000_000)
    a = ap.parse_args(argv)
    data, truth = build_stream(a.samples, seq_start=a.seq_start, t0_us=a.t0_us, garbage_every=a.garbage_every, corrupt_every=a.corrupt_every, drop_every=a.drop_every)
    open(a.out, "wb").write(data); print(len(data), "bytes", truth); return 0


if __name__ == "__main__":
    raise SystemExit(main())
