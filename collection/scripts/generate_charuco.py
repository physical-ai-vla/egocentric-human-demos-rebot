#!/usr/bin/env python3
"""Write a printable calibration board as an A4 PDF/PNG at a given DPI — ChArUco, or a plain checkerboard.

    python scripts/generate_charuco.py --out outputs/charuco --dpi 300
    python scripts/generate_charuco.py --plain --cols 9 --rows 6 --square-mm 25 --out outputs/checkerboard_9x6_25mm

Print at 100 % / actual size, glue to a rigid flat board, then verify one square against a ruler.

`--plain` exists for Kalibr: its checkerboard detector wants an unadorned board, and the ArUco markers inside a
ChArUco break it. Kalibr is how the camera-IMU extrinsic `T_camera_imu` gets measured, which is the one number
standing between the finished OpenVINS path and running it. `--cols`/`--rows` count SQUARES, matching what
ego_teleop.tools.kalibr_export writes into target.yaml (its default is 9x6 at 25 mm).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ego_collector.camera.calibration import CharucoSpec

A4_MM = (210.0, 297.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--charuco", type=Path, default=Path("configs/charuco.yaml"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("outputs/charuco"), help="Output stem (-> <stem>.pdf and <stem>.png)")
    p.add_argument("--plain", action="store_true", help="plain checkerboard, no markers (what Kalibr's detector wants)")
    p.add_argument("--cols", type=int, default=9, help="--plain: squares across")
    p.add_argument("--rows", type=int, default=6, help="--plain: squares down")
    p.add_argument("--square-mm", type=float, default=25.0, help="--plain: square size")
    a = p.parse_args()
    px = a.dpi / 25.4
    if a.plain:
        sq = int(round(a.square_mm * px))
        board = np.full((a.rows * sq, a.cols * sq), 255, np.uint8)
        for r in range(a.rows):
            for c in range(a.cols):
                if (r + c) % 2 == 0:
                    board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 0
        bw_mm, bh_mm = a.cols * a.square_mm, a.rows * a.square_mm
        caption = (f"checkerboard {a.cols}x{a.rows} squares  square {a.square_mm:.0f} mm  "
                   f"board {bw_mm:.0f}x{bh_mm:.0f} mm  (Kalibr targetCols={a.cols} targetRows={a.rows})")
    else:
        spec = CharucoSpec.from_yaml(a.charuco) if a.charuco.exists() else CharucoSpec()
        board = spec.image(px_per_m=a.dpi / 0.0254, margin_px=0)
        bw_mm, bh_mm = spec.squares_x * spec.square_length_m * 1000, spec.squares_y * spec.square_length_m * 1000
        caption = (f"ChArUco {spec.squares_x}x{spec.squares_y}  square {spec.square_length_m*1000:.0f} mm  "
                   f"marker {spec.marker_length_m*1000:.0f} mm  {spec.dictionary}  board {bw_mm:.0f}x{bh_mm:.0f} mm")
    W, H = int(round(A4_MM[0] * px)), int(round(A4_MM[1] * px))
    fits = lambda b: b.shape[1] <= W - 2 * int(10 * px) and b.shape[0] <= H - int(40 * px)
    if not fits(board) and fits(np.rot90(board)):
        # A 9x6 board at 25 mm is 225x150 mm: wider than A4 is, and shorter than A4 is tall. Turning it costs
        # nothing -- a checkerboard has no up -- and the caption records the square size, which is what matters.
        board = np.ascontiguousarray(np.rot90(board))
        bw_mm, bh_mm = bh_mm, bw_mm
        caption += "  (rotated to fit A4)"
    if not fits(board):
        raise SystemExit(f"board {bw_mm:.0f}x{bh_mm:.0f} mm does not fit A4 with margins even rotated; "
                         f"reduce the square size or print on A3")
    page = np.full((H, W), 255, dtype=np.uint8)
    x0, y0 = int(10 * px), int(10 * px)
    cv2.rectangle(page, (x0, y0), (x0 + int(100 * px), y0 + int(2 * px)), 0, -1)
    cv2.putText(page, "100 mm scale bar - verify with a ruler after printing at 100% / actual size", (x0, y0 + int(8 * px)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
    bx, by = (W - board.shape[1]) // 2, y0 + int(20 * px)
    page[by : by + board.shape[0], bx : bx + board.shape[1]] = board
    cv2.putText(page, caption, (x0, by + board.shape[0] + int(8 * px)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(page)
    im.save(a.out.with_suffix(".png"), dpi=(a.dpi, a.dpi))
    im.convert("L").save(a.out.with_suffix(".pdf"), resolution=a.dpi)
    print(f"{a.out.with_suffix('.pdf')}  (A4 @ {a.dpi} dpi, board {bw_mm:.0f}x{bh_mm:.0f} mm, print at 100 %)")
    print(f"{a.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
