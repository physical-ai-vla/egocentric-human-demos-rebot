"""Table frame from the ChArUco board that is actually on this desk.

    sudo .venv/bin/python scripts/calibrate_table_frame_charuco.py --live --plane-from-depth \
        --height-cm 82 --edge-cm 35 --pitch-deg 38 --write

Thin wrapper: it supplies a ChArUco detector to handumi_collector.tools.calibrate_table_frame and does nothing else.
That package may not import marker code -- tests/handumi/test_pose_pipeline.py holds it to "no AprilTag / ArUco /
ChArUco anywhere in the production package" -- and the rule is worth keeping intact rather than amending for a one-off
calibration, so the board lives out here and is injected, the same shape tools.calibrate_fisheye already uses.

Board defaults are what scripts/identify_board.py found on the desk: 6x8 squares, 31 mm square, 23 mm marker. The
dictionary genuinely does not matter for this board -- its marker ids reach 23, inside DICT_5X5_50 as well as
DICT_5X5_100 -- so the smaller one is the honest thing to record."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ego_collector.camera.calibration import CharucoSpec, detect_charuco     # noqa: E402  (outside the guarded package)
from handumi_collector.tools import calibrate_table_frame as ctf             # noqa: E402


class CharucoDetectorAdapter:
    def __init__(self, spec: CharucoSpec, min_corners: int = 12) -> None:
        self.spec, self.min_corners = spec, min_corners
        self.name = (f"charuco {spec.squares_x}x{spec.squares_y} "
                     f"{spec.square_length_m*1000:.0f}/{spec.marker_length_m*1000:.0f}mm {spec.dictionary}")

    def __call__(self, bgr: np.ndarray):
        d = detect_charuco(bgr, self.spec, min_corners=self.min_corners)
        if d is None: return None
        return d.image_points.reshape(-1, 2).astype(np.float64), d.object_points.reshape(-1, 3).astype(np.float64)


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--squares-x", type=int, default=6)
    pre.add_argument("--squares-y", type=int, default=8)
    pre.add_argument("--square-mm", type=float, default=31.0)
    pre.add_argument("--marker-mm", type=float, default=23.0)
    pre.add_argument("--dictionary", default="DICT_5X5_50")
    pre.add_argument("--min-corners", type=int, default=12)
    a, rest = pre.parse_known_args(argv)
    spec = CharucoSpec(a.squares_x, a.squares_y, a.square_mm / 1000.0, a.marker_mm / 1000.0, a.dictionary)
    det = CharucoDetectorAdapter(spec, a.min_corners)
    print(f"target: {det.name}")
    return ctf.main(rest + ["--board-name", det.name], detector=det)


if __name__ == "__main__":
    sys.exit(main())
