#!/usr/bin/env python3
"""Write printable AprilTag PNGs for the world (100-103) and wrist (10, 20) tags.

    python scripts/generate_apriltags.py --family tagStandard41h12 --world-size-mm 70 --wrist-size-mm 50

Bit images come from the official AprilRobotics/apriltag-imgs repository (tiny PNGs,
one per id, cached under --cache) and are upscaled with nearest-neighbour. The size
you pass is the **detector quad** (AprilTag ``tag_size``: 8 of 10 modules for tag36h11, the
inner 5 of 9 modules for tagStandard41h12), which is what ``tag_size_m`` in the configs means.
The printed footprint is larger (9/5 x for tagStandard41h12) - the script prints it.
Offline fallback: tag36h11 can be rendered locally with OpenCV.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

APRILTAG_IMGS = "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/{family}/tag{code}_{id:05d}.png"
FAMILY_CODES = {"tagStandard41h12": "41_12", "tag36h11": "36_11", "tagStandard52h13": "52_13", "tag25h9": "25_09", "tag16h5": "16_05"}
# (modules across the apriltag-imgs bit image, modules across the quad the detector reports = AprilTag `tag_size`)
# AprilTag 3 family definitions: tag36h11 total 10 / width_at_border 8; tagStandard41h12 total 9 / width_at_border 5
# (reversed border: data bits also live outside the border, the detected quad is the inner 5x5 square);
# tagStandard52h13 10/6; tag25h9 9/7; tag16h5 8/6.  Sizes you pass = the detector quad, i.e. `tag_size_m` in the configs.
FAMILY_LAYOUT = {"tagStandard41h12": (9, 5), "tag36h11": (10, 8), "tagStandard52h13": (10, 6), "tag25h9": (9, 7), "tag16h5": (8, 6)}


def bit_image(family: str, tag_id: int, cache: Path) -> np.ndarray:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{family}_{tag_id:05d}.png"
    if not path.exists():
        url = APRILTAG_IMGS.format(family=family, code=FAMILY_CODES[family], id=tag_id)
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:  # noqa: BLE001
            if family != "tag36h11":
                raise SystemExit(f"could not download {url}: {exc}")
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            marker = cv2.aruco.generateImageMarker(d, tag_id, 8)  # 8 modules at 1 px each
            img = np.full((10, 10), 255, dtype=np.uint8)
            img[1:9, 1:9] = marker
            cv2.imwrite(str(path), img)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not read {path}")
    expected = FAMILY_LAYOUT[family][0]
    if img.shape[0] != expected:
        raise SystemExit(f"{path}: expected a {expected}x{expected} bit image, got {img.shape}")
    return img


def render(family: str, tag_id: int, size_mm: float, dpi: int, cache: Path, label: str) -> tuple[np.ndarray, str]:
    bits = bit_image(family, tag_id, cache)
    modules_total, modules_black = FAMILY_LAYOUT[family]
    px_per_m = dpi / 0.0254
    module_px = (size_mm / 1000.0 * px_per_m) / modules_black
    total_px = int(round(modules_total * module_px))
    tag = cv2.resize(bits, (total_px, total_px), interpolation=cv2.INTER_NEAREST)
    quiet = int(round(1 * module_px))  # extra white quiet zone around the bit image (>= 1 module)
    canvas = np.full((total_px + 2 * quiet + 40, total_px + 2 * quiet), 255, dtype=np.uint8)
    canvas[quiet : quiet + total_px, quiet : quiet + total_px] = tag
    footprint_mm = modules_total * module_px / px_per_m * 1000.0
    cv2.putText(canvas, f"{family} id{tag_id} {label} tag_size {size_mm:.0f}mm (footprint {footprint_mm:.0f}mm)", (5, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    note = f"print at 100 %: detector quad (tag_size) = {size_mm:.1f} mm = {modules_black} modules x {module_px:.1f} px; printed footprint {footprint_mm:.0f} mm"
    return canvas, note


A4_MM = (210.0, 297.0)


def compose_a4_sheets(tiles: list[tuple[np.ndarray, str]], dpi: int, *, margin_mm: float = 8.0, gap_mm: float = 10.0) -> list[np.ndarray]:
    """Pack tag tiles onto A4 pages (portrait) with a 100 mm scale bar for print verification."""
    px = dpi / 25.4
    W, H = int(round(A4_MM[0] * px)), int(round(A4_MM[1] * px))
    margin, gap = int(round(margin_mm * px)), int(round(gap_mm * px))
    bar_h = int(round(12 * px))
    pages: list[np.ndarray] = []
    page = np.full((H, W), 255, dtype=np.uint8)
    x, y, row_h = margin, margin + bar_h, 0
    remaining = list(tiles)

    def add_scale_bar(pg: np.ndarray) -> None:
        x0, y0, L = margin, margin, int(round(100 * px))
        cv2.rectangle(pg, (x0, y0), (x0 + L, y0 + int(round(2 * px))), 0, -1)
        cv2.putText(pg, "100 mm scale bar - verify with a ruler after printing at 100% / actual size", (x0, y0 + int(round(8 * px))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    add_scale_bar(page)
    while remaining:
        tile, _ = remaining[0]
        th, tw = tile.shape[:2]
        if tw > W - 2 * margin:
            raise SystemExit("tile wider than the page: lower --dpi or the tag size")
        if x + tw > W - margin:  # next row
            x, y, row_h = margin, y + row_h + gap, 0
        if y + th > H - margin:  # next page
            pages.append(page)
            page = np.full((H, W), 255, dtype=np.uint8)
            add_scale_bar(page)
            x, y, row_h = margin, margin + bar_h, 0
            continue
        page[y : y + th, x : x + tw] = tile
        x += tw + gap
        row_h = max(row_h, th)
        remaining.pop(0)
    pages.append(page)
    return pages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="tagStandard41h12", choices=sorted(FAMILY_CODES))
    p.add_argument("--world-ids", default="100,101,102,103")
    p.add_argument("--world-size-mm", type=float, default=70.0)
    p.add_argument("--wrist-ids", default="10,20")
    p.add_argument("--wrist-size-mm", type=float, default=50.0)
    p.add_argument("--finger-ids", default="", help="Thumb/index dorsal (fingernail-side) tags, e.g. '11,12,21,22' = left thumb/index, right thumb/index")
    p.add_argument("--finger-size-mm", type=float, default=16.0, help="Detector quad; tag36h11 printed footprint = 10/8 x (16 mm -> 20 mm)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("outputs/tags"))
    p.add_argument("--cache", type=Path, default=Path("outputs/tags/.bits"))
    p.add_argument("--a4", action="store_true", help="Also pack the tags onto A4 print sheets with a 100 mm scale bar")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(int(i), a.world_size_mm, "world") for i in a.world_ids.split(",") if i.strip()]
    jobs += [(int(i), a.wrist_size_mm, "left_wrist" if int(i) < 20 else "right_wrist") for i in a.wrist_ids.split(",") if i.strip()]
    for i in (int(s) for s in a.finger_ids.split(",") if s.strip()):
        side = "left" if i < 20 else "right"
        finger = "thumb" if i % 10 == 1 else "index"
        jobs.append((i, a.finger_size_mm, f"{side}_{finger}"))
    tiles = []
    for tag_id, size, label in jobs:
        img, note = render(a.family, tag_id, size, a.dpi, a.cache, label)
        path = a.out / f"{a.family}_{tag_id:03d}_{label}_{int(size)}mm.png"
        cv2.imwrite(str(path), img)
        tiles.append((img, label))
        print(f"{path}  {note}")
    if a.a4:
        from PIL import Image

        pages = compose_a4_sheets(tiles, a.dpi)
        pil_pages = []
        for i, page in enumerate(pages, start=1):
            path = a.out / f"print_sheet_A4_{i}.png"
            im = Image.fromarray(page)
            im.save(path, dpi=(a.dpi, a.dpi))  # DPI metadata so "actual size" printing is exact
            pil_pages.append(im.convert("L"))
            print(f"{path}  (A4 @ {a.dpi} dpi, print at 100% / actual size, no 'fit to page')")
        pdf = a.out / "print_sheets_A4.pdf"
        pil_pages[0].save(pdf, save_all=True, append_images=pil_pages[1:], resolution=a.dpi)
        print(f"{pdf}  ({len(pil_pages)} A4 pages, 300 dpi; print at 100 %)")


if __name__ == "__main__":
    main()
