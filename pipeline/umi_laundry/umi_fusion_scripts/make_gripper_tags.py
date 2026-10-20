#!/usr/bin/env python3
"""[2026-10-13] Printable AprilTag test sheet for the HandUMI gripper bodies (A4, print at 100 % / "actual size").
Family tag36h11 (OpenCV DICT_APRILTAG_36h11 == apriltag tag36h11). Left unit IDs 10/11/12, right unit IDs 20/21/22.
"size" = outer edge of the BLACK square (what the pose solver needs); each tag carries a 1-cell white quiet zone + a light cut box.
Rows = sizes so the detection limit at the Orbbec distance can be found on real frames before choosing one.
usage: make_gripper_tags.py [out.pdf] [--sizes 20 25 30 40]
"""
import argparse, pathlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default=str(pathlib.Path.home() / "Desktop/gripper_apriltags_36h11_test.pdf"))
ap.add_argument("--sizes", nargs="+", type=float, default=[20, 25, 30, 40]); a = ap.parse_args()
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
IDS = [("L", 10), ("L", 11), ("L", 12), ("R", 20), ("R", 21), ("R", 22)]
W, H = 210.0, 297.0
from matplotlib.backends.backend_pdf import PdfPages


def new_page(pdf, page):
    fig = plt.figure(figsize=(W / 25.4, H / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.text(12, 12, f"HandUMI gripper AprilTags  —  tag36h11  —  print at 100 % (actual size), no scaling   (page {page})", fontsize=9, va="top")
    ax.text(12, 17, "size = black square edge.  Left unit: 10 11 12   Right unit: 20 21 22.   Cut on the grey box (keeps the white quiet zone).",
            fontsize=6.5, va="top", color="#52514e")
    ax.plot([12, 62], [24, 24], color="k", lw=1); ax.plot([12, 12], [22.5, 25.5], color="k", lw=1); ax.plot([62, 62], [22.5, 25.5], color="k", lw=1)
    ax.text(37, 27.5, "50 mm check bar", fontsize=6, ha="center", va="top")
    return fig, ax, 36.0


def rows_height(s):
    q = s / 8.0; box = s + 2 * q; per = int((W - 22 + 6.0) // (box + 6.0)); nrow = -(-len(IDS) // per)
    return nrow * box + (nrow - 1) * 8.0 + 12.0


with PdfPages(a.out) as pdf:
    page = 1; fig, ax, y = new_page(pdf, page)
    for s in a.sizes:
        if y + rows_height(s) > H - 8:                                   # next size does not fit: new page
            pdf.savefig(fig); plt.close(fig); page += 1; fig, ax, y = new_page(pdf, page)
        q = s / 8.0; box = s + 2 * q; gap = 6.0
        ax.text(12, y - 1.5, f"{s:g} mm", fontsize=8, va="bottom", weight="bold"); x = 12.0
        for side, i in IDS:
            if x + box > W - 10:                                         # wrap: never let a tag run off the page
                x = 12.0; y += box + 8.0
            img = cv2.aruco.generateImageMarker(D, i, 8)                 # 8x8 cells incl. the black border, 1 px per cell
            ax.add_patch(plt.Rectangle((x, y), box, box, fill=False, ec="#c8c7c2", lw=0.4))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(x + q, x + q + s, y + q + s, y + q))
            ax.text(x + box / 2, y + box + 1.2, f"{side}{i}", fontsize=5.5, ha="center", va="top", color="#52514e")
            x += box + gap
        y += box + 12.0
    pdf.savefig(fig); plt.close(fig)
print("wrote", a.out, "pages", page)
