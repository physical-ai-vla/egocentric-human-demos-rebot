#!/usr/bin/env python3
"""Fig. 10 of the Part I full report (v4.1): our modified HandUMI parts next to upstream.
Built from the renders in hardware/mods/images/. Run from the repo root: python3 report/figures/make_fig10_handumi_mods.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]
a = mpimg.imread(R / "hardware/mods/images/camera_mount_servo_cover_vs_upstream.png")
b = mpimg.imread(R / "hardware/mods/images/controller_support_connector.png")
H, W = a.shape[:2]
h2, w2 = b.shape[:2]
panels = [a[35:H - 60, i * W // 4:(i + 1) * W // 4] for i in range(4)]       # drop the render's own titles
panels += [b[60:h2 - 40, i * w2 // 2 + 40:(i + 1) * w2 // 2 - 40] for i in range(2)]
titles = ["(a) camera mount,\nupstream", "(b) camera mount,\nours (2nd window)", "(c) servo cover,\nupstream",
          "(d) servo cover,\nours (wider cable slot)", "(e) controller-support connector,\nours (85.7×21.4×42.1 mm)",
          "(f) connector, plate y +10 mm,\nours (85.7×31.4×42.1 mm)"]
fig, ax = plt.subplots(1, 6, figsize=(14, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.2, 1.2]))
for k, (im, t) in enumerate(zip(panels, titles)):
    ax[k].imshow(im)
    ax[k].set_title(t, fontsize=9)
    ax[k].axis("off")
plt.tight_layout(w_pad=0.3)
fig.savefig(Path(__file__).parent / "fig10_handumi_mods.pdf", bbox_inches="tight", dpi=200)
