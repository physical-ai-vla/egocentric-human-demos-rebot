#!/usr/bin/env bash
# Publish-safe copy of a trajectory video from viz_mast3r_episode.py / viz_hra_mast3r.py (1280x720 layout):
# the wrist fisheye panels (x 0-480, y 240-720) show the office background and monitor screens, so they are covered
# by a grey label image; the head view, 3D trajectory and plots are kept. Output 960x540 H.264.
# usage: mask_wrist_panels.sh <in.mp4> <label.png 480x480> <out.mp4>
set -euo pipefail
ffmpeg -v error -y -i "$1" -i "$2" -filter_complex "[0:v][1:v]overlay=0:240,scale=960:540" -an \
  -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p -movflags +faststart "$3"
