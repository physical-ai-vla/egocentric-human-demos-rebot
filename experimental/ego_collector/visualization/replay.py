"""Frame-by-frame replay with tracking overlays (mandatory check before mass collection).

Keys: space play/pause, n/p step, j/k ±30 frames, q quit.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.io.parquet import read_parquet
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.transforms import pose7_to_T

COLORS = {"world": (60, 220, 60), "left": (255, 140, 40), "right": (40, 160, 255), "unknown": (200, 200, 200)}
POSE = ("x", "y", "z", "qx", "qy", "qz", "qw")


class EpisodeReplay:
    def __init__(self, paths: EpisodePaths, intr: CameraIntrinsics | None = None) -> None:
        self.paths = paths
        self.intr = intr
        self.tags = read_parquet(paths.apriltags) if paths.apriltags.exists() else None
        self.cam = read_parquet(paths.camera_pose) if paths.camera_pose.exists() else None
        self.wrist = read_parquet(paths.wrist_pose) if paths.wrist_pose.exists() else None
        self.finger = read_parquet(paths.finger_pose) if paths.finger_pose.exists() else None
        self.hands = read_parquet(paths.hand_pose) if paths.hand_pose.exists() else None
        self.actions = read_parquet(paths.pseudo_actions) if paths.pseudo_actions.exists() else None
        self.meta = paths.read_metadata()
        self._tags_by_frame = {k: g for k, g in self.tags.groupby("frame_index")} if self.tags is not None else {}

    def overlay(self, idx: int, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        for _, r in self._tags_by_frame.get(idx, pd.DataFrame()).iterrows():
            c = np.asarray(r["corners"], dtype=np.float64).reshape(4, 2)
            color = COLORS.get(r["role"], COLORS["unknown"])
            cv2.polylines(out, [c.astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
            cv2.putText(out, str(int(r["tag_id"])), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self.wrist is not None and idx < len(self.wrist) and self.intr is not None:
            w = self.wrist.iloc[idx]
            for side in ("left", "right"):
                if bool(w.get(f"{side}_tag_visible", False)):
                    T = pose7_to_T([w[f"{side}_wrist_cam_{k}"] for k in POSE])
                    rvec, _ = cv2.Rodrigues(T[:3, :3])
                    cv2.drawFrameAxes(out, self.intr.camera_matrix, self.intr.distortion_coefficients, rvec, T[:3, 3], 0.04, 2)
        if self.finger is not None and idx < len(self.finger) and self.intr is not None:
            f = self.finger.iloc[idx]
            for side in ("left", "right"):
                pts = {}
                for fg in ("thumb", "index"):
                    if bool(f.get(f"{side}_{fg}_visible", False)):
                        p = np.array([[f[f"{side}_{fg}_cam_x"], f[f"{side}_{fg}_cam_y"], f[f"{side}_{fg}_cam_z"]]])
                        uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), self.intr.camera_matrix, self.intr.distortion_coefficients)
                        pts[fg] = uv.reshape(2)
                        cv2.circle(out, tuple(pts[fg].astype(int)), 6, COLORS[side], 2)
                if len(pts) == 2 and bool(f.get(f"{side}_aperture_valid", False)):
                    a, b = pts["thumb"].astype(int), pts["index"].astype(int)
                    cv2.line(out, tuple(a), tuple(b), (255, 255, 255), 2)
                    cv2.putText(out, f"{f[f'{side}_aperture_m']*1000:.0f}mm", tuple(((pts['thumb'] + pts['index']) / 2).astype(int) + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[side], 2)
        if self.hands is not None and idx < len(self.hands):
            h = self.hands.iloc[idx]
            for side in ("left", "right"):
                if bool(h.get(f"{side}_hand_visible", False)):
                    lm = np.asarray(h[f"{side}_landmarks_2d"], dtype=np.float64).reshape(-1, 2)
                    for p in lm:
                        cv2.circle(out, (int(p[0]), int(p[1])), 3, COLORS[side], -1)
                    for a, b in ((4, 8),):
                        cv2.line(out, tuple(lm[a].astype(int)), tuple(lm[b].astype(int)), (255, 255, 255), 2)
        y = 30
        lines = [f"frame {idx}  {self.meta.get('task', '')}: \"{self.meta.get('instruction', '')}\""]
        if self.cam is not None and idx < len(self.cam):
            c = self.cam.iloc[idx]
            lines.append(f"world {'OK ' if c['world_pose_valid'] else '-- '} tags {int(c['world_tag_count'])} reproj {c['world_reprojection_error']:.2f}px")
        for side in ("left", "right"):
            parts = [side.upper()]
            if self.wrist is not None and idx < len(self.wrist):
                w = self.wrist.iloc[idx]
                if bool(w[f"{side}_wrist_valid"]):
                    parts.append(f"xyz {w[f'{side}_wrist_x']:+.3f} {w[f'{side}_wrist_y']:+.3f} {w[f'{side}_wrist_z']:+.3f}")
                else:
                    parts.append("wrist --")
            if self.finger is not None and idx < len(self.finger):
                f = self.finger.iloc[idx]
                ap = f.get(f"{side}_aperture_m", float("nan"))
                parts.append(f"grip {ap*1000:.0f}mm" if np.isfinite(ap) else "grip --")
                parts.append(f"T{'+' if f.get(f'{side}_thumb_visible') else '-'}I{'+' if f.get(f'{side}_index_visible') else '-'}")
            if self.actions is not None and idx < len(self.actions):
                a = self.actions.iloc[idx]
                g = a.get(f"{side}_grasp", float("nan"))
                parts.append(f"grasp {g:.2f}" if np.isfinite(g) else "grasp --")
            lines.append("  ".join(parts))
        for text in lines:
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 28
        return out

    def run(self, *, scale: float = 0.6, start: int = 0) -> None:
        cap = cv2.VideoCapture(str(self.paths.video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = max(0, min(start, n - 1))
        playing = False
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                break
            shown = self.overlay(idx, img)
            if scale != 1.0:
                shown = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(f"replay {self.paths.root.name} (space/n/p/j/k/q)", shown)
            key = cv2.waitKey(33 if playing else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                playing = not playing
            elif key == ord("n") or playing:
                idx = min(n - 1, idx + 1)
            elif key == ord("p"):
                idx = max(0, idx - 1)
            elif key == ord("j"):
                idx = max(0, idx - 30)
            elif key == ord("k"):
                idx = min(n - 1, idx + 30)
        cap.release()
        cv2.destroyAllWindows()

    def render_video(self, out: Path, *, scale: float = 0.5) -> Path:
        """Write the overlay as a video (headless review / sharing)."""
        cap = cv2.VideoCapture(str(self.paths.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            shown = self.overlay(idx, img)
            if scale != 1.0:
                shown = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (shown.shape[1], shown.shape[0]))
            writer.write(shown)
            idx += 1
        cap.release()
        if writer is not None:
            writer.release()
        return out


__all__ = ["EpisodeReplay"]
