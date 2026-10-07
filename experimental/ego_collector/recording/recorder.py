"""Raw episode recorder: capture thread -> encoder thread -> head.mp4 + frame_timestamps.parquet.

Recording never depends on AprilTag or MediaPipe; the raw video is the source of truth.
Keys in the preview window: R start, S stop/save, D discard, Q quit.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np

from ego_collector.camera.capture import CameraCapture, CaptureConfig, Frame
from ego_collector.io.parquet import write_parquet
from ego_collector.recording.episode import EpisodePaths, episode_dir_name, next_episode_id
from ego_collector.recording.metadata import EpisodeMetadata
from ego_collector.recording.episode import EpisodePaths as _EP  # noqa: F401 (type use)

log = logging.getLogger("ego_collector.recorder")

CODEC_CANDIDATES = ("avc1", "mp4v")  # avc1 (H.264) when the OpenCV build has it, else MPEG-4 part 2


@dataclass
class EpisodeWriter:
    """Encodes frames on its own thread and keeps the per-frame timestamp table."""

    paths: EpisodePaths
    width: int
    height: int
    fps: int
    codec: str = ""
    depth_intrinsics: dict | None = None  # {fx,fy,cx,cy} from the RGB-D color stream; enables depth PNGs + intrinsics.yaml
    depth_scale: float | None = None  # mm per depth unit (RGB-D)
    frame_indices: list[int] = field(default_factory=list)
    timestamps_ns: list[int] = field(default_factory=list)
    capture_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._writer: cv2.VideoWriter | None = None
        for codec in CODEC_CANDIDATES:
            w = cv2.VideoWriter(str(self.paths.video), cv2.VideoWriter_fourcc(*codec), float(self.fps), (self.width, self.height))
            if w.isOpened():
                self._writer, self.codec = w, codec
                break
            w.release()
        if self._writer is None:
            raise RuntimeError("no working mp4 codec (tried %s)" % ",".join(CODEC_CANDIDATES))
        self._queue: Queue[Frame | None] = Queue(maxsize=600)
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=True)
        self._thread.start()
        self.started_ns = time.monotonic_ns()
        self.first_ts_ns: int | None = None
        self.last_ts_ns: int | None = None
        self._depth_dir = self.paths.root / "depth"  # created lazily on the first depth frame
        self._wrote_depth = False

    def put(self, frame: Frame) -> None:
        self._queue.put(frame)  # blocks if the encoder falls >20 s behind (never silently drops)

    def _run(self) -> None:
        assert self._writer is not None
        n = 0
        while True:
            item = self._queue.get()
            if item is None:
                break
            img = item.image
            if img.shape[1] != self.width or img.shape[0] != self.height:
                img = cv2.resize(img, (self.width, self.height))
            self._writer.write(img)
            if item.depth is not None:  # RGB-D: save 16-bit depth aligned to color (color stays the mp4 source of truth)
                if not self._wrote_depth:
                    self._depth_dir.mkdir(parents=True, exist_ok=True)
                    self._wrote_depth = True
                cv2.imwrite(str(self._depth_dir / f"frame_{n:04d}.png"), item.depth)
            self.frame_indices.append(n)
            self.capture_indices.append(item.index)
            self.timestamps_ns.append(item.timestamp_ns)
            if self.first_ts_ns is None:
                self.first_ts_ns = item.timestamp_ns
            self.last_ts_ns = item.timestamp_ns
            n += 1

    def close(self) -> dict[str, float | int | str]:
        self._queue.put(None)
        self._thread.join()
        assert self._writer is not None
        self._writer.release()
        if self._wrote_depth and self.depth_intrinsics is not None:
            di = self.depth_intrinsics
            (self.paths.root / "intrinsics.yaml").write_text(
                f"color: {{fx: {di['fx']:.4f}, fy: {di['fy']:.4f}, cx: {di['cx']:.4f}, cy: {di['cy']:.4f}}}\n"
                f"resolution: [{self.width}, {self.height}]\n"
                f"depth_scale_mm: {self.depth_scale}\n"
                "note: depth is aligned to color; use color intrinsics for both\n"
            )
        n = len(self.frame_indices)
        duration = ((self.last_ts_ns - self.first_ts_ns) / 1e9) if n > 1 else 0.0
        write_parquet(
            self.paths.timestamps,
            {
                "frame_index": np.asarray(self.frame_indices, dtype=np.int64),
                "capture_index": np.asarray(self.capture_indices, dtype=np.int64),
                "timestamp_ns": np.asarray(self.timestamps_ns, dtype=np.int64),
            },
        )
        gaps = int(np.sum(np.diff(self.capture_indices) - 1)) if n > 1 else 0
        dt_ms = np.diff(np.asarray(self.timestamps_ns, dtype=np.float64)) / 1e6 if n > 1 else np.zeros(0)
        timing = {
            "frame_interval_ms_p50": float(np.percentile(dt_ms, 50)) if dt_ms.size else 0.0,
            "frame_interval_ms_p99": float(np.percentile(dt_ms, 99)) if dt_ms.size else 0.0,
            "frame_interval_ms_max": float(dt_ms.max()) if dt_ms.size else 0.0,
            "frames_over_1p5x_interval": int(np.sum(dt_ms > 1.5 * 1000.0 / self.fps)) if dt_ms.size else 0,
        }
        fps_measured = float((n - 1) / duration) if duration > 0 else 0.0
        if n > 10 and fps_measured < 0.9 * self.fps:
            log.warning(
                "measured %.1f fps < target %d: the camera itself slowed down (C922 lowers frame rate in dim light "
                "via auto-exposure) - add light or fix exposure; no frames were lost by the recorder (%d capture gaps).",
                fps_measured, self.fps, gaps,
            )
        return {
            "frame_count": n,
            "duration_s": float(duration),
            "fps_measured": fps_measured,
            "dropped_frames": gaps,  # capture indices skipped between written frames
            "video_codec": self.codec,
            **timing,
        }


class BatchTimer:
    """Timed batch collection: N episodes of record_s seconds with rest_s pauses between.

    Drives the interactive loop: phase() returns (phase, seconds_left, episode_no).
    Phases: "armed" (waiting for R), "rec", "rest", "done".
    """

    def __init__(self, episodes: int, record_s: float, rest_s: float) -> None:
        self.episodes = episodes
        self.record_s = record_s
        self.rest_s = rest_s
        self.count = 0  # completed episodes
        self.phase_name = "armed"
        self.deadline = 0.0

    def start(self, now: float) -> None:
        self.phase_name = "rec"
        self.deadline = now + self.record_s

    def phase(self, now: float) -> tuple[str, float, int]:
        return self.phase_name, max(0.0, self.deadline - now), self.count

    def tick(self, now: float) -> str | None:
        """Advance on deadline; returns an action: 'stop' (save + enter rest), 'start' (begin recording), None."""
        if self.phase_name == "rec" and now >= self.deadline:
            self.count += 1
            if self.count >= self.episodes:
                self.phase_name = "done"
                return "stop"
            self.phase_name = "rest"
            self.deadline = now + self.rest_s
            return "stop"
        if self.phase_name == "rest" and now >= self.deadline:
            self.start(now)
            return "start"
        return None

    def discard_current(self, now: float) -> None:
        """Discard the running episode and retake the same index after a rest."""
        if self.phase_name == "rec":
            self.phase_name = "rest"
            self.deadline = now + self.rest_s


def _beep() -> None:
    import subprocess

    try:
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


class Recorder:
    def __init__(
        self,
        *,
        raw_root: Path,
        capture: CaptureConfig,
        task: str,
        instruction: str,
        operator_id: str = "op01",
        camera_name: str = "Logitech C922",
        camera_calibration: str | None = None,
        tag_family: str | None = None,
        preview: bool = True,
        preview_scale: float = 0.5,
        tag_monitor=None,
        hand_monitor=None,
        diag: bool = False,
        layout: str = "",
        order: str = "",
        batch: "BatchTimer | None" = None,
        batch_orders: list[str] | None = None,
        colors: dict[str, str] | None = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.capture_cfg = capture
        self.task = task
        self.instruction = instruction
        self.operator_id = operator_id
        self.camera_name = camera_name
        self.camera_calibration = camera_calibration
        self.tag_family = tag_family
        self.preview = preview
        self.preview_scale = preview_scale
        self.current: EpisodeWriter | None = None
        self.current_meta: EpisodeMetadata | None = None
        self.saved: list[Path] = []
        self.tag_monitor = tag_monitor
        self.hand_monitor = hand_monitor
        self.show_tags = tag_monitor is not None
        self.show_hands = hand_monitor is not None
        self.last_saved: EpisodePaths | None = None
        self.last_visibility: dict[str, float] | None = None
        self.flash: tuple[str, float] = ("", 0.0)  # transient status message, expiry time
        self.diag = diag
        self.layout = layout
        self.order = order
        self.batch = batch
        self.batch_orders = batch_orders or []
        self.colors = colors or {}  # initial -> color name, e.g. {"R": "red"}

    def order_for(self, ep_index: int) -> str | None:
        if not self.batch_orders:
            return None
        return self.batch_orders[ep_index % len(self.batch_orders)]

    def order_display(self, order: str) -> str:
        arrows = "\u2192".join(order)
        names = "\u2192".join(self.colors.get(ch, ch) for ch in order)
        return f"{arrows}  ({names})"

    def _apply_batch_order(self, ep_index: int) -> None:
        order = self.order_for(ep_index)
        if order is None:
            return
        self.order = order
        seq = [self.colors.get(ch, ch) for ch in order]
        self.instruction = f"Stack the cubes in this order: {seq[0]} at the bottom, {seq[1]} in the middle, {seq[2]} on top."

    # -- episode control ---------------------------------------------------------

    def start_episode(self, cam: CameraCapture) -> EpisodePaths:
        eid = next_episode_id(self.raw_root)
        paths = EpisodePaths(self.raw_root / episode_dir_name(eid))
        w, h = int(cam.actual.get("width", self.capture_cfg.width)), int(cam.actual.get("height", self.capture_cfg.height))
        self.current = EpisodeWriter(
            paths=paths, width=w, height=h, fps=self.capture_cfg.fps,
            depth_intrinsics=getattr(cam, "intrinsics", None),  # set only by OrbbecCapture (RGB-D)
            depth_scale=getattr(cam, "depth_scale", None),
        )
        self.current_meta = EpisodeMetadata(
            episode_id=eid,
            task=self.task,
            instruction=self.instruction,
            operator_id=self.operator_id,
            camera=self.camera_name,
            resolution=[w, h],
            fps_target=self.capture_cfg.fps,
            camera_index=int(self.capture_cfg.index) if isinstance(self.capture_cfg.index, int) else 0,
            camera_calibration=self.camera_calibration,
            tag_family=self.tag_family,
            layout=self.layout,
            order=self.order,
        )
        cam.drain()  # start the episode from fresh frames
        if self.tag_monitor is not None:
            self.tag_monitor.start_counting()
        log.info("episode %06d recording -> %s", eid, paths.root)
        return paths

    def stop_episode(self, *, discard: bool = False) -> Path | None:
        if self.current is None or self.current_meta is None:
            return None
        stats = self.current.close()
        paths = self.current.paths
        visibility = self.tag_monitor.stop_counting() if self.tag_monitor is not None else None
        if discard:
            shutil.rmtree(paths.root, ignore_errors=True)
            log.info("episode %06d discarded", self.current_meta.episode_id)
            self.current = self.current_meta = None
            return None
        meta = self.current_meta
        meta.frame_count = int(stats["frame_count"])
        meta.dropped_frames = int(stats["dropped_frames"])
        meta.duration_s = float(stats["duration_s"])
        meta.fps_measured = float(stats["fps_measured"])
        meta.video_codec = str(stats["video_codec"])
        data = meta.to_dict()
        data["timing"] = {k: v for k, v in stats.items() if k.startswith("frame_interval") or k.startswith("frames_over")}
        if visibility is not None:
            data["live_visibility"] = visibility  # preview-rate estimate; process-tags gives the exact numbers
        paths.write_metadata(data)
        self.last_saved = paths
        self.last_visibility = visibility
        log.info(
            "episode %06d saved: %d frames, %.1fs @ %.2f fps, %d dropped (%s)",
            meta.episode_id, meta.frame_count, meta.duration_s, meta.fps_measured, meta.dropped_frames, meta.video_codec,
        )
        if visibility is not None and visibility.get("frames", 0) > 0:
            log.info("live visibility: L %.0f%%  R %.0f%%  world %.0f%%  ALL %.0f%%", visibility["left"] * 100, visibility["right"] * 100, visibility["world"] * 100, visibility["all_required"] * 100)
        self.saved.append(paths.root)
        self.current = self.current_meta = None
        return paths.root

    # -- main loops ----------------------------------------------------------------

    def _label_last(self, outcome: str | None = None, toggle_recovery: bool = False) -> None:
        if self.last_saved is None:
            self._say("no saved episode to label")
            return
        meta = self.last_saved.read_metadata()
        fields = {}
        if outcome is not None:
            fields["outcome"] = outcome
        if toggle_recovery:
            fields["recovery"] = not bool(meta.get("recovery", False))
        meta = self.last_saved.update_metadata(**fields)
        self._say(f"ep {meta.get('episode_id', 0):06d}: outcome={meta.get('outcome')} recovery={meta.get('recovery')}")

    def _say(self, text: str, seconds: float = 3.0) -> None:
        self.flash = (text, time.monotonic() + seconds)
        log.info(text)

    _KOR_VOICE: str | None = None
    _KOR_VOICE_CHECKED: bool = False
    _KOR_COLOR = {"red": "빨강", "blue": "파랑", "purple": "보라", "yellow": "노랑",
                  "green": "초록", "orange": "주황"}

    def _korean_voice(self) -> str | None:
        cls = type(self)
        if not cls._KOR_VOICE_CHECKED:
            cls._KOR_VOICE_CHECKED = True
            try:
                import subprocess
                out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    if "ko_KR" in line:
                        cls._KOR_VOICE = line.split()[0]
                        break
            except Exception:  # noqa: BLE001
                cls._KOR_VOICE = None
        return cls._KOR_VOICE

    def _speak(self, text: str) -> None:
        """Non-blocking macOS TTS (voice cue for eyes-free recording)."""
        try:
            import subprocess
            cmd = ["say"]
            v = self._korean_voice()
            if v:
                cmd += ["-v", v]
            cmd.append(text)
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            pass
        log.info("SPEAK: %s", text)

    def _order_colors_kr(self, order: str | None) -> list[str]:
        if not order:
            return []
        colors = getattr(self, "colors", {}) or {}
        out = []
        for ch in order:
            name = colors.get(ch, ch)
            out.append(self._KOR_COLOR.get(name, name))
        return out

    def _stack_phrase(self, order: str | None) -> str:
        """쌓는 순서를 아래/가운데/위로 읽어주는 문구 (order = bottom,middle,top)."""
        c = self._order_colors_kr(order)
        if len(c) >= 3:
            return f"아래 {c[0]}, 가운데 {c[1]}, 위 {c[2]}"
        return " ".join(c)

    def _prompt_instruction(self) -> None:
        """Change task/instruction/layout/order between episodes (typed in the terminal)."""
        try:
            task = input(f"task id [{self.task}]: ").strip()
            instr = input(f"instruction [{self.instruction}]: ").strip()
            layout = input(f"layout [{self.layout or '-'}]: ").strip()
            order = input(f"order [{self.order or '-'}]: ").strip()
        except EOFError:
            return
        if task:
            self.task = task
        if instr:
            self.instruction = instr
        if layout:
            self.layout = "" if layout == "-" else layout
        if order:
            self.order = "" if order == "-" else order.upper()
        self._say(f"next episodes: {self.task} / \"{self.instruction}\" layout={self.layout or '-'} order={self.order or '-'}")

    def _make_camera(self):
        """CameraCapture for UVC (C922); OrbbecCapture when --camera orbbec was requested."""
        if str(self.capture_cfg.index) == "orbbec":
            from ego_collector.camera.orbbec_capture import OrbbecCapture  # lazy: only imports pyorbbecsdk on demand
            return OrbbecCapture(self.capture_cfg)
        return CameraCapture(self.capture_cfg)

    def run_interactive(self) -> list[Path]:
        with self._make_camera() as cam:
            log.info("camera %s: %s", self.capture_cfg.index, cam.actual)
            if self.tag_monitor is not None:
                self.tag_monitor.cam = cam
                self.tag_monitor.start()
            if self.hand_monitor is not None:
                self.hand_monitor.cam = cam
                self.hand_monitor.start()
            fps_t, fps_n, fps_val = time.monotonic(), 0, 0.0
            diag_t, loop_n, t_grab_max, t_show_max = time.monotonic(), 0, 0.0, 0.0
            log.info("capture mode: %s", "threaded" if self.capture_cfg.threaded else "main-thread (poll)")
            while True:
                t_a = time.monotonic()
                frame = cam.poll() if not self.capture_cfg.threaded else cam.get(timeout=0.5)
                t_b = time.monotonic()
                if frame is None:
                    if cam.read_failures > 30 and cam.frames_grabbed == 0:
                        raise RuntimeError("camera produces no frames")
                    time.sleep(0.002)
                    continue
                if self.current is not None:
                    self.current.put(frame)
                fps_n += 1
                loop_n += 1
                now = time.monotonic()
                if now - fps_t >= 1.0:
                    fps_val, fps_t, fps_n = fps_n / (now - fps_t), now, 0
                if self.diag and now - diag_t >= 1.0:
                    tm = self.tag_monitor
                    log.info(
                        "diag: loop %d/s  grab max %.0f ms  show max %.0f ms  cam grabbed %d fail %d  writer q %s  tags %s",
                        loop_n, t_grab_max * 1000, t_show_max * 1000, cam.frames_grabbed, cam.read_failures,
                        self.current._queue.qsize() if self.current is not None else "-",
                        f"{tm.fps:.1f}Hz" if tm is not None else "off",
                    )
                    diag_t, loop_n, t_grab_max, t_show_max = now, 0, 0.0, 0.0
                t_grab_max = max(t_grab_max, t_b - t_a)
                if self.batch is not None:
                    action = self.batch.tick(time.monotonic())
                    if action == "stop" and self.current is not None:
                        self.stop_episode()
                        _beep()
                        if self.batch.phase_name == "done":
                            self._say(f"batch done: {self.batch.count}/{self.batch.episodes} episodes", 10)
                            self._speak(f"레스트. 수집 완료. {self.batch.count}개.")
                        else:
                            nxt = self._stack_phrase(self.order_for(self.batch.count))
                            self._speak(f"레스트. 다음 {self.batch.count + 1}번. {nxt}.")
                    elif action == "start" and self.current is None:
                        self._apply_batch_order(self.batch.count)
                        self.start_episode(cam)
                        _beep()
                        self._speak("스타트.")
                if self.preview:
                    t_c = time.monotonic()
                    key = self._show(frame, fps_val, cam)
                    t_show_max = max(t_show_max, time.monotonic() - t_c)
                    if key in (ord("q"), ord("Q")):
                        if self.current is not None:
                            self.stop_episode()
                        break
                    if key in (ord("r"), ord("R")) and self.current is None:
                        if self.batch is not None:
                            if self.batch.phase_name in ("armed", "done"):
                                self.batch.count = 0
                                self.batch.start(time.monotonic())
                                self._apply_batch_order(0)
                                self.start_episode(cam)
                                _beep()
                                self._speak(f"스타트. 1번. {self._stack_phrase(self.order)}.")
                        else:
                            self.start_episode(cam)
                    elif key in (ord("s"), ord("S")) and self.current is not None:
                        self.stop_episode()
                        if self.batch is not None:
                            self.batch.discard_current(time.monotonic())
                    elif key in (ord("d"), ord("D")) and self.current is not None:
                        self.stop_episode(discard=True)
                        if self.batch is not None:
                            self.batch.discard_current(time.monotonic())
                            self._say("discarded - retaking the same episode after the rest")
                    elif key in (ord("t"), ord("T")) and self.tag_monitor is not None:
                        self.show_tags = not self.show_tags
                        self.tag_monitor.enabled = self.show_tags
                    elif key in (ord("h"), ord("H")) and self.hand_monitor is not None:
                        self.show_hands = not self.show_hands
                        self.hand_monitor.enabled = self.show_hands
                    elif key == ord("1"):
                        self._label_last("success")
                    elif key == ord("2"):
                        self._label_last("failure")
                    elif key == ord("3"):
                        self._label_last(toggle_recovery=True)
                    elif key in (ord("i"), ord("I")) and self.current is None:
                        self._prompt_instruction()
            for m in (self.tag_monitor, self.hand_monitor):
                if m is not None:
                    m.stop()
            if self.preview:
                cv2.destroyAllWindows()
        return self.saved

    def run_timed(self, duration_s: float, *, start_delay_s: float = 0.0) -> Path | None:
        """Headless recording of one episode (integration tests / smoke checks)."""
        with self._make_camera() as cam:
            log.info("camera %s: %s", self.capture_cfg.index, cam.actual)
            if start_delay_s > 0:
                time.sleep(start_delay_s)
            self.start_episode(cam)
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                frame = cam.get(timeout=0.5)
                if frame is not None and self.current is not None:
                    self.current.put(frame)
            out = self.stop_episode()
            if out is not None:
                EpisodePaths(out).update_metadata(capture_dropped_frames=cam.frames_dropped, capture_read_failures=cam.read_failures)
            return out

    def _show(self, frame: Frame, fps: float, cam: CameraCapture) -> int:
        cv2.imshow("ego_collector record", self.compose_preview(frame, fps, cam))
        return cv2.waitKey(1) & 0xFF

    def compose_preview(self, frame: Frame, fps: float, cam: CameraCapture | None) -> np.ndarray:
        """Preview image with overlays + status panel (pure function of current state; testable)."""
        cam_drop = cam.frames_dropped if cam is not None else 0
        sc = self.preview_scale
        img = cv2.resize(frame.image, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc != 1.0 else frame.image.copy()
        rec = self.current is not None
        colors = {"left": (255, 140, 40), "right": (40, 160, 255), "world": (60, 220, 60), "unknown": (180, 180, 180)}

        # -- tag overlay ---------------------------------------------------------------
        tag_line = "tags: off (T)"
        vis_now = {"left": False, "right": False, "world": 0}
        live_hands: dict = {}
        if self.tag_monitor is not None and self.show_tags:
            res = self.tag_monitor.latest()
            if res is not None:
                vis_now = {"left": res.left_visible, "right": res.right_visible, "world": res.world_visible}
                live_hands = res.hands or {}
                for d in res.detections:
                    role = res.roles.get(d.tag_id, "unknown")
                    col = colors.get(role, colors.get(role.split("_")[0], colors["unknown"]))
                    pts = (d.corners * sc).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [pts], True, col, 2)
                    label = f"{role} {d.tag_id}  {self.tag_monitor.quad_px(d):.0f}px"
                    dist = self.tag_monitor.distance_m(d, self.tag_monitor.wrist_size_m) if role in ("left", "right") else None
                    if dist is not None:
                        label += f" ~{dist:.2f}m"
                    if np.isfinite(d.decision_margin):
                        label += f" m{d.decision_margin:.0f}"
                    cv2.putText(img, label, tuple((d.corners[0] * sc).astype(int) + np.array([0, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                stale = frame.index - res.frame_index
                tag_line = f"tags {self.tag_monitor.fps:4.1f}Hz {res.process_ms:4.0f}ms" + (f" (lag {stale}f)" if stale > 10 else "")
        # -- hand overlay --------------------------------------------------------------
        hand_line = "hands: off (H)" if self.hand_monitor is not None else ""
        if self.hand_monitor is not None and self.show_hands:
            hres = self.hand_monitor.latest()
            if self.hand_monitor.error:
                hand_line = f"hands: {self.hand_monitor.error[:40]}"
            elif hres is not None:
                k = sc / hres.scale
                seen = []
                for side, h in hres.hands.items():
                    if not getattr(h, "visible", False):
                        continue
                    seen.append(side)
                    pts = (h.landmarks_2d * k).astype(int)
                    for p in pts:
                        cv2.circle(img, (int(p[0]), int(p[1])), 3, colors[side], -1)
                    cv2.line(img, tuple(pts[4]), tuple(pts[8]), (255, 255, 255), 2)
                hand_line = f"hands {self.hand_monitor.fps:3.1f}Hz: {', '.join(seen) or 'none'}"

        # -- status panel ----------------------------------------------------------------
        color = (40, 40, 230) if rec else (200, 200, 200)
        batch_line = None
        if self.batch is not None:
            ph, left, cnt = self.batch.phase(time.monotonic())
            cur_o = self.order_for(cnt)
            if ph == "armed":
                first = f"   first: {self.order_display(cur_o)}" if cur_o else ""
                batch_line = (f"BATCH ready: {self.batch.episodes} eps x {self.batch.record_s:.0f}s rec / {self.batch.rest_s:.0f}s rest - press R{first}", (0, 255, 255))
            elif ph == "rec":
                this_o = f"   THIS: {self.order_display(cur_o)}" if cur_o else ""
                batch_line = (f"REC {cnt + 1}/{self.batch.episodes}  {left:4.1f}s{this_o}", (40, 40, 230))
            elif ph == "rest":
                nxt = f"   NEXT: {self.order_display(cur_o)}" if cur_o else ""
                batch_line = (f"REST - reset cubes!  ep {cnt + 1}/{self.batch.episodes} in {left:4.1f}s{nxt}", (0, 200, 255))
            else:
                batch_line = (f"BATCH DONE: {cnt}/{self.batch.episodes} episodes saved", (0, 255, 0))
        fps_warn = fps > 0 and fps < 28.0
        lines = [
            (f"REC  ep {self.current_meta.episode_id:06d}" if rec else f"IDLE  next ep {next_episode_id(self.raw_root):06d}", color),
            (f"cam {fps:4.1f} fps{'  << LOW: add light / fix exposure' if fps_warn else ''}   frames {len(self.current.frame_indices) if rec else 0}   drop {cam_drop}", (40, 200, 255) if fps_warn else (255, 255, 255)),
            (f"task: {self.task}   \"{self.instruction}\"" + (f"   layout {self.layout}" if self.layout else "") + (f"  order {self.order}" if self.order else ""), (255, 255, 255)),
            (tag_line + ("   " + hand_line if hand_line else ""), (255, 255, 255)),
        ]
        for side in ("left", "right"):
            h = live_hands.get(side)
            if h is None:
                continue
            parts = [f"{side[0].upper()}:"]
            if h["xyz"] is not None:
                x, y, z = h["xyz"]
                r, pch, yw = h["rpy_deg"]
                parts.append(f"wrist [{x:+.3f} {y:+.3f} {z:+.3f}]m  rpy [{r:+4.0f} {pch:+4.0f} {yw:+4.0f}]deg ({h['frame']})")
            else:
                parts.append("wrist --")
            ap = h["aperture_mm"]
            parts.append(f"grip {ap:5.1f}mm" if np.isfinite(ap) else "grip   --")
            parts.append(f"thumb {'OK' if h['thumb'] else '--'}  index {'OK' if h['index'] else '--'}")
            lines.append(("  ".join(parts), colors[side]))
        def flag(name, ok):
            return f"{name} {'OK' if ok else '--'}"
        need_world = self.tag_monitor is not None and bool(self.tag_monitor.world_ids)
        if self.tag_monitor is not None:
            now_line = f"now: {flag('L', vis_now['left'])}  {flag('R', vis_now['right'])}" + (f"  world {vis_now['world']}" if need_world else "")
            if rec and self.tag_monitor.counting:
                r = self.tag_monitor.counter.ratios()
                now_line += f"   episode: L {r['left']*100:3.0f}%  R {r['right']*100:3.0f}%" + (f"  W {r['world']*100:3.0f}%" if need_world else "") + f"  ALL {r['all_required']*100:3.0f}%"
            elif self.last_visibility:
                r = self.last_visibility
                now_line += f"   last ep: L {r['left']*100:3.0f}%  R {r['right']*100:3.0f}%" + (f"  W {r['world']*100:3.0f}%" if need_world else "") + f"  ALL {r['all_required']*100:3.0f}%"
            lines.append((now_line, (255, 255, 255)))
        if batch_line is not None:
            lines.insert(0, batch_line)
        if self.flash[0] and time.monotonic() < self.flash[1]:
            lines.append((self.flash[0], (0, 255, 255)))
        keys = ["R start   S stop/save   D discard   Q quit   T tags   H hands   I task/instruction", "1 success   2 failure   3 toggle recovery   (labels apply to the last saved episode)"]
        panel_h = 22 * len(lines) + 12
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], panel_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, img.shape[0] - 22 * len(keys) - 10), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, (text, col) in enumerate(lines):
            cv2.putText(img, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        for i, text in enumerate(keys):
            cv2.putText(img, text, (10, img.shape[0] - 22 * (len(keys) - i) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
        if rec:
            cv2.circle(img, (img.shape[1] - 30, 30), 12, (40, 40, 230), -1)
        # rest/armed 중 다음 쌓는 순서를 화면 중앙에 큰 색 큐브로 (한 줄 텍스트 잘림 방지)
        try:
            phase = self.batch.phase_name if self.batch is not None else None
        except Exception:
            phase = None
        if phase in ("rest", "armed"):
            cnt = len(self.saved)
            nxt = self.order_for(cnt)
            if nxt:
                COL = {"R": (40, 40, 220), "B": (200, 90, 40), "P": (200, 40, 160),
                       "Y": (40, 200, 220), "G": (40, 180, 40)}
                cw, chh = 120, 46; gap = 8
                cx = img.shape[1] // 2 - cw // 2
                top_y = img.shape[0] // 2 - (chh * 3 + gap * 2) // 2 - 10
                cv2.putText(img, "top", (cx - 46, top_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                for i, ch in enumerate(nxt[::-1]):  # top..bottom → 화면 위에서 아래
                    y = top_y + i * (chh + gap)
                    bg = COL.get(ch, (120, 120, 120))
                    cv2.rectangle(img, (cx, y), (cx + cw, y + chh), bg, -1)
                    cv2.rectangle(img, (cx, y), (cx + cw, y + chh), (0, 0, 0), 2)
                    nm = self.colors.get(ch, ch)
                    cv2.putText(img, nm, (cx + 8, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                by = top_y + 3 * (chh + gap)
                cv2.putText(img, "bottom (plate)", (cx - 46, by + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return img


__all__ = ["CODEC_CANDIDATES", "EpisodeWriter", "Recorder"]
