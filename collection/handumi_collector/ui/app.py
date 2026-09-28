"""PySide6 production UI (M1.5, pre-IMU): one screen with 3 live previews (+ optional Orbbec), per-hand grip/IMU panels, task
/ order + dataset selectors, counters, disk, preflight checklist, HOME READY countdown, REC live stats, REVIEW summary with head
replay, KEEP / DISCARD, HARDWARE CHECK, gripper calibration; POSE and QA tabs. Raw is never touched by anything drawn here
(overlays are preview-only). Colours: GREEN ok, YELLOW optional missing / degraded, RED required failure, GREY not installed."""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W
from ..collector.session import CollectorSession
from ..config import AppConfig

GREEN, RED, GREY, AMBER = "#2ecc71", "#e74c3c", "#7f8c8d", "#f39c12"
COL = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED, "GRAY": GREY, "GREY": GREY}


def _qimage(bgr) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()


class Dot(W.QLabel):
    def set(self, color: str, text: str) -> None:
        self.setText(f'<span style="color:{color};font-size:16px">●</span> {text}')


class Preview(W.QWidget):
    """Camera tile: title + image + fps dot. Optional workspace-guide overlay (head) and undistort toggle (wrists, only
    with intrinsics). `depth=True` tiles render the depth map as a colour ramp instead of the colour image — the Orbbec
    was previously status-dot-only, which is how a pilot can be recorded pointing at the wrong thing: with no depth
    preview there is no way to see what the sensor under test is actually looking at."""

    def __init__(self, name: str, title: str, *, overlay: dict | None = None, rectifier=None, depth: bool = False,
                 near_m: float = 0.30, far_m: float = 1.20) -> None:
        super().__init__(); self.name, self.overlay, self.rect = name, overlay, rectifier
        self.is_depth, self.near_m, self.far_m = depth, float(near_m), float(far_m)
        v = W.QVBoxLayout(self); v.setContentsMargins(2, 2, 2, 2)
        top = W.QHBoxLayout(); self.title = W.QLabel(f"<b>{title}</b>"); top.addWidget(self.title); top.addStretch()
        self.dot = Dot(); top.addWidget(self.dot)
        self.undist = W.QCheckBox("undistort"); self.undist.setEnabled(rectifier is not None); self.undist.setToolTip("preview only; raw video stays fisheye")
        if name != "head" and not depth: top.addWidget(self.undist)
        v.addLayout(top)
        self.lab = W.QLabel(); self.lab.setMinimumSize(300, 200); self.lab.setStyleSheet("background:#111"); self.lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter); v.addWidget(self.lab)
        self.warn = W.QLabel(""); self.warn.setStyleSheet(f"color:{AMBER};font-weight:bold"); v.addWidget(self.warn)

    def show_depth(self, depth_raw: np.ndarray, depth_scale_mm: float | None = None) -> None:
        """Colour-ramp the depth map over [near_m, far_m]; BLACK means no depth return, which is the thing a colour
        image cannot tell you. The overlaid numbers are what you aim by: what fraction of the frame has valid depth,
        and how far the nearest surface actually is."""
        d = np.asarray(depth_raw, np.float32) * (float(depth_scale_mm or 1.0) * 1e-3)
        dn = np.clip((d - self.near_m) / max(self.far_m - self.near_m, 1e-6), 0.0, 1.0)
        vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[d <= 0] = 0
        in_range = (d > self.near_m) & (d < self.far_m)
        valid = float(np.mean(d > 0))
        med = float(np.median(d[in_range])) if in_range.any() else float("nan")
        cv2.putText(vis, f"valid {100*valid:4.1f}%   in-range {100*float(in_range.mean()):4.1f}%   median {med:.2f} m",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.warn.setText("" if valid > 0.5 else f"LOW DEPTH RETURN {valid:.0%} — too close, too dark or out of range")
        self.show_frame(vis)

    def show_frame(self, img: np.ndarray) -> None:
        if self.rect is not None and self.undist.isChecked():
            try: img = self.rect.rectify(img)
            except Exception: pass
        if img.shape[1] > 640: img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
        if self.overlay and self.name == "head":
            img = img.copy(); h, w = img.shape[:2]; o = self.overlay
            cv2.rectangle(img, (int(o["x0"] * w), int(o["y0"] * h)), (int(o["x1"] * w), int(o["y1"] * h)), (0, 255, 255), 2)
            cv2.putText(img, "WORKSPACE", (int(o["x0"] * w) + 6, int(o["y0"] * h) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        pm = QtGui.QPixmap.fromImage(_qimage(img)).scaled(self.lab.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio); self.lab.setPixmap(pm)


class GripGauge(W.QWidget):
    """Live jaw visualisation: bar = normalized opening (0 closed … 1 open) with raw ticks, sparkline = last ~6 s of history.
    Uncalibrated (norm NaN): bar shows raw ticks over 0..4095 in grey."""

    def __init__(self, side: str, history_s: float = 6.0, hz: float = 15.0) -> None:
        super().__init__(); self.side = side; self.hist: list[tuple[float, float]] = []; self.history_s = history_s
        self.raw = None; self.norm = None; self.err = ""; self.setMinimumHeight(96)

    def push(self, raw, norm, err: str = "") -> None:
        self.raw, self.norm, self.err = raw, norm, err
        v = norm if (norm is not None and norm == norm) else (None if raw is None else raw / 4095.0)
        if v is not None:
            t = time.monotonic(); self.hist.append((t, float(v))); self.hist = [(a, b) for a, b in self.hist if t - a <= self.history_s]
        self.update()

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing); w, h = self.width(), self.height()
        cal = self.norm is not None and self.norm == self.norm
        p.fillRect(0, 0, w, h, QtGui.QColor("#1b1b1b"))
        # bar
        bar = QtCore.QRectF(8, 8, w - 16, 26); p.setPen(QtGui.QPen(QtGui.QColor("#555"), 1)); p.drawRect(bar)
        v = (self.norm if cal else (None if self.raw is None else self.raw / 4095.0))
        if v is not None:
            fill = QtCore.QRectF(bar.left(), bar.top(), bar.width() * max(0.0, min(1.0, v)), bar.height())
            p.fillRect(fill, QtGui.QColor(GREEN if cal else GREY))
        p.setPen(QtGui.QColor("#eee")); p.setFont(QtGui.QFont("Menlo", 10))
        txt = (f"{self.side.upper()}  raw {self.raw}   norm {self.norm:.3f}  {'OPEN' if self.norm > 0.5 else 'CLOSED'}" if cal else f"{self.side.upper()}  raw {self.raw}   (uncalibrated: SET CLOSED / SET OPEN)") if self.raw is not None else f"{self.side.upper()}  no gripper samples"
        p.drawText(QtCore.QRectF(12, 8, w - 24, 26), QtCore.Qt.AlignmentFlag.AlignVCenter, txt)
        p.setPen(QtGui.QColor("#aaa")); p.setFont(QtGui.QFont("Menlo", 8)); p.drawText(10, 48, "closed 0"); p.drawText(w - 46, 48, "open 1")
        if self.err: p.setPen(QtGui.QColor(AMBER)); p.drawText(w // 2 - 60, 48, f"servo flags: {self.err}")
        # sparkline
        area = QtCore.QRectF(8, 52, w - 16, h - 60); p.setPen(QtGui.QPen(QtGui.QColor("#333"), 1)); p.drawRect(area)
        if len(self.hist) > 1:
            t1 = self.hist[-1][0]; pts = [QtCore.QPointF(area.left() + area.width() * (1 - (t1 - t) / self.history_s), area.bottom() - area.height() * max(0.0, min(1.0, v))) for t, v in self.hist]
            p.setPen(QtGui.QPen(QtGui.QColor(GREEN if cal else GREY), 1.5)); p.drawPolyline(QtGui.QPolygonF(pts))
        p.end()


class Replay(QtCore.QObject):
    """Loops head.mp4 in a Preview during REVIEW (decode thread → latest frame)."""

    def __init__(self, fps: int) -> None:
        super().__init__(); self.fps = fps; self._stop = threading.Event(); self._t = None; self.frame = None; self.path = None

    def start(self, path: str) -> None:
        self.stop(); self.path = path; self._stop.clear(); self._t = threading.Thread(target=self._run, daemon=True); self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t: self._t.join(1.0); self._t = None
        self.frame = None

    def _run(self) -> None:
        import av
        while not self._stop.is_set():
            try:
                with av.open(self.path) as c:
                    for fr in c.decode(video=0):
                        if self._stop.is_set(): return
                        self.frame = fr.to_ndarray(format="bgr24"); time.sleep(1.0 / self.fps)
            except Exception:
                time.sleep(0.5)


class MainWindow(W.QMainWindow):
    def __init__(self, session: CollectorSession) -> None:
        super().__init__(); self.s = session; cc = self.s.cfg.collector
        self.setWindowTitle("HandUMI Egocentric Collector")
        self.replay = Replay(cc.replay_fps); self._prev_head = None
        tabs = W.QTabWidget(); self.setCentralWidget(tabs)
        tabs.addTab(self._record_tab(), "RECORD"); tabs.addTab(self._pose_tab(), "POSE"); tabs.addTab(self._qa_tab(), "QA")
        for key, fn in (("Space", self.on_space), ("D", self.on_discard), ("K", self.on_keep), ("H", self.on_mark), ("A", self.on_auto)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, fn)
        self.set_order(self.s.order)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(int(1000 / cc.preview_hz))

    # ------------------------------------------------------------- layout
    def _record_tab(self) -> W.QWidget:
        root = W.QWidget(); v = W.QVBoxLayout(root); cc = self.s.cfg.collector
        hdr = W.QHBoxLayout(); hdr.addWidget(W.QLabel("<b>HANDUMI EGOCENTRIC COLLECTOR</b>")); hdr.addStretch()
        self.clock = W.QLabel(); hdr.addWidget(self.clock); v.addLayout(hdr)
        rectifiers = {}
        for side in ("left", "right"):
            try:
                from ..pose.calibration import SideCalibration
                from ..pose.backends.opencv_vo import FisheyeRectifier
                cal = SideCalibration(side)
                if cal.intrinsics is not None and cal.intrinsics.get("model") != "pinhole": rectifiers[f"{side}_wrist"] = FisheyeRectifier(cal.intrinsics, fov_deg=100)
            except Exception: pass
        # One tile per CONFIGURED camera, in profile order — not a hardcoded head/left/right triple, so a profile
        # without a C922 (the Stage-A RGB-D pilot) still previews what it is actually recording.
        row = W.QHBoxLayout(); self.previews: dict[str, Preview] = {}
        titles = {"head": "HEAD C922", "left_wrist": "LEFT WRIST", "right_wrist": "RIGHT WRIST", "head_depth": "HEAD DEPTH (Orbbec)"}
        dp = cc.depth_preview
        # [2026-09-18] The Orbbec carries an RGB stream alongside depth, and that colour view is what the head actually
        # sees; showing it first makes it obvious whether the depth tile is aimed where the operator thinks it is.
        ordered = list(self.s.cfg.hardware.cameras)
        for cam_cfg in ordered:
            if cam_cfg.role == "aux_depth":
                p = Preview(cam_cfg.name + "_rgb", "HEAD RGB (Orbbec)", rectifier=None, depth=False)
                row.addWidget(p); self.previews[cam_cfg.name + "_rgb"] = p
                break
        for cam_cfg in ordered:
            name = cam_cfg.name; is_depth = cam_cfg.role == "aux_depth"
            title = titles.get(name) or (name.upper().replace("_", " ") + (" (DEPTH)" if is_depth else ""))
            p = Preview(name, title, overlay=cc.head_overlay if name == "head" else None, rectifier=rectifiers.get(name),
                        depth=is_depth, near_m=float(dp.get("near_m", 0.30)), far_m=float(dp.get("far_m", 1.20)))
            row.addWidget(p); self.previews[name] = p
        v.addLayout(row)
        aux = W.QHBoxLayout(); self.aux_dot = Dot(); aux.addWidget(W.QLabel("<b>AUX ORBBEC RGB-D</b>")); aux.addWidget(self.aux_dot); aux.addWidget(W.QLabel("OPTIONAL")); aux.addStretch(); v.addLayout(aux)
        g3 = W.QGroupBox("LEFT HAND / RIGHT HAND"); g3l = W.QGridLayout(g3); self.side_panels: dict[str, W.QLabel] = {}; self.gauges: dict[str, GripGauge] = {}
        for col, side in enumerate(("left", "right")):
            lab = W.QLabel(); lab.setStyleSheet("font-family:monospace"); lab.setTextFormat(QtCore.Qt.TextFormat.RichText)
            g3l.addWidget(W.QLabel(f"<b>{side.upper()}</b>"), 0, col); g3l.addWidget(lab, 1, col); self.side_panels[side] = lab
            gg = GripGauge(side); g3l.addWidget(gg, 2, col); self.gauges[side] = gg
            hb2 = W.QHBoxLayout()
            for which in ("closed", "open"):
                b = W.QPushButton(f"SET {which.upper()} ({side[0].upper()})"); b.clicked.connect(lambda _=False, sd=side, wh=which: self.on_cal(sd, wh)); hb2.addWidget(b)
            g3l.addLayout(hb2, 3, col)
        bsave = W.QPushButton("SAVE gripper calibration (new version)"); bsave.clicked.connect(self.on_cal_save); g3l.addWidget(bsave, 4, 0, 1, 2)
        self.cal_label = W.QLabel(); g3l.addWidget(self.cal_label, 5, 0, 1, 2)
        # The test stays available for diagnosing a suspect servo, but it only occupies the panel when it gates REC.
        pol_gate = bool(getattr(self.s.cfg.collector, "polarity_test_required", False))
        bpol = W.QPushButton("POLARITY TEST (open→close→open both jaws ×3, 8 s)"
                             + (" — required before REC" if pol_gate else " — optional")); bpol.setStyleSheet("font-weight:bold")
        bpol.clicked.connect(self.on_polarity_test); g3l.addWidget(bpol, 6, 0, 1, 2)
        self.pol_label = W.QLabel("polarity: not tested"); g3l.addWidget(self.pol_label, 7, 0, 1, 2); v.addWidget(g3)
        if not pol_gate:
            bpol.setVisible(False); self.pol_label.setVisible(False)
        mid = W.QHBoxLayout()
        g = W.QGroupBox("TASK"); gl = W.QVBoxLayout(g); br = W.QHBoxLayout(); self.order_btns: dict[str, W.QPushButton] = {}
        for o in self.s.cfg.tasks.orders:
            b = W.QPushButton(o); b.setCheckable(True); b.clicked.connect(lambda _=False, oo=o: self.set_order(oo)); br.addWidget(b); self.order_btns[o] = b
        nb = W.QPushButton("next least-collected"); nb.clicked.connect(lambda: self.set_order(self.s.manager.next_order())); br.addWidget(nb)
        gl.addLayout(br); self.instr = W.QLabel(); self.instr.setWordWrap(True); gl.addWidget(self.instr); self.suggest = W.QLabel(); gl.addWidget(self.suggest)
        proto = W.QLabel("Protocol: HEAD C922 fixed mount · plate fixed · cubes red/blue/purple · layout ≈ robot R150 · 6 balanced orders · HOME still 1–2 s at start and end")
        proto.setStyleSheet("color:#555"); proto.setWordWrap(True); gl.addWidget(proto); mid.addWidget(g, 3)
        gd = W.QGroupBox("DATASET"); gdl = W.QVBoxLayout(gd); self.dataset_lab = W.QLabel(f"<b>{self.s.manager.dataset}</b>  target/order {self.s.cfg.tasks.target_per_order}"); gdl.addWidget(self.dataset_lab)
        self.counter = W.QLabel(); self.counter.setStyleSheet("font-family:monospace"); gdl.addWidget(self.counter)
        self.disk_lab = Dot(); gdl.addWidget(self.disk_lab); mid.addWidget(gd, 2)
        gp = W.QGroupBox("PREFLIGHT"); gpl = W.QVBoxLayout(gp); self.preflight: list[W.QCheckBox] = []
        for item in cc.preflight:
            cb = W.QCheckBox(item); cb.stateChanged.connect(self._preflight_changed); gpl.addWidget(cb); self.preflight.append(cb)
        if not cc.preflight: gpl.addWidget(W.QLabel("(no checklist configured)"))
        mid.addWidget(gp, 3); v.addLayout(mid)
        g2 = W.QGroupBox("DEVICES"); grid = W.QGridLayout(g2); self.dots: dict[str, Dot] = {}
        names = [d.name for d in self.s.devices.all_devices()]
        for side in ("left", "right"):
            if f"imu_{side}" not in names: names.append(f"imu_{side}")
        for i, n in enumerate(names):
            d = Dot(); self.dots[n] = d; grid.addWidget(d, i // 4, i % 4)
        v.addWidget(g2)
        self.ident = W.QLabel(); self.ident.setStyleSheet("font-family:monospace;color:#555"); self.ident.setWordWrap(True); v.addWidget(self.ident)
        self.rec = W.QLabel(); self.rec.setStyleSheet("font-size:20px;font-weight:bold"); v.addWidget(self.rec)
        self.live = W.QLabel(); self.live.setStyleSheet("font-family:monospace"); v.addWidget(self.live)
        self.review = W.QLabel(); self.review.setStyleSheet("font-family:monospace;background:#fdf6e3;padding:6px"); self.review.setTextFormat(QtCore.Qt.TextFormat.RichText); self.review.hide(); v.addWidget(self.review)
        hb = W.QHBoxLayout()
        st_cfg = cc.stillness; self.b_home = W.QPushButton(f"● START — waits for {float(st_cfg.get('min_still_s', 3)):.0f} s stillness on both IMUs"); self.b_start = W.QPushButton("● START  (Space)"); self.b_stop = W.QPushButton("■ STOP / CANCEL  (Space)")
        self.b_mark = W.QPushButton("MARK HOME leave/return (H)"); self.b_keep = W.QPushButton("KEEP  (K)"); self.b_discard = W.QPushButton("DISCARD  (D)"); self.b_hw = W.QPushButton("HARDWARE CHECK (5 s)")
        al = cc.auto_loop; self.b_auto = W.QPushButton(f"▶▶ AUTO  (A)  {float(al.get('episode_s', 10)):.0f} s stack / {float(al.get('reset_s', 10)):.0f} s reset, voice"); self.b_auto.setCheckable(True); self.b_auto.clicked.connect(self.on_auto)
        for b in (self.b_home, self.b_start, self.b_stop, self.b_mark, self.b_keep, self.b_discard, self.b_hw): b.setMinimumHeight(44); hb.addWidget(b)
        self.b_home.clicked.connect(self.on_home_ready); self.b_start.clicked.connect(self.on_start); self.b_stop.clicked.connect(self.on_stop); self.b_mark.clicked.connect(self.on_mark)
        self.b_keep.clicked.connect(self.on_keep); self.b_discard.clicked.connect(self.on_discard); self.b_hw.clicked.connect(self.on_hwcheck); v.addLayout(hb)
        v.addWidget(self.b_auto)
        v.addWidget(W.QLabel("Keyboard: SPACE = start/stop · K = keep · D = discard · H = HOME mark · A = AUTO loop on/off"))
        self.logbox = W.QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumHeight(90); v.addWidget(self.logbox); self._nlog = 0
        return root

    def _pose_tab(self) -> W.QWidget:
        root = W.QWidget(); v = W.QVBoxLayout(root)
        pc = self.s.cfg.pose; imu = bool(self.s.devices.imus); mode = "VISUAL-INERTIAL" if imu else "VISUAL ONLY (pre-IMU experimental)"
        v.addWidget(W.QLabel(f"<b>Pose tracking</b> — offline, never a recording dependency.  LEFT tracking: {mode}   RIGHT tracking: {mode}"))
        v.addWidget(W.QLabel(f"configured backend: <b>{pc.backend if pc else '—'}</b>  mode: {pc.tracking_mode if pc else '—'}  lead {pc.lead if pc else '—'} / horizon {pc.horizon if pc else '—'} @ {pc.fps if pc else '—'} Hz"))
        try:
            from ..pose.calibration import SideCalibration
            for side in ("left", "right"):
                c = SideCalibration(side).summary(); v.addWidget(W.QLabel(f"{side}: intrinsics {c['versions']['fisheye'] or 'NONE'} · T_camera_imu {c['versions']['camera_imu'] or 'NONE'} · T_camera_tcp {c['versions']['camera_tcp'] or 'NONE'}"))
        except Exception as exc: v.addWidget(W.QLabel(f"calibration: {exc}"))
        v.addWidget(W.QLabel("Process:  .venv/bin/python -m handumi_collector.tools.pose_process <session>   ·   Inspect:  tools.pose_inspector --episode <ep>\n"
                             "Per-hand relative TCP only (no shared L/R world). IMU arrival: plug the Teensy units in, switch to hardware_handumi_v1.yaml, set pose.yaml backend to a visual-inertial one."))
        self.pose_live = W.QLabel("live tracking: not running (continuous_session mode is optional; raw acquisition never waits for it)"); v.addWidget(self.pose_live); v.addStretch(); return root

    def _qa_tab(self) -> W.QWidget:
        root = W.QWidget(); v = W.QVBoxLayout(root)
        b = W.QPushButton("Load latest pose_qa.json of this session"); b.clicked.connect(self.on_load_qa); v.addWidget(b)
        self.qa_text = W.QPlainTextEdit(); self.qa_text.setReadOnly(True); v.addWidget(self.qa_text); return root

    # ------------------------------------------------------------- actions
    def set_order(self, o: str) -> None:
        if self.s.state != "IDLE" or not o: return
        self.s.order = o
        for k, b in self.order_btns.items(): b.setChecked(k == o)
        self.instr.setText(f"<b>Current: {o}</b> — <i>{self.s.cfg.tasks.instruction(o)}</i>")

    def _preflight_changed(self, *_): self.s.preflight_ok = all(cb.isChecked() for cb in self.preflight) if self.preflight else True

    def on_space(self) -> None:
        if self.s.state == "IDLE": self.on_start()
        elif self.s.state == "WAITING_FOR_STILLNESS": self.s.cancel_start()
        elif self.s.state == "RECORDING": self.on_stop()

    def on_home_ready(self) -> None: self.on_start()

    def on_auto(self) -> None:
        """Hands-free loop: RESET -> still -> REC -> three two one go -> stack -> stop -> KEEP -> RESET ... (collector/autoloop.py)."""
        if self.s.state not in ("IDLE", "WAITING_FOR_STILLNESS", "RECORDING"): self.s.say("AUTO: finish KEEP/DISCARD first, then A resumes"); self.b_auto.setChecked(False); return
        on = self.s.auto_toggle(); self.b_auto.setChecked(on)
        if on: self.replay.stop(); self.review.hide()

    def on_start(self) -> None:
        """START arms the stillness gate; recording begins from session.poll() once every IMU has been still long enough."""
        ok, why = self.s.can_record()
        if not ok: self.s.say(f"cannot start: {why}"); return
        self.replay.stop(); self.review.hide(); self.s.request_start()

    def on_stop(self) -> None:
        if self.s.state == "RECORDING": self.s.stop(); self._enter_review()
        elif self.s.state == "WAITING_FOR_STILLNESS": self.s.cancel_start()

    def _enter_review(self) -> None:
        try:
            r = self.s.review_summary()
            st = "".join(f"<tr><td>{n}</td><td align=right>{d['frames']}</td><td align=right>{d['drops']}</td><td align=right>{d['fps']}</td></tr>" for n, d in r["streams"].items())
            gr = " · ".join(f"grip {k} {v}" for k, v in r["grippers"].items()); im = " · ".join(f"imu {k} {v}" for k, v in r["imus"].items()) or "IMU: not installed"
            col = GREEN if r["verdict"] == "PASS" else AMBER
            self.review.setText(f"<b>REVIEW {r['episode']}</b> order {r['order']} · {r['duration_s']} s · HW errors {r['hw_errors']} · events {r['events']}<br>"
                                f"<table cellspacing=6><tr><th align=left>stream</th><th>frames</th><th>drops</th><th>fps</th></tr>{st}</table>{gr} · {im}<br>"
                                f"Preliminary QA: <span style='color:{col};font-weight:bold'>{r['verdict']}</span> {'; '.join(r['notes'])}<br><b>KEEP (K)</b> or <b>DISCARD (D)</b> — head replay is looping in the HEAD tile")
            self.review.show()
            if Path(r["head_video"]).exists(): self.replay.start(r["head_video"])
        except Exception as exc: self.s.say(f"review summary: {exc}")

    def on_mark(self) -> None:
        if self.s.state == "RECORDING":
            try: self.s.mark()
            except Exception as exc: self.s.say(f"mark: {exc}")

    def on_keep(self) -> None:
        if self.s.state in ("REVIEW", "ERROR_REVIEW"):
            m = self.s.keep(); self.replay.stop(); self.review.hide(); self.s.say(f"KEPT {m['episode_dir']} quality={m['quality']}"); self.set_order(self.s.order)

    def on_discard(self) -> None:
        if self.s.state in ("REVIEW", "ERROR_REVIEW"):
            p = self.s.discard(); self.replay.stop(); self.review.hide(); self.s.say(f"DISCARDED -> {p.name}"); self.set_order(self.s.order)

    def on_hwcheck(self) -> None:
        if self.s.state != "IDLE": return
        self.s.say("HARDWARE CHECK: recording 5 s test episode …"); W.QApplication.processEvents()
        try:
            r = self.s.hardware_check(5.0)
            txt = ("<span style='color:%s;font-weight:bold'>%s</span>" % ((GREEN, "READY FOR COLLECTION") if r["ready"] else (RED, "HARDWARE CHECK FAILED")))
            det = " · ".join(f"{n}: {d.get('video_frames', '?')} f @ {d.get('fps', '?')} fps (max gap {d.get('max_gap_ms', '?')} ms)" for n, d in r.get("streams", {}).items())
            self.review.setText(f"{txt}<br>{det}<br>grippers {r.get('grippers_live')} · optional missing {r.get('optional_missing')}<br>{'; '.join(r.get('problems', []))}<br><i>{r.get('path', '')}</i>"); self.review.show()
        except Exception as exc: self.s.say(f"hardware check: {exc}")

    def on_polarity_test(self) -> None:
        try: self.s.polarity_test_begin()
        except Exception as exc: self.s.say(f"polarity test: {exc}"); return
        self.pol_label.setText("polarity: TESTING — open/close both jaws fully, 3×, then leave closed"); self._pol_timer = QtCore.QTimer(self); self._pol_timer.setInterval(200)
        def _tick():
            if self.s.polarity_test_tick(): return
            self._pol_timer.stop(); r = self.s.polarity_test_finish()
            self.pol_label.setText(("polarity: <b style='color:green'>PASSED</b> " if r["ok"] else "polarity: <b style='color:red'>FAILED — REC blocked</b> ") + " | ".join(f"{sd} {x.get('min','?')}..{x.get('max','?')} ×{x.get('cycles','?')} {x['why']}" for sd, x in r["sides"].items()))
        self._pol_timer.timeout.connect(_tick); self._pol_timer.start()

    def on_cal(self, side: str, which: str) -> None:
        try: self.s.calibrate_gripper(side, which)
        except Exception as exc: self.s.say(f"calibration: {exc}")

    def on_cal_save(self) -> None:
        try: self.s.save_gripper_calibration(notes="UI")
        except Exception as exc: self.s.say(f"calibration save: {exc}")

    def on_load_qa(self) -> None:
        files = sorted(self.s.manager.session_dir.glob("episode_*/derived/pose_*/pose_qa.json"))
        self.qa_text.setPlainText(files[-1].read_text() if files else "no pose_qa.json in this session yet (run tools.pose_process)")

    # ------------------------------------------------------------- panels
    def _side_html(self, side: str) -> str:
        cc = self.s.cfg.collector; imu = self.s.devices.imus.get(side); gr = self.s.devices.grippers.get(side); rows = []
        if gr is not None:
            q = gr.quality(); gcol = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED}[q.grade(cc.gripper_quality)]
            age = f"{q.age_ms:.0f}" if q.age_ms is not None else "—"; norm = "uncal" if q.normalized is None or q.normalized != q.normalized else f"{q.normalized:.3f}"
            # The servo's own error byte, not just link health: the STS3215 flags its supply here, and a jaw reading
            # that arrives at a healthy rate can still come from a servo that is complaining.
            bits = (gr.status().detail or {}).get("servo_error_bits") or ""
            errs = f'  errors {q.errors}' if q.errors else ""
            bad = f'  <span style="color:{RED}">servo {bits}</span>' if bits else ""
            rows.append(f'<span style="color:{gcol}">●</span> Grip raw {q.raw}  unwrapped {q.unwrapped}  '
                        f'norm {norm}  {q.hz:5.1f} Hz  age {age} ms{errs}{bad}')
        else: rows.append(f'<span style="color:{GREY}">●</span> Grip: not configured')
        if imu is not None:
            q = imu.quality(); gcol = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED}[q.grade(cc.imu_quality)]
            rows.append(f'<span style="color:{gcol}">●</span> IMU {q.hz:6.1f} Hz  loss {q.loss_ratio*100:.2f}%  crc {q.crc_errors}  gaps {q.seq_gaps}')
            smp = getattr(imu, "last_sample", None)
            if smp: rows.append(f"  a {smp.ax:+6.2f} {smp.ay:+6.2f} {smp.az:+6.2f}  g {smp.gx:+6.2f} {smp.gy:+6.2f} {smp.gz:+6.2f}")
        else: rows.append(f'<span style="color:{GREY}">●</span> IMU: WAITING FOR HW (not installed)')
        return "<br>".join(rows).replace(" ", "&nbsp;")

    def refresh(self) -> None:
        s = self.s; cc = s.cfg.collector
        self.clock.setText(time.strftime("%H:%M:%S"))
        s.poll()                                                  # drives the stillness gate -> REC, and device-error -> ERROR_REVIEW
        if s.state == "ERROR_REVIEW" and not self.review.isVisible(): self._enter_review()
        replay_frame = self.replay.frame if s.state in ("REVIEW", "ERROR_REVIEW") else None
        for name, pv in self.previews.items():
            src = name[:-4] if name.endswith("_rgb") else name        # the head RGB tile reads the same device as the depth tile
            cam = s.devices.cameras.get(src); st = cam.status() if cam else None
            f = cam.latest() if cam else None
            if pv.is_depth:
                if f is not None and getattr(f, "depth", None) is not None:
                    pv.show_depth(f.depth, (st.detail.get("depth_scale") if st else None))
            else:
                img = replay_frame if (name == "head" and replay_frame is not None) else (f.image if f is not None else None)
                if img is not None: pv.show_frame(img)
            if st is None: pv.dot.set(GREY, "not configured")
            else:
                col = RED if (not st.connected or st.error) else AMBER if (st.age_ms is not None and st.age_ms > 500) else GREEN
                pv.dot.set(col, f"{st.rate_hz:4.1f} fps" if st.connected else (st.error or "disconnected")[:40])
            if name == "head" and f is not None and replay_frame is None:
                g = cv2.resize(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY), (80, 60)).astype(np.int16)
                if self._prev_head is not None:
                    pv.warn.setText("HEAD MOTION HIGH" if float(np.abs(g - self._prev_head).mean()) > cc.head_motion_warn else "")
                self._prev_head = g
        aux = s.devices.cameras.get("head_depth"); ast = aux.status() if aux else None
        self.aux_dot.set(GREY, "not configured") if ast is None else self.aux_dot.set(GREEN if ast.connected and not ast.error else AMBER, f"RGB+DEPTH {ast.rate_hz:.0f} fps" if ast.connected else "absent (optional)")
        for n, d in self.dots.items():
            dev = next((x for x in s.devices.all_devices() if x.name == n), None)
            if dev is None: d.set(GREY, f"{n}: not installed"); continue
            st = dev.status(); stale = st.age_ms is not None and st.age_ms > 500
            col = (RED if dev.required else AMBER) if (not st.connected or st.error) else AMBER if stale else GREEN
            d.set(col, f"{n}{f' {st.rate_hz:.0f} Hz' if st.connected and st.rate_hz else ''}{f' — {st.error[:40]}' if st.error else ''}{'' if dev.required else ' (optional)'}")
        for side, lab in self.side_panels.items():
            lab.setText(self._side_html(side)); gr = s.devices.grippers.get(side)
            if gr is not None:
                q = gr.quality(); self.gauges[side].push(q.raw, q.normalized, gr.status().detail.get("servo_error_bits", "") or "")
        ident = s.devices.identity(); self.ident.setText("  |  ".join(f"{k} = {v}" for k, v in ident.items()) + f"  |  profile {s.cfg.hardware.profile}")
        pend = getattr(s, "_pending_cal", {}); self.cal_label.setText(f"active gripper calibration: {s.devices.gripper_calibration_version or 'NONE (norm = NaN until saved)'}   pending: {pend or '—'}")
        c = s.manager.counts; tgt = s.cfg.tasks.target_per_order
        self.counter.setText(" | ".join(f"{o} {c[o]:2d}/{tgt}" for o in s.cfg.tasks.orders) + f"\nValid {sum(c.values())} / Raw {s.manager.raw_total} / Discarded {s.manager.rejected}\n{s.manager.session_dir.name}")
        self.suggest.setText(f"Suggested next order: <b>{s.manager.next_order()}</b>")
        dk = s.disk(); self.disk_lab.set(COL[dk["state"]], f"Free disk {dk['free_gb']:.0f} GB  (~{dk['capacity_min']:.0f} min of recording)")
        if s.state == "RECORDING" and s.recorder:
            r = s.recorder; el = (time.monotonic_ns() - r.t_start_ns) / 1e9
            if s.holding_still:
                self.rec.setText(f'<span style="color:{RED}">REC ●</span>  <span style="color:{AMBER};font-weight:bold">HOLD STILL {s.hold_left_s():.1f} s</span>'
                                 + ('  <span style="color:{RED}">moved!</span>'.format(RED=RED) if s._hold_broken else '') + f'   {r.episode_dir.name}')
            elif el < float(s.cfg.collector.stillness.get("hold_after_rec_s", 0)) + 1.5 and s.cfg.collector.stillness.get("hold_after_rec_s", 0) > 0 and s.stillness_enabled():
                self.rec.setText(f'<span style="color:{RED}">REC ●</span>  <span style="color:{GREEN};font-weight:bold">GO</span>   {r.episode_dir.name}   {int(el // 60):02d}:{el % 60:05.2f}')
            else:
                self.rec.setText(f'<span style="color:{RED}">REC ●</span>  {r.episode_dir.name}   {int(el // 60):02d}:{el % 60:05.2f}')
            fr = "  ".join(f"{n}: {st.frames} f (drop {st.skipped_capture_indices})" for n, st in r.stats.items())
            gz = "  ".join(f"grip {sd}: {g.quality().hz:.0f} Hz" for sd, g in s.devices.grippers.items())
            self.live.setText(f"{fr}\n{gz}   events {len(r.events)} {sorted({e['kind'] for e in r.events})}")
        elif s.state in ("REVIEW", "ERROR_REVIEW"):
            self.rec.setText(f'<span style="color:{AMBER}">{s.state}</span> — KEEP (K) or DISCARD (D)'); self.live.setText("")
        elif s.state == "WAITING_FOR_STILLNESS" and s.gate is not None:
            lab = s.state_label(); ready = s.gate.ready
            per = "   ".join(f"{side[0].upper()} {st.still_s:.1f}s" + (f" ({st.last_window.get('reason')})" if st.last_window.get("reason") else "") for side, st in s.gate.state.items())
            self.rec.setText(f'<span style="color:{GREEN if ready else AMBER}">{lab}</span>   {per}'); self.live.setText("Hold both units still. SPACE / STOP cancels.")
        else:
            ok, why = s.can_record(); lab = s.state_label()
            self.rec.setText(f'<span style="color:{GREEN if ok else GREY}">{lab}</span> {why}'); self.live.setText("")
        if s.auto is not None:
            if s.auto.active: self.live.setText(s.auto.label() + "   (A / AUTO button stops)")
            elif s.auto.phase == "PAUSED": self.live.setText(s.auto.label()); self.b_auto.setChecked(False)
            elif self.b_auto.isChecked(): self.b_auto.setChecked(False)
        idle = s.state == "IDLE"; can = idle and s.can_record()[0]
        self.b_start.setEnabled(can); self.b_home.setEnabled(can); self.b_hw.setEnabled(can); self.b_stop.setEnabled(s.state in ("RECORDING", "WAITING_FOR_STILLNESS"))
        self.b_mark.setEnabled(s.state == "RECORDING"); self.b_keep.setEnabled(s.state in ("REVIEW", "ERROR_REVIEW")); self.b_discard.setEnabled(s.state in ("REVIEW", "ERROR_REVIEW"))
        for b in self.order_btns.values(): b.setEnabled(idle)
        if len(s.log_lines) != self._nlog: self.logbox.setPlainText("\n".join(s.log_lines[-6:])); self._nlog = len(s.log_lines)

    def closeEvent(self, ev) -> None:
        self.timer.stop(); self.replay.stop(); self.s.close(); ev.accept()


def run(cfg: AppConfig, *, session_dir=None, dataset: str | None = None) -> int:
    import sys
    app = W.QApplication.instance() or W.QApplication(sys.argv)
    session = CollectorSession.create(cfg, session_dir=session_dir, dataset=dataset)
    w = MainWindow(session); w.resize(1500, 1000); w.show()
    return app.exec()
