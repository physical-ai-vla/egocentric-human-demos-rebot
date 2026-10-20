"""Sensor hub (2026-10-14): wrist IMUs + fixed RGB-D -> fused human hand per side -> UDP (link.hand_port). Robot-free.
Runs in ego_collector/.venv; the Orbbec needs sudo on macOS:

    cd ~/ego_collector && sudo .venv/bin/python -m visual_teleop.tools.sensor_hub [--no-rgbd] [--no-imu]

Threads: one ImuReader per side (AHRS at the IMU rate), one RGB-D loop (MediaPipe + depth -> wrist in W, 30 Hz), and the fusion
publisher (100 Hz: HandFusion.estimate per side). Commands from control (link.cmd_port):
    {cmd: sync, seconds}        camera<->IMU offset per side: RGB-D palm rotation vs gyro over the next N s (wave both wrists)
    {cmd: align}                init: both hands at X_L / X_R in the known orientation -> IMU yaw psi per side (+ residual tilt)
    {cmd: rec_start, dir}       record RGB-D colour mp4 + depth png + IMU raw into dir;  {cmd: rec_stop}
Every result is published back as {kind: hub_event} and lands in the episode metadata via control.
Latest annotated RGB-D JPEG -> /tmp/vt_rgbd.jpg (dashboard)."""
from __future__ import annotations
import argparse
import threading
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .. import config as vcfg
from ..build import build_hand_fusion
from ..imu.imu_reader import ImuReader, open_teensy
from ..link import Latest, Publisher
from ..fusion.wrist_pose_fusion import HandStatus

SIDES = ("left", "right")


class Hub:
    def __init__(self, c, *, use_rgbd: bool = True, use_imu: bool = True) -> None:
        self.c = c; self.fus = build_hand_fusion(c); self.pub = Publisher(c.link["hand_port"]); self.cmd = Latest(c.link["cmd_port"], keep_kinds=("cmd",))
        self.imu: dict = {}; self.devs: dict = {}; self.stop = threading.Event(); self.lock = threading.Lock()
        self.rec_dir: Path | None = None; self.rec_dir_last: Path | None = None; self._labels: list = []
        self._pose_log: list = []; self._det_log: list = []; self._qhist = {s: {} for s in SIDES}
        self.push: dict | None = None             # INIT push window (method motion): {t_end, sides: {side: {R, acc, p}}}
        self.last_rgbd_ms = None; self.rgbd_fps = 0.0
        if use_imu:
            for s in SIDES:
                sn = c.imu[s]["serial_number"]
                if not sn: self.event("imu_missing", side=s, reason="imu.<side>.serial_number not set (LEFT/RIGHT identity comes from the USB serial)"); continue
                off = c.time_sync["camera_imu_offset_ms"][s]
                dev = open_teensy(s, c.imu); r = ImuReader(s, dev, c.imu["ahrs"], camera_imu_offset_ms=off, max_age_ms=c.imu["max_age_ms"],
                                                           min_rate_hz=c.imu["min_rate_hz"]); r.start(); self.imu[s] = r; self.devs[s] = dev
        self.use_rgbd = use_rgbd; self.tracker = None
        if c.fusion["position"]["source"] == "tracker":
            from ..vision.tracker6dof import Tracker6DoFProvider
            self.tracker = Tracker6DoFProvider.from_cfg(c.fusion["tracker"]); self.tracker.start()
            mon = c.fusion["tracker"].get("rgbd_monitor") or {}
            # 10-15 FastUMI-style: tracker primary, the markerless RGB-D keypoint only cross-checks it (residual monitor)
            self.use_rgbd = bool(use_rgbd and mon.get("enabled", False)); self.monitor = None; self._kpt_obs = {}
            if self.use_rgbd:
                from ..vision.residual_monitor import ResidualMonitor
                self.monitor = ResidualMonitor.from_cfg(mon); self.monitor_action = mon.get("action", "degrade")
            self.event("tracker_listen", port=self.tracker.port, preset=c.fusion["tracker"]["world_preset"],
                       T_TB_TCP_measured=c.fusion["tracker"]["T_TB_TCP"]["measured"])

    def event(self, what: str, **kw) -> None:
        print(f"[hub] {what} {kw}", flush=True); self.pub.send("hub_event", kw.get("side"), what=what, **{k: v for k, v in kw.items() if k != "side"})

    # ---- RGB-D --------------------------------------------------------------------------------------------------
    def rgbd_loop(self) -> None:
        import cv2
        from ego_teleop.hand3d.head_camera import OrbbecHeadCamera, ColorExposure
        from ..vision.rgbd_pose import RgbdWristEstimator
        cc = self.c.cameras["rgbd"]; T_W_C, measured = vcfg.T_W_C(self.c)
        src = self.c.fusion["position"]["source"]
        if self.tracker is not None: src = "kpt"           # tracker source: RGB-D runs ONLY as the keypoint residual monitor
        big = src in ("tag", "umi", "kpt")
        # 10-15: colour 1280x720 for the small UMI tag, depth 848x480 (both at 1280x720 delivered no aligned frames)
        def open_cam():
            return OrbbecHeadCamera(width=cc["width"] if big else cc["depth_width"], height=cc["height"] if big else cc["depth_height"], fps=cc["fps"],
                                    depth_width=cc["depth_width"], depth_height=cc["depth_height"],
                                    exposure=ColorExposure(auto=False, exposure=cc["color_exposure"]) if cc.get("color_exposure") else None).open()
        cam = open_cam()
        P = {"source": src}; kpt_mode = P["source"] == "kpt"; umi_mode = P["source"] == "umi"
        if kpt_mode:                                   # markerless UMI jaw tip (user 10-15: no tag, no tape at runtime)
            from ..vision.umi_kpt import UmiKptTracker
            kc = self.c.fusion["kpt"]; kpt_side = kc.get("side", "right")
            kpt = UmiKptTracker(kc["onnx"], T_W_C, imgsz=kc["imgsz"], conf=kc["conf"], max_jump_m=kc["max_jump_m"], v_max_m_s=kc["v_max_m_s"], margin_m=kc["margin_m"])
            self.tag_stats = dict(seen=0, missed=0)
        tag_mode = P["source"] == "tag" or umi_mode; tape_mode = P["source"] == "tape" or umi_mode
        if umi_mode:
            from ..vision.umi_pose import UmiPositionFusion
            self.umi_fusion = UmiPositionFusion(learn_n=self.c.fusion["umi"]["offset_learn_n"])
        if tape_mode:
            from ..vision.tape_pose import UmiTapeTracker
            tp = self.c.fusion["tape"]; tape_side = tp.get("side", "right")
            tape = UmiTapeTracker(T_W_C, hsv_lo=tuple(tp["hsv_lo"]), hsv_hi=tuple(tp["hsv_hi"]), min_area_px=tp["min_area_px"], max_jump_px=tp["max_jump_px"])
            self.tag_stats = dict(seen=0, missed=0)
        if tag_mode:                                   # UMI AprilTag: position primary + weak yaw anchor (user 10-15); no MediaPipe
            from ..vision.tag_pose import UmiTagTracker
            tg = self.c.fusion["tag"]; tag_side = tg.get("side", "right")
            tracker = UmiTagTracker(T_W_C, family=tg["family"], tag_id=tg["id"], size_m=tg["size_m"], min_margin=tg["min_margin"],
                                    use_depth=tg["use_depth"], T_tag_tcp=np.asarray(tg["T_tag_tcp"]["matrix"], np.float64))
            est = None; self.tag_stats = dict(seen=0, missed=0, yaw="skip")
        else:
            est = RgbdWristEstimator(T_W_C, selfie_mirrored=self.c.cameras["rgbd_selfie_mirrored"])
        self.event("rgbd_open", T_W_C_measured=measured, exposure=cam.applied_exposure)
        self._rs = dict(n=0, t_fps=time.monotonic(), last_jpg=0.0, vw=None, ts=[], dep_dir=None)     # loop state used by _rgbd_step
        self._last_frame_t = time.monotonic()
        diag = dict(none=0, ok=0, err=0); t_diag = time.monotonic()
        while not self.stop.is_set():
            if time.monotonic() - t_diag > 2.0:          # 10-15: the loop stalled silently twice -> always say what it is doing
                qh = self._qhist.get("right", {}); tot = max(1, sum(qh.values()))
                print(f"[hub] rgbd 2s: frames {diag['ok']}, empty reads {diag['none']}, errors {diag['err']}, stats {getattr(self, 'tag_stats', None)} | "
                      f"quality(right, since start) " + " ".join(f"{k} {100 * v / tot:.1f}%" for k, v in sorted(qh.items(), key=lambda kv: -kv[1])), flush=True)
                diag = dict(none=0, ok=0, err=0); t_diag = time.monotonic()
            f = cam.read()
            if f is None:
                diag["none"] += 1
                # 10-15: an Orbbec reopened right after the previous owner released it delivers NOTHING (probe: 0 framesets) --
                # reopen instead of waiting forever at 0 fps
                if time.monotonic() - getattr(self, "_last_frame_t", time.monotonic() - 0.0) > 3.0:
                    print("[hub] rgbd: no frames for 3 s -> reopening the Orbbec", flush=True)
                    try: cam.close()
                    except Exception: pass
                    time.sleep(1.0); cam = open_cam(); self._last_frame_t = time.monotonic()
                continue
            self._last_frame_t = time.monotonic(); diag["ok"] += 1
            try:
                self._rgbd_step(f, cam, cc, est, tag_mode, tape_mode, umi_mode, locals())
            except Exception as e:
                import traceback
                diag["err"] += 1; print(f"[hub] rgbd ERROR {type(e).__name__}: {e}", flush=True); traceback.print_exc()
        cam.close()

    def _rgbd_step(self, f, cam, cc, est, tag_mode, tape_mode, umi_mode, L):
        import cv2
        tracker, tape = L.get("tracker"), L.get("tape"); tag_side, tape_side = L.get("tag_side", "right"), L.get("tape_side", "right")
        if True:
            t0 = time.monotonic_ns()
            kpt, kpt_side = L.get("kpt"), L.get("kpt_side", "right"); kpt_mode = kpt is not None
            if kpt_mode:
                ob = kpt.detect(f); obs = {}
                if ob is not None:
                    self.tag_stats["seen"] += 1
                    if self.tracker is not None: self._kpt_obs[kpt_side] = (int(ob.t_ns), np.asarray(ob.p_W, np.float64))   # monitor only
                    else:
                        with self.lock: self.fus[kpt_side].observe_position(ob.p_W, ob.conf, ob.t_ns)
                    self.tag_stats.update(conf=round(ob.conf, 2), z=round(ob.depth_m, 3), p=np.round(ob.p_W, 3).tolist())
                else: self.tag_stats["missed"] += 1
                self._reacquiring = getattr(self, "_reacquiring", {}); self._reacquiring[kpt_side] = kpt.prev_p is None or (time.monotonic() - kpt._last_t) > kpt.lost_max
                if self.rec_dir is not None:
                    dg = kpt.last_diag; pp = ob.p_W if ob is not None else (np.nan,) * 3; uv = ob.uv if ob is not None else (np.nan, np.nan)
                    self._det_log.append((f.timestamp_ns, dg["n_det"], dg["best_conf"], dg["rejected_jump"], dg["rejected_depth"], int(dg["accepted"]), *uv, *pp))
            elif umi_mode:                                   # tape tip primary, tag + learned offset when the tape is hidden (cloth)
                side = tape_side; ob_tag = tracker.detect(f); obs = {}
                ob_tape = tape.detect(f, hint_uv=tracker._last_uv if ob_tag is not None else None)   # tag-guided tape (10-15)
                imu = self.imu[side].latest() if side in self.imu else None
                with self.lock:
                    R = self.fus[side].orient.predict(imu.R_G_I) if (imu is not None and imu.healthy and self.fus[side].orient.psi is not None) else None
                    p_, conf_, which = self.umi_fusion.update(ob_tape, ob_tag, R)
                    if p_ is not None: self.fus[side].observe_position(p_, conf_, f.timestamp_ns)
                    if ob_tag is not None: self.tag_stats["yaw"] = self.fus[side].observe_tag_orientation(ob_tag.T_W_tag[:3, :3], imu, ob_tag.t_ns)
                k = "seen" if p_ is not None else "missed"; self.tag_stats[k] = self.tag_stats.get(k, 0) + 1
                if self.rec_dir is not None:                # 10-15: pseudo-labels for a later markerless UMI keypoint model
                    self._labels.append((f.timestamp_ns, *(ob_tape.uv if ob_tape is not None else (np.nan, np.nan)),
                                         *(tracker._last_uv if ob_tag is not None else (np.nan, np.nan)),
                                         *(p_ if p_ is not None else (np.nan, np.nan, np.nan)), {"tape": 1, "tag": 2}.get(which, 0)))
                self.tag_stats.update(src=which, tape=ob_tape is not None, tag=ob_tag is not None, offset_learned=self.umi_fusion.r_H is not None,
                                      p=None if p_ is None else np.round(p_, 3).tolist())
            elif tape_mode:
                ob = tape.detect(f); obs = {}
                if ob is not None:
                    self.tag_stats["seen"] += 1
                    with self.lock: self.fus[tape_side].observe_position(ob.p_W, ob.conf, ob.t_ns)
                    self.tag_stats.update(area=ob.area_px, z=round(ob.depth_m, 3), p=np.round(ob.p_W, 3).tolist())
                else: self.tag_stats["missed"] += 1
            elif tag_mode:
                ob = tracker.detect(f); obs = {}
                if ob is not None:
                    self.tag_stats["seen"] += 1
                    imu = self.imu[tag_side].latest() if tag_side in self.imu else None
                    with self.lock:
                        self.fus[tag_side].observe_position(ob.p_W_tcp, ob.conf, ob.t_ns)
                        self.tag_stats["yaw"] = self.fus[tag_side].observe_tag_orientation(ob.T_W_tag[:3, :3], imu, ob.t_ns)
                    self.tag_stats.update(margin=round(ob.margin), depth=ob.depth_refined, p=np.round(ob.p_W_tcp, 3).tolist())
                else: self.tag_stats["missed"] += 1
            else:
                obs = est.estimate(f)
                for s, o in obs.items():
                    with self.lock: self.fus[s].observe_position(o.wrist_W, o.conf, o.t_ns, o.tips_mid_W)
            R_ = self._rs; self.last_rgbd_ms = (time.monotonic_ns() - t0) / 1e6; R_["n"] += 1
            if time.monotonic() - R_["t_fps"] > 1.0: self.rgbd_fps = R_["n"] / (time.monotonic() - R_["t_fps"]); R_["n"] = 0; R_["t_fps"] = time.monotonic()
            if self.rec_dir is not None:
                if R_["vw"] is None:
                    (self.rec_dir / "frames").mkdir(parents=True, exist_ok=True); R_["dep_dir"] = self.rec_dir / "frames" / "rgbd_depth_mm"; R_["dep_dir"].mkdir(exist_ok=True)
                    R_["vw"] = cv2.VideoWriter(str(self.rec_dir / "frames" / "rgbd_color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cc["fps"], (f.color_bgr.shape[1], f.color_bgr.shape[0]))
                R_["vw"].write(f.color_bgr); R_["ts"].append(f.timestamp_ns)
                cv2.imwrite(str(R_["dep_dir"] / f"{f.timestamp_ns}.png"), np.nan_to_num(f.depth_m * 1000.0, nan=0).astype(np.uint16))
            elif R_["vw"] is not None:
                R_["vw"].release(); np.save(self.rec_dir_last / "frames" / "rgbd_color_t_ns.npy", np.asarray(R_["ts"], np.int64)); R_["vw"] = None; R_["ts"] = []
            if time.monotonic() - R_["last_jpg"] > 0.1:
                R_["last_jpg"] = time.monotonic(); v = f.color_bgr.copy()
                for s, o in obs.items():
                    col = (255, 128, 0) if s == "left" else (0, 128, 255)
                    cv2.putText(v, f"{s.upper()} conf={o.conf:.2f} {o.health}", (10, 30 if s == "left" else 60), 0, 0.8, col, 2)
                if tag_mode or tape_mode or kpt_mode: cv2.putText(v, f"{'KPT' if kpt_mode else 'UMI' if umi_mode else 'TAPE' if tape_mode else 'TAG'} {self.tag_stats}", (10, 30), 0, 0.45, (0, 255, 255), 2)
                if kpt_mode and kpt.prev_p is not None and ob is not None: cv2.circle(v, (int(ob.uv[0]), int(ob.uv[1])), 14, (0, 255, 0), 3)
                if tape_mode and tape.prev_uv is not None: cv2.circle(v, (int(tape.prev_uv[0]), int(tape.prev_uv[1])), 14, (0, 255, 255), 2)
                cv2.imwrite("/tmp/vt_rgbd.jpg", v)

    # ---- fixed RGB camera + UMI AprilTag (no depth, no sudo) -------------------------------------------------------
    def tag_rgb_loop(self) -> None:
        import cv2
        from ..vision.tag_camera import TagCamera
        from ..vision.tag_pose import UmiTagTracker
        T_W_C, measured = vcfg.T_W_C(self.c); tg = self.c.fusion["tag"]; side = tg.get("side", "right")
        cam = TagCamera(self.c.cameras["tag_cam"])
        tracker = UmiTagTracker(T_W_C, family=tg["family"], tag_id=tg["id"], size_m=tg["size_m"], min_margin=tg["min_margin"], use_depth=False,
                                T_tag_tcp=np.asarray(tg["T_tag_tcp"]["matrix"], np.float64))
        self.tag_stats = dict(seen=0, missed=0, yaw="skip")
        self.event("tag_cam_open", index=cam.index, focus=cam.focus, T_W_C_measured=measured)
        n = 0; t_fps = time.monotonic(); last_jpg = 0.0; vw = None; ts = []
        while not self.stop.is_set():
            f = cam.read()
            if f is None: continue
            t0 = time.monotonic_ns(); ob = tracker.detect(f)
            if ob is not None:
                self.tag_stats["seen"] += 1; imu = self.imu[side].latest() if side in self.imu else None
                with self.lock:
                    self.fus[side].observe_position(ob.p_W_tcp, ob.conf, ob.t_ns)
                    self.tag_stats["yaw"] = self.fus[side].observe_tag_orientation(ob.T_W_tag[:3, :3], imu, ob.t_ns)
                self.tag_stats.update(margin=round(ob.margin), p=np.round(ob.p_W_tcp, 3).tolist())
            else: self.tag_stats["missed"] += 1
            self.last_rgbd_ms = (time.monotonic_ns() - t0) / 1e6; n += 1
            if time.monotonic() - t_fps > 1.0: self.rgbd_fps = n / (time.monotonic() - t_fps); n = 0; t_fps = time.monotonic()
            if self.rec_dir is not None:
                if vw is None:
                    (self.rec_dir / "frames").mkdir(parents=True, exist_ok=True)
                    vw = cv2.VideoWriter(str(self.rec_dir / "frames" / "tagcam_color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (f.color_bgr.shape[1], f.color_bgr.shape[0]))
                vw.write(f.color_bgr); ts.append(f.timestamp_ns)
            elif vw is not None:
                vw.release(); np.save(self.rec_dir_last / "frames" / "tagcam_color_t_ns.npy", np.asarray(ts, np.int64)); vw = None; ts = []
            if time.monotonic() - last_jpg > 0.1:
                last_jpg = time.monotonic(); v = f.color_bgr.copy()
                cv2.putText(v, ("TAG OK " if ob is not None else "TAG LOST ") + str(self.tag_stats), (8, 22), 0, 0.45, (0, 255, 255) if ob is not None else (0, 0, 255), 1)
                cv2.imwrite("/tmp/vt_rgbd.jpg", v)
        cam.close()

    def tracker_diag(self) -> None:
        """2 s line: quality histogram, rx rate, seq losses / reorders / rejections, bridge->hub latency (t_rx - t_send)."""
        tr = self.tracker; qh = self._qhist.get("right", {}); tot = max(1, sum(qh.values())); self._qhist["right"] = {}
        n = tr.n_rx - getattr(self, "_tdiag_n", 0); self._tdiag_n = tr.n_rx; tp = tr.latest_raw("right")
        lat = (tp.t_ns - tp.t_send_ns) / 1e6 if tp is not None and tp.t_send_ns else float("nan")
        print(f"[트래커] 수신 {n / 2:.0f} Hz | " + " ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(qh.items()))
              + f" | 누락seq {tr.stats['lost_seq']} 역순 {tr.stats['reordered']} 무효 {tr.stats['invalid']} 점프거부 {tr.stats['jump_rejected']}"
              + f" | 브리지→허브 {lat:.1f} ms" + (f" | 최근 사유 {tp.why}" if tp is not None and tp.why else "")
              + (f" | RGB-D 대조 {self.monitor.last}" if getattr(self, "monitor", None) is not None else ""), flush=True)

    # ---- fusion publisher ---------------------------------------------------------------------------------------
    def fusion_loop(self) -> None:
        period = 0.01
        while not self.stop.is_set():
            t = time.monotonic_ns()
            for s in SIDES:
                imu = self.imu[s].latest(t) if s in self.imu else None
                if self.tracker is not None:                   # dedicated 6DoF tracker: orientation AND position from it
                    e, quality = self.tracker.estimate(s, t); mon = None
                    if self.monitor is not None and s == self.c.fusion["kpt"].get("side", "right"):
                        mon = self.monitor.update(t, e.T_W_H[:3, 3] if e.T_W_H is not None else None, self._kpt_obs.get(s))
                        if mon["state"] == "warn" and quality == "TRACKING":
                            quality = "DEGRADED"; e.reasons = e.reasons + [f"rgbd_residual:{mon['residual_m'] * 100:.1f}cm"]
                        elif mon["state"] == "hold" and self.monitor_action == "hold":
                            quality = "LOST"; e.reasons = e.reasons + [f"rgbd_residual_hold:{mon['residual_m'] * 100:.1f}cm"]
                            e.status = HandStatus.HOLD; e.T_W_H = None; e.wrist_W = None
                    self._qhist[s][quality] = self._qhist[s].get(quality, 0) + 1
                    if self.rec_dir is not None:
                        pp = e.T_W_H[:3, 3] if e.T_W_H is not None else (np.nan,) * 3
                        q_ = Rotation.from_matrix(e.T_W_H[:3, :3]).as_quat() if e.T_W_H is not None else (np.nan,) * 4
                        self._pose_log.append((t, s == "right", ["INIT", "TRACKING", "DEGRADED", "LOST", "REACQUIRE"].index(quality), e.position_conf,
                                               e.position_age_ms if np.isfinite(e.position_age_ms) else -1.0, *pp, *q_))
                    if s == "right" and t - getattr(self, "_tdiag_t", 0) > 2e9: self._tdiag_t = t; self.tracker_diag()
                    self.pub.send("hand", s, t_ns=e.t_ns, status=e.status.value, quality=quality, reasons=e.reasons, T_W_H=e.T_W_H, wrist_W=e.wrist_W,
                                  position_conf=e.position_conf, orientation_source="tracker", position_age_ms=e.position_age_ms,
                                  imu_ok=bool(imu is not None and imu.healthy), imu_rate_hz=self.imu[s].rate_hz() if s in self.imu else 0.0, aligned=True,
                                  rgbd_fps=self.rgbd_fps if self.use_rgbd else 0.0, rgbd_ms=None, tag=None, tag_R_learned=False, rgbd_monitor=mon)
                    continue
                with self.lock: e = self.fus[s].estimate(t, imu)
                # pose-quality state (FastUMI-style, 10-15): one word that says why a downstream consumer gets / does not get a pose
                if self.fus[s].orient.psi is None: quality = "INIT"
                elif e.status.value == "TRACKING": quality = "TRACKING"
                elif e.status.value == "WARN": quality = "DEGRADED"
                elif getattr(self, "_reacquiring", {}).get(s): quality = "REACQUIRE"
                else: quality = "LOST"
                self._qhist[s][quality] = self._qhist[s].get(quality, 0) + 1
                if self.rec_dir is not None:
                    from scipy.spatial.transform import Rotation as _Rot
                    q_ = _Rot.from_matrix(imu.R_G_I).as_quat() if imu is not None else (np.nan,) * 4
                    pp = e.T_W_H[:3, 3] if e.T_W_H is not None else (np.nan,) * 3
                    self._pose_log.append((t, s == "right", ["INIT", "TRACKING", "DEGRADED", "LOST", "REACQUIRE"].index(quality), e.position_conf, e.position_age_ms, *pp, *q_))
                self.pub.send("hand", s, t_ns=e.t_ns, status=e.status.value, quality=quality, reasons=e.reasons, T_W_H=e.T_W_H, wrist_W=e.wrist_W,
                              position_conf=e.position_conf, orientation_source=e.orientation_source, position_age_ms=e.position_age_ms,
                              imu_ok=bool(imu is not None and imu.healthy), imu_rate_hz=self.imu[s].rate_hz() if s in self.imu else 0.0,
                              imu_tilt_deg=self.imu[s].ahrs.tilt_deg() if s in self.imu and self.imu[s].ahrs.ready else None,
                              aligned=self.fus[s].orient.psi is not None, rgbd_fps=self.rgbd_fps, rgbd_ms=self.last_rgbd_ms,
                              tag=getattr(self, "tag_stats", None), tag_R_learned=self.fus[s].R_tag_H is not None)
            if self.push is not None: self.push_step(t)
            for m in self.cmd.pop_all(): self.handle(m)
            time.sleep(max(0.0, period - (time.monotonic_ns() - t) / 1e9))

    def push_step(self, t: int) -> None:
        """Collect (R_G_I, gravity-free accel, RGB-D wrist) during the INIT push; solve the yaw when the window ends."""
        P = self.push
        for s, buf in P["sides"].items():
            imu = self.imu[s].latest(t) if s in self.imu else None
            if imu is None or not self.imu[s].ahrs.ready: continue
            buf["R"].append(imu.R_G_I)
            if imu.acc_m_s2 is not None: buf["acc"].append(imu.acc_m_s2 - imu.R_G_I.T @ np.array([0, 0, self.imu[s].ahrs.g_rest]))
            with self.lock: p, pt = self.fus[s]._p, self.fus[s]._p_t
            if p is not None and pt != buf.get("last_pt"): buf["p"].append(p.copy()); buf["Rp"].append(imu.R_G_I); buf["last_pt"] = pt
        if t < P["t_end"]: return
        self.push = None
        for s, buf in P["sides"].items():
            if not buf["Rp"]: self.event("align_failed", side=s, reason="no RGB-D wrist positions during the push (hand visible? IMU ready?)"); continue
            with self.lock: rep = self.fus[s].orient.align_from_motion(buf["Rp"], buf["p"], t, acc_lin_I=buf["acc"] or None)
            self.event("aligned" if rep.get("ok") else "align_failed", side=s, method="motion", **{k: v for k, v in rep.items() if k != "ok"})

    def handle(self, m: dict) -> None:
        cmd = m.get("cmd")
        if cmd == "align" and self.tracker is not None:     # a 6DoF tracker is already in a gravity-up world: nothing to align
            for s in m.get("sides", SIDES): self.event("aligned", side=s, method="tracker")
        elif cmd == "align" and self.c.init["method"] == "motion":
            sides = [s for s in m.get("sides", SIDES) if s in self.imu]
            if not sides: self.event("align_failed", reason="no IMU for the requested sides"); return
            for s in sides: self.fus[s].reset_tag_learning()
            if hasattr(self, "umi_fusion"): self.umi_fusion.reset()
            self.push = dict(t_end=time.monotonic_ns() + int(self.c.init["push_window_s"] * 1e9),
                             sides={s: dict(R=[], acc=[], p=[], Rp=[]) for s in sides})
            self.event("push_now", sides=sides, seconds=self.c.init["push_window_s"],
                       instruction="push the hand ~10 cm forward ALONG THE FINGERS and back, palm down")
        elif cmd == "align":
            for s in SIDES:
                imu = self.imu[s].latest() if s in self.imu else None
                if imu is None or not self.imu[s].ahrs.ready: self.event("align_failed", side=s, reason="IMU/AHRS not ready"); continue
                tilt = self.fus[s].orient.align(imu.R_G_I, vcfg.R_W_H_init(self.c, s), time.monotonic_ns())
                ok = tilt <= self.c.fusion["init_tilt_warn_deg"]
                self.event("aligned" if ok else "align_tilt_warn", side=s, init_tilt_deg=tilt, psi=self.fus[s].orient.psi)
        elif cmd == "rec_start":
            self.rec_dir = Path(m["dir"]).expanduser(); self.rec_dir_last = self.rec_dir; self._labels = []; self._pose_log = []; self._det_log = []
            for r in self.imu.values(): r.samples_for_recorder.clear()
            if self.tracker is not None: self.tracker.log = []
            self.event("rec_started", dir=str(self.rec_dir))
        elif cmd == "rec_stop":
            d = self.rec_dir; self.rec_dir = None
            if d is not None and self._pose_log:
                P = np.asarray(self._pose_log, np.float64)
                np.savez(d / "pose_log.npz", t_ns=P[:, 0].astype(np.int64), is_right=P[:, 1].astype(bool), quality=P[:, 2].astype(np.int8), position_conf=P[:, 3],
                         position_age_ms=P[:, 4], p_W=P[:, 5:8], imu_quat_xyzw=P[:, 8:12], quality_names=np.array(["INIT", "TRACKING", "DEGRADED", "LOST", "REACQUIRE"]))
                self._pose_log = []
            if d is not None and self._det_log:
                Q = np.asarray(self._det_log, np.float64)
                np.savez(d / "det_log.npz", t_ns=Q[:, 0].astype(np.int64), n_det=Q[:, 1].astype(int), best_conf=Q[:, 2], rejected_jump=Q[:, 3].astype(int),
                         rejected_depth=Q[:, 4].astype(int), accepted=Q[:, 5].astype(bool), uv=Q[:, 6:8], p_W=Q[:, 8:11])
                self._det_log = []
            if d is not None and getattr(self, "_labels", None):
                L = np.asarray(self._labels, np.float64)      # t_ns, tape_u, tape_v, tag_u, tag_v, p_W xyz, source (1 tape, 2 tag, 0 none)
                np.savez(d / "umi_pseudo_labels.npz", t_ns=L[:, 0].astype(np.int64), tape_uv=L[:, 1:3], tag_uv=L[:, 3:5], p_W=L[:, 5:8], source=L[:, 8].astype(np.int8),
                         note="Orbbec colour px (frames/rgbd_color.mp4 + rgbd_color_t_ns.npy); for a markerless UMI keypoint model")
                self._labels = []
            if d is not None and self.tracker is not None and self.tracker.log:
                with self.tracker._lock: K = np.asarray(self.tracker.log, np.float64); self.tracker.log = None
                np.savez(d / "tracker_log.npz", t_rx_ns=K[:, 0].astype(np.int64), seq=K[:, 1].astype(np.int64), t_src_ns=K[:, 2].astype(np.int64),
                         t_send_ns=K[:, 3].astype(np.int64), valid=K[:, 4].astype(bool), conf=K[:, 5], p_TW=K[:, 6:9], q_xyzw_TW=K[:, 9:13],
                         T_W_TW=self.tracker.T_W_TW, T_TB_TCP=self.tracker.T_TB_TCP, note="raw tracker datagrams; -1 = field absent")
            if d is not None:
                for s, r in self.imu.items():
                    S = list(r.samples_for_recorder)
                    if S: np.savez(d / f"imu_{s}.npz", device_us=np.array([x.device_timestamp_us for x in S], np.int64),
                                   host_ns=np.array([x.host_receive_ns for x in S], np.int64), acc=np.array([[x.ax, x.ay, x.az] for x in S]),
                                   gyro=np.array([[x.gx, x.gy, x.gz] for x in S]))
            self.event("rec_stopped", dir=str(d))
        elif cmd == "sync":
            self.event("sync_unavailable", reason="RGB-D palm-rotation history not wired yet; estimate offline with imu/time_sync.py on a recorded wave")
        else: self.event("unknown_cmd", cmd=cmd)

    def run(self, seconds: float = 0.0) -> None:
        th = [threading.Thread(target=self.fusion_loop, daemon=True)]
        if self.use_rgbd:
            loop = self.tag_rgb_loop if self.c.fusion["position"]["source"] == "tag_rgb" else self.rgbd_loop
            th.append(threading.Thread(target=loop, daemon=True))
        for t in th: t.start()
        t0 = time.monotonic()
        try:
            while all(t.is_alive() for t in th) and (seconds <= 0 or time.monotonic() - t0 < seconds): time.sleep(0.2)
        except KeyboardInterrupt: pass
        self.stop.set()
        for r in self.imu.values(): r.stop()
        for d in self.devs.values(): d.stop(); d.close()
        if self.tracker is not None: self.tracker.stop()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--no-rgbd", action="store_true")
    ap.add_argument("--no-imu", action="store_true"); ap.add_argument("--seconds", type=float, default=0.0)
    a = ap.parse_args(); c = vcfg.load(a.config)
    Hub(c, use_rgbd=not a.no_rgbd, use_imu=not a.no_imu).run(a.seconds)


if __name__ == "__main__":
    main()
