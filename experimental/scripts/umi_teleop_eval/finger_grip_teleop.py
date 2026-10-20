#!/usr/bin/env python3
"""[2026-10-13] Bare-hand GRIPPER-ONLY teleop: red tape on the thumb and index fingertips, seen by a wrist-band camera, drives ONE
reBot gripper. No arm joint moves: the arm joints are LATCHED once at start (leader frame, from /observe) and re-sent unchanged;
only the chosen jaw's COMMAND (0 = closed .. 45 = open, obs = -6 * cmd) changes, the other jaw holds its current command.
This is the same "current pose with one jaw element replaced" primitive as mac_v4_smoke_ui /arm_ident.

Safety: DRY-RUN by default (shows what would be sent); --live posts to the robot service. Refuses to start unless /status says
connected and not E-STOP. Fingertips lost > LOST_S -> HOLD (last command kept). SPACE = pause/hold, q / ESC = quit (jaw stays).
Command smoothing: EMA + slew limit (MAX_CMD_PER_S) + deadband; posts at most SEND_HZ.

usage: finger_grip_teleop.py --cam FisheyeCamLeft --arm right [--robot http://localhost:8021] [--live]
Calibration: the first CAL_S seconds -> open and close the two fingers fully a couple of times.
"""
import argparse, math, sys, time, pathlib
import cv2, numpy as np, requests

sys.path.insert(0, str(pathlib.Path.home() / "ego_collector"))
# copied from ~/softfold_infer/rebot/infer_core_v4.py (py3.12 syntax, not importable from this 3.11 env) -- keep in sync
GRIP_IDX = [6, 13]; FLIP_IDX = [0, 1, 5, 7, 8, 12]; GRIP_OBS_PER_CMD = -6.0; GRIP_CMD_MAX = 45.0


def unwrap_grip(raw):
    r = float(raw); return r - 360.0 if r > 180.0 else r       # a closed jaw can read 358 instead of -2


def jaw_hold_cmd(raw):
    """/execute_step jaw COMMAND (0..45) that keeps the jaw where /observe reads it (obs = cmd * -6)."""
    return float(np.clip(unwrap_grip(raw) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))

CMD_MAX, CAL_S, LOST_S, SEND_HZ, MAX_CMD_PER_S, EMA, DEADBAND = 45.0, 4.0, 0.5, 10.0, 60.0, 0.35, 0.5


def red_mask(img):
    b, g, r = (img[..., k].astype(np.int16) for k in range(3))
    m = (((r - np.maximum(g, b)) > 40) & (r > 100)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def candidates(img, min_area, ref_lab=None, lab_tol=22.0):
    """red blobs -> [(area, centroid, mean Lab)]; if ref_lab is set, keep only blobs whose colour is within lab_tol of it."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(red_mask(img)); out = []
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if n > 1 else None
    for i in range(1, n):
        if st[i, 4] < min_area or st[i, 4] > 0.01 * img.shape[0] * img.shape[1]: continue   # fingertip tape is small
        c = L[lab == i].mean(0)
        if ref_lab is not None and np.linalg.norm(c[1:] - ref_lab[1:]) > lab_tol: continue
        out.append((st[i, 4], cen[i], c))
    return out


class TipTracker:
    """[2026-10-13 user: "빨간색은 다 따라가"] follow THE TWO fingertips, not every red thing: initialise on the two largest blobs,
    then each frame take, for each tip, the nearest candidate within GATE px of its last position (the two must differ);
    tips lost for > LOST_S -> caller holds. 'r' re-initialises."""
    GATE = 140.0

    def __init__(self): self.p = None; self.ref_lab = None

    def reset(self): self.p = None

    def update(self, cands):
        if self.p is None:
            # fingertips = a PAIR of similar-size, small, mutually close blobs (not "the two largest red things")
            pairs = [(np.linalg.norm(x[1] - y[1]) + 300.0 * abs(np.log(x[0] / y[0])), x, y) for i, x in enumerate(cands) for y in cands[i + 1:]
                     if max(x[0], y[0]) / min(x[0], y[0]) < 3.0]
            if not pairs: return None, []
            _, x, y = min(pairs, key=lambda q: q[0]); two = [x, y]; self.p = [x[1], y[1]]
            if self.ref_lab is None: self.ref_lab = np.mean([two[0][2], two[1][2]], axis=0)
            return float(np.linalg.norm(self.p[0] - self.p[1])), two
        best = None
        for i, a in enumerate(cands):
            for j, b in enumerate(cands):
                if i == j: continue
                c = np.linalg.norm(a[1] - self.p[0]) + np.linalg.norm(b[1] - self.p[1])
                if np.linalg.norm(a[1] - self.p[0]) < self.GATE and np.linalg.norm(b[1] - self.p[1]) < self.GATE and (best is None or c < best[0]):
                    best = (c, a, b)
        if best is None: return None, []
        _, a, b = best; self.p = [a[1], b[1]]
        return float(np.linalg.norm(a[1] - b[1])), [a, b]


def open_camera(name):
    from handumi_collector.devices.camera import list_video_devices, probe_index_map
    listing = list_video_devices(); hit = next((n for n in listing if name.lower() in n.lower()), None)
    if hit is None: sys.exit(f"camera '{name}' not in {listing}")
    idx = probe_index_map([hit]).get(hit)
    if idx is None: sys.exit(f"could not resolve the OpenCV index of '{hit}'")
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened(): sys.exit(f"cannot open camera index {idx} ({hit})")
    return cap, hit, idx


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cam", default="FisheyeCamLeft"); ap.add_argument("--arm", choices=["left", "right"], default="right")
    ap.add_argument("--robot", default="http://localhost:8021"); ap.add_argument("--live", action="store_true"); ap.add_argument("--min-area", type=int, default=60)
    ap.add_argument("--mode", choices=["sigmoid", "linear", "binary"], default="sigmoid",
                    help="finger opening f (0..1) -> jaw: sigmoid = soft binary (like the BCE gripper head), binary = hysteresis 0.4/0.6")
    ap.add_argument("--k", type=float, default=10.0, help="sigmoid steepness (10 = soft step around f=0.5)")
    ap.add_argument("--no-undistort", action="store_true", help="feed the raw fisheye frame to MediaPipe")
    ap.add_argument("--hand", choices=["auto", "left", "right"], default="auto", help="wilor: wearer's hand label (auto = the camera's side)")
    ap.add_argument("--pinch", choices=["3d", "2d"], default="3d", help="mediapipe: metric world-landmark distance (whole hand visible) or pixels")
    ap.add_argument("--full-frame", action="store_true", help="mediapipe/wilor: no finger box, whole frame (camera worn near the elbow)")
    ap.add_argument("--student", default=str(pathlib.Path.home() / "finger_distill/student.pt"), help="YOLO-pose 4-kpt model distilled from WiLoR")
    ap.add_argument("--pick-roi", action="store_true", help="re-draw the finger box (else the saved one per camera is reused)")
    ap.add_argument("--detector", choices=["student", "appearance", "mediapipe", "wilor", "redtape"], default="student",
                    help="mediapipe = thumb-tip (4) to index-tip (8) distance from MediaPipe's metric world landmarks (no tape); redtape = red fingertip tape")
    a = ap.parse_args(); gi = GRIP_IDX[0 if a.arm == "left" else 1]; go = GRIP_IDX[1 if a.arm == "left" else 0]
    sess = requests.Session(); latched = None
    if a.live:
        st = sess.get(a.robot + "/status", timeout=5).json()
        if not st.get("connected") or st.get("estop"):
            sys.exit(f"robot not ready: {st}")
        o = sess.get(a.robot + "/observe", timeout=10).json()
        q = list(o["joints_rad"])
        for k in FLIP_IDX: q[k] = -q[k]                                     # follower -> leader frame (execute_step units)
        q[gi] = jaw_hold_cmd(o["joints_rad"][gi]); q[go] = jaw_hold_cmd(o["joints_rad"][go])   # jaws as COMMANDS 0..45
        latched = q; print(f"[live] arm joints latched (leader rad); {a.arm} jaw starts at cmd {q[gi]:.1f}, other jaw holds {q[go]:.1f}")
    cap, name, idx = open_camera(a.cam); print(f"[cam] {name} -> index {idx}")
    t0 = time.time(); cal = []; dmin = dmax = None; f_ema = None; cmd = latched[gi] if latched else 0.0; last_send = 0.0; last_seen = time.time()
    paused = False; win = "finger -> gripper  (SPACE pause, r re-pick tips, q quit)"; trk = TipTracker()
    lm = None; n_frames = n_hand = 0
    if a.detector == "mediapipe":
        from ego_collector.hands.mediapipe_tracker import HandLandmarker
        lm = HandLandmarker(num_hands=2, min_detection=0.3, min_presence=0.3, min_tracking=0.3)
    if a.detector == "appearance": lm = "appearance"
    stu = None
    if a.detector == "student":
        # [2026-10-13 user: "wilor는 잘 되는데 너무 느려"] WiLoR-distilled YOLO-pose (finger_distill_*.py): 4 kpts
        # [wrist, thumb tip, index MCP, index tip]; pinch = |thumb tip - index tip| / |wrist - index MCP| (scale-free)
        from ultralytics import YOLO
        stu = YOLO(a.student); lm = "student"
    wl = None; own_side = a.hand
    if a.detector == "wilor":
        # [2026-10-13 user: "Wilor가자"] MediaPipe kept taking the far (other) hand when only the wearer's thumb + index are in
        # view close to the lens; WiLoR detects partial close hands and labels left/right itself (not selfie-mirrored)
        lm = "wilor"
        if own_side == "auto": own_side = "left" if "left" in name.lower() else "right"
    # [2026-10-13 user: "반대손을 잡아"] fisheye -> pinhole (the wearer's hand is close to the lens and badly distorted at the
    # edge) and pick the WEARER's hand: the one entering from the bottom of the frame and appearing large; never the far hand.
    und = None
    if lm not in (None, "appearance") and not a.no_undistort:   # student: same undistortion as its WiLoR labels
        import yaml
        calmap = {"fisheyecamleft": "fisheye_left_v002.yaml", "arducam": "fisheye_right_v003.yaml"}
        key = next((k for k in calmap if k in name.lower().replace(" ", "")), None)
        if key:
            y = yaml.safe_load(open(pathlib.Path.home() / "ego_collector/configs/calibration" / calmap[key]))
            K = np.asarray(y["K"], float); D = np.asarray(y["D"], float).reshape(4, 1); cw, chh = y["image_size"]
            ok0, f0 = cap.read(); h0, w0 = f0.shape[:2]  # noqa; K[0] *= w0 / cw; K[1] *= h0 / chh
            Kp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w0, h0), np.eye(3), balance=0.6)
            if lm == "wilor": Kp[0, 2], Kp[1, 2] = w0 / 2.0, h0 / 2.0         # WiLoR assumes the principal point at the centre
            und = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kp, (w0, h0), cv2.CV_16SC2); print(f"[cam] undistort with {calmap[key]}")
    # [2026-10-13 user: "여전히 반대손을 더 잘 잡아 그리고 너무 느려"] the camera is rigid on the wrist, so the wearer's thumb + index
    # always sit in the same image region: detect ONLY inside a fixed box (drawn once, saved per camera). The other hand cannot be
    # picked outside it, and the small crop is much faster than the full frame.
    roi = None
    if lm is not None:
        import json as _json
        roi_file = pathlib.Path.home() / f".finger_grip_roi_{name.lower().replace(' ', '_')}{'_appearance' if lm == 'appearance' else ''}.json"
        if a.full_frame and lm != "appearance":
            okr, fr = cap.read(); roi = (0, 0, fr.shape[1], fr.shape[0]); print("[roi] full frame")
        elif roi_file.exists() and not a.pick_roi:
            roi = tuple(_json.loads(roi_file.read_text())); print(f"[roi] reuse {roi} from {roi_file} (--pick-roi to redraw)")
        else:
            for _ in range(5): okr, fr = cap.read()
            if und is not None: fr = cv2.remap(fr, *und, interpolation=cv2.INTER_LINEAR)
            sc = 960.0 / fr.shape[1]; small = cv2.resize(fr, None, fx=sc, fy=sc)
            cv2.putText(small, "drag a box around YOUR thumb + index (open hand), ENTER", (10, 30), 0, 0.7, (0, 255, 255), 2)
            bx = cv2.selectROI("finger box", small, showCrosshair=False); cv2.destroyWindow("finger box")
            if bx[2] < 10 or bx[3] < 10: sys.exit("no box drawn")
            roi = tuple(int(round(v / sc)) for v in bx); roi_file.write_text(_json.dumps(roi)); print(f"[roi] saved {roi} -> {roi_file}")
    if lm == "wilor":
        from ego_collector.hands3d.wilor import WilorTracker
        wl = WilorTracker(focal_px=float(Kp[0, 0]) if und is not None else 960.0, image_size=(roi[2], roi[3]), hand_conf=0.2)
        # [2026-10-13 user: "너무 느려"] WiLoR runs ~7 Hz on MPS (model-bound, not image size): run it in a worker thread on
        # the latest frame so the window stays at camera rate; the pinch updates at ~7 Hz (the jaw needs ~3 s full stroke anyway)
        import threading
        wst = {"img": None, "res": None, "seq": 0, "dt": 0.0}
        def _wilor_worker():
            while True:
                im = wst["img"]
                if im is None: time.sleep(0.003); continue
                wst["img"] = None; ta = time.time(); r = wl(im)
                seen_ = [h for h in r.values() if h.visible and h.kp3d is not None]
                hf_ = max(seen_, key=lambda h: (h.bbox[2] - h.bbox[0]) * (h.bbox[3] - h.bbox[1]), default=None)   # largest hand
                wst["res"] = (time.time(), None if hf_ is None else float(np.linalg.norm(hf_.kp3d[4] - hf_.kp3d[8])) * 1000.0,
                              None if hf_ is None else hf_.kp2d, [h.kp2d for h in seen_ if h is not hf_])
                wst["dt"] = time.time() - ta; wst["seq"] += 1
        threading.Thread(target=_wilor_worker, daemon=True).start(); wseq = 0; wkp = None; wkp_t = 0.0; wother = []
    app_w = None
    if lm == "appearance":
        # [2026-10-13] the near hand shows only thumb + index, no palm: MediaPipe / WiLoR found nothing (or the other hand).
        # No hand model: the camera is rigid on the wrist, so the finger box looks like "open" or "pinched". Two labelled
        # phases (move the wrist around in each) -> ridge-LDA direction on the normalised grey crop -> score 0 (pinched)..100 (open).
        def feat(im):
            rx, ry, rw, rh = roi; g = cv2.cvtColor(cv2.resize(im[ry:ry + rh, rx:rx + rw], (40, 30), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
            return (g - g.mean()) / (g.std() + 1e-3)
        X = {}
        for lab, msg in (("open", "OPEN: spread thumb + index wide"), ("closed", "PINCH: thumb + index together")):
            for ph, dur in (("get ready", 2.0), ("RECORDING", 4.0)):
                te = time.time() + dur
                while time.time() < te:
                    ok, im = cap.read()
                    if not ok: continue
                    if ph == "RECORDING": X.setdefault(lab, []).append(feat(im))
                    v = im.copy(); cv2.rectangle(v, roi[:2], (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 255), 3)
                    cv2.putText(v, f"{msg}  [{ph} {te - time.time():.1f}s]  keep it while moving the wrist", (20, 60), 0, 1.1, (0, 255, 255), 3)
                    cv2.imshow(win, cv2.resize(v, (960, 540))); cv2.waitKey(1)
        Xo, Xc = np.array(X["open"]), np.array(X["closed"]); Xa = np.vstack([Xo - Xo.mean(0), Xc - Xc.mean(0)])
        C = Xa.T @ Xa / len(Xa); lam = 0.1 * np.trace(C) / C.shape[0]
        w = np.linalg.solve(C + lam * np.eye(C.shape[0]), Xo.mean(0) - Xc.mean(0))
        so, sc_ = Xo @ w, Xc @ w; s0, s1 = np.median(sc_), np.median(so); app_w = (w, s0, s1)
        sep = float(((so - s0) / (s1 - s0) > 0.5).mean() + ((sc_ - s0) / (s1 - s0) < 0.5).mean()) / 2
        print(f"[appearance] {len(Xo)} open / {len(Xc)} pinched frames, training separation {100 * sep:.0f}%")
        dmin, dmax = 0.0, 100.0
    pick_far = False; t_fps = time.time(); fps = 0.0; t0 = time.time(); d_disp = None; d_disp_t = 0.0      # calibration starts after the model loads
    if app_w is not None: t0 -= 1000; cal = [0.0] * 10                  # already calibrated by the labelled phases
    while True:
        ok, img = cap.read()
        if not ok: continue
        now = time.time(); n_frames += 1; hand2d = None
        if stu is not None:
            if und is not None: img = cv2.remap(img, *und, interpolation=cv2.INTER_LINEAR)
            cands = []; blobs = []; dets = []; d = None
            res = stu.predict(img, imgsz=640, conf=0.4, verbose=False, device="mps")[0]
            if res.keypoints is not None and len(res.boxes):
                b = res.boxes.xyxy.cpu().numpy(); j = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))    # largest hand
                k4 = res.keypoints.xy.cpu().numpy()[j]; ref = np.linalg.norm(k4[0] - k4[2])
                if ref > 5: d = 100.0 * float(np.linalg.norm(k4[1] - k4[3])) / ref; n_hand += 1
                hand2d = np.zeros((21, 2)); hand2d[[0, 4, 5, 8]] = k4; hand2d[[1, 2, 3, 6, 7] + list(range(9, 21))] = k4[0]
        elif app_w is not None:
            w, s0, s1 = app_w; d = float(100.0 * (feat(img) @ w - s0) / (s1 - s0)); cands = []; blobs = []; dets = []; n_hand += 1
        elif wl is not None:
            if und is not None: img = cv2.remap(img, *und, interpolation=cv2.INTER_LINEAR)
            rx, ry, rw, rh = roi; cands = []; blobs = []; dets = []; d = None
            if wst["img"] is None: wst["img"] = np.ascontiguousarray(img[ry:ry + rh, rx:rx + rw])
            if wst["seq"] != wseq:                                  # a new WiLoR result (else no new measurement this frame)
                wseq = wst["seq"]; _, d, kp, wother = wst["res"]
                if kp is not None: wkp = kp + np.array([rx, ry]); wkp_t = now; n_hand += 1
            if wkp is not None and now - wkp_t < 0.3: hand2d = wkp
        elif lm is not None:
            # [2026-10-13 user] pinch from a hand-pose detector: thumb tip (4) - index tip (8) in MediaPipe's metric world
            # landmarks (mm), independent of the camera-hand distance; no tape, no red distractors
            if und is not None: img = cv2.remap(img, *und, interpolation=cv2.INTER_LINEAR)
            rx, ry, rw, rh = roi; dets = lm.detect_all(np.ascontiguousarray(img[ry:ry + rh, rx:rx + rw]), int(now * 1000)); cands = []; blobs = []
            for x in dets: x.landmarks_2d[:] += np.array([rx, ry])          # crop -> full-frame pixels for drawing
            ok_d = dets
            if ok_d:
                dd = max(ok_d, key=lambda x: float(np.sqrt(np.ptp(x.landmarks_2d, axis=0).prod()))); n_hand += 1      # largest hand in the box
                # [2026-10-13] 2D pixel distance: the world landmarks of the partial near hand were squashed (cal 14..21 mm);
                # the camera is rigid on the wrist, so the tip-to-tip pixel distance tracks the pinch directly
                d = (float(np.linalg.norm(dd.landmarks_3d[4] - dd.landmarks_3d[8])) * 1000.0 if a.pinch == "3d"
                     else float(np.linalg.norm(dd.landmarks_2d[4] - dd.landmarks_2d[8]))); hand2d = dd.landmarks_2d
            else:
                d = None
        else:
            cands = candidates(img, a.min_area, trk.ref_lab if dmin is not None else None)
            d, blobs = trk.update(cands)
        if d is None and now - last_seen > LOST_S: trk.reset()      # re-acquire the two largest matching-colour blobs
        if d is not None: last_seen = now
        if now - t0 < CAL_S or (len(cal) < 10 and now - t0 < 30):    # wait (<=30 s) until the hand was seen 10x
            if d is not None: cal.append(d)
            phase = f"CALIBRATING {CAL_S - (now - t0):.1f}s: open + close the fingers fully"
        else:
            if dmin is None:
                if len(cal) < 10: sys.exit("calibration failed: fingertips/hand not seen in 30 s")
                dmin, dmax = np.percentile(cal, 5), np.percentile(cal, 95)
                unit = "mm" if (lm == "wilor" or (a.detector == "mediapipe" and a.pinch == "3d")) else ("%hand" if lm == "student" else "px")
                if dmax - dmin < (15 if unit == "mm" else 20):                  # retry instead of quitting
                    print(f"[cal] range too small ({dmin:.0f}..{dmax:.0f} {unit}): open and close more -> recalibrating"); t0 = now; cal = []; dmin = None; continue
                print(f"[cal] closed {dmin:.0f} {unit}  open {dmax:.0f} {unit}")
            lost = (now - last_seen) > LOST_S
            if d is not None and not paused:
                f = float(np.clip((d - dmin) / (dmax - dmin), 0, 1)); f_ema = f if f_ema is None else EMA * f + (1 - EMA) * f_ema
            if f_ema is not None:
                if a.mode == "sigmoid":                                 # rescaled so f=0 -> 0 and f=1 -> 1
                    sg = lambda x: 1.0 / (1.0 + math.exp(-a.k * (x - 0.5)))
                    u = (sg(f_ema) - sg(0.0)) / (sg(1.0) - sg(0.0))
                elif a.mode == "binary":
                    u = 1.0 if f_ema > 0.6 else 0.0 if f_ema < 0.4 else (1.0 if cmd > CMD_MAX / 2 else 0.0)
                else:
                    u = f_ema
            target = cmd if (paused or lost or f_ema is None) else CMD_MAX * u
            dt = 1.0 / 30; cmd = cmd + float(np.clip(target - cmd, -MAX_CMD_PER_S * dt, MAX_CMD_PER_S * dt))
            phase = "PAUSED (hold)" if paused else "LOST -> HOLD" if lost else "TRACKING"
            if latched is not None and not paused and now - last_send >= 1.0 / SEND_HZ and abs(cmd - latched[gi]) >= DEADBAND:
                out = list(latched); out[gi] = float(np.clip(cmd, 0, CMD_MAX))
                try:
                    r = sess.post(a.robot + "/execute_step", json={"action": out}, timeout=2).json()
                    if isinstance(r, dict) and r.get("error"): print("[robot]", r["error"]); paused = True
                    else: latched = out
                except Exception as ex:
                    print("[robot] send failed:", ex); paused = True
                last_send = now
        dt = time.time() - t_fps; t_fps = time.time(); fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
        vis = img.copy()
        if roi is not None: cv2.rectangle(vis, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 255), 3)
        for _, c, _l in cands: cv2.circle(vis, tuple(int(x) for x in c), 10, (150, 150, 150), 2)     # ignored red things: grey
        for _, c, _l in blobs: cv2.circle(vis, tuple(int(x) for x in c), 14, (0, 255, 0), 3)
        if hand2d is not None:
            for j in range(21): cv2.circle(vis, tuple(int(x) for x in hand2d[j]), 4, (255, 200, 0), -1)
            cv2.line(vis, tuple(int(x) for x in hand2d[4]), tuple(int(x) for x in hand2d[8]), (0, 255, 0), 3)
        if wl is not None:
            for kp in wother: [cv2.circle(vis, tuple(int(v) for v in kp[j] + np.array(roi[:2])), 5, (120, 120, 120), -1) for j in (0, 4, 8)]
            cv2.putText(vis, f"WiLoR {1 / max(wst['dt'], 1e-3):.1f} Hz (hand found = per WiLoR result)", (20, 200), 0, 0.8, (0, 255, 255), 2)

        if lm not in (None, "wilor") and dets:
            for x in dets:
                if hand2d is None or x.landmarks_2d is not hand2d:
                    for j in (0, 4, 8): cv2.circle(vis, tuple(int(v) for v in x.landmarks_2d[j]), 5, (120, 120, 120), -1)   # other hand: grey
        if lm is not None: cv2.putText(vis, f"{a.detector} {fps:.1f} fps (box only)", (20, 165), 0, 0.8, (0, 255, 255), 2)
        if d is not None: d_disp = d; d_disp_t = now
        dshow = d_disp if now - d_disp_t < LOST_S else None; nden = (wst["seq"] if wl is not None else n_frames)
        if lm is not None: cv2.putText(vis, f"hand found {100 * n_hand / max(nden, 1):.0f}%  pinch {dshow:.0f} {'mm' if (lm == 'wilor' or (a.detector == 'mediapipe' and a.pinch == '3d')) else ('%' if lm == 'appearance' else 'px')}" if dshow else f"hand found {100 * n_hand / max(nden, 1):.0f}%  (no hand)", (20, 130), 0, 0.8, (0, 255, 255), 2)
        if len(blobs) == 2 and lm is None: cv2.line(vis, tuple(int(x) for x in blobs[0][1]), tuple(int(x) for x in blobs[1][1]), (0, 255, 0), 2)
        cv2.rectangle(vis, (20, 20), (420, 60), (40, 40, 40), -1); cv2.rectangle(vis, (20, 20), (20 + int(400 * cmd / CMD_MAX), 60), (0, 200, 255), -1)
        cv2.putText(vis, f"{'LIVE' if latched is not None else 'DRY-RUN'} {a.arm} jaw cmd {cmd:4.1f}/45  [{a.mode}]  {phase}", (20, 95), 0, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, cv2.resize(vis, (960, 540))); k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27): break
        if k == ord(" "): paused = not paused
        if k == ord("r"): trk.reset(); trk.ref_lab = None; print("[tips] re-pick: the two largest red blobs")
        if k == ord("h"): print("[roi] to move the finger box restart with --pick-roi")
    cap.release(); cv2.destroyAllWindows(); print(f"[exit] jaw left at cmd {cmd:.1f} (not snapped)")


if __name__ == "__main__":
    main()
