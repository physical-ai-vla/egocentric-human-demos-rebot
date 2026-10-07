"""ego16: one HandUMI episode -> a 16-D EEF trajectory in the reBot base frame, ready for the continuity-IK retarget.

    python -m ego_teleop.tools.ego16 EPISODE [--orientation fixed|relative] [--vio pose_openvins_seg]

Output  EPISODE/derived/humanik/ego16.npz
    S16       (T,16) float32  [L x y z qx qy qz qw grip01 | R ...]  robot-base frame (m, xyzw), on the HEAD timeline
    valid_L/R (T,)   bool     pose usable at t (VIO valid or static hold; never an init-lag frame)
    grip_valid (T,2) bool     gripper stream healthy this episode (per jaw)
    pose_conf (T,2) / grip_conf (T,2) float  method-4 reliability weights in [0,1] (pose: valid x tracker state, degraded 0.5;
                              grip: healthy x 0.5 within +-3 frames of a label transition) -- exported as arm_conf / grip_conf
    t_ns (T,) int64, go_idx / stop_idx int, plus meta.json with every frame convention used.

This is the seam between our capture and the HumanIK retarget path (~/humanik_retarget: build_eef16 -> retarget.solve_trajectory
-> assemble -> to_lerobot). What it decides, and why:
  frame     the segmented VIO output frame is the camera frame at the first still frame, not gravity-aligned. The first IMU
            static window gives gravity in the camera frame (R_camera_imu . mean accel), which becomes robot +z; the camera
            optical axis at GO, projected on the horizontal, becomes robot +x (the operator faces the plate, so does the robot).
            Yaw beyond that is unknowable from one wrist camera and is not invented.
  anchor    p_robot(t) = p_home_side + R_align (p_world(t) - p_world(t_GO)). At GO both hands rest at HOME by protocol; the
            robot's home TCP (rebot_b601 home_q FK, configs/handumi/retarget_shakedown.yaml) is the matching anchor. Scale 1.
  rotation  `fixed` (default): the robot home orientation, arm-only IK as in the 2026-09-04 IK audit (IK_VALID 76 %).
            `relative`: home orientation composed with the camera's rotation since GO, expressed in the aligned frame.
  gripper   binary, FROZEN semantics identical to the R150 recipe (frame-verified on both devices): 0 = jaw closed & empty
            (rest), 1 = jaw at least half open (approach / holding / open). Rest mode from the pre-GO hold, widest opening
            from the session band's extreme, threshold half-way with hysteresis 0.40/0.60. See grip_binary. (v1/v1b)
            `--grip-rule command` (v1c): what the R150 model really outputs -- 1 = jaw commanded open (opening / kept wide),
            0 = closing / HOLDING / idle -- from the aperture trajectory (run-up / drawdown). See grip_command_label.
  timeline  head frames (the training master clock). Wrist poses are taken at the nearest wrist frame within max_gap_ms.
Nothing here is a relative-motion objective yet; the exporter/training patch (humanik_delta) builds t+5 deltas from q."""
from __future__ import annotations
import os
import argparse
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
from handumi_collector.pose.episode_io import RawEpisode
from handumi_collector.pose.calibration import SideCalibration
from handumi_collector.pose.backends.segmented import static_windows, DEFAULTS as SEG_DEFAULTS

CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "handumi" / "retarget_shakedown.yaml"
SIDES = ("left", "right")
_CLUSTERS: dict = {}          # (session, side) -> session_grip_clusters, computed once per run


def rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9: return np.eye(3) if c > 0 else Rotation.from_rotvec(np.pi * np.array([1.0, 0, 0])).as_matrix()
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1, 1))).as_matrix()


MOD = 4096


def circular_unwrap(raw: np.ndarray, center: float | None = None) -> tuple[np.ndarray, float]:
    """Ticks modulo 4096, re-centred so the jaw's working band is contiguous. The driver's unwrapped counter jumps by whole
    turns at power cycles and glitches to ~32800 (2026-09-16), so nothing here trusts it beyond its remainder."""
    m = np.asarray(raw, float) % MOD
    if center is None:
        ang = m / MOD * 2 * np.pi; center = float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi) * MOD)
    return ((m - center + MOD / 2) % MOD) - MOD / 2 + center, center


def kmeans2(x: np.ndarray, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """1-D two-cluster k-means; returns sorted centres and occupancy."""
    c = np.percentile(x, [10, 90]).astype(float)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None]).argmin(1); c = np.array([x[lab == k].mean() if (lab == k).any() else c[k] for k in range(2)])
    lab = np.abs(x[:, None] - c[None]).argmin(1); return c, np.bincount(lab, minlength=2) / len(x)


def session_grip_clusters(session_dir: Path, side: str, ref_episode: Path | None = None, min_sep: float = 200.0, min_occ: float = 0.05, band_gap: float = 600.0) -> dict:
    """Data-driven jaw modes for one servo BAND of a session (the same rule is applied to the robot teleop set).
    A session can hold more than one servo band when the counter jumps mid-session (125236: episodes 1-10 sat at ~50-1000,
    14-18 at ~2360-2850), so episodes are grouped by their median raw (circular, band_gap ticks) and only the band the
    reference episode belongs to is clustered: 1-D two-mode k-means on the pooled samples. Valid when the modes are
    >= min_sep apart and the minority mode holds >= min_occ; otherwise the caller falls back to per-episode extremes."""
    eps = []
    for em in sorted(glob.glob(str(session_dir / "episode_*" / "episode_meta.json"))):
        try:
            r = RawEpisode.load(Path(em).parent); g = r.grip.get(side)
            if g is not None and len(g): eps.append((Path(em).parent, np.asarray(g.raw_position, float)))
        except Exception: continue
    if not eps: return dict(valid=False, center=0.0, modes=[0.0, 1.0], occupancy=[1.0, 0.0], separation=0.0, n=0, band_episodes=0)
    med = np.array([np.median(x % MOD) for _, x in eps]); ref_i = next((i for i, (e, _) in enumerate(eps) if ref_episode is not None and e.resolve() == ref_episode.resolve()), 0)
    d = np.abs(((med - med[ref_i] + MOD / 2) % MOD) - MOD / 2); in_band = d <= band_gap
    raw = np.concatenate([x for (_, x), ok in zip(eps, in_band) if ok]); u, center = circular_unwrap(raw)
    c, occ = kmeans2(u); sep = float(abs(c[1] - c[0]))
    return dict(center=center, modes=[float(c[0]), float(c[1])], occupancy=[float(occ[0]), float(occ[1])], separation=sep, p01=float(np.percentile(u, 1)), p99=float(np.percentile(u, 99)),
                valid=bool(sep >= min_sep and occ.min() >= min_occ), n=int(len(raw)), band_episodes=int(in_band.sum()), band_median=float(med[ref_i]))


def grip_binary(raw: np.ndarray, t_ns: np.ndarray, hold_t0_ns: int, hold_t1_ns: int, clusters: dict, lo: float = 0.40, hi: float = 0.60) -> dict:
    """Binary grip, FROZEN semantics identical to the R150 robot recipe (verified on frames, 2026-09-16):
        0 = jaw closed and empty (the rest state)          1 = jaw at least half open (approach, holding the cube, or wide open)
    Both devices have the same three physical levels and the same rest state -- reBot: fingers together at raw 0, holding a
    5 cm cube at ~-136, wide open at -270; HandUMI: prongs together at the rest mode, holding at ~0.5-0.85 of the way to the
    widest opening, wide open at the far extreme. The robot recipe labels raw <= -135 (half way) as 1 and reached real-robot
    success at 250k with it, so the ego label is the same half-way rule between the rest mode and the widest opening
    (band p99), with hysteresis lo/hi to stop chatter at the threshold (a held cube sits right around half on both devices).
    The longer human bouts (jaw kept open while waiting) are a behaviour difference under the same label -- v2 alignment."""
    u, _ = circular_unwrap(raw, clusters["center"]); m = (t_ns >= hold_t0_ns) & (t_ns <= hold_t1_ns)
    rest = float(np.median(u[m])) if m.sum() >= 5 else float(np.median(u[: max(len(u) // 10, 5)]))
    c0, c1 = clusters["modes"]; rest_m, far_m = (c0, c1) if abs(rest - c0) <= abs(rest - c1) else (c1, c0)
    open_m = float(clusters.get("p99", far_m)) if far_m >= rest_m else float(clusters.get("p01", far_m))     # widest opening, not the far mode
    norm = np.clip((u - rest_m) / (open_m - rest_m), 0.0, 1.0) if abs(open_m - rest_m) > 1e-6 else np.zeros(len(u))
    g = np.zeros(len(u), np.float32); state = 0.0
    for i, v in enumerate(norm):
        if v > hi: state = 1.0
        elif v < lo: state = 0.0
        g[i] = state
    cycles = int(((g[1:] == 1) & (g[:-1] == 0)).sum())
    return dict(grip01=g, norm=norm.astype(np.float32), idle_mode=rest_m, engaged_mode=open_m, rest_level=rest, engaged_frac=float(g.mean()), cycles=cycles,
                travel=float(abs(open_m - rest_m)), direction="up" if open_m > rest_m else "down")


def grip_command_label(norm: np.ndarray, d_open: float = 0.10, d_up: float = 0.03, d_down: float = 0.05, d_grasp: float = 0.08, sustain: int = 6, hold_min: float = 0.35, open_min: float = 0.25, slip: float = 0.15, smooth: int = 5) -> dict:
    """Command-state gripper label, matching what the R150 model actually outputs (verified on R675 frames + traces, 2026-09-16):
        1 = jaw commanded OPEN and empty: opening, kept wide before the grasp, re-opened for the release, or idling open
        0 = jaw commanded CLOSED: closing onto the cube, HOLDING it, or resting closed and empty
    On the robot the leader trigger (cmd >= 27) drives the follower to -270 (wide open); relaxing it closes the jaw, which stalls
    on the cube (~-118) -> holding is 0 on both arms. HandUMI has no trigger, so the label is read from the aperture TRAJECTORY
    (levels cannot do it: a held cube sits at ~0.7, an idling open jaw at ~0.8): the smoothed openness is segmented into plateaus
    and moves (a departure of >= d_up / d_down from the current plateau that lasts `sustain` frames); a state machine walks them:
        CLOSED --up (>= d_open)--> OPEN ;  OPEN --down ending >= hold_min--> HOLD ;  OPEN --down ending < hold_min--> CLOSED
        HOLD --up (>= d_up, even a few % -- the human releases by barely opening)--> OPEN ;  HOLD --down past the cube--> CLOSED
    (OPEN needs a drop >= d_grasp to become HOLD; HOLD survives a further squeeze that stays >= hold_min and within `slip`)
    Every transition is back-dated to the onset of the move (offline labelling may look ahead). Human idle-open after a release
    stays 1 (jaw open, empty = the robot's open command); that they rarely close back to rest is a behaviour difference."""
    o = pd.Series(norm).rolling(smooth, center=True, min_periods=1).median().to_numpy(); n = len(o)
    g = np.zeros(n, np.float32); state = "CLOSED" if o[0] < hold_min else "OPEN"; level = float(o[0]); i = 0; onset = None; moves = []
    while i < n:
        dev = o[i] - level; up_th = d_open if state == "CLOSED" else d_up          # leaving rest needs a real opening; a release can be a few %
        if (dev >= up_th or dev <= -d_down) and i + sustain <= n and (np.sign(o[i:i + sustain] - level) == np.sign(dev)).all() and np.abs(o[i:i + sustain] - level).min() >= min(d_up, d_down):
            j = i                                                   # ride the move until it plateaus
            while j + 1 < n and abs(o[j + 1] - o[j]) > 0.004: j += 1
            j = max(j, i + sustain - 1); new_level = float(np.median(o[j:min(j + sustain, n)])); up = new_level > level
            if state == "CLOSED" and up: state = "OPEN" if new_level >= open_min else "CLOSED"     # a fidget of a few % is not an open command
            elif state == "OPEN" and not up:                        # a real drop onto something = grasp; a few % of relaxation is still open
                if new_level < hold_min: state = "CLOSED"
                elif level - new_level >= d_grasp: state = "HOLD"
            elif state == "HOLD" and up: state = "OPEN"             # any sustained re-opening releases (the human barely opens to let go)
            elif state == "HOLD" and not up:                        # squeezing harder keeps holding; closing past where the cube was = it is gone
                if new_level < max(hold_min, level - slip): state = "CLOSED"
            moves.append((i, j, round(level, 3), round(new_level, 3), state)); g[i:j + 1] = 1.0 if state == "OPEN" else 0.0; level = new_level; i = j + 1
        else:
            g[i] = 1.0 if state == "OPEN" else 0.0; level = 0.9 * level + 0.1 * float(o[i]) if abs(dev) < min(d_up, d_down) / 2 else level; i += 1
    bouts = int(((g[1:] == 1) & (g[:-1] == 0)).sum() + (g[0] == 1))
    return dict(grip01=g, open_frac=float(g.mean()), bouts=bouts, moves=moves, hold_frac=float(np.mean([m[4] == "HOLD" for m in moves])) if moves else 0.0)


_CUBE_HSV = {"R": [((0, 80, 60), (8, 255, 255)), ((168, 80, 60), (180, 255, 255))], "B": [((96, 80, 50), (111, 255, 255))], "P": [((113, 60, 40), (135, 255, 255))]}


def object_in_jaw_series(ep_dir: Path, side: str, T: int, step: int = 2) -> np.ndarray:
    """Contact cue from the WRIST camera: fraction of the jaw ROI (bottom-centre of the fisheye view, where the prongs meet) covered by
    the largest red/blue/purple blob, per head frame (nearest wrist frame; every `step`-th frame decoded, linearly held in between).
    Measured 2026-09-16: HOLD p50 8-16 %, idle CLOSED 0.3-0.6 %, OPEN 4-7 % (a cube seen between open prongs on approach)."""
    import cv2
    cap = cv2.VideoCapture(str(ep_dir / f"{side}_wrist.mp4")); n = int(cap.get(7)); out = np.full(T, np.nan, np.float32)
    for i in range(0, T, step):
        cap.set(1, int(i * n / max(T, 1))); ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (480, 270)); H, W = f.shape[:2]; roi = f[int(H * 0.55):, int(W * 0.30):int(W * 0.70)]; hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV); best = 0.0
        for rngs in _CUBE_HSV.values():
            m = np.zeros(roi.shape[:2], np.uint8)
            for lo, hi in rngs: m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); k, _, st, _ = cv2.connectedComponentsWithStats(m)
            if k > 1: best = max(best, float(st[1:, cv2.CC_STAT_AREA].max()) / m.size)
        out[i] = best
    cap.release(); return pd.Series(out).interpolate(limit_direction="both").to_numpy(np.float32)


def grip_contact_label(norm: np.ndarray, obj: np.ndarray, hold_obj_min: float = 0.02, empty_obj_max: float = 0.01, confirm_s: float = 0.5, fps: int = 30, **kw) -> dict:
    """Contact-aware gripper backend (E): the v1c command-state machine decides from the aperture TRAJECTORY, then the object-in-jaw cue
    corrects the two cases aperture alone cannot see, with a `confirm_s` persistence so a single frame never flips a state:
        HOLD with no object for confirm_s        -> IDLE_CLOSED   (jaw squeezed on nothing)
        OPEN/CLOSED at hold aperture (0.35..0.92) with an object between the prongs for confirm_s -> HOLD_OBJECT (grasp the trajectory missed)
    Internal states IDLE_CLOSED / OPEN_EMPTY / HOLD_OBJECT / RELEASE_OPEN; label = 1 for OPEN_EMPTY / RELEASE_OPEN, 0 otherwise -- the same
    R90 command semantics as v1c. Robot-side inference is untouched; this only changes the ego pseudo-label."""
    base = grip_command_label(norm, **kw); n = len(norm); st = np.array(["IDLE_CLOSED"] * n, dtype=object)
    cur = "IDLE_CLOSED" if norm[0] < 0.35 else "OPEN_EMPTY"; k = 0; mv = base["moves"]
    for i in range(n):
        while k < len(mv) and mv[k][0] <= i:
            cur = {"OPEN": "OPEN_EMPTY", "HOLD": "HOLD_OBJECT", "CLOSED": "IDLE_CLOSED"}[mv[k][4]]; k += 1
        st[i] = cur
    o = pd.Series(norm).rolling(5, center=True, min_periods=1).median().to_numpy(); cf = max(1, int(confirm_s * fps)); changed = 0
    ob = pd.Series(obj).rolling(cf, min_periods=1).median().to_numpy()                  # persistence: median over the last confirm_s
    for i in range(n):
        if st[i] == "HOLD_OBJECT" and ob[i] < empty_obj_max and i >= cf: st[i] = "IDLE_CLOSED"; changed += 1
        elif st[i] in ("OPEN_EMPTY", "IDLE_CLOSED") and 0.35 <= o[i] <= 0.92 and ob[i] >= hold_obj_min and i >= cf and abs(o[i] - o[max(0, i - cf)]) < 0.03:
            st[i] = "HOLD_OBJECT"; changed += 1
    # a re-opening straight out of HOLD is a RELEASE_OPEN until the next plateau
    for i in range(1, n):
        if st[i] == "OPEN_EMPTY" and st[i - 1] in ("HOLD_OBJECT", "RELEASE_OPEN") and o[i] >= o[i - 1] - 1e-3: st[i] = "RELEASE_OPEN"
    g = np.isin(st, ["OPEN_EMPTY", "RELEASE_OPEN"]).astype(np.float32)
    return dict(grip01=g, states=st, open_frac=float(g.mean()), bouts=int(((g[1:] == 1) & (g[:-1] == 0)).sum() + (g[0] == 1)), moves=mv, changed_frames=int(changed),
                agreement_with_v1c=float((g == base["grip01"]).mean()))


def object_in_jaw_cue(ep_dir: Path, side: str, T: int, step: int = 2, occ_min: float = 0.15, s_min: int = 70, v_min: int = 50) -> np.ndarray:
    """v2 object-in-jaw EVIDENCE per head frame (bool), colour-agnostic: the largest SATURATED blob (S > s_min, V > v_min -- a cube of any colour;
    the table is white, the prongs are black) covers >= occ_min of the jaw ROI (bottom-centre of the wrist view). A held cube a few cm from
    the lens fills the ROI; a cube on the table seen through the open jaw is small. Measured 2026-09-17 on v1c states (6 eps, both wrists):
    HOLD frames pass 63 %, OPEN 29 % (approach frames with the cube between the prongs), CLOSED 2 %. Hue-specific masks were dropped
    because close-range purple/orange cubes fell outside the HSV ranges and produced false demotions on the left wrist."""
    import cv2
    cap = cv2.VideoCapture(str(ep_dir / f"{side}_wrist.mp4")); n = int(cap.get(7)); out = np.zeros(T, np.float32); seen = np.zeros(T, bool)
    for i in range(0, T, step):
        cap.set(1, int(i * n / max(T, 1))); ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (480, 270)); H, W = f.shape[:2]; roi = f[int(H * 0.55):, int(W * 0.30):int(W * 0.70)]; hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] > s_min) & (hsv[..., 2] > v_min)).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        k, _, st, _ = cv2.connectedComponentsWithStats(m); big = float(st[1:, cv2.CC_STAT_AREA].max()) / m.size if k > 1 else 0.0
        out[i] = float(big >= occ_min); seen[i] = True
    cap.release(); idx = np.flatnonzero(seen)
    return np.interp(np.arange(T), idx, out[idx]) >= 0.5 if len(idx) else np.zeros(T, bool)


def grip_contact_v2(norm: np.ndarray, cue: np.ndarray, hold_confirm: float = 0.30, promote_min: float = 1.01, fps: int = 30, **kw) -> dict:
    """Contact-aware gripper v2: the v1c trajectory machine proposes plateaus (moves = transitions), then PLATEAU-LEVEL object-in-jaw evidence
    decides HOLD (temporal persistence instead of per-frame flips, which made v0 produce false HOLDs):
        v1c HOLD plateau with evidence fraction >= hold_confirm      -> HOLD_OBJECT (confirmed)
        v1c HOLD plateau with evidence <  hold_confirm               -> OPEN_EMPTY if aperture >= 0.35 else IDLE_CLOSED (jaw squeezed nothing)
        v1c OPEN plateau at hold aperture (0.35..0.85), evidence >= promote_min -> HOLD_OBJECT  -- DISABLED by default (promote_min > 1):
            eye-check 2026-09-17: promotions were stacks on the plate seen through the open jaw (wide blobs), not grasps
        down-moves -> CLOSING ; up-moves out of HOLD_OBJECT -> RELEASE_OPEN ; other up-moves keep the next plateau's state
    Binary label: OPEN_EMPTY / RELEASE_OPEN -> 1 ; IDLE_CLOSED / CLOSING / HOLD_OBJECT -> 0 (R90 command semantics, unchanged)."""
    base = grip_command_label(norm, **kw); n = len(norm); mv = base["moves"]; o = pd.Series(norm).rolling(5, center=True, min_periods=1).median().to_numpy()
    st = np.array(["IDLE_CLOSED" if o[0] < 0.35 else "OPEN_EMPTY"] * n, dtype=object); changes = []
    # plateau segments between moves, each carrying the v1c state that starts at that move
    bounds = [(0, mv[0][0] - 1, "CLOSED" if o[0] < 0.35 else "OPEN")] if mv else [(0, n - 1, "CLOSED" if o[0] < 0.35 else "OPEN")]
    for k, m in enumerate(mv):
        a = m[1] + 1; b = mv[k + 1][0] - 1 if k + 1 < len(mv) else n - 1
        if b >= a: bounds.append((a, b, m[4]))
    prev = None
    for k, m in enumerate(mv):                                            # moves themselves
        s0, s1 = m[0], m[1]; up = m[3] > m[2]
        st[s0:s1 + 1] = ("RELEASE_OPEN" if (up and prev == "HOLD_OBJECT") else ("OPEN_EMPTY" if up else "CLOSING"))
        prev = None
    for a, b, v1c in bounds:
        ev = float(cue[a:b + 1].mean()) if b >= a else 0.0; ap = float(np.median(o[a:b + 1]))
        if v1c == "HOLD": new = "HOLD_OBJECT" if ev >= hold_confirm else ("OPEN_EMPTY" if ap >= 0.35 else "IDLE_CLOSED")
        elif v1c == "OPEN": new = "HOLD_OBJECT" if (0.35 <= ap <= 0.85 and ev >= promote_min) else "OPEN_EMPTY"     # a held 5 cm cube sits at ~0.7-0.8; >0.85 = wide open over the stack
        else: new = "IDLE_CLOSED"
        st[a:b + 1] = new
        if (v1c == "HOLD") != (new == "HOLD_OBJECT"): changes.append(dict(start=int(a), end=int(b), v1c=v1c, v2=new, evidence=round(ev, 2), aperture=round(ap, 2)))
    # a move leaving a confirmed HOLD upward is a release
    for k, m in enumerate(mv):
        if m[3] > m[2] and m[0] > 0 and st[m[0] - 1] == "HOLD_OBJECT": st[m[0]:m[1] + 1] = "RELEASE_OPEN"
    g = np.isin(st, ["OPEN_EMPTY", "RELEASE_OPEN"]).astype(np.float32)
    return dict(grip01=g, states=st, open_frac=float(g.mean()), bouts=int(((g[1:] == 1) & (g[:-1] == 0)).sum() + (g[0] == 1)), moves=mv, changes=changes,
                agreement_with_v1c=float((g == base["grip01"]).mean()), hold_frac=float((st == "HOLD_OBJECT").mean()),
                plateaus=dict(hold_confirmed=sum(1 for a, b, v in bounds if v == "HOLD" and st[a] == "HOLD_OBJECT"), hold_demoted=sum(1 for a, b, v in bounds if v == "HOLD" and st[a] != "HOLD_OBJECT"),
                              open_promoted=sum(1 for a, b, v in bounds if v == "OPEN" and st[a] == "HOLD_OBJECT")))


def nearest(ts_src: np.ndarray, ts_dst: np.ndarray, max_gap_ns: int) -> np.ndarray:
    """index into ts_src nearest each ts_dst, -1 when farther than max_gap."""
    j = np.searchsorted(ts_src, ts_dst); j = np.clip(j, 1, len(ts_src) - 1)
    left, right = ts_src[j - 1], ts_src[j]; pick = np.where(np.abs(ts_dst - left) <= np.abs(right - ts_dst), j - 1, j)
    gap = np.abs(ts_src[pick] - ts_dst); pick[gap > max_gap_ns] = -1
    return pick


def build(ep_dir: Path, *, orientation: str = "fixed", vio: str = "pose_openvins_seg", vio_left: str = "auto", vio_right: str = "fixed", vio_mode: str = "any", cfg_path: Path = CFG_PATH, grip_rule: str = "aperture") -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    home = {s: np.asarray(cfg["robot_home_tcp"][s], float) for s in SIDES}          # 7: xyz + quat xyzw (rebot_b601 home_q FK)
    scale = float(cfg.get("scale", 1.0)); max_gap_ns = int(float(cfg.get("max_pose_gap_ms", 100)) * 1e6)
    ep = RawEpisode.load(ep_dir); t_head = np.asarray(ep.frames["head"].capture_ns, np.int64); T = len(t_head)
    events = ep.events; go = [e for e in events if e["kind"] == "auto_loop" and e["detail"].get("cue") == "go"]
    stop = [e for e in events if e["kind"] == "auto_loop" and e["detail"].get("cue") == "auto_stop"]
    go_idx = int(np.argmin(np.abs(t_head - go[0]["t_ns"]))) if go else int(np.argmin(np.abs(t_head - (ep.t_start_ns + int(3.0e9)))))
    stop_idx = int(np.argmin(np.abs(t_head - stop[0]["t_ns"]))) if stop else T - 1
    session_dir = ep_dir.parent; smeta = json.loads((session_dir / "session_meta.json").read_text())
    S16 = np.full((T, 16), np.nan, np.float32); valid = {}; gvalid = np.zeros((T, 2), bool); pose_conf = np.zeros((T, 2), np.float32); grip_conf = np.zeros((T, 2), np.float32); open_cont = np.zeros((T, 2), np.float32); meta = dict(episode=ep_dir.name, session=session_dir.name, orientation=orientation, vio=vio,
                                                                                                          go_idx=go_idx, stop_idx=stop_idx, sides={})
    for si, side in enumerate(SIDES):
        vio_dir = vio
        auto = (side == "left" and vio_left == "auto") or (side == "right" and vio_right == "auto")
        if auto:                                                       # per-episode source selection: the run with more valid poses wins
            # left: the camera-IMU offset is session-dependent (30.17 / 35.97 runs); both sides: pose_mast3r (MASt3R-Fusion replay) when present
            names = (("pose_openvins_seg_l30", "pose_openvins_seg_l36") if side == "left" else ()) + ("pose_mast3r", vio)
            cands = [d for d in names if (ep_dir / "derived" / d / f"{side}_camera_pose.parquet").exists()]
            if cands:
                fr = {d: float(pd.read_parquet(ep_dir / "derived" / d / f"{side}_camera_pose.parquet", columns=["valid"]).valid.mean()) for d in cands}
                vio_dir = max(fr, key=fr.get)
                if side == "left": meta["left_vio_choice"] = dict(chosen=vio_dir, valid_frac=fr)
        if vio_mode == "mast3r_only" and "mast3r" not in vio_dir:
            raise FileNotFoundError(f"{ep_dir.name} {side}: VIO_BACKEND=mast3r_only but no pose_mast3r* trajectory ({vio_dir} would be used)")
        meta.setdefault("pose_source", {})[side] = dict(dir=vio_dir, backend=("mast3r_fusion" if "mast3r" in vio_dir else "openvins"), selection="auto" if auto else "fixed")
        meta["pose_backend"] = "mast3r_fusion" if all("mast3r" in v["dir"] for v in meta["pose_source"].values()) else ("mixed" if any("mast3r" in v["dir"] for v in meta["pose_source"].values()) else "openvins")
        cam = pd.read_parquet(ep_dir / "derived" / vio_dir / f"{side}_camera_pose.parquet")
        tv = cam.t_ns.to_numpy(np.int64); v = cam.valid.to_numpy(bool)
        P = cam[["x", "y", "z"]].to_numpy(float); Q = cam[["qx", "qy", "qz", "qw"]].to_numpy(float)
        # --- gravity in the VIO output frame: first IMU static window -> mean accel -> camera frame; the output frame at that
        #     moment is the (stitched) camera pose, so rotate by it too
        cal = SideCalibration(side); Rci = np.eye(3) if cal.T_camera_imu is None else np.asarray(cal.T_camera_imu)[:3, :3]
        imu = ep.imu[side]
        wins = static_windows(imu.host_ns, imu.gyro, imu.accel, SEG_DEFAULTS)
        if wins:
            w = wins[0]; m = (imu.host_ns >= w["t0_ns"]) & (imu.host_ns <= w["t1_ns"]); a_imu = imu.accel[m].mean(axis=0)
            i_w = int(np.argmin(np.abs(tv - (w["t0_ns"] + w["t1_ns"]) // 2))); R_cam_at = Rotation.from_quat(Q[i_w]).as_matrix() if v[i_w] else np.eye(3)
        else:
            a_imu = imu.accel[:200].mean(axis=0); R_cam_at = np.eye(3)
        g_out = R_cam_at @ (Rci @ a_imu)                       # accelerometer reads +g upward when still -> this points UP in the output frame
        R_up = rot_a_to_b(g_out, np.array([0.0, 0.0, 1.0]))
        # --- yaw: camera optical axis (+z) at GO, projected on the horizontal, -> robot +x
        j_go = nearest(tv, t_head[go_idx:go_idx + 1], max_gap_ns)[0]
        if j_go < 0 or not v[j_go]:
            cand = np.nonzero(v)[0]; j_go = int(cand[np.argmin(np.abs(tv[cand] - t_head[go_idx]))]) if len(cand) else 0
        R_cam_go = Rotation.from_quat(Q[j_go]).as_matrix()
        fwd = R_up @ (R_cam_go @ np.array([0.0, 0.0, 1.0])); fwd[2] = 0.0
        if np.linalg.norm(fwd) < 1e-6: fwd = np.array([1.0, 0.0, 0.0])
        yaw = float(np.arctan2(fwd[1], fwd[0])); R_yaw = Rotation.from_euler("z", -yaw).as_matrix()
        R_align = R_yaw @ R_up
        p_go = P[j_go]; R_home = Rotation.from_quat(home[side][3:7]).as_matrix()
        # --- resample wrist poses onto the head timeline
        idx = nearest(tv, t_head, max_gap_ns); ok = (idx >= 0)
        ok[ok] &= v[idx[ok]]
        pos = np.full((T, 3), np.nan); quat = np.tile(home[side][3:7], (T, 1)).astype(float)
        for t in np.nonzero(ok)[0]:
            j = idx[t]; pos[t] = home[side][:3] + scale * (R_align @ (P[j] - p_go))
            if orientation == "relative":
                dR = R_align @ (Rotation.from_quat(Q[j]).as_matrix() @ R_cam_go.T) @ R_align.T
                quat[t] = Rotation.from_matrix(dR @ R_home).as_quat()
        # plausibility: a diverging VIO segment (left wrist, 2026-09-16: metres of "motion" at 5-30 m/s) must not reach the IK
        far = np.linalg.norm(np.nan_to_num(pos - home[side][:3], nan=0.0), axis=1) > float(cfg.get("max_reach_from_home_m", 0.8))
        dt = np.diff(t_head) / 1e9; step = np.linalg.norm(np.diff(np.nan_to_num(pos, nan=0.0), axis=0), axis=1); fast = np.concatenate([[False], (step / np.maximum(dt, 1e-3)) > float(cfg.get("max_speed_m_s", 2.0))])
        bad = ok & (far | fast); ok &= ~bad; pos[bad] = np.nan; n_implausible = int(bad.sum())
        # sign continuity of the quaternion
        for t in range(1, T):
            if np.dot(quat[t], quat[t - 1]) < 0: quat[t] = -quat[t]
        o = si * 8; S16[:, o:o + 3] = pos; S16[:, o + 3:o + 7] = quat; valid[side] = ok
        # method-4 reliability weight for this arm: valid frames weighted by the tracker state (degraded = half); everything else 0
        st = cam.tracking_state.to_numpy(str); w_state = np.where(st == "degraded", 0.5, 1.0)
        pose_conf[ok, si] = w_state[idx[ok]]
        # --- gripper: session-level two-mode clustering + hysteresis (grip_binary); hold window = REC start .. GO
        g = ep.grip[side]; raw = np.asarray(g.raw_position, float); tg = np.asarray(g.t_ns, np.int64)
        cl = _CLUSTERS.setdefault((str(session_dir), side, str(ep_dir)), session_grip_clusters(session_dir, side, ref_episode=ep_dir))
        gi = nearest(tg, t_head, int(150e6)); graw = np.where(gi >= 0, raw[np.clip(gi, 0, len(raw) - 1)], np.nan)
        graw = pd.Series(graw).ffill().bfill().to_numpy()
        if cl["valid"]: gb = grip_binary(graw, t_head, int(t_head[0]), int(t_head[go_idx]), cl); source = "session_kmeans2"
        else:                                                           # fallback: this episode's own two extremes as the modes
            u, c = circular_unwrap(graw); ep_cl = dict(center=c, modes=[float(np.percentile(u, 2)), float(np.percentile(u, 98))], p01=float(np.percentile(u, 1)), p99=float(np.percentile(u, 99)))
            gb = grip_binary(graw, t_head, int(t_head[0]), int(t_head[go_idx]), ep_cl); source = "episode_extremes_fallback"
        if grip_rule == "command":                                   # v1c: 1 = commanded open (opening / kept wide), 0 = closing / holding / idle
            gc = grip_command_label(gb["norm"]); gb["grip01"] = gc["grip01"]; gb["engaged_frac"] = gc["open_frac"]; gb["cycles"] = gc["bouts"]
        elif grip_rule == "contact":                                 # E: v1c trajectory machine + wrist-camera object-in-jaw cue -> ego16_con.npz
            obj = object_in_jaw_series(ep_dir, side, T); gc = grip_contact_label(gb["norm"], obj)
            gb["grip01"] = gc["grip01"]; gb["engaged_frac"] = gc["open_frac"]; gb["cycles"] = gc["bouts"]
            gb["contact"] = dict(changed_frames=gc["changed_frames"], agreement_with_v1c=gc["agreement_with_v1c"], hold_frac=float((gc["states"] == "HOLD_OBJECT").mean()),
                                 obj_area_p50_hold=float(np.nanmedian(obj[gc["states"] == "HOLD_OBJECT"])) if (gc["states"] == "HOLD_OBJECT").any() else None)
        elif grip_rule == "v2contact":                               # E v2: plateau-level object-in-jaw evidence -> ego16_v2c.npz
            cue = object_in_jaw_cue(ep_dir, side, T); gc = grip_contact_v2(gb["norm"], cue)
            gb["grip01"] = gc["grip01"]; gb["engaged_frac"] = gc["open_frac"]; gb["cycles"] = gc["bouts"]
            gb["contact"] = dict(version="v2", agreement_with_v1c=gc["agreement_with_v1c"], hold_frac=gc["hold_frac"], plateaus=gc["plateaus"], changes=gc["changes"])
        elif grip_rule != "aperture": raise ValueError(f"grip_rule {grip_rule!r}")
        S16[:, o + 7] = gb["grip01"]; open_cont[:, si] = gb["norm"]
        ep_excursion = float(np.ptp(circular_unwrap(graw, cl["center"] if cl["valid"] else None)[0]))     # this EPISODE's own jaw travel
        healthy = (raw.std() > 3.0) and (len(raw) >= 20 * (t_head[-1] - t_head[0]) / 1e9) and gb["travel"] >= 250 and ep_excursion >= 250
        gvalid[:, si] = healthy & (gi >= 0)
        # gripper reliability: the label is least certain around its own transitions (+-3 frames = 0.5), 1 elsewhere, 0 when the stream is unhealthy
        g01 = np.asarray(gb["grip01"], np.float32); tr = np.flatnonzero(np.diff(g01) != 0); near = np.zeros(T, bool)
        for k in tr: near[max(0, k - 2):min(T, k + 4)] = True
        grip_conf[:, si] = gvalid[:, si] * np.where(near, 0.5, 1.0)
        ext = dict(source=source, rule=grip_rule, contact=gb.get("contact"), session_modes=cl["modes"], session_occupancy=cl["occupancy"], separation=cl["separation"], idle_mode=gb["idle_mode"], engaged_mode=gb["engaged_mode"],
                   direction=gb["direction"], travel=gb["travel"], episode_excursion=ep_excursion, healthy=bool(healthy), engaged_frac=gb["engaged_frac"], cycles=gb["cycles"])
        meta["sides"][side] = dict(R_align=R_align.round(6).tolist(), yaw_deg=float(np.degrees(yaw)), gravity_out_frame=g_out.round(4).tolist(),
                                   p_world_go=p_go.round(4).tolist(), pose_valid_frac=float(ok.mean()), implausible_frames=n_implausible, grip=ext,
                                   pos_range_m=[float(np.nanmin(pos[:, k]) if ok.any() else np.nan) for k in range(3)] + [float(np.nanmax(pos[:, k]) if ok.any() else np.nan) for k in range(3)])
    # invalid frames: hold the last known pose so IK has a continuous target; validity tells the loss what to trust
    for si, side in enumerate(SIDES):
        o = si * 8; df = pd.DataFrame(S16[:, o:o + 7]).ffill().bfill(); S16[:, o:o + 7] = df.to_numpy(np.float32)
        S16[np.isnan(S16[:, o]), o:o + 3] = home[side][:3]
    out_dir = ep_dir / "derived" / "humanik"; out_dir.mkdir(parents=True, exist_ok=True)
    stem = "ego16" if grip_rule == "aperture" else ("ego16_v2c" if grip_rule == "v2contact" else f"ego16_{grip_rule[:3]}")   # ego16 / ego16_com (v1c) / ego16_con (v0) / ego16_v2c (v2)
    stem += os.environ.get("EGO16_STEM_SUFFIX", "")   # [2026-09-17] e.g. "_rel" for --orientation relative variants (audit only; frozen outputs untouched)
    np.savez(out_dir / f"{stem}.npz", S16=S16, valid_L=valid["left"], valid_R=valid["right"], grip_valid=gvalid, openness=open_cont, t_ns=t_head, go_idx=go_idx, stop_idx=stop_idx,
             pose_conf=pose_conf, grip_conf=grip_conf)
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--orientation", default="fixed", choices=["fixed", "relative"])
    ap.add_argument("--vio", default="pose_openvins_seg"); ap.add_argument("--vio-left", default="auto", help="auto: pick pose_openvins_seg_l30 / _l36 by valid fraction (left offset is session-dependent)")
    ap.add_argument("--vio-right", default="fixed", choices=["fixed", "auto"], help="auto: pick pose_mast3r over the --vio dir when it has more valid poses (recorded in meta.pose_source)")
    ap.add_argument("--vio-mode", default="any", choices=["any", "mast3r_only"], help="mast3r_only (production E100/E150): both wrists must come from a pose_mast3r* dir, else the episode fails; `any` keeps the auto/OpenVINS fallback for debug")
    ap.add_argument("--grip-rule", default="aperture", choices=["aperture", "command", "contact", "v2contact"], help="aperture: v1/v1b half-way rule -> ego16.npz; command: opening/kept-open=1, closing/holding/idle=0 -> ego16_com.npz")
    a = ap.parse_args(argv)
    m = build(Path(a.episode), orientation=a.orientation, vio=a.vio, vio_left=a.vio_left, vio_right=a.vio_right, vio_mode=a.vio_mode, grip_rule=a.grip_rule)
    for s in SIDES:
        d = m["sides"][s]; g = d["grip"]; print(f"[{s}] pose valid {d['pose_valid_frac']:.3f} | yaw {d['yaw_deg']:+.1f} deg | grip {g['source']} modes {np.round(g['session_modes'],0)} occ {np.round(g['session_occupancy'],2)} idle {g['idle_mode']:.0f} engaged {g['engaged_mode']:.0f} healthy {g['healthy']} engaged_frac {g['engaged_frac']:.2f} cycles {g['cycles']} | implausible {d['implausible_frames']} | xyz {np.round(d['pos_range_m'][:3],2)}..{np.round(d['pos_range_m'][3:],2)}")
    print(f"go_idx {m['go_idx']} stop_idx {m['stop_idx']} -> derived/humanik/{'ego16' if a.grip_rule == 'aperture' else 'ego16_com'}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
