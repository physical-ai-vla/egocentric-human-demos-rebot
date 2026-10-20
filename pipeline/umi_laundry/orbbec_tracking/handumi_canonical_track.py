#!/usr/bin/env python3
"""[2026-10-12] Canonical rigid-point tracker for the two HandUMI units on top of the Orbbec laundry_v1 detector.

Why: laundry_v1 reports, per side, the trimmed median of the VISIBLE dark unit points. That point slides 1-2 cm
across the unit as the viewpoint / occlusion changes, so it is not a fixed physical point (see the memory note
handumi-centroid-not-rigid and scripts/body_canonical_tracker.py, whose lessons this file applies):

  * de-rotate every cloud by a TRUSTED rotation, accumulate one canonical model in the unit (IMU) body frame, and
    register every frame to that model (frame-to-MODEL, never frame-to-frame: nothing accumulates);
  * the model is a pose-binned UNION (a voxel is rigid if it is seen repeatedly from at least one viewing direction),
    never an intersection -- a solid body occludes itself;
  * positions live in the Orbbec TABLE world, rotations in the IMU GRAVITY world: they share +Z and differ by an
    unknown YAW that must be solved, not assumed zero.

Pipeline per episode (needs <ep>/derived/body_track/track.npz from `handumi_body_track.py --profile laundry_v1
--save-npz`):

  1. AHRS per wrist IMU (Mahony: gyro + accel-gravity for roll/pitch, accel gated when |a| != g), device clock mapped
     onto the host clock by a lower-envelope fit, gyro bias from the REC stillness hold. -> R_g_imu(t) at Orbbec t.
  2. Clouds: the laundry_v1 detector is re-run per frame with the tracked centre as ROI; each side keeps the
     candidate points within CROP_M of its tracked centre (nearest side wins where both crops overlap).
  3. Yaw: global search of the yaw between gravity world and table world (sharpest canonical model = lowest
     registration residual), then a SLOW yaw drift term (gyro heading drift is unobservable to the accelerometer):
     per-frame 4-DOF (yaw + t) registration, heavily smoothed (YAW_SMOOTH_S) so it can only absorb drift, never
     per-frame position error.
  4. Canonical model in the IMU body frame (pose-binned union), recentred so the canonical point = model centroid;
     every frame registered TRANSLATION ONLY with R = Rz(yaw(t)) R_g_imu(t). Rounds are printed.
  5. Validity from registration fitness (residual, inlier ratio, visible model fraction, points).
  6. Temporal assignment: per frame, the straight (L->L, R->R) vs swapped labelling is costed with
     predicted-position distance + lambda * velocity inconsistency; plus a max-speed gate (V_MAX_MPS) and an IMU
     cross-check (each cloud against the OTHER unit's model + attitude). Suspected swaps mark BOTH sides INVALID;
     an implausible jump marks that side INVALID and the other WARN. Nothing is ever relabelled -- no guessing.
  7. Outputs: <ep>/derived/body_track/canonical.npz + canonical_summary.json, and per side
     orbbec_valid_wristclock_<side>.npy (structured: wrist video_frame, frame_index, capture_ns, orbbec_valid, ...)
     for OR/AND with MASt3R validity on the wrist timeline.

    .venv/bin/python scripts/handumi_canonical_track.py --episode <ep> [--debug-dir DIR --debug-frames 4]

All parameters live in CANON_V1 and were FROZEN on 2026-10-12 after tuning on Hfold 163800/ep1, 163312/ep1,
152646/ep2 only (held-out UMIpilot_20261012_170552/ep1 never used for tuning).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import median_filter, gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.pose.rgbd_io import RgbdEpisode                              # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                            # noqa: E402
from scripts import handumi_body_track as bt                                        # noqa: E402

SIDES = ("left", "right")
CANON_V1 = dict(
    version="canonical_v1",
    # --- AHRS (Mahony)
    AHRS_KP=1.0,             # rad/s per unit error: ~1 s gravity time constant
    AHRS_KI=0.005,           # slow roll/pitch gyro-bias adaptation
    AHRS_ACC_TOL=1.2,        # m/s^2: accel correction only when | |a| - g_rest | < this (else it is motion, not gravity)
    AHRS_GYRO_MAX=4.0,       # rad/s: and the unit is not spinning fast
    BIAS_WIN_S=1.0,          # gyro bias = mean over the stillest window of this length in the first 3 s (REC hold)
    # --- clouds
    CROP_M=0.12,             # canonical ROI radius around the detector's trimmed centre
    PTS_PER_FRAME=1500,      # random subsample per frame and side
    MIN_PTS=150,
    # --- canonical model
    VOXEL_M=0.006,
    N_VIEW_BINS=42,          # fibonacci-sphere viewing-direction bins in the body frame (~30 deg apart)
    BIN_MIN_FRAMES=6,        # a view bin votes only when it has this many frames
    VIS_BIN=0.30,            # a voxel is rigid if it is occupied in >= this share of the frames of ANY view bin (union)
    MAX_MODEL_FRAMES=1200,   # frames used to build the model (evenly subsampled, good-fitness ones after round 1)
    # --- registration
    ICP_ITERS=15,
    TRIM=0.70,               # correspondence trim: hand, shirt and the gripper fingers are not the rigid body
    INLIER_M=0.010,          # inlier / visible-model distance
    # --- yaw
    YAW_GRID_DEG=20.0, YAW_FINE_DEG=1.0, YAW_SEARCH_FRAMES=300, YAW_SEARCH_PTS=400,
    ANCHOR_S=20.0, ANCHOR_MIN_FRAMES=150,   # yaw anchor window at the start (least heading drift)
    YAW_TRACK_PTS=600,       # cloud subsample for the 4-DOF yaw passes
    YAW_STEP_MAX_DEG=12.0,   # a per-frame yaw correction beyond this is not drift -> discarded
    YAW_HIST=45,             # running yaw = median of the last 45 accepted per-frame estimates
    YAW_SMOOTH_S=6.0,        # yaw(t): median over this, then Gaussian sigma = YAW_SMOOTH_S/3
    # --- validity (canonical registration fitness)
    VALID_RESID_MM=7.0, VALID_INLIER=0.50, VALID_VISFRAC=0.05,   # visfrac of a UNION model is small by nature (p10 0.06-0.15)
    # rigid-quality tier: below these the frame stays VALID (the unit is there) but is flagged LOW_VIS (WARN): too little
    # of the model is seen to pin the rigid point -- on the tuning set these frames carry 2-3x the high-frequency
    # jitter (Spearman rho(visfrac, jitter) -0.50..-0.63). rigid_ok = valid & !LOW_VIS.
    RIGID_VISFRAC=0.12, RIGID_INLIER=0.75,
    # --- temporal assignment
    V_MAX_MPS=1.5,           # maximum plausible unit speed (hand-held, folding): beyond this between accepted samples
    JUMP_SLACK_M=0.03,       # + this slack (detector noise) on the distance allowance V_MAX * dt
    REANCHOR_S=0.5,          # after this long without an accepted sample the track re-anchors (flag WARN)
    LAMBDA_VEL_S=0.10,       # cost = |p - pred| + LAMBDA * |v_new - v_prev|   (s: velocity term in metres)
    SWAP_MARGIN_M=0.03,      # swapped labelling must be cheaper by this to raise SWAP_SUSPECT
    XCHECK_RATIO=0.60,       # IMU cross-check: other unit's model+attitude fits >= this much better -> WARN
    VEL_EMA=0.5,
    WRIST_MAX_DT_MS=40.0,    # wrist-clock resampling: bracketing Orbbec frames must be within this
)

# flag bits
F_DET_INVALID = 1 << 0    # detector had no valid track
F_FEW_PTS = 1 << 1
F_FIT_BAD = 1 << 2        # registration fitness below the validity thresholds
F_JUMP = 1 << 3           # this side broke the speed gate -> INVALID
F_PARTNER_JUMP = 1 << 4   # the other side broke the speed gate -> WARN
F_SWAP = 1 << 5           # straight vs swapped labelling: swapped is cheaper -> both INVALID
F_XCHECK = 1 << 6         # IMU cross-check prefers the other unit's model -> both WARN
F_REANCHOR = 1 << 7       # first accepted sample after a gap > REANCHOR_S -> WARN
F_NO_IMU = 1 << 8
F_LOW_VIS = 1 << 9        # too little of the model visible / weak inliers -> rigid point uncertain (WARN)
INVALID_MASK = F_DET_INVALID | F_FEW_PTS | F_FIT_BAD | F_JUMP | F_SWAP | F_NO_IMU
WARN_MASK = F_PARTNER_JUMP | F_XCHECK | F_REANCHOR | F_LOW_VIS
FLAG_NAMES = {F_DET_INVALID: "DET_INVALID", F_FEW_PTS: "FEW_PTS", F_FIT_BAD: "FIT_BAD", F_JUMP: "JUMP",
              F_PARTNER_JUMP: "PARTNER_JUMP", F_SWAP: "SWAP_SUSPECT", F_XCHECK: "XCHECK_WARN",
              F_REANCHOR: "REANCHOR", F_NO_IMU: "NO_IMU", F_LOW_VIS: "LOW_VIS"}


# ============================================================================================ AHRS
def imu_host_time(im) -> np.ndarray:
    """Device micros -> host monotonic ns. Host receive = true time + USB latency >= 0, so the LOWER envelope of
    host - slope*device is the offset (1st percentile; robust to the batched, repeated host stamps)."""
    dev = im.device_us.astype(np.float64)
    b, a = np.polyfit(dev, im.host_ns.astype(np.float64), 1)
    r = im.host_ns - (b * dev + a)
    return (b * dev + a + np.percentile(r, 1)).astype(np.float64)


def mahony(t_ns: np.ndarray, gyro: np.ndarray, acc: np.ndarray, P: dict) -> tuple[np.ndarray, dict]:
    """R_g_imu (body->gravity world, +Z up, yaw arbitrary = 0 at start) at every IMU sample."""
    t = (t_ns - t_ns[0]) / 1e9
    # gyro bias + g_rest from the stillest BIAS_WIN_S window in the first 3 s (the collector's REC stillness hold)
    n = len(t)
    best = None
    for s0 in np.arange(0.0, max(0.01, min(3.0, t[-1]) - P["BIAS_WIN_S"]) + 1e-9, 0.1):
        m = (t >= s0) & (t < s0 + P["BIAS_WIN_S"])
        if m.sum() < 20:
            continue
        sc = float(np.linalg.norm(gyro[m].std(0)) + np.linalg.norm(gyro[m].mean(0)) * 0.2)
        if best is None or sc < best[0]:
            best = (sc, m)
    m = best[1]
    bias = gyro[m].mean(0)
    g_rest = float(np.linalg.norm(acc[m], axis=1).mean())
    a0 = acc[m].mean(0); a0 /= np.linalg.norm(a0)
    # initial attitude: rotate body 'up' (a0) onto world +Z
    R = Rotation.align_vectors([[0, 0, 1.0]], [a0])[0].as_matrix()
    Rs = np.empty((n, 3, 3))
    integ = np.zeros(3)
    used = 0
    for k in range(n):
        Rs[k] = R
        if k == n - 1:
            break
        dt = float(t[k + 1] - t[k])
        if dt <= 0 or dt > 0.05:
            dt = min(max(dt, 0.0), 0.05)
        w = gyro[k] - bias
        an = np.linalg.norm(acc[k])
        if abs(an - g_rest) < P["AHRS_ACC_TOL"] and np.linalg.norm(w) < P["AHRS_GYRO_MAX"]:
            v = R[2]                      # world +Z expressed in body = R^T e_z = third row of R
            e = np.cross(acc[k] / an, v)
            integ += P["AHRS_KI"] * e * dt
            w = w + P["AHRS_KP"] * e + integ
            used += 1
        R = R @ Rotation.from_rotvec(w * dt).as_matrix()
        if k % 200 == 0:                   # re-orthonormalise
            U, _, Vt = np.linalg.svd(R); R = U @ Vt
    return Rs, dict(gyro_bias=bias.tolist(), g_rest=g_rest, acc_corr_share=used / max(n - 1, 1),
                    bias_window_s=[float(t[m][0]), float(t[m][-1])])


def ahrs_at(raw: RawEpisode, side: str, t_query: np.ndarray, P: dict):
    im = raw.imu.get(side)
    if im is None or len(im) < 100:
        return None, None
    th = imu_host_time(im)
    o = np.argsort(th, kind="stable"); th = th[o]
    keep = np.r_[True, np.diff(th) > 0]
    th, gy, ac = th[keep], im.gyro[o][keep], im.accel[o][keep]
    Rs, info = mahony(th, gy, ac, P)
    sl = Slerp(th, Rotation.from_matrix(Rs))
    tq = np.clip(t_query.astype(np.float64), th[0], th[-1])
    out = sl(tq).as_matrix()
    ok = (t_query >= th[0] - 5e7) & (t_query <= th[-1] + 5e7)
    info["imu_span_ns"] = [int(th[0]), int(th[-1])]
    return out, dict(info, ok_share=float(ok.mean()), ok=ok)


# ============================================================================================ geometry
def table_frame(nrm: np.ndarray, pd: float) -> tuple[np.ndarray, np.ndarray]:
    """Orbbec TABLE world: +Z = plane normal (up, towards the camera), z = height above the table, x along the
    camera x-axis projected onto the table. p_W = R_WC p_C + t_WC."""
    n = nrm / np.linalg.norm(nrm)
    ex = np.array([1.0, 0, 0]) - n * n[0]; ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    R = np.stack([ex, ey, n])
    return R, np.array([0.0, 0.0, pd])


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


# ============================================================================================ clouds
def collect_clouds(ep_path: Path, z: dict, P: dict, cache: Path, log=print) -> dict:
    if cache.exists():
        c = np.load(cache)
        out = {}
        for s in SIDES:
            off = c[f"off_{s}"]; flat = c[f"pts_{s}"]
            out[s] = dict(frames=c[f"frames_{s}"], clouds=[flat[off[k]:off[k + 1]] for k in range(len(off) - 1)])
        log(f"  clouds from cache {cache.name}")
        return out
    ep = RgbdEpisode.load(ep_path)
    K = ep.intrinsics.K
    nrm, pd = z["plane_normal"], float(z["plane_d"])
    L = bt.LAUNDRY_V1
    scene = bt.laundry_scene_model(ep, nrm, pd, L)
    xyz = {s: z[f"xyz_{s}"] for s in SIDES}; val = {s: z[f"valid_{s}"] for s in SIDES}
    res = {s: dict(frames=[], clouds=[]) for s in SIDES}
    t0 = time.time()
    for f in ep.iter_frames(0, ep.n_frames, 1):
        i = f.index
        live = [s for s in SIDES if val[s][i]]
        if not live:
            continue
        dets, _ = bt.detect_bodies_laundry(f, K, nrm, pd, scene, L, [xyz[s][i] for s in live])
        if not dets:
            continue
        Pts = np.concatenate([d["points"] for d in dets])
        dist = {s: np.linalg.norm(Pts - xyz[s][i], axis=1) for s in live}
        for s in live:
            m = dist[s] < P["CROP_M"]
            o = "right" if s == "left" else "left"
            if o in dist:
                m &= dist[s] <= dist[o]
            Q = Pts[m]
            if len(Q) < P["MIN_PTS"]:
                continue
            if len(Q) > P["PTS_PER_FRAME"]:
                Q = Q[np.random.default_rng(i * 2 + (s == "right")).choice(len(Q), P["PTS_PER_FRAME"], replace=False)]
            res[s]["frames"].append(i); res[s]["clouds"].append(Q.astype(np.float32))
        if i % 500 == 0:
            print(f"    clouds f{i}/{ep.n_frames}  {time.time()-t0:.0f}s", flush=True)
    sv = {}
    for s in SIDES:
        res[s]["frames"] = np.asarray(res[s]["frames"], np.int64)
        sv[f"frames_{s}"] = res[s]["frames"]
        sv[f"off_{s}"] = np.cumsum([0] + [len(c) for c in res[s]["clouds"]])
        sv[f"pts_{s}"] = (np.concatenate(res[s]["clouds"]) if res[s]["clouds"] else np.zeros((0, 3), np.float32))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **sv)
    return res


# ============================================================================================ registration
def reg_translation(X: np.ndarray, tree: cKDTree, P: dict, t0=None):
    """Translation-only trimmed ICP of body-frame points X onto the model. Returns (t, resid_m, inlier, nn_idx)."""
    t = np.zeros(3) if t0 is None else t0.copy()
    d = j = None
    for _ in range(P["ICP_ITERS"]):
        d, j = tree.query(X + t, k=1)
        keep = d <= max(np.quantile(d, P["TRIM"]), 1e-4)
        if keep.sum() < 30:
            break
        step = (tree.data[j[keep]] - (X[keep] + t)).mean(0)
        t = t + step
        if np.linalg.norm(step) < 2e-5:
            break
    d, j = tree.query(X + t, k=1)
    keep = d <= max(np.quantile(d, P["TRIM"]), 1e-4)
    return t, float(d[keep].mean()), float((d < P["INLIER_M"]).mean()), j[d < P["INLIER_M"]]


def reg_yaw_translation(Pw: np.ndarray, c: np.ndarray, R: np.ndarray, tree: cKDTree, P: dict):
    """4-DOF: rotation about WORLD z through c + translation, registering the world cloud onto the model placed at
    (c, R). Returns (delta_yaw_deg, new c, resid). Closed-form yaw (2D Procrustes on xy) inside an ICP loop."""
    Mw = lambda idx: c + tree.data[idx] @ R.T             # model points in world for matched indices
    A = Pw.copy()
    Rtot = np.eye(3); ttot = np.zeros(3)
    resid = np.inf
    for _ in range(P["ICP_ITERS"]):
        X = (A - c) @ R
        d, j = tree.query(X, k=1)
        keep = d <= max(np.quantile(d, P["TRIM"]), 1e-4)
        if keep.sum() < 30:
            break
        a = A[keep]; b = Mw(j[keep])
        ma, mb = a.mean(0), b.mean(0)
        ac, bc = a - ma, b - mb
        ang = np.arctan2((ac[:, 0] * bc[:, 1] - ac[:, 1] * bc[:, 0]).sum(), (ac[:, 0] * bc[:, 0] + ac[:, 1] * bc[:, 1]).sum())
        Rz = rz(np.degrees(ang))
        tt = mb - Rz @ ma
        A = A @ Rz.T + tt
        Rtot = Rz @ Rtot; ttot = Rz @ ttot + tt
        resid = float(d[keep].mean())
        if abs(ang) < 1e-4 and np.linalg.norm(tt - (ma - Rz @ ma) * 0) < 2e-5:
            break
    # the cloud moved by (Rtot, ttot); equivalently the BODY moved by the inverse: c' = Rtot^T (c - ttot), R' = Rtot^T R
    dyaw = -np.degrees(np.arctan2(Rtot[1, 0], Rtot[0, 0]))
    c_new = Rtot.T @ (c - ttot)
    return dyaw, c_new, resid


# ============================================================================================ canonical model
def build_model(clouds, frames, c, Rb, cam_W, P, sel=None):
    """Pose-binned UNION. Returns model points (body frame), voxel stats."""
    idx = np.arange(len(frames)) if sel is None else np.asarray(sel)
    if len(idx) > P["MAX_MODEL_FRAMES"]:
        idx = idx[np.linspace(0, len(idx) - 1, P["MAX_MODEL_FRAMES"]).astype(int)]
    bins = fib_sphere(P["N_VIEW_BINS"])
    Xs, fid, vb = [], [], []
    for k in idx:
        X = (clouds[k] - c[k]) @ Rb[k]
        Xs.append(X); fid.append(np.full(len(X), k))
        v = Rb[k].T @ (cam_W - c[k]); v /= np.linalg.norm(v)
        vb.append(int(np.argmax(bins @ v)))
    X = np.concatenate(Xs); fid = np.concatenate(fid)
    vbin_of = dict(zip(idx.tolist(), vb))
    key = np.floor(X / P["VOXEL_M"]).astype(np.int64)
    off = key.min(0); key -= off
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uvox, vinv = np.unique(lin, return_inverse=True)
    # occupancy per (frame, voxel), then per (bin, voxel)
    fv = np.unique(np.stack([fid, vinv], 1), axis=0)
    fb = np.array([vbin_of[f] for f in fv[:, 0]])
    nb_frames = np.bincount(np.array(vb), minlength=len(bins))
    occ = np.zeros((len(bins), len(uvox)))
    np.add.at(occ, (fb, fv[:, 1]), 1.0)
    rate = occ / np.maximum(nb_frames[:, None], 1)
    rate[nb_frames < P["BIN_MIN_FRAMES"]] = 0.0
    keepv = rate.max(0) >= P["VIS_BIN"]
    sums = np.zeros((len(uvox), 3)); cnt = np.bincount(vinv, minlength=len(uvox)).astype(float)
    np.add.at(sums, vinv, X)
    model = sums[keepv] / cnt[keepv, None]
    return model, dict(n_vox_seen=int(len(uvox)), n_vox_kept=int(keepv.sum()),
                       bins_used=int((nb_frames >= P["BIN_MIN_FRAMES"]).sum()), n_frames=int(len(idx)))


def register_all(clouds, c, Rb, model, P, idx=None):
    tree = cKDTree(model)
    n = len(clouds)
    idx = range(n) if idx is None else idx
    c_new = c.copy(); res = np.full(n, np.nan); inl = np.full(n, np.nan); vis = np.full(n, np.nan)
    for k in idx:
        X = (clouds[k] - c[k]) @ Rb[k]
        t, e, r, jj = reg_translation(X, tree, P)
        c_new[k] = c[k] - Rb[k] @ t
        res[k] = e; inl[k] = r; vis[k] = len(np.unique(jj)) / len(model)
    return c_new, res, inl, vis


def recentre(model, c, Rb):
    m = model.mean(0)
    return model - m, c + np.einsum("nij,j->ni", Rb, m)


# ============================================================================================ per side
def _smooth_yaw(t_s, est, P):
    """Robust slow yaw(t) from sparse per-frame estimates (deg): unwrap, time-interpolate, median + Gaussian."""
    ok = np.isfinite(est)
    if ok.sum() < 10:
        return None
    u = np.degrees(np.unwrap(np.radians(est[ok])))
    dd = np.interp(t_s, t_s[ok], u)
    fps = 1.0 / max(np.median(np.diff(t_s)), 1e-3)
    w = max(3, int(P["YAW_SMOOTH_S"] * fps) | 1)
    return gaussian_filter1d(median_filter(dd, size=w, mode="nearest"), sigma=w / 3, mode="nearest")


def track_side(side, clouds, frames, seeds_W, Rg, cam_W, t_s, P, log=print):
    """Yaw between the IMU gravity world and the table world, then the canonical model and per-frame position.

    Heading from a 6-axis AHRS drifts (residual gyro z-bias, unobservable to the accelerometer), measured on the
    tuning set as jagged / high-floor yaw curves over whole 60-120 s episodes. So:
      a) ANCHOR: grid-search the yaw on the first ANCHOR_S seconds (bias was calibrated at REC start -> least drift);
      b) TRACK: walk through the episode in time order, 4-DOF (yaw + t) registration of each cloud against the anchor
         model starting from the running yaw; accepted per-frame yaws -> heavily smoothed yaw(t) (drift only);
      c) REFINE: full-episode union model with yaw(t), second 4-DOF pass around yaw(t) against it, smooth again;
      d) FINAL: rebuild + translation-only registration rounds with R = Rz(yaw(t)) R_g_imu(t)."""
    n = len(frames)
    T0 = time.time()
    c0 = seeds_W.copy()
    rng = np.random.default_rng(0)
    small = [c[rng.choice(len(c), min(len(c), P["YAW_SEARCH_PTS"]), replace=False)] for c in clouds]
    mid = [c[rng.choice(len(c), min(len(c), P["YAW_TRACK_PTS"]), replace=False)] for c in clouds]
    anchor = np.flatnonzero(t_s <= t_s[0] + P["ANCHOR_S"])
    if len(anchor) < P["ANCHOR_MIN_FRAMES"]:
        anchor = np.arange(min(n, P["ANCHOR_MIN_FRAMES"]))
    sub = anchor[np.linspace(0, len(anchor) - 1, min(len(anchor), P["YAW_SEARCH_FRAMES"])).astype(int)]

    def score(yaw, rounds=3):
        Rb = np.einsum("ij,njk->nik", rz(yaw), Rg)
        c = c0.copy()
        res = None
        for _ in range(rounds):
            mdl, _ = build_model(small, frames, c, Rb, cam_W, P, sel=sub)
            if len(mdl) < 50:
                return np.inf
            mdl, c = recentre(mdl, c, Rb)
            c, res, _, _ = register_all(small, c, Rb, mdl, P, idx=sub)
        return float(np.nanmedian(res[sub]))

    grid = np.arange(0, 360, P["YAW_GRID_DEG"])
    sc = np.array([score(y) for y in grid])
    best = float(grid[int(np.argmin(sc))])
    for half, step in ((P["YAW_GRID_DEG"], P["YAW_GRID_DEG"] / 4), (P["YAW_GRID_DEG"] / 4, P["YAW_FINE_DEG"])):
        cand = np.arange(best - half, best + half + 1e-9, step)
        scc = np.array([score(y) for y in cand])
        best = float(cand[int(np.argmin(scc))])
    yaw0 = best % 360
    s_best, s_flip = score(yaw0), score(yaw0 + 180)
    depth = float(np.median(sc) / s_best)
    log(f"  [{side}] anchor yaw {yaw0:.1f} deg on {len(sub)} frames of the first {t_s[anchor[-1]]-t_s[0]:.0f} s: "
        f"resid {1e3*s_best:.2f} mm, +180 deg {1e3*s_flip:.2f} mm, grid median {1e3*np.median(sc):.2f} mm (x{depth:.2f})  [{time.time()-T0:.0f}s]")

    # anchor model at yaw0
    Rb = np.einsum("ij,njk->nik", rz(yaw0), Rg)
    c = c0.copy()
    for _ in range(3):
        mdl, _ = build_model(clouds, frames, c, Rb, cam_W, P, sel=anchor)
        mdl, c = recentre(mdl, c, Rb)
        c, res, inl, vis = register_all(clouds, c, Rb, mdl, P, idx=anchor)
    tree = cKDTree(mdl)
    # sequential yaw tracking against the anchor model
    est = np.full(n, np.nan)
    cur, hist = yaw0, []
    for k in range(n):
        R = rz(cur) @ Rg[k]
        t, e, _, _ = reg_translation((mid[k] - c0[k]) @ R, tree, P)
        ck = c0[k] - R @ t
        dy, _, e2 = reg_yaw_translation(mid[k], ck, R, tree, P)
        if e2 * 1e3 <= P["VALID_RESID_MM"] and abs(dy) <= P["YAW_STEP_MAX_DEG"]:
            est[k] = cur + dy
            hist.append(est[k])
            cur = float(np.median(hist[-P["YAW_HIST"]:]))
    yaw_t = _smooth_yaw(t_s, est, P)
    if yaw_t is None:
        yaw_t = np.full(n, yaw0)
    n_seq = int(np.isfinite(est).sum())
    # refine: full union model with yaw(t), 4-DOF around yaw(t), smooth again
    Rb = np.einsum("nij,njk->nik", np.stack([rz(y) for y in yaw_t]), Rg)
    for _ in range(2):
        mdl, st = build_model(clouds, frames, c0 if _ == 0 else c, Rb, cam_W, P)
        mdl, c = recentre(mdl, c0 if _ == 0 else c, Rb)
        c, res, inl, vis = register_all(clouds, c, Rb, mdl, P)
    tree = cKDTree(mdl)
    est2 = np.full(n, np.nan)
    for k in range(n):
        if not (res[k] * 1e3 <= P["VALID_RESID_MM"] and inl[k] >= P["VALID_INLIER"]):
            continue
        dy, _, e2 = reg_yaw_translation(mid[k], c[k], Rb[k], tree, P)
        if abs(dy) <= P["YAW_STEP_MAX_DEG"]:
            est2[k] = yaw_t[k] + dy
    y2 = _smooth_yaw(t_s, est2, P)
    if y2 is not None:
        yaw_t = y2
    dev = est2 - yaw_t
    log(f"  [{side}] yaw(t): sequential {n_seq}/{n} frames, refine {int(np.isfinite(est2).sum())}/{n}; drift over episode "
        f"{np.min(yaw_t)-yaw0:+.1f}..{np.max(yaw_t)-yaw0:+.1f} deg; per-frame scatter |dev| p50 {np.nanmedian(np.abs(dev)):.1f} deg  [{time.time()-T0:.0f}s]")

    rounds = []
    Rb = np.einsum("nij,njk->nik", np.stack([rz(y) for y in yaw_t]), Rg)
    for r in range(3):
        good = np.flatnonzero((res * 1e3 <= P["VALID_RESID_MM"]) & (inl >= P["VALID_INLIER"]))
        mdl, st = build_model(clouds, frames, c, Rb, cam_W, P, sel=good if len(good) > 50 else None)
        mdl, c = recentre(mdl, c, Rb)
        c, res, inl, vis = register_all(clouds, c, Rb, mdl, P)
        rounds.append(dict(stage="final", round=r, model_vox=st["n_vox_kept"], bins=st["bins_used"],
                           resid_p50_mm=1e3 * float(np.nanmedian(res)), resid_p90_mm=1e3 * float(np.nanpercentile(res, 90)),
                           inlier_p50=float(np.nanmedian(inl)), visfrac_p50=float(np.nanmedian(vis))))
        log(f"  [{side}] final r{r}: model {st['n_vox_kept']:5d} vox ({st['bins_used']} view bins)  resid p50 "
            f"{rounds[-1]['resid_p50_mm']:.2f} p90 {rounds[-1]['resid_p90_mm']:.2f} mm  inlier p50 {rounds[-1]['inlier_p50']:.2f}"
            f"  visfrac p50 {rounds[-1]['visfrac_p50']:.2f}  [{time.time()-T0:.0f}s]")
    return dict(c=c, Rb=Rb, model=mdl, res=res, inl=inl, vis=vis, yaw_global=yaw0, yaw_t=yaw_t, yaw_sharpness=depth,
                yaw_flip_resid_mm=1e3 * s_flip, yaw_best_resid_mm=1e3 * s_best,
                yaw_grid=grid, yaw_grid_resid=sc, dyaw_raw=dev, rounds=rounds)


# ============================================================================================ temporal pass
def temporal_pass(t_s, p, valid0, flags, xbetter, P):
    """p[s]: (N,3) world, valid0[s]: bool after fitness. Mutates flags; returns final valid per side."""
    n = len(t_s)
    st = {s: dict(p=None, v=np.zeros(3), t=None, rej=0) for s in SIDES}
    events = dict(jump=0, swap=0, xcheck=0, reanchor=0)
    for i in range(n):
        cand = {s: valid0[s][i] for s in SIDES}
        # --- swap test (both candidates present and both tracks recent)
        if cand["left"] and cand["right"] and all(st[s]["p"] is not None and t_s[i] - st[s]["t"] < P["REANCHOR_S"] for s in SIDES):
            def cost(a, b):     # a's track takes b's point
                dt = max(t_s[i] - st[a]["t"], 1e-3)
                pred = st[a]["p"] + st[a]["v"] * dt
                vn = (p[b][i] - st[a]["p"]) / dt
                return float(np.linalg.norm(p[b][i] - pred) + P["LAMBDA_VEL_S"] * np.linalg.norm(vn - st[a]["v"]))
            straight = cost("left", "left") + cost("right", "right")
            swapped = cost("left", "right") + cost("right", "left")
            if swapped + P["SWAP_MARGIN_M"] < straight:
                for s in SIDES:
                    flags[s][i] |= F_SWAP
                events["swap"] += 1
                continue
        if cand["left"] and cand["right"] and xbetter[i]:
            for s in SIDES:
                flags[s][i] |= F_XCHECK
            events["xcheck"] += 1
        for s in SIDES:
            if not cand[s]:
                continue
            o = "right" if s == "left" else "left"
            S = st[s]
            if S["p"] is None or t_s[i] - S["t"] >= P["REANCHOR_S"]:
                if S["p"] is not None:
                    flags[s][i] |= F_REANCHOR; events["reanchor"] += 1
                S.update(p=p[s][i].copy(), v=np.zeros(3), t=t_s[i], rej=0)
                continue
            dt = t_s[i] - S["t"]
            if np.linalg.norm(p[s][i] - S["p"]) > P["V_MAX_MPS"] * dt + P["JUMP_SLACK_M"]:
                flags[s][i] |= F_JUMP; events["jump"] += 1
                if valid0[o][i]:
                    flags[o][i] |= F_PARTNER_JUMP
                continue
            vn = (p[s][i] - S["p"]) / max(dt, 1e-3)
            S.update(v=P["VEL_EMA"] * vn + (1 - P["VEL_EMA"]) * S["v"], p=p[s][i].copy(), t=t_s[i])
    final = {s: valid0[s] & ((flags[s] & INVALID_MASK) == 0) for s in SIDES}
    return final, events


# ============================================================================================ wrist clock
def wrist_validity(raw: RawEpisode, side: str, t_orb: np.ndarray, valid: np.ndarray, P: dict,
                   rigid: np.ndarray | None = None) -> np.ndarray:
    fm = raw.frames.get(f"{side}_wrist")
    dt_max = P["WRIST_MAX_DT_MS"] * 1e6
    tw = fm.capture_ns
    j = np.clip(np.searchsorted(t_orb, tw), 1, len(t_orb) - 1)
    a, b = j - 1, j
    near = np.where(np.abs(tw - t_orb[a]) <= np.abs(t_orb[b] - tw), a, b)
    bracket = (t_orb[a] <= tw) & (tw <= t_orb[b]) & (t_orb[b] - t_orb[a] <= dt_max)
    v_interp = bracket & valid[a] & valid[b]
    v_near = (np.abs(tw - t_orb[near]) <= dt_max / 2) & valid[near]
    v = np.where(bracket, v_interp, v_near)
    rg = valid if rigid is None else rigid
    vr = np.where(bracket, bracket & rg[a] & rg[b], (np.abs(tw - t_orb[near]) <= dt_max / 2) & rg[near])
    rec = np.zeros(len(tw), dtype=[("video_frame", "i8"), ("frame_index", "i8"), ("capture_ns", "i8"),
                                    ("orbbec_valid", "?"), ("orbbec_rigid_ok", "?"), ("orbbec_idx_near", "i8"),
                                    ("dt_near_ms", "f4")])
    rec["video_frame"] = fm.video_frame; rec["frame_index"] = fm.frame_index; rec["capture_ns"] = tw
    rec["orbbec_valid"] = v; rec["orbbec_rigid_ok"] = vr; rec["orbbec_idx_near"] = near; rec["dt_near_ms"] = (tw - t_orb[near]) / 1e6
    return rec


def interp_points(t_orb, p, valid, tq, max_dt_ns):
    """Linear interpolation of an Orbbec-rate track at query times; NaN unless both bracketing frames are valid."""
    j = np.clip(np.searchsorted(t_orb, tq), 1, len(t_orb) - 1)
    a, b = j - 1, j
    w = ((tq - t_orb[a]) / np.maximum(t_orb[b] - t_orb[a], 1))[:, None]
    ok = valid[a] & valid[b] & (t_orb[a] <= tq) & (tq <= t_orb[b]) & (t_orb[b] - t_orb[a] <= max_dt_ns)
    out = (1 - w) * p[a] + w * p[b]
    out[~ok] = np.nan
    return out


# ============================================================================================ main
def run_episode(ep_path: Path, P: dict = CANON_V1, debug_dir: Path | None = None, debug_frames: int = 4, log=print) -> dict:
    t_start = time.time()
    bdir = ep_path / "derived" / "body_track"
    z = dict(np.load(bdir / "track.npz"))
    if str(z.get("profile", "")) != "laundry_v1":
        raise SystemExit(f"{bdir/'track.npz'}: profile {z.get('profile')} -- run handumi_body_track.py --profile laundry_v1 --save-npz")
    t_ns = z["t_ns"]; n = len(t_ns); t_s = (t_ns - t_ns[0]) / 1e9
    nrm, pd = z["plane_normal"], float(z["plane_d"])
    R_WC, t_WC = table_frame(nrm, pd)
    cam_W = t_WC.copy()                                   # camera centre in the table world
    toW = lambda X: X @ R_WC.T + t_WC
    raw = RawEpisode.load(ep_path)
    log(f"\n{ep_path.parent.name}/{ep_path.name}: {n} Orbbec frames, {t_s[-1]:.1f} s")
    clouds = collect_clouds(ep_path, z, P, bdir / "canonical_clouds.npz", log=log)

    out = dict(t_ns=t_ns, R_WC=R_WC, t_WC=t_WC, plane_normal=nrm, plane_d=np.array(pd))
    summ = dict(episode=f"{ep_path.parent.name}/{ep_path.name}", n_frames=int(n), duration_s=float(t_s[-1]),
                params=P, sides={})
    p_all, v0, flags, fit = {}, {}, {}, {}
    side_res = {}
    for s in SIDES:
        fl = np.zeros(n, np.uint16)
        fl[~z[f"valid_{s}"]] |= F_DET_INVALID
        Rg, info = ahrs_at(raw, s, t_ns, P)
        fr = clouds[s]["frames"]; cl = [toW(c.astype(np.float64)) for c in clouds[s]["clouds"]]
        pW = np.full((n, 3), np.nan); RWB = np.full((n, 3, 3), np.nan)
        res = np.full(n, np.nan); inl = np.full(n, np.nan); vis = np.full(n, np.nan); npts = np.zeros(n, int)
        if Rg is None or len(fr) < 50:
            fl |= F_NO_IMU if Rg is None else 0
            log(f"  [{s}] no IMU or too few clouds ({len(fr)}) -- side skipped")
            p_all[s] = pW; v0[s] = np.zeros(n, bool); flags[s] = fl; fit[s] = (res, inl, vis, npts)
            continue
        fl[~info["ok"]] |= F_NO_IMU
        seeds = toW(z[f"xyz_{s}"][fr])
        R = track_side(s, cl, fr, seeds, Rg[fr], cam_W, t_s[fr], P, log=log)
        side_res[s] = dict(R, frames=fr, clouds=cl)
        pW[fr] = R["c"]; RWB[fr] = R["Rb"]
        res[fr] = R["res"]; inl[fr] = R["inl"]; vis[fr] = R["vis"]; npts[fr] = [len(c) for c in cl]
        have = np.zeros(n, bool); have[fr] = True
        fl[~have & z[f"valid_{s}"]] |= F_FEW_PTS
        bad = have & ~((res * 1e3 <= P["VALID_RESID_MM"]) & (inl >= P["VALID_INLIER"]) & (vis >= P["VALID_VISFRAC"]))
        fl[bad] |= F_FIT_BAD
        fl[have & ~bad & ((vis < P["RIGID_VISFRAC"]) | (inl < P["RIGID_INLIER"]))] |= F_LOW_VIS
        # full-length yaw / attitude arrays (attitude everywhere the IMU is, yaw interpolated over time)
        yaw_full = np.interp(t_s, t_s[fr], R["yaw_t"])
        Rb_full = np.einsum("nij,njk->nik", np.stack([rz(y) for y in yaw_full]), Rg)
        p_all[s] = pW; flags[s] = fl; fit[s] = (res, inl, vis, npts)
        v0[s] = have & ((fl & INVALID_MASK) == 0)
        out.update({f"R_world_body_{s}": Rb_full, f"R_gravity_imu_{s}": Rg, f"yaw_offset_deg_{s}": yaw_full,
                    f"model_body_{s}": R["model"], f"centroid_W_{s}": toW(z[f"xyz_{s}"])})
        summ["sides"][s] = dict(ahrs={k: v for k, v in info.items() if k != "ok"}, yaw_anchor_deg=R["yaw_global"],
                                yaw_sharpness=R["yaw_sharpness"], yaw_anchor_resid_mm=R["yaw_best_resid_mm"],
                                yaw_anchor_plus180_resid_mm=R["yaw_flip_resid_mm"],
                                yaw_drift_deg_p5_p95=[float(np.percentile(R["yaw_t"] - R["yaw_global"], q)) for q in (5, 95)],
                                rounds=R["rounds"], model_voxels=int(len(R["model"])),
                                model_extent_mm=(1e3 * (R["model"].max(0) - R["model"].min(0))).tolist(),
                                n_cloud_frames=int(len(fr)))

    # IMU cross-check on frames where both have clouds: does each cloud fit the OTHER unit better?
    xbetter = np.zeros(n, bool)
    if len(side_res) == 2:
        trees = {s: cKDTree(side_res[s]["model"]) for s in SIDES}
        pos = {s: {int(f): k for k, f in enumerate(side_res[s]["frames"])} for s in SIDES}
        for i in range(n):
            if not (v0["left"][i] and v0["right"][i]):
                continue
            own = cross = 0.0
            for s in SIDES:
                o = "right" if s == "left" else "left"
                k = pos[s][i]; X = side_res[s]["clouds"][k]
                own += fit[s][0][i]
                # this cloud registered against the other unit's model with the other unit's attitude
                _, e, _, _ = reg_translation((X - p_all[s][i]) @ out[f"R_world_body_{o}"][i], trees[o], P)
                cross += e
            xbetter[i] = cross < P["XCHECK_RATIO"] * own
    final, ev = temporal_pass(t_s, p_all, v0, flags, xbetter, P)

    for s in SIDES:
        res, inl, vis, npts = fit[s]
        out.update({f"p_W_{s}": p_all[s], f"p_C_{s}": (p_all[s] - t_WC) @ R_WC, f"valid_{s}": final[s],
                    f"valid_prepass_{s}": v0[s], f"flags_{s}": flags[s],
                    f"rigid_ok_{s}": final[s] & ((flags[s] & F_LOW_VIS) == 0), f"resid_mm_{s}": res * 1e3,
                    f"inlier_{s}": inl, f"visfrac_{s}": vis, f"npts_{s}": npts})
        warn = final[s] & ((flags[s] & WARN_MASK) != 0)
        if s in summ["sides"]:
            summ["sides"][s].update(coverage_det=float(z[f"valid_{s}"].mean()), coverage_prepass=float(v0[s].mean()),
                                    coverage_final=float(final[s].mean()), warn_share=float(warn.mean()),
                                    coverage_rigid_ok=float((final[s] & ((flags[s] & F_LOW_VIS) == 0)).mean()),
                                    resid_mm_p50=float(np.nanmedian(res[final[s]] * 1e3)) if final[s].any() else None,
                                    inlier_p50=float(np.nanmedian(inl[final[s]])) if final[s].any() else None,
                                    flag_counts={nm: int(((flags[s] & b) != 0).sum()) for b, nm in FLAG_NAMES.items()})
    summ["temporal_events"] = ev
    summ["both_valid_share"] = float((final["left"] & final["right"]).mean())
    summ["flag_bits"] = {nm: int(b) for b, nm in FLAG_NAMES.items()}

    # wrist-clock validity
    for s in SIDES:
        rec = wrist_validity(raw, s, t_ns, final[s], P, final[s] & ((flags[s] & F_LOW_VIS) == 0))
        np.save(bdir / f"orbbec_valid_wristclock_{s}.npy", rec)
        summ["sides"].setdefault(s, {})["wristclock_valid_share"] = float(rec["orbbec_valid"].mean())
        summ["sides"][s]["wristclock_frames"] = int(len(rec))

    out["params_json"] = np.array(json.dumps(P))
    np.savez(bdir / "canonical.npz", **out)
    summ["runtime_s"] = time.time() - t_start
    (bdir / "canonical_summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"  -> {bdir/'canonical.npz'}  ({summ['runtime_s']:.0f} s)")
    for s in SIDES:
        d = summ["sides"].get(s, {})
        log(f"  [{s}] coverage det {d.get('coverage_det', 0)*100:.1f}%  prepass {d.get('coverage_prepass', 0)*100:.1f}%  "
            f"final {d.get('coverage_final', 0)*100:.1f}%  warn {d.get('warn_share', 0)*100:.1f}%  flags {d.get('flag_counts')}")
    log(f"  events {ev}  both valid {summ['both_valid_share']*100:.1f}%")
    if debug_dir is not None and debug_frames:
        write_debug(ep_path, out, z, debug_dir, debug_frames)
    return summ


def write_debug(ep_path: Path, out: dict, z: dict, debug_dir: Path, k: int):
    debug_dir.mkdir(parents=True, exist_ok=True)
    ep = RgbdEpisode.load(ep_path)
    K = ep.intrinsics.K
    n = len(out["t_ns"])
    picks = list(np.linspace(0, n - 1, k + 2)[1:-1].astype(int))
    for s in SIDES:      # one flagged frame per side if any
        fl = out[f"flags_{s}"]
        bad = np.flatnonzero((fl & (F_JUMP | F_SWAP | F_XCHECK)) != 0)
        if len(bad):
            picks.append(int(bad[len(bad) // 2]))
    picks = sorted(set(picks))
    tag = f"{ep_path.parent.name}_{ep_path.name}"
    for f in ep.iter_frames(0, n):
        if f.index not in picks:
            continue
        vis = f.rgb.copy()
        for s, col in (("left", (0, 165, 255)), ("right", (255, 140, 0))):
            i = f.index
            if z[f"valid_{s}"][i]:
                u, v = bt._uv_of(z[f"xyz_{s}"][i], K)
                cv2.drawMarker(vis, (int(u), int(v)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            pc = out[f"p_C_{s}"][i]
            if np.all(np.isfinite(pc)):
                u, v = bt._uv_of(pc, K)
                ok = bool(out[f"valid_{s}"][i])
                cv2.circle(vis, (int(u), int(v)), 10, col if ok else (0, 0, 255), 2)
                # body axes (IMU frame) 5 cm
                Rb = out[f"R_world_body_{s}"][i]
                for ax, ac in zip(range(3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    tip = out["R_WC"].T @ (out[f"p_W_{s}"][i] + 0.05 * Rb[:, ax] - out["t_WC"])
                    ut, vt = bt._uv_of(tip, K)
                    cv2.line(vis, (int(u), int(v)), (int(ut), int(vt)), ac, 2)
                # model projected
                mW = out[f"p_W_{s}"][i] + out[f"model_body_{s}"] @ Rb.T
                mC = (mW - out["t_WC"]) @ out["R_WC"]
                uv = np.stack([K[0, 0] * mC[:, 0] / mC[:, 2] + K[0, 2], K[1, 1] * mC[:, 1] / mC[:, 2] + K[1, 2]], 1)
                for q in uv[::3]:
                    if 0 <= q[0] < vis.shape[1] and 0 <= q[1] < vis.shape[0]:
                        vis[int(q[1]), int(q[0])] = col
                fl = int(out[f"flags_{s}"][i])
                names = ",".join(nm for b, nm in FLAG_NAMES.items() if fl & b)
                cv2.putText(vis, f"{s} {'OK' if ok else 'INV'} r{out[f'resid_mm_{s}'][i]:.1f}mm {names}",
                            (int(u) + 12, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
        cv2.putText(vis, f"{tag} f{f.index} canonical(circle+axes+model dots) vs detector centroid(x)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, f"{tag} f{f.index} canonical(circle+axes+model dots) vs detector centroid(x)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(debug_dir / f"{tag}_f{f.index:05d}_canonical.jpg"), vis)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True, nargs="+")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--debug-frames", type=int, default=4)
    a = ap.parse_args()
    for ep in a.episode:
        run_episode(ep, CANON_V1, a.debug_dir, a.debug_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
