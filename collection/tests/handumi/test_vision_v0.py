"""IMU-less V0: where metric scale comes from when there is no IMU.

A monocular wrist VO gives trajectory SHAPE but no metres, so its translations cannot be an action label. The head
Orbbec already measures metric 3D, so any frame where both cameras see the same surface yields a metric pose anchor:

    head RGB + depth  -> 3D points  ---RANSAC PnP--->  wrist 2D pixels  =>  T_head_wrist, metric

These tests pin the two pieces that make that work — the anchor itself, and the sim(3) that carries a scale-free
trajectory onto sparse anchors — against ground truth, on a synthetic textured plane rendered into both cameras."""
import numpy as np
import cv2
import pytest
from scipy.spatial.transform import Rotation

from handumi_collector.pose.cross_view import CrossViewAnchor, anchor_from_pair
from handumi_collector.pose.metric_align import align_to_anchors, scale_consistency, umeyama_sim3
from handumi_collector.pose.rgbd_io import CameraIntrinsics
from handumi_collector.pose.se3 import inv_T, make_T

K_HEAD = np.array([[461.2, 0, 423.5], [0, 461.4, 239.5], [0, 0, 1.0]])
K_WRIST = np.array([[430.0, 0, 639.5], [0, 430.0, 359.5], [0, 0, 1.0]])


def _texture(n=900, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((n, n, 3), 240, np.uint8)
    for _ in range(600):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p = tuple(int(x) for x in rng.integers(0, n, 2))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(4, 26)), c, -1)
        else:
            cv2.rectangle(img, p, tuple(int(x) for x in np.array(p) + rng.integers(8, 60, 2)), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _render(tex, K, T_wc, size, plane_m=1.0):
    """The textured plane z=0 seen from a camera at T_wc -> (BGR image, metric depth). A plane makes the geometry exact,
    so any error in the test is the code under test, not the renderer."""
    w, h = size
    T_cw = inv_T(T_wc)
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    S = np.array([[plane_m / tex.shape[1], 0, -plane_m / 2], [0, plane_m / tex.shape[0], -plane_m / 2], [0, 0, 1]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    img = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    denom = d @ R[:, 2]
    lam = np.where(np.abs(denom) > 1e-9, (t @ R[:, 2]) / np.where(np.abs(denom) > 1e-9, denom, 1), 0.0)
    depth = np.where(lam > 0, lam * d[..., 2], 0.0).astype(np.float32)
    depth[img.sum(-1) == 0] = 0.0
    return img, depth


@pytest.fixture(scope="module")
def scene():
    tex = _texture()
    T_wh = make_T(Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix(), [0.0, 0.0, 0.80])
    T_ww = make_T(Rotation.from_euler("xyz", [165, 12, -20], degrees=True).as_matrix(), [0.14, -0.09, 0.55])
    head_rgb, head_depth = _render(tex, K_HEAD, T_wh, (848, 480))
    wrist_rgb, _ = _render(tex, K_WRIST, T_ww, (1280, 720))
    return dict(head_rgb=head_rgb, head_depth=head_depth, wrist_rgb=wrist_rgb, T_head_wrist=inv_T(T_wh) @ T_ww)


# ------------------------------------------------------------------------------------ cross-view metric anchor
def test_cross_view_anchor_recovers_a_known_metric_pose(scene):
    intr = CameraIntrinsics(K_HEAD[0, 0], K_HEAD[1, 1], K_HEAD[0, 2], K_HEAD[1, 2], 848, 480)
    a = anchor_from_pair(scene["head_rgb"], scene["head_depth"], intr, scene["wrist_rgb"], K_WRIST, t_ns=0, side="left")
    assert a.valid, a.reason
    assert a.n_inliers >= 100 and a.inlier_ratio > 0.5 and a.reprojection_error_px < 2.0
    E = inv_T(a.T_head_wrist) @ scene["T_head_wrist"]
    assert np.linalg.norm(E[:3, 3]) * 1e3 < 5.0, f"{np.linalg.norm(E[:3,3])*1e3:.2f} mm"
    assert np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude()) < 0.5


def test_cross_view_records_why_there_is_no_anchor():
    """A failed anchor is data: the availability ratio and the REASON are headline V0 metrics, so nothing may be
    silently dropped."""
    intr = CameraIntrinsics(K_HEAD[0, 0], K_HEAD[1, 1], K_HEAD[0, 2], K_HEAD[1, 2], 64, 48)
    blank = np.full((48, 64, 3), 128, np.uint8)
    a = anchor_from_pair(blank, np.full((48, 64), 0.5, np.float32), intr, blank.copy(), K_WRIST, t_ns=7, side="right")
    assert not a.valid and a.T_head_wrist is None and a.reason
    assert a.to_row()["t_ns"] == 7 and np.isnan(a.to_row()["x"])


def test_cross_view_rejects_when_depth_is_missing(scene):
    """Matches without depth are useless: no depth, no metric anchor — and it must say so rather than fall back to an
    arbitrary scale."""
    intr = CameraIntrinsics(K_HEAD[0, 0], K_HEAD[1, 1], K_HEAD[0, 2], K_HEAD[1, 2], 848, 480)
    a = anchor_from_pair(scene["head_rgb"], np.zeros_like(scene["head_depth"]), intr, scene["wrist_rgb"], K_WRIST,
                         t_ns=0, side="left")
    assert not a.valid and "depth" in a.reason and a.n_matches > 0


# ------------------------------------------------------------------------------------ sim(3) alignment
def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(50, 3))
    s_true, R_true, t_true = 0.37, Rotation.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix(), np.array([0.2, -0.1, 0.6])
    s, R, t, rmse = umeyama_sim3(S, (s_true * (R_true @ S.T).T + t_true))
    assert abs(s - s_true) < 1e-9 and rmse < 1e-9
    assert np.degrees(Rotation.from_matrix(R.T @ R_true).magnitude()) < 1e-9


def _metric_traj(n=120, fps=30.0):
    t = (np.arange(n) * int(1e9 / fps)).astype(np.int64)
    Ts = np.tile(np.eye(4), (n, 1, 1))
    u = np.linspace(0, 1, n)
    Ts[:, :3, 3] = np.stack([0.25 * np.sin(2 * np.pi * u), 0.15 * u, 0.6 + 0.1 * np.cos(2 * np.pi * u)], 1)
    for i, ui in enumerate(u):
        Ts[i, :3, :3] = Rotation.from_euler("xyz", [20 * ui, -15 * ui, 35 * ui], degrees=True).as_matrix()
    return t, Ts


def test_alignment_puts_a_scale_free_trajectory_into_metres():
    """The VO is in its own arbitrary units and its own world. Sparse anchors must be enough to recover both."""
    t, Ts_metric = _metric_traj()
    s_true, R_true, t_true = 0.42, Rotation.from_euler("xyz", [8, 22, -13], degrees=True).as_matrix(), np.array([-0.3, 0.2, 1.1])
    Ts_vo = np.tile(np.eye(4), (len(t), 1, 1))                       # metric -> VO units (the inverse of what we fit)
    Ts_vo[:, :3, 3] = (R_true.T @ (Ts_metric[:, :3, 3] - t_true).T).T / s_true
    Ts_vo[:, :3, :3] = R_true.T @ Ts_metric[:, :3, :3]
    anchors = [CrossViewAnchor(t_ns=int(t[i]), side="left", T_head_wrist=Ts_metric[i], valid=True) for i in (5, 30, 60, 95, 115)]
    r = align_to_anchors(t, Ts_vo, np.ones(len(t), bool), anchors)
    assert r.ok, r.reason
    assert abs(r.scale - s_true) / s_true < 1e-6
    assert r.position_rmse_mm < 0.01 and r.rotation_rmse_deg < 1e-3
    back = r.apply(Ts_vo)
    assert np.allclose(back, Ts_metric, atol=1e-9)                   # the whole trajectory, not only the anchors


def test_alignment_refuses_when_scale_is_not_observable():
    """Anchors bunched into a few millimetres cannot determine a scale. Refusing beats returning a confident number."""
    t, Ts_metric = _metric_traj()
    anchors = [CrossViewAnchor(t_ns=int(t[i]), side="left", T_head_wrist=Ts_metric[i], valid=True) for i in (40, 41, 42, 43)]
    r = align_to_anchors(t, Ts_metric, np.ones(len(t), bool), anchors)
    assert not r.ok and "scale is not observable" in r.reason


def test_alignment_needs_enough_anchors_and_ignores_invalid_vo():
    t, Ts_metric = _metric_traj()
    anchors = [CrossViewAnchor(t_ns=int(t[i]), side="left", T_head_wrist=Ts_metric[i], valid=True) for i in (5, 60, 115)]
    valid = np.ones(len(t), bool); valid[[5, 60]] = False            # VO lost exactly where the anchors are
    r = align_to_anchors(t, Ts_metric, valid, anchors)
    assert not r.ok and "line up with a valid VO sample" in r.reason
    assert not align_to_anchors(t, Ts_metric, np.ones(len(t), bool), anchors[:2]).ok


def test_scale_consistency_exposes_a_drifting_scale():
    """One global sim(3) assumes the VO scale is constant. This is the measurement that tests the assumption instead
    of letting a drift hide inside an averaged number."""
    t, Ts_metric = _metric_traj()
    idx = np.array([5, 30, 60, 95, 115])
    Ts_const = Ts_metric.copy(); Ts_const[:, :3, 3] /= 0.42
    good = scale_consistency(t, Ts_const, idx, Ts_metric[idx][:, :3, 3])
    assert good["n_pairs"] >= 3 and good["spread_pct"] < 1e-6
    Ts_drift = Ts_metric.copy()
    Ts_drift[:, :3, 3] /= (0.42 * np.linspace(1.0, 1.6, len(t)))[:, None]
    bad = scale_consistency(t, Ts_drift, idx, Ts_metric[idx][:, :3, 3])
    assert bad["spread_pct"] > 10.0, bad


# ------------------------------------------------------------------------------------ head RGB-D temporal odometry
def _head_traj(n=24):
    """A gentle head motion over the plane: small enough per frame that KLT tracks, large enough to accumulate."""
    Ts = []
    for i in range(n):
        u = i / (n - 1)
        R = Rotation.from_euler("xyz", [180 + 4 * u, 6 * u, -5 * u], degrees=True).as_matrix()
        Ts.append(make_T(R, [0.06 * u, -0.04 * u, 0.80 + 0.05 * u]))
    return Ts


def test_head_odometry_recovers_a_known_metric_trajectory():
    """Depth makes this metric with no scale to estimate: the accumulated chain must match ground truth in millimetres."""
    from handumi_collector.pose.head_odometry import HeadRgbdOdometry
    tex = _texture(seed=3)
    intr = CameraIntrinsics(K_HEAD[0, 0], K_HEAD[1, 1], K_HEAD[0, 2], K_HEAD[1, 2], 848, 480)
    gt = _head_traj()
    odo = HeadRgbdOdometry(intr)
    errs = []
    for i, T_wc in enumerate(gt):
        rgb, depth = _render(tex, K_HEAD, T_wc, (848, 480))
        e = odo.push(rgb, depth, t_ns=i * 33_333_333, frame_index=i)
        assert e.valid, f"frame {i}: {e.reason}"
        assert not e.chain_broken
        E = inv_T(e.T_world_head) @ (inv_T(gt[0]) @ T_wc)        # odometry world = the first frame
        errs.append((np.linalg.norm(E[:3, 3]) * 1e3, np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude())))
    tr, rot = np.array(errs).T
    assert odo.summary()["valid_ratio"] == 1.0 and odo.summary()["broken_links"] == 0
    assert tr[-1] < 8.0, f"accumulated translation error {tr[-1]:.2f} mm"
    assert rot[-1] < 1.0, f"accumulated rotation error {rot[-1]:.3f} deg"


def test_head_odometry_breaks_the_chain_instead_of_bridging_a_gap():
    """A frame with no solvable relative pose must yield NO pose, and the next solved frame must be flagged: motion
    across the gap is unmeasured, and silently bridging it would corrupt the very drift number V0-B measures."""
    from handumi_collector.pose.head_odometry import HeadRgbdOdometry
    tex = _texture(seed=4)
    intr = CameraIntrinsics(K_HEAD[0, 0], K_HEAD[1, 1], K_HEAD[0, 2], K_HEAD[1, 2], 848, 480)
    gt = _head_traj(12)
    odo = HeadRgbdOdometry(intr)
    blank = np.full((480, 848, 3), 120, np.uint8)
    states = []
    for i, T_wc in enumerate(gt):
        if i == 5:                                               # a featureless frame in the middle
            states.append(odo.push(blank, np.full((480, 848), 0.8, np.float32), t_ns=i, frame_index=i))
            continue
        rgb, depth = _render(tex, K_HEAD, T_wc, (848, 480))
        states.append(odo.push(rgb, depth, t_ns=i, frame_index=i))
    assert not states[5].valid and states[5].T_world_head is None and states[5].reason
    assert np.isnan(states[5].to_row()["x"])
    after = next(s for s in states[6:] if s.valid)
    assert after.chain_broken and odo.summary()["broken_links"] == 1
    assert "UNMEASURED" in odo.summary()["note"]


# --------------------------------------------------------------------------------------------- end-to-end orchestration
def _scene(tex, tex2, K, T_wc, size, T_patch):
    """Ground plane plus a raised finite patch, composited by depth.

    One plane is NOT enough here: monocular VO is degenerate on a purely planar scene (the essential matrix is
    ambiguous when all points lie on a plane), so a single-plane fixture would exercise everything EXCEPT the wrist
    VO and quietly report nonsense residuals for it. The patch gives real parallax structure."""
    from handumi_collector.pose.se3 import inv_T
    img1, d1 = _render(tex, K, T_wc, size, plane_m=1.2)
    img2, d2 = _render(tex2, K, inv_T(T_patch) @ T_wc, size, plane_m=0.35)
    near = (d2 > 0) & ((d1 <= 0) | (d2 < d1))
    img = np.where(near[..., None], img2, img1)
    depth = np.where(near, d2, d1).astype(np.float32)
    return img.astype(np.uint8), depth


def _write_episode(root, tex, K, n=90, fps=30.0, still=25):
    """A recorder-layout episode both cameras can actually be run on: head RGB-D and one wrist, watching the same
    textured plane, still at both ends with operator HOME marks. Small, but it is the wiring that is under test here —
    the estimators themselves are covered above."""
    import json
    import cv2
    import numpy as np
    from handumi_collector.collector.mcap_writer import SensorMcapWriter
    from handumi_collector.collector.video_writer import VideoStreamWriter
    from handumi_collector.pose.se3 import make_T
    from scipy.spatial.transform import Rotation

    size = (640, 480)
    tex2 = _texture(n=400, seed=77)
    T_patch = make_T(np.eye(3), [0.16, -0.11, 0.20])      # a block standing proud of the ground plane
    root.mkdir(parents=True, exist_ok=True)
    t0 = 10_000_000_000
    streams = ["head_depth", "left_wrist"]
    mcap = SensorMcapWriter(root / "sensors.mcap", streams=streams)
    vids = {s: VideoStreamWriter(root / f"{s}.mp4", width=size[0], height=size[1], fps=int(fps), codec="libx264",
                                 bitrate_kbps=4000) for s in streams}
    ddir = root / "head_depth_depth"; ddir.mkdir(exist_ok=True)
    (ddir / "intrinsics.json").write_text(json.dumps(dict(
        intrinsics=dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=size[0], height=size[1]),
        depth_scale=1.0)))

    def at(i):
        """Still -> smooth motion -> still, so the HOME windows are genuinely static."""
        if i < still:
            u = 0.0
        elif i >= n - still:
            u = 1.0
        else:
            u = (i - still) / max(1, n - 2 * still - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * u)       # ease in/out; returns NEAR but not exactly to the start
        return s

    for i in range(n):
        t = t0 + int(i * 1e9 / fps)
        s = at(i)
        # cameras look DOWN at the z=0 plane: the 180 deg x-flip is what makes the optical axis face it (see _render)
        R_h = Rotation.from_euler("xyz", [180 + 3 * s, 4 * s, -3 * s], degrees=True).as_matrix()
        R_w = Rotation.from_euler("xyz", [170 + 4 * s, 10 + 5 * s, -14 * s], degrees=True).as_matrix()
        # A LOOP, not a straight line: anchors strung along one axis leave the sim(3) rotation about that axis
        # unobservable, which would make the rotation residual meaningless (metric_align reports that geometry).
        a = 2 * np.pi * s
        T_head = make_T(R_h, [0.05 * s, 0.03 * s, 0.80])
        T_wrist = make_T(R_w, [-0.10 + 0.06 * np.sin(a), 0.05 + 0.05 * (1 - np.cos(a)), 0.55 + 0.04 * np.sin(2 * a)])
        rgb_h, depth_h = _scene(tex, tex2, K, T_head, size, T_patch)
        rgb_w, _ = _scene(tex, tex2, K, T_wrist, size, T_patch)
        vf = vids["head_depth"].put(rgb_h); mcap.frame_meta("head_depth", i, vf, t, 0)
        cv2.imwrite(str(ddir / f"{vf:06d}.png"), (depth_h * 1000.0).astype(np.uint16))
        vf = vids["left_wrist"].put(rgb_w); mcap.frame_meta("left_wrist", i, vf, t, 0)

    events = []
    for kind, tt in (("home_leave", t0 + int(still * 1e9 / fps)), ("home_return", t0 + int((n - still) * 1e9 / fps))):
        events.append(dict(t_ns=tt, t_rel_s=(tt - t0) / 1e9, kind=kind, device="operator", detail={}))
        mcap.event(tt, kind, "operator", {})
    for v in vids.values():
        v.close()
    mcap.close()
    (root / "events.json").write_text(json.dumps(events, indent=1))
    (root / "episode_meta.json").write_text(json.dumps(dict(
        schema="handumi_episode_meta/v1", episode_dir=root.name, order="TEST", status="KEEP", quality="PASS",
        t_start_monotonic_ns=t0, t_stop_monotonic_ns=t0 + int(n * 1e9 / fps), duration_s=n / fps,
        streams={s: dict(frames=n) for s in streams}, n_events=len(events),
        event_kinds=sorted({e["kind"] for e in events}))))
    return root


def _write_pinhole_cal(cal_dir, K, size=(640, 480), side="left"):
    import yaml
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / f"fisheye_{side}_v001.yaml").write_text(yaml.safe_dump(dict(
        schema="handumi_fisheye_intrinsics/v1", model="pinhole", side=side,
        K=[[float(x) for x in r] for r in K], D=[0.0, 0.0, 0.0, 0.0], image_size=list(size),
        physical_side_verified=True, source="synthetic test camera")))
    return cal_dir


def test_analyze_vpilot_runs_the_whole_v0_chain_on_an_episode(tmp_path):
    """The orchestration itself: episode -> head odometry -> cross-view -> wrist VO -> sim(3) -> HOME, with the frame
    joins and the head-frame-to-world composition that only exist in the runner."""
    import numpy as np
    from handumi_collector.tools.analyze_vpilot import analyze, format_report

    K = np.array([[420.0, 0, 319.5], [0, 420.0, 239.5], [0, 0, 1.0]])
    ep = _write_episode(tmp_path / "episode_000000", _texture(seed=11), K)
    cal = _write_pinhole_cal(tmp_path / "cal", K)

    out, anchors, (al, t_vo, Ts_vo, valid_vo) = analyze(ep, "left", stride=5, cal_dir=cal, progress=lambda *_: None)

    # the three headline numbers must exist and be self-consistent, whatever their values
    cv = out["cross_view"]
    assert cv["attempted_frames"] > 0, "no synchronized head/wrist pair was even attempted"
    assert 0.0 <= cv["A_crossview"] <= 1.0
    assert cv["valid_anchors"] == sum(a.valid for a in anchors)
    assert cv["A_crossview"] == pytest.approx(cv["valid_anchors"] / cv["attempted_frames"])
    assert cv["placed_in_world"] <= cv["valid_anchors"], "an anchor cannot be placed in world more often than it is valid"

    for chain in ("head", "wrist_vo"):
        c = out[chain]
        assert c["n_valid"] <= c["n"]
        assert c["longest_valid_run"] <= c["n"]
        assert c["broken_links"] >= 0

    assert out["home"]["marks_found"] >= 2, "the operator HOME marks were not read back"
    assert format_report(out)          # the report renders whatever happened, including total failure


def test_analyze_vpilot_reports_missing_home_marks_instead_of_inventing_them(tmp_path):
    import numpy as np
    from handumi_collector.tools.analyze_vpilot import analyze

    K = np.array([[420.0, 0, 319.5], [0, 420.0, 239.5], [0, 0, 1.0]])
    ep = _write_episode(tmp_path / "episode_000001", _texture(seed=12), K)
    (ep / "events.json").write_text("[]")
    cal = _write_pinhole_cal(tmp_path / "cal", K)

    out, _, _ = analyze(ep, "left", stride=8, cal_dir=cal, progress=lambda *_: None)
    assert out["home"].get("translation_mm") is None
    assert "reason" in out["home"] and "HOME" in out["home"]["reason"]
