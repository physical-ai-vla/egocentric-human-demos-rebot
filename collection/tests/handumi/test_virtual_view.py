"""C922-like virtual wrist view: exact K_target projection, exact pure-rotation correction, versioned target, tool on a synthetic episode."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from handumi_collector.pose.calibration import save_versioned
from handumi_collector.pose.virtual_view import VirtualPinholeView, c922_like_target, load_virtual_wrist, write_virtual_wrist


def _dot_image(K, size, pts3d):
    import cv2
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for X in pts3d:
        u = K @ X; u = u[:2] / u[2]; cv2.circle(img, (int(round(u[0])), int(round(u[1]))), 3, (255, 255, 255), -1)
    return img


def _centroid(img):
    ys, xs = np.nonzero(img[..., 0] > 128); return np.array([xs.mean(), ys.mean()])


def test_virtual_view_matches_target_intrinsics_and_rotation():
    K_src = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]]); src = dict(model="pinhole", K=K_src, D=np.zeros(5), image_size=(640, 480))
    K_t = c922_like_target(); assert K_t.shape == (3, 3) and 650 < K_t[0, 0] < 760 and abs(K_t[1, 2] - 480 * 440.7 / 1080) < 1
    X = np.array([0.15, -0.05, 1.0])
    view = VirtualPinholeView(src, K_t, (640, 480))
    out = view.render(_dot_image(K_src, (640, 480), [X]))
    exp = (K_t @ X)[:2] / X[2]; assert np.linalg.norm(_centroid(out) - exp) < 1.0, (_centroid(out), exp)
    # pure rotation of the virtual camera: the point must land where the rotated camera would see it
    rpy = (5.0, -8.0, 3.0); R = Rotation.from_euler("xyz", np.radians(rpy)).as_matrix()
    view_r = VirtualPinholeView(src, K_t, (640, 480), rpy_deg=rpy); out_r = view_r.render(_dot_image(K_src, (640, 480), [X]))
    Xv = R.T @ X; exp_r = (K_t @ Xv)[:2] / Xv[2]
    assert np.linalg.norm(_centroid(out_r) - exp_r) < 1.5, (_centroid(out_r), exp_r)
    assert 45 < view.hfov_deg() < 50


def test_virtual_wrist_versioned_target(tmp_path):
    v, d = load_virtual_wrist(cal_dir=tmp_path); assert v is None and d["size"] == (640, 480) and "default" in d["source"] and set(d["K_target"]) == {"left", "right"}
    write_virtual_wrist(np.eye(3) * 700 + np.array([[0, 0, 320], [0, 0, 240], [0, 0, -699]]), (640, 480), rpy_deg={"left": [0, 10, 0], "right": [0, -10, 0]}, source="test", cal_dir=tmp_path)
    v, d = load_virtual_wrist(cal_dir=tmp_path); assert v == "virtual_wrist_v001" and d["rpy_deg"]["right"] == [0, -10, 0] and d["K_target"]["left"][0, 2] == 320 and d["D_target"]["left"] is None
    # per-side K/D (what tools.calibrate_c922 writes) round-trips
    write_virtual_wrist({"left": np.eye(3) * 700, "right": np.eye(3) * 710}, (640, 480), D_target={"left": [0.1, -0.2, 0, 0, 0], "right": None}, cal_dir=tmp_path)
    v, d = load_virtual_wrist(cal_dir=tmp_path); assert v == "virtual_wrist_v002" and d["K_target"]["right"][0, 0] == 710 and d["D_target"]["left"][1] == -0.2 and d["D_target"]["right"] is None
    # the repo default file loads too
    v0, d0 = load_virtual_wrist(); assert v0 is not None and d0["size"] == (640, 480) and abs(d0["K_target"]["left"][1, 1] - 705) < 1


def test_render_tool_on_synthetic_episode(tmp_path):
    from handumi_collector.tools.render_virtual_wrist import render_episode
    from handumi_collector.tools.synth_episode import synth_episode
    ep = tmp_path / "episode_000001"; gt = synth_episode(ep, seconds=1.0, sides=("left",))
    cal = tmp_path / "cal"; save_versioned("fisheye_left", dict(model="pinhole", K=gt["intrinsics"]["K"], D=gt["intrinsics"]["D"], image_size=gt["intrinsics"]["image_size"]), cal_dir=cal)
    m = render_episode(ep, sides=("left", "right"), cal_dir=cal, codec="libx264", preview=tmp_path / "prev.png", mount_revision="mount_v2")
    assert m["sides"]["left"]["frames"] == 30 and "error" in m["sides"]["right"] and (ep / "derived" / "virtual_wrist" / "left_wrist_c922like.mp4").exists()
    assert m["mount_revision"] == "mount_v2" and m["sides"]["left"]["source_fisheye_intrinsics"] == "fisheye_left_v001" and m["target_virtual_wrist"] == "DEFAULT_ESTIMATE (no virtual_wrist_vNNN.yaml)"
    assert (ep / "left_wrist.mp4").exists()          # raw fisheye untouched
    import av
    with av.open(str(ep / "derived" / "virtual_wrist" / "left_wrist_c922like.mp4")) as c:
        fr = next(c.decode(video=0)); assert (fr.width, fr.height) == (640, 480)
    assert (tmp_path / "prev.png").exists()


def test_target_distortion_and_rotation_fit():
    from handumi_collector.pose.virtual_view import compose_rpy, fit_rotation
    K_src = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]]); src = dict(model="pinhole", K=K_src, D=np.zeros(5), image_size=(640, 480))
    K_t = c922_like_target(); D_t = np.array([0.05, -0.1, 0.001, -0.001, 0.0])
    X = np.array([0.12, 0.08, 1.0])
    view = VirtualPinholeView(src, K_t, (640, 480), D_target=D_t); out = view.render(_dot_image(K_src, (640, 480), [X]))
    import cv2
    exp, _ = cv2.projectPoints(X.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K_t, D_t)       # where a real C922 (with D_t) would see it
    assert np.linalg.norm(_centroid(out) - exp.reshape(2)) < 1.0
    # rotation fit: robot camera = virtual camera rotated by rpy_true; recover it from ≥3 landmark correspondences
    rpy_true = np.array([4.0, -6.0, 2.5]); R = Rotation.from_euler("xyz", np.radians(rpy_true)).as_matrix()
    pts = np.array([[0.1, 0.05, 1.0], [-0.15, 0.1, 1.2], [0.05, -0.12, 0.9], [-0.05, -0.05, 1.5]])
    p_v = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_t, D_t)[0].reshape(-1, 2)
    p_r = cv2.projectPoints((R.T @ pts.T).T.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_t, D_t)[0].reshape(-1, 2)
    f = fit_rotation(p_v, p_r, K_t, D_t)
    assert f["E_view_before_px"] > 20 and f["E_view_after_px"] < 0.5 and np.allclose(f["rpy_delta_deg"], -rpy_true, atol=0.05) or np.allclose(
        Rotation.from_euler("xyz", np.radians(f["rpy_delta_deg"])).as_matrix(), R.T, atol=1e-3)
    assert np.allclose(compose_rpy([0, 0, 0], f["rpy_delta_deg"]), f["rpy_delta_deg"])
    # translation (parallax) cannot be absorbed: shifting the camera 5 cm leaves a residual after the fit
    p_r2 = cv2.projectPoints((pts + np.array([0.05, 0, 0])).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_t, D_t)[0].reshape(-1, 2)
    assert fit_rotation(p_v, p_r2, K_t, D_t)["E_view_after_px"] > 2.0


def _scene(K, size, cubes, plate_center=(320, 300), plate_wh=(260, 120)):
    import cv2
    img = np.full((size[1], size[0], 3), (60, 60, 60), np.uint8)
    cv2.rectangle(img, (plate_center[0] - plate_wh[0] // 2, plate_center[1] - plate_wh[1] // 2), (plate_center[0] + plate_wh[0] // 2, plate_center[1] + plate_wh[1] // 2), (235, 235, 235), -1)
    for (u, v, s), col in cubes: cv2.rectangle(img, (u - s, v - s), (u + s, v + s), col, -1)
    return img


def test_view_match_qa_on_synthetic_scene(tmp_path):
    import cv2
    from handumi_collector.pose.calibration import save_versioned
    from handumi_collector.tools.view_match_qa import compare, detect_landmarks, load_spec, run
    spec = load_spec()
    red, blue, purple = (40, 40, 220), (220, 90, 40), (200, 60, 170)
    cubes = [((250, 330, 18), red), ((320, 330, 18), blue), ((390, 330, 18), purple)]
    robot = _scene(None, (640, 480), cubes)
    lm = detect_landmarks(robot, spec)
    assert {"red_cube", "blue_cube", "purple_cube", "plate"} <= set(lm), lm.keys()
    assert abs(lm["red_cube"]["center"][0] - 250) < 1.5 and 34 <= lm["red_cube"]["w"] <= 38 and lm["plate"]["depth"] == "far"
    shifted = _scene(None, (640, 480), [((u + 6, v + 8, s), c) for (u, v, s), c in cubes], plate_center=(326, 308))
    c = compare(lm, detect_landmarks(shifted, spec), (640, 480), spec)
    assert c["counts"]["all"] == 4 and abs(c["E_view_all_px"] - 10.0) < 0.6
    assert c["counts"]["near"] == 3 and c["counts"]["far"] == 1 and 0.9 < c["scale_ratio_mean"] < 1.1
    # end-to-end: "human raw" through a pinhole source whose K equals the target -> virtual == robot
    cal = tmp_path / "cal"; K_t = c922_like_target()
    save_versioned("fisheye_left", dict(model="pinhole", K=K_t.tolist(), D=[0, 0, 0, 0], image_size=[640, 480]), cal_dir=cal)
    write_virtual_wrist(K_t, (640, 480), cal_dir=cal)
    r = run(robot, robot.copy(), "left", out=tmp_path / "qa", cal_dir=cal, spec=spec, fit_rpy=True, mount_revision="mount_v2", scene="layout_01")
    assert r["verdict"] == "PASS" and r["E_view_all_px"] < 1.0 and r["counts"]["all"] == 4 and not r["flags"]
    assert abs(r["rotation_fit"]["rpy_delta_deg"][0]) < 0.2 and (tmp_path / "qa" / "side_by_side.png").exists()
    pv = r["provenance"]
    assert pv["source_fisheye_intrinsics"] == "fisheye_left_v001" and pv["target_virtual_wrist"] == "virtual_wrist_v001"
    assert pv["mount_revision"] == "mount_v2" and pv["n_correspondences"] == 4 and pv["virtual_size"] == [640, 480] and pv["target_distortion"] is None
    assert r["scene"] == "layout_01" and r["per_landmark"]["plate"]["depth"] == "far"
    # compact report block (quoted in write-ups) + its file
    rep = r["report"]
    for line in ("E_view mean", "E_view median", "E_view p95", "near/far", "scale ratio", "fit RPY", "flags             NONE",
                 "gate: mean < 8.0 px", "median/p95 diagnostic only"):
        assert line in rep, (line, rep)
    g = r["gate"]; assert g["statistic"] == "mean" and g["metric"] == "E_view_all_px" and g["diagnostic_only"] == ["E_view_median_px", "E_view_p95_px"]
    assert r["E_view_median_px"] is not None and r["E_view_p95_px"] >= r["E_view_median_px"]
    assert (tmp_path / "qa" / "report.txt").read_text().startswith("side left")


def _two_camera_points(K_t, D_t, *, R=np.eye(3), t=np.zeros(3), near_z=0.3, far_z=1.5):
    """Landmark pixels seen by the virtual camera and by a robot camera offset by (R, t): near + far groups."""
    import cv2
    near = np.array([[0.02, 0.01, near_z], [-0.03, 0.02, near_z], [0.01, -0.02, near_z], [0.03, 0.03, near_z * 1.05]])
    far = np.array([[0.10, 0.05, far_z], [-0.12, 0.06, far_z], [0.05, -0.08, far_z * 1.1]])
    pts = np.vstack([near, far]); depth = ["near"] * len(near) + ["far"] * len(far)
    p_v = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_t, D_t)[0].reshape(-1, 2)
    p_r = cv2.projectPoints((R.T @ (pts - t).T).T.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_t, D_t)[0].reshape(-1, 2)
    return dict(names=[f"p{i}" for i in range(len(pts))], robot=p_r.tolist(), virtual=p_v.tolist(), depth=depth)


def test_parallax_shows_up_as_near_much_worse_than_far(tmp_path):
    from handumi_collector.pose.calibration import save_versioned
    from handumi_collector.tools.view_match_qa import load_spec, run
    spec = load_spec(); cal = tmp_path / "cal"; K_t = c922_like_target(); D_t = np.zeros(5)
    save_versioned("fisheye_left", dict(model="pinhole", K=K_t.tolist(), D=[0, 0, 0, 0], image_size=[640, 480]), cal_dir=cal)
    write_virtual_wrist(K_t, (640, 480), cal_dir=cal)
    img = np.zeros((480, 640, 3), np.uint8)
    # 4 cm mount translation offset: geometry alone, no rotation
    pts = _two_camera_points(K_t, D_t, t=np.array([0.04, 0.0, 0.0]))
    r = run(img, img.copy(), "left", out=tmp_path / "qa_par", cal_dir=cal, spec=spec, points=pts, fit_rpy=True)
    assert r["E_view_near_px"] > r["E_view_far_px"] * 2 and r["near_far_ratio"] > 2
    assert any(f.startswith("PARALLAX") for f in r["flags"]) and r["verdict"] == "REJECT"
    assert r["rotation_fit"]["E_view_after_px"] > 5.0                      # rotation cannot absorb a translation
    assert r["scale_ratio_mean"] is None and r["counts"]["near"] == 4 and r["counts"]["far"] == 3
    # pure rotation instead: near and far are affected alike, and the fit removes it
    from scipy.spatial.transform import Rotation as R_
    pts2 = _two_camera_points(K_t, D_t, R=R_.from_euler("xyz", np.radians([1.5, -2.0, 0.5])).as_matrix())
    r2 = run(img, img.copy(), "left", out=tmp_path / "qa_rot", cal_dir=cal, spec=spec, points=pts2, fit_rpy=True)
    assert r2["near_far_ratio"] < 2 and not any(f.startswith("PARALLAX") for f in r2["flags"])
    assert r2["rotation_fit"]["E_view_after_px"] < 0.5 and r2["rotation_fit"]["recommended"] is True


def test_large_rotation_fit_is_flagged_as_mount_mismatch(tmp_path):
    from handumi_collector.pose.calibration import save_versioned
    from handumi_collector.tools.view_match_qa import load_spec, run
    from scipy.spatial.transform import Rotation as R_
    spec = load_spec(); cal = tmp_path / "cal"; K_t = c922_like_target(); D_t = np.zeros(5)
    save_versioned("fisheye_left", dict(model="pinhole", K=K_t.tolist(), D=[0, 0, 0, 0], image_size=[640, 480]), cal_dir=cal)
    write_virtual_wrist(K_t, (640, 480), cal_dir=cal)
    img = np.zeros((480, 640, 3), np.uint8)
    pts = _two_camera_points(K_t, D_t, R=R_.from_euler("xyz", np.radians([0, 12.0, 0])).as_matrix())
    r = run(img, img.copy(), "left", out=tmp_path / "qa_mm", cal_dir=cal, spec=spec, points=pts, fit_rpy=True)
    f = r["rotation_fit"]
    assert max(abs(x) for x in f["rpy_delta_deg"]) > 5 and f["recommended"] is False
    assert any(x.startswith("MOUNT_MISMATCH") for x in r["flags"]) and f["E_view_after_px"] < 0.5   # fit works, but is not recommended
    assert "rpy_proposed_deg" in f


def test_manual_points_and_depth_labels():
    from handumi_collector.tools.view_match_qa import compare, load_spec, points_to_landmarks
    spec = load_spec()
    pts = dict(names=["red", "plate_far"], robot=[[100, 100], [500, 400]], virtual=[[103, 104], [501, 401]], depth=["near", "far"])
    lm_r, lm_v = points_to_landmarks(pts)
    m = compare(lm_r, lm_v, (640, 480), spec)
    assert m["counts"]["all"] == 2 and abs(m["E_view_near_px"] - 5.0) < 1e-9 and abs(m["E_view_far_px"] - np.sqrt(2)) < 1e-9
    assert abs(m["E_view_median_px"] - (5.0 + np.sqrt(2)) / 2) < 1e-9 and m["E_view_p95_px"] <= 5.0
    assert m["per_landmark"]["red"]["region"] == "edge" and m["per_landmark"]["plate_far"]["region"] == "edge"
    assert m["scale_ratio_mean"] is None            # hand-picked points carry no size -> ratios are not invented
    import pytest as _pt
    with _pt.raises(ValueError): points_to_landmarks(dict(robot=[[0, 0]], virtual=[]))


def test_calibrate_pinhole_on_rendered_checkerboard():
    import cv2
    from handumi_collector.tools.calibrate_c922 import calibrate_pinhole
    cols, rows, sq = 9, 6, 0.025; K = np.array([[700.0, 0, 320], [0, 705.0, 240], [0, 0, 1]]); D = np.array([0.03, -0.05, 0.0005, -0.0005, 0.0])
    rng = np.random.default_rng(0); imgs = []
    for i in range(14):
        rvec = np.radians(rng.uniform(-25, 25, 3)); tvec = np.array([rng.uniform(-0.08, 0.02), rng.uniform(-0.06, 0.02), rng.uniform(0.45, 0.8)])
        img = np.full((480, 640), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2: continue
                sqr = np.array([[c, r, 0], [c + 1, r, 0], [c + 1, r + 1, 0], [c, r + 1, 0]], np.float64) * sq - np.array([sq, sq, 0])
                pts, _ = cv2.projectPoints(sqr.reshape(-1, 1, 3), rvec, tvec, K, D); cv2.fillConvexPoly(img, np.round(pts.reshape(-1, 2)).astype(np.int32), 0, lineType=cv2.LINE_AA)
        imgs.append(cv2.GaussianBlur(img, (3, 3), 0))
    r = calibrate_pinhole(imgs, cols, rows, sq)
    assert r["views"] >= 8 and r["rms_px"] < 1.0 and abs(r["K"][0, 0] - 700) < 15 and abs(r["K"][0, 2] - 320) < 10, r
    from handumi_collector.tools.calibrate_c922 import fov_deg
    hf, vf = fov_deg(K, (640, 480)); assert abs(hf - 49.2) < 0.5 and abs(vf - 37.7) < 0.5      # the v001 provisional target is ~49.1 deg
    assert fov_deg(np.array([[350.0, 0, 320], [0, 350.0, 240], [0, 0, 1]]), (640, 480))[0] > 80
