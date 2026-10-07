"""Stage-A RGB-D rigid-body tracker: episode I/O, mesh conventions, ICP core, the tracker interface, QA and the run
driver. The end-to-end tests track a small SYNTHETIC RGB-D episode whose ground truth is known, so a convention error
(depth units, principal point, pose direction, body frame) fails here instead of silently biasing a pilot."""
import json
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from handumi_collector.pose import depth_qa, icp_core
from handumi_collector.pose.depth_config import DepthPoseCfg, load_depth_pose_cfg
from handumi_collector.pose.depth_hand_tracker import (RigidPoseEstimate, TrackerUnavailable, TrackingState,
                                                      available_backends, backend_info, make_tracker)
from handumi_collector.pose.depth_run import (derived_depth_dir, load_run, poses_from_table, run_side,
                                              static_segments_from_events)
from handumi_collector.pose.rgbd_io import CameraIntrinsics, RgbdEpisode, backproject, project
from handumi_collector.pose.se3 import inv_T, make_T, poses7_to_T
from handumi_collector.pose.tracking_mesh import TrackingMesh, classify_part, load_tracking_mesh
from handumi_collector.tools import synth_rgbd_pilot as synth

CFG = load_depth_pose_cfg()          # the shipped configs/handumi/depth_pose.yaml, not a test-local copy


@pytest.fixture(scope="module")
def synth_episode(tmp_path_factory):
    """A synthetic pilot at the real 30 Hz cadence (still -> §14 motions -> still) with ground truth."""
    out = tmp_path_factory.mktemp("synth_rgbd")
    synth.main([str(out), "--seconds", "8", "--fps", "30", "--no-occluder"])
    return out


@pytest.fixture(scope="module")
def synth_fast_episode(tmp_path_factory):
    """The same motions squeezed into 1.5 s — inter-frame steps far beyond the ICP correspondence radius. Used to pin
    the one behaviour that matters more than coverage: when the baseline cannot follow, it must say LOST, never emit a
    confident wrong pose."""
    out = tmp_path_factory.mktemp("synth_rgbd_fast")
    synth.main([str(out), "--seconds", "1.5", "--fps", "30", "--no-occluder"])
    return out


# ------------------------------------------------------------------------------------------------ intrinsics / I/O
def test_intrinsics_K_and_scaling():
    k = CameraIntrinsics(400.0, 400.0, 319.5, 239.5, 640, 480)
    assert k.K[0, 0] == 400.0 and k.K[0, 2] == 319.5 and k.K[2, 2] == 1.0
    h = k.scaled(0.5)
    assert (h.width, h.height) == (320, 240)
    assert h.fx == 200.0 and h.cx == pytest.approx(159.5)      # (cx+0.5)*s-0.5 keeps the pixel centre convention


def test_depth_unit_comes_from_the_file_never_a_guess(tmp_path):
    (tmp_path / "color").mkdir(); (tmp_path / "depth").mkdir()
    import cv2
    for i in range(2):
        cv2.imwrite(str(tmp_path / "color" / f"{i:03d}.png"), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(tmp_path / "depth" / f"{i:03d}.png"), np.full((8, 8), 1500, np.uint16))
    with pytest.raises(FileNotFoundError):                      # no intrinsics => refuse, never assume a focal length
        RgbdEpisode.load(tmp_path)
    # Orbbec recorder form: depth_scale is MILLIMETRES per count
    (tmp_path / "intrinsics.json").write_text(json.dumps(dict(intrinsics=dict(fx=460, fy=460, cx=4, cy=4, width=8, height=8),
                                                              depth_scale=1.0)))
    ep = RgbdEpisode.load(tmp_path)
    assert ep.intrinsics.depth_unit_m == pytest.approx(1e-3)
    f = next(ep.iter_frames())
    assert f.depth_m[0, 0] == pytest.approx(1.5)                # 1500 counts * 1 mm = 1.5 m


def test_backproject_project_roundtrip():
    k = CameraIntrinsics(461.2, 461.4, 424.9, 240.9, 848, 480)
    depth = np.zeros((480, 848), np.float32)
    depth[100:110, 200:210] = 0.7
    pts, px = backproject(depth, k.K)
    assert len(pts) == 100 and np.allclose(pts[:, 2], 0.7)
    uv = project(pts, k.K)
    assert np.allclose(uv[:, 0], px[:, 1], atol=1e-6) and np.allclose(uv[:, 1], px[:, 0], atol=1e-6)


def test_synth_episode_shape_and_gt(synth_episode):
    ep = RgbdEpisode.load(synth_episode)
    gt = json.loads((synth_episode / "ground_truth.json").read_text())
    assert ep.layout == "flat" and ep.n_frames == gt["n_frames"] == 240
    assert ep.fps == pytest.approx(30.0, abs=0.5)
    f = next(ep.iter_frames())
    assert f.depth_raw.dtype == np.uint16 and f.rgb.shape == (480, 848, 3)
    assert 0.3 < f.depth_m[f.depth_m > 0].min() < 1.3


# ------------------------------------------------------------------------------------------------ tracking mesh
def test_part_classification_matches_the_handumi_bom():
    assert classify_part("left_thumb_link.stl") == "moving"
    assert classify_part("right_index_middle_finger_link.stl") == "moving"
    assert classify_part("crank_mechanism_plate.stl") == "moving"
    assert classify_part("connecting_link_1.stl") == "moving"
    assert classify_part("fisheye_camera_main_support.stl") == "rigid"
    assert classify_part("left_controller_support.stl") == "rigid"
    assert classify_part("something_new.stl") == "unknown"      # never silently rigid


def test_mesh_units_autodetect_and_scale_problem(tmp_path):
    m = trimesh.creation.box((70.0, 45.0, 120.0))               # millimetres
    p = tmp_path / "body_mm.stl"
    m.export(str(p))
    tm = load_tracking_mesh(p)
    assert tm.units_in == "mm" and tm.scale_applied == 1e-3
    assert tm.diagnostics()["longest_edge_m"] == pytest.approx(0.120)
    assert tm.diagnostics()["problems"] == []
    big = TrackingMesh(trimesh.creation.box((2.0, 2.0, 2.0)), None, "left", "m", 1.0, {})
    assert any("plausible HandUMI range" in x for x in big.diagnostics()["problems"])


def test_assembly_without_poses_names_the_unresolved_parts(tmp_path):
    import yaml
    part = tmp_path / "fisheye_camera_main_support.stl"
    trimesh.creation.box((0.07, 0.045, 0.12)).export(str(part))
    spec = dict(schema="handumi_tracking_mesh_assembly/v1", side="left", units="m",
                parts=[dict(file=part.name, include=True, pose=None),
                       dict(file="left_thumb_link.stl", include=False)])
    y = tmp_path / "asm.yaml"
    y.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError) as e:
        load_tracking_mesh(y)
    assert "fisheye_camera_main_support.stl" in str(e.value) and "no `pose`" in str(e.value)
    spec["parts"][0]["pose"] = dict(translation_m=[0.01, 0, 0], quaternion_xyzw=[0, 0, 0, 1])
    y.write_text(yaml.safe_dump(spec))
    tm = load_tracking_mesh(y)
    assert tm.meta["included"] == [part.name] and tm.meta["excluded"] == ["left_thumb_link.stl"]
    assert np.allclose(tm.mesh.bounds.mean(axis=0), [0.01, 0, 0], atol=1e-9)


# ------------------------------------------------------------------------------------------------ ICP core
def test_icp_core_recovers_a_known_transform():
    m = trimesh.creation.box((0.07, 0.05, 0.12))
    pts, fid = trimesh.sample.sample_surface(m, 20000, seed=0)
    model = icp_core.ModelCloud(np.asarray(pts), m.face_normals[fid])
    T_true = make_T(Rotation.from_euler("xyz", [12, -8, 25], degrees=True).as_matrix(), [0.03, -0.02, 0.55])
    rng = np.random.default_rng(0)
    scene = (T_true[:3, :3] @ np.asarray(pts[:4000]).T + T_true[:3, 3:4]).T + rng.normal(0, 8e-4, (4000, 3))
    T0 = make_T(Rotation.from_euler("xyz", [8, -3, 18], degrees=True).as_matrix(), [0.02, -0.01, 0.54])
    r = icp_core.icp_point_to_plane(icp_core.voxel_down_sample(scene, 0.003), model, inv_T(T0), max_corr_m=0.02)
    T = inv_T(r.T)
    assert r.fitness > 0.9 and r.inlier_rmse_m < 2e-3
    assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) * 1e3 < 1.0
    assert np.degrees(Rotation.from_matrix(T[:3, :3].T @ T_true[:3, :3]).magnitude()) < 0.5


def test_voxel_down_sample_reduces_and_stays_inside():
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.1, 0.1, (5000, 3))
    D = icp_core.voxel_down_sample(P, 0.02)
    assert 100 < len(D) < len(P)
    assert D.min() >= P.min() - 1e-9 and D.max() <= P.max() + 1e-9


# ------------------------------------------------------------------------------------------------ tracker interface
def test_rigid_pose_estimate_roundtrip_and_lost_is_nan():
    T = make_T(Rotation.from_euler("z", 30, degrees=True).as_matrix(), [0.1, 0.2, 0.5])
    e = RigidPoseEstimate.from_T(T, timestamp_ns=7, side="left", frame_index=3)
    assert e.tracking_valid and np.allclose(e.T_depthcam_body, T, atol=1e-12)
    row = e.to_row()
    assert row["t_ns"] == 7 and row["tracking_state"] == "tracking"
    lost = RigidPoseEstimate.from_T(None, timestamp_ns=8, side="left", frame_index=4)
    assert not lost.tracking_valid and lost.tracking_state is TrackingState.LOST
    assert np.isnan(lost.to_row()["x"]) and lost.T_depthcam_body is None


def test_backend_registry_and_foundationpose_unavailability():
    assert set(available_backends()) == {"mock", "icp", "foundationpose"}
    assert backend_info("icp").runs_on == "cpu" and backend_info("foundationpose").runs_on == "cuda"
    with pytest.raises(TrackerUnavailable):                     # no mesh
        make_tracker("icp")
    with pytest.raises(TrackerUnavailable):                     # test-only backend refuses to run without poses
        make_tracker("mock")
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        with pytest.raises(TrackerUnavailable, match="not runnable here"):
            make_tracker("foundationpose", mesh=TrackingMesh(trimesh.creation.box((0.07, 0.05, 0.12)), None, "left", "m", 1.0, {}))


def test_icp_tracker_recovers_the_synthetic_ground_truth(synth_episode):
    import cv2
    ep = RgbdEpisode.load(synth_episode)
    gt = json.loads((synth_episode / "ground_truth.json").read_text())
    G = poses7_to_T(np.array(gt["poses7_T_depthcam_body"]))
    mask = cv2.imread(str(synth_episode / "init_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    tm = load_tracking_mesh(synth_episode / "body_mesh.obj", side="left")
    tr = make_tracker("icp", mesh=tm, **CFG.options_for("icp"))
    errs = []
    for i, f in enumerate(ep.iter_frames()):
        e = (tr.initialize(f.rgb, f.depth_m, ep.intrinsics, "left", initial_mask=mask, timestamp_ns=f.t_ns, frame_index=f.index)
             if i == 0 else tr.track(f.rgb, f.depth_m, f.t_ns, frame_index=f.index))
        assert e.tracking_valid, f"frame {i} lost on noiseless synthetic depth"
        E = inv_T(e.T_depthcam_body) @ G[i]
        errs.append((np.linalg.norm(E[:3, 3]) * 1e3, np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude())))
    tr_err, rot_err = np.array(errs).T
    assert tr_err.max() < 3.0, f"translation error up to {tr_err.max():.2f} mm"
    assert rot_err.max() < 1.5, f"rotation error up to {rot_err.max():.2f} deg"
    assert tr.get_status()["valid"] == ep.n_frames


def test_icp_baseline_limit_is_surfaced_not_hidden(synth_fast_episode):
    """The §14 motions squeezed into 1.5 s step further per frame than the ICP correspondence radius. The baseline then
    either loses the object OR converges on the wrong part of the model while fitness (1.00), residual (1.7 mm) and
    model coverage (0.94) all still look healthy — measured, not hypothetical. Nothing inside the backend detects that,
    so QA must say so from the observed motion, and the verdict must not be PASS."""
    res = run_side(synth_fast_episode, "left", cfg=CFG, backend="icp",
                   mesh=synth_fast_episode / "body_mesh.obj", write=False,
                   initial_mask=__import__("cv2").imread(str(synth_fast_episode / "init_mask.png"), 0) > 0)
    assert res.error is None
    qa = res.qa
    assert qa.frames_over_trust_radius and qa.frames_over_trust_radius > 0
    assert qa.verdict != "PASS"
    assert any("trust radius" in r for r in qa.reasons)
    assert "steps over the trust radius" in res.report


# ------------------------------------------------------------------------------------------------ QA
def _straight(n=90, fps=30.0, step=0.005):
    t = (np.arange(n) * int(1e9 / fps)).astype(np.int64)
    Ts = np.tile(np.eye(4), (n, 1, 1))
    Ts[:, 0, 3] = np.arange(n) * step
    return t, Ts, np.ones(n, bool)


def test_qa_pass_and_jump_detection():
    t, Ts, valid = _straight()
    qa = depth_qa.evaluate(side="left", backend="icp", t_ns=t, Ts=Ts, valid=valid,
                           states=["tracking"] * len(t), cfg=CFG.qa, fps=30.0, runtime_s=1.0)
    assert qa.verdict == "PASS" and qa.tracking_valid_ratio == 1.0
    Ts[30:, 0, 3] += 0.4                                        # a 400 mm teleport
    qa = depth_qa.evaluate(side="left", backend="icp", t_ns=t, Ts=Ts, valid=valid,
                           states=["tracking"] * len(t), cfg=CFG.qa, fps=30.0)
    assert qa.catastrophic_jumps_translation == 1 and qa.verdict == "FAIL"


def test_qa_fails_on_low_coverage_and_long_loss():
    t, Ts, valid = _straight()
    valid[10:55] = False
    states = ["tracking" if v else "lost" for v in valid]
    qa = depth_qa.evaluate(side="left", backend="icp", t_ns=t, Ts=Ts, valid=valid, states=states, cfg=CFG.qa, fps=30.0)
    assert qa.lost_events == 1 and qa.lost_frames == 45 and qa.reacquired is True
    assert qa.verdict == "FAIL" and any("lost for" in r for r in qa.reasons)


def test_qa_static_stats_ignore_invalid_placeholder_poses():
    """Invalid frames carry an identity placeholder; counting them as poses would report a metre of 'jitter'."""
    n = 40
    t = (np.arange(n) * int(1e9 / 30)).astype(np.int64)
    Ts = np.tile(np.eye(4), (n, 1, 1))
    Ts[:, :3, 3] = [0.0, 0.0, 0.6]
    valid = np.ones(n, bool)
    valid[20] = False
    Ts[20] = np.eye(4)
    seg = depth_qa.StaticSegment(0, n - 1, int(t[0]), int(t[-1]), "operator_event")
    s = depth_qa.segment_stats(Ts.copy(), seg, valid)
    assert s.translation_std_mm == pytest.approx(0.0, abs=1e-9) and s.n == n - 1


def test_identity_swap_detection():
    n = 20
    Tl = np.tile(np.eye(4), (n, 1, 1)); Tr = np.tile(np.eye(4), (n, 1, 1))
    Tl[:, 0, 3] = -0.15; Tr[:, 0, 3] = 0.15
    v = np.ones(n, bool)
    assert depth_qa.interhand_identity_swaps(Tl, v, Tr, v) == 0
    Tl[5:8, 0, 3], Tr[5:8, 0, 3] = 0.15, -0.15                  # the two hands swap identity for 3 frames
    assert depth_qa.interhand_identity_swaps(Tl, v, Tr, v) == 3


# ------------------------------------------------------------------------------------------------ HOME windows
def test_still_windows_come_from_the_marks_the_collector_actually_emits(synth_episode, tmp_path):
    """The collector writes home_leave / home_return (Session.mark()) and nothing else — reading only
    static_begin/static_end would silently fall back to the self-referential auto-detector on every real episode."""
    ep = RgbdEpisode.load(synth_episode)
    t_ns = ep.t_ns
    d = tmp_path / "ep"
    (d / "color").mkdir(parents=True); (d / "depth").mkdir()
    (d / "events.json").write_text(json.dumps([
        dict(t_ns=int(t_ns[30]), kind="home_leave", device="operator", detail={}),
        dict(t_ns=int(t_ns[200]), kind="home_return", device="operator", detail={})]))
    fake = type("E", (), {"path": d})()
    segs = static_segments_from_events(fake, t_ns, CFG)
    assert [x.role for x in segs] == ["home_start", "home_end"]
    assert all(x.source == "operator_event" for x in segs)
    assert segs[0].i0 >= 0 and segs[0].i1 <= 30 and segs[1].i0 >= 200 and segs[1].i1 <= len(t_ns) - 1
    assert segs[0].i0 > 0                               # edge_trim_s is applied at the episode start


def test_home_return_drift_is_measured_and_flagged_as_an_upper_bound():
    n = 120
    t = (np.arange(n) * int(1e9 / 30)).astype(np.int64)
    Ts = np.tile(np.eye(4), (n, 1, 1))
    Ts[:, :3, 3] = [0.0, 0.0, 0.6]
    Ts[80:, 0, 3] = 0.020                               # the body comes back 20 mm to the side of where it started
    Ts[80:, :3, :3] = Rotation.from_euler("z", 6, degrees=True).as_matrix()
    valid = np.ones(n, bool)
    segs = [depth_qa.StaticSegment(0, 30, int(t[0]), int(t[30]), "operator_event", role="home_start"),
            depth_qa.StaticSegment(85, 119, int(t[85]), int(t[119]), "operator_event", role="home_end")]
    qa = depth_qa.evaluate(side="left", backend="icp", t_ns=t, Ts=Ts, valid=valid, states=["tracking"] * n,
                           cfg=CFG.qa, fps=30.0, static_segments=segs)
    assert qa.return_translation_mm == pytest.approx(20.0, abs=0.2)
    assert qa.return_rotation_deg == pytest.approx(6.0, abs=0.1)
    assert "home_start" in qa.return_windows and "home_end" in qa.return_windows
    assert qa.verdict == "NEEDS_WORK"                   # 20 mm > pass 15 mm, < warn 35 mm
    assert any("placement repeatability" in f for f in qa.flags)     # never sold as pure tracker drift
    assert "HOME return drift" in depth_qa.format_report(qa, episode="e")


def test_home_return_drift_absent_without_two_still_windows():
    t, Ts, valid = _straight()
    qa = depth_qa.evaluate(side="left", backend="icp", t_ns=t, Ts=Ts, valid=valid, states=["tracking"] * len(t),
                           cfg=CFG.qa, fps=30.0, static_segments=[])
    assert qa.return_translation_mm is None and qa.return_windows is None


# ------------------------------------------------------------------------------------------------ run driver
def test_run_side_writes_derived_and_leaves_raw_untouched(synth_episode):
    before = {p.name for p in synth_episode.iterdir()}
    gt = json.loads((synth_episode / "ground_truth.json").read_text())
    res = run_side(synth_episode, "left", cfg=CFG, backend="mock",
                   backend_options=dict(poses=gt["poses7_T_depthcam_body"], drop_frames=(5, 6)))
    assert res.error is None and res.qa is not None
    assert res.qa.n_frames == gt["n_frames"] and res.qa.lost_frames == 2
    out = derived_depth_dir(synth_episode, "mock")
    assert (out / "left_depth_track.json").exists() and (out / "left_report.txt").exists()
    assert {p.name for p in synth_episode.iterdir()} - before == {"derived"}     # raw untouched apart from derived/
    df, prov = load_run(synth_episode, "left", "mock")
    t, Ts, valid = poses_from_table(df)
    assert valid.sum() == gt["n_frames"] - 2
    G = poses7_to_T(np.array(gt["poses7_T_depthcam_body"]))
    assert np.allclose(Ts[valid], G[valid], atol=1e-9)          # written table roundtrips to the same poses
    assert prov["backend_info"]["name"] == "mock" and prov["episode_info"]["layout"] == "flat"
    assert "Verdict:" in res.report


def test_run_side_records_a_backend_failure_instead_of_raising(synth_episode):
    """A missing start-up prerequisite must be RECORDED on the result, never raised out of run_side.

    Which prerequisite is missing depends on the configured mesh (2026-09-11: depth_pose.yaml now points at the
    frozen single-part tracking body, so the mesh resolves and the initial ROI becomes the first thing missing).
    The contract under test is "reports instead of raising", so do not pin the specific message."""
    res = run_side(synth_episode, "left", cfg=CFG, backend="icp", mesh=None, write=False)
    assert res.error is not None and res.qa is None
    assert any(k in res.error.lower() for k in ("tracking mesh", "bbox or mask"))


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "depth_pose.yaml"
    p.write_text("backend: icp\nnot_a_key: 3\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_depth_pose_cfg(p)
    assert DepthPoseCfg().backend == "icp"
