"""M2 integration tests on a synthetic raw episode: offline pipeline (mock + real opencv_vo backends), derived outputs,
QA verdicts, benchmark tool, live runner isolation from raw recording, session HOME marks, no-fiducial guard."""
import json
import re
import time
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from handumi_collector.config import PoseCfg
from handumi_collector.pose import se3
from handumi_collector.pose.backends import available_backends, make_backend
from handumi_collector.pose.backends.mock import MockBackend
from handumi_collector.pose.calibration import SideCalibration, save_versioned, versions, write_camera_tcp
from handumi_collector.pose.episode_io import RawEpisode, derived_dir, read_table
from handumi_collector.pose.estimator import BackendUnavailable
from handumi_collector.pose.run import process_episode
from handumi_collector.tools.synth_episode import INTRINSICS, synth_episode

PKG = Path(__file__).resolve().parents[2] / "handumi_collector"


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth"); ep = root / "episode_000001"
    gt = synth_episode(ep, seconds=6.0, camera_imu_offset_ms=12.0, seed=0)
    cal = root / "cal"
    for s in ("left", "right"):
        save_versioned(f"fisheye_{s}", dict(model="pinhole", K=gt["intrinsics"]["K"], D=gt["intrinsics"]["D"], image_size=gt["intrinsics"]["image_size"]), cal_dir=cal)
    return dict(ep=ep, gt=gt, cal=cal, root=root)


def _gt_factory(gt, side=None):
    """Mock backend replaying the GT trajectory; with side=None the factory picks the side the pipeline asks for."""
    def make(side_=None, **_):
        sd = side or side_ or "left"; ts = np.array(gt["sides"][sd]["t_ns"]); P = np.array(gt["sides"][sd]["pose7"])
        def traj(t): return se3.pose7_to_T(P[int(np.argmin(np.abs(ts - t)))])
        return MockBackend(traj, noise_pos_m=0.0003, noise_rot_deg=0.03)
    return lambda side=None, **kw: make(side)


def test_raw_episode_loads_and_frames_join(synth):
    ep = RawEpisode.load(synth["ep"])
    assert set(ep.frames) == {"head", "left_wrist", "right_wrist"} and len(ep.imu["left"]) == 2400 and len(ep.grip["right"]) == 600
    frames = list(ep.iter_frames("left_wrist", downscale=2))
    assert len(frames) == 180 and frames[0][3].shape == (240, 320, 3) and frames[5][2] == int(ep.frames["left_wrist"].capture_ns[5])
    assert ep.protocol_events("home_leave") and ep.protocol_events("home_return")


def test_pipeline_mock_backend_passes_and_writes_derived(synth):
    cfg = PoseCfg(); ep = synth["ep"]
    s = process_episode(ep, cfg, backend_name="mock", cal_dir=synth["cal"], backend_factory=_gt_factory(synth["gt"]))
    assert s["verdict"] == "PASS" and set(s["sides"]) == {"left", "right"}
    for side in ("left", "right"):
        q = s["sides"][side]
        assert q["verdict"] == "PASS", q["reasons"]
        assert q["return_translation_mm"] < 2 and q["return_rotation_deg"] < 0.3 and q["canonical_anchor"] == "home_start_mean"
        assert np.allclose(q["gyro_bias_dps"], synth["gt"]["gyro_bias_dps"], atol=0.05)       # bias recovered from the HOME still window
        assert q["imu_residual_deg_p95"] < 1.0 and q["lost_events"] == 0 and q["home_start"]["duration_s"] > 0.8
        assert any("tcp_uncalibrated" in f for f in q["flags"]) and any("offset_unmeasured" in f for f in q["flags"])
    d = derived_dir(ep, "mock")
    assert (d / "pose_qa.json").exists()
    can = read_table(d / "canonical"); cam = read_table(d / "left_camera_pose"); tcp = read_table(d / "left_tcp_pose")
    assert len(can) == 180 and len(cam) == 180 == len(tcp) and can["left_tcp_valid"].mean() > 0.95 and can["left_grip_valid"].all()
    assert set(can["left_segment"]) == {"home_start", "manipulation", "home_end"}
    # episode-local canonicalisation: HOME start ≈ identity
    hs = can[can["left_segment"] == "home_start"]; assert np.abs(hs[["left_tcp_x", "left_tcp_y", "left_tcp_z"]].dropna().values).max() < 0.002
    # relative TCP from the canonical table equals inv(T_t) T_{t+5} of the GT trajectory (up to mock noise), in the LOCAL frame
    from handumi_collector.pose.relative import relative_tcp
    P = can[["left_tcp_x", "left_tcp_y", "left_tcp_z", "left_tcp_qx", "left_tcp_qy", "left_tcp_qz", "left_tcp_qw"]].values; v = can["left_tcp_valid"].values
    P = P.copy(); P[~v] = se3.IDENTITY7                                   # invalid rows carry NaN; masked out by `valid`
    d5, ok = relative_tcp(se3.poses7_to_T(P), 5, v); assert ok.sum() > 150
    gt = synth["gt"]["sides"]["left"]; Tg = se3.poses7_to_T(np.array(gt["pose7"])); i = 90
    ref = se3.local_delta(Tg[i], Tg[i + 5]); assert np.allclose(d5[i], ref, atol=0.004), (d5[i], ref)
    # raw untouched
    assert not (ep / ".incomplete").exists() and json.loads((ep / "episode_meta.json").read_text())["status"] == "KEEP"


def test_pipeline_real_opencv_vo_on_synthetic_video(synth):
    cfg = PoseCfg(image_downscale=1); ep = synth["ep"]
    s = process_episode(ep, cfg, backend_name="opencv_vo", sides=["left"], cal_dir=synth["cal"])
    q = s["sides"]["left"]
    assert q["error"] is None and q["metric_scale"] is False
    assert q["valid_ratio"] > 0.6 and q["lost_events"] == 0 and q["jumps_rotation"] == 0, q
    assert q["imu_residual_deg_p95"] < 5.0                                          # rotation agrees with the raw gyro
    # rotation of the manipulation bump is recovered (non-metric translation is not judged)
    cam = read_table(derived_dir(ep, "opencv_vo") / "left_camera_pose"); ok = cam["valid"].values.astype(bool)
    gt = synth["gt"]["sides"]["left"]; Tg = se3.poses7_to_T(np.array(gt["pose7"]))
    Te = se3.poses7_to_T(cam[["x", "y", "z", "qx", "qy", "qz", "qw"]].values[ok]); idx = np.nonzero(ok)[0]
    A = Tg[idx[0]] @ se3.inv_T(Te[0]); errs = [se3.rotation_angle_deg((A @ Te[k])[:3, :3], Tg[i][:3, :3]) for k, i in enumerate(idx)]
    assert np.median(errs) < 2.0 and max(errs) < 8.0, (np.median(errs), max(errs))
    assert any("no valid poses inside the HOME start window" in r for r in q["reasons"])   # monocular VO cannot initialise while still: reported, not hidden


def test_pipeline_survives_backend_crash_and_unavailable_backend(synth):
    cfg = PoseCfg(); ep = synth["ep"]
    s = process_episode(ep, cfg, backend_name="mock", sides=["left"], cal_dir=synth["cal"], backend_factory=lambda: MockBackend(raise_on_frame=60))
    q = s["sides"]["left"]; assert q["verdict"] == "REJECT" and "backend exception" in q["error"] and q["n_frames"] == 180
    s = process_episode(ep, cfg, backend_name="mock", sides=["right"], cal_dir=synth["cal"], backend_factory=lambda: MockBackend(fail_on_init=True))
    assert s["sides"]["right"]["verdict"] == "REJECT" and (derived_dir(ep, "mock") / "pose_qa.json").exists()
    with pytest.raises(BackendUnavailable): make_backend("dpvo").initialize(intrinsics=INTRINSICS)
    with pytest.raises(BackendUnavailable): make_backend("orbslam3").initialize(intrinsics=INTRINSICS, T_camera_imu=np.eye(4))
    with pytest.raises(BackendUnavailable): make_backend("opencv_vo").initialize(intrinsics=None)
    for b in available_backends(): s2 = process_episode(ep, cfg, backend_name=b, sides=["left"], cal_dir=synth["cal"]) if b in ("dpvo", "orbslam3") else None
    for b in ("dpvo", "orbslam3"):
        q = json.loads((derived_dir(ep, b) / "pose_qa.json").read_text())["sides"]["left"]; assert q["verdict"] == "REJECT" and "unavailable" in q["error"]


def test_orbslam3_export_and_tum_import(synth, tmp_path):
    from handumi_collector.pose.backends.orbslam3 import OrbSlam3Backend, align_trajectory_to_frames, read_tum_trajectory
    ep = RawEpisode.load(synth["ep"]); frames = list(ep.iter_frames("left_wrist", downscale=4))[:5]
    gt = synth["gt"]["sides"]["left"]; lines = ["# t x y z qx qy qz qw"] + [f"{t / 1e9:.9f} " + " ".join(f"{v:.6f}" for v in p) for t, p in zip(gt["t_ns"][:4], gt["pose7"][:4])]
    tf = tmp_path / "f_run.txt"; tf.write_text("\n".join(lines))
    traj = read_tum_trajectory(tf); assert len(traj) == 4 and np.allclose(traj[1][1], se3.pose7_to_T(gt["pose7"][1]), atol=1e-5)
    be = OrbSlam3Backend(trajectory_file=str(tf)); be.initialize(intrinsics=None, T_camera_imu=None)
    for vf, fi, t, img in frames: be.push_image(t, fi, img)
    out = be.finish(); assert [e.valid for e in out] == [True, True, True, True, False] and out[4].tracking_state.value == "lost"
    be2 = OrbSlam3Backend(binary="/nonexistent", vocabulary="/nonexistent")
    with pytest.raises(BackendUnavailable): be2.initialize(intrinsics=INTRINSICS, T_camera_imu=np.eye(4))
    be3 = OrbSlam3Backend(trajectory_file=str(tf)); be3.initialize(); be3._intr, be3._T_c_i, be3._noise, be3._size = INTRINSICS, np.eye(4), {}, (640, 480)
    for vf, fi, t, img in frames[:2]: be3.push_image(t, fi, img)
    be3.push_imu(frames[0][2], (0, 0, 0.1), (0, 0, 9.81)); wd = tmp_path / "wd"; be3.export_euroc(wd); be3.write_settings(wd / "s.yaml")
    assert (wd / "mav0" / "cam0" / "data.csv").exists() and (wd / "mav0" / "imu0" / "data.csv").read_text().count("\n") == 2 and "KannalaBrandt8" in (wd / "s.yaml").read_text()


def test_benchmark_tool_aggregates(synth, tmp_path):
    from handumi_collector.tools.pose_benchmark import run_benchmark
    cfg = PoseCfg(); session = synth["ep"].parent
    r = run_benchmark(session, ["mock"], cfg, out=tmp_path, cal_dir=synth["cal"], allow_mock=True, backend_factories={"mock": _gt_factory(synth["gt"], "left")})   # LEFT GT replayed for BOTH sides on purpose
    a = r["backends"]["mock"]["aggregate"]; assert a["episodes"] == 1 and a["valid"] > 0.9 and (tmp_path / "benchmark.md").read_text().startswith("# Pose backend benchmark")
    assert a["reject"] == 1          # right side gets LEFT's trajectory -> IMU consistency check rejects it (the QA catches a wrong-hand pose stream)
    from handumi_collector.tools.camera_imu_offset import offset_for_episode
    process_episode(synth["ep"], cfg, backend_name="mock", sides=["left"], cal_dir=synth["cal"], backend_factory=_gt_factory(synth["gt"], "left"))
    o = offset_for_episode(synth["ep"], "left", "mock"); assert abs(o["offset_ms"] - 12.0) < 2.0 and o["confidence"] > 0.9


def test_inspector_renders(synth, tmp_path):
    import matplotlib; matplotlib.use("Agg")
    from handumi_collector.tools.pose_inspector import figure, load
    fig = figure(load(synth["ep"], "mock"), "synthetic"); out = tmp_path / "insp.png"; fig.savefig(out); assert out.stat().st_size > 10_000


def test_live_runner_failure_never_breaks_raw_recording(mock_cfg):
    from handumi_collector.collector.episode_manager import EpisodeManager
    from handumi_collector.collector.recorder import EpisodeRecorder
    from handumi_collector.devices.manager import DeviceManager
    from handumi_collector.pose.live import LivePoseRunner
    dm = DeviceManager(mock_cfg.hardware); dm.build(); dm.connect_all(); mgr = EpisodeManager(mock_cfg)
    crashing = LivePoseRunner("left", dm.cameras["left_wrist"], dm.imus["left"], MockBackend(raise_on_frame=2, init_frames=0), downscale=1)
    dead = LivePoseRunner("right", dm.cameras["right_wrist"], dm.imus["right"], MockBackend(fail_on_init=True), downscale=1)
    crashing.start(); dead.start()
    assert dead.status.error and not dead.status.running
    rec = EpisodeRecorder(mock_cfg, dm, mgr.next_episode_dir(), "RBP", "x", mgr.anchor); rec.start(); time.sleep(0.6); rec.stop(); meta = rec.finalize(status="KEEP")
    crashing.stop(); dm.close_all()
    assert crashing.status.error and "crash" in crashing.status.error
    assert meta["streams"]["left_wrist"]["frames"] > 10 and meta["streams"]["head"]["frames"] > 10 and "recorder_error" not in meta["event_kinds"]
    healthy_ok = MockBackend(init_frames=0)
    dm2 = DeviceManager(mock_cfg.hardware); dm2.build(); dm2.connect_all()
    r = LivePoseRunner("left", dm2.cameras["left_wrist"], dm2.imus["left"], healthy_ok, downscale=1); r.start(); time.sleep(0.4); r.stop(); dm2.close_all()
    assert r.status.frames > 3 and r.buffer.latest().valid and r.status.error is None


def test_session_home_marks_written_to_events(mock_cfg):
    from handumi_collector.collector.session import CollectorSession
    s = CollectorSession.create(mock_cfg)
    with pytest.raises(AssertionError): s.mark()
    s.start(); time.sleep(0.15); assert s.mark() == "home_leave"; time.sleep(0.15); assert s.mark() == "home_return"; time.sleep(0.1)
    s.stop(); meta = s.keep(); sdir = s.manager.session_dir; s.close()
    ev = json.loads((sdir / meta["episode_dir"] / "events.json").read_text())
    kinds = [e["kind"] for e in ev]; assert kinds == ["home_leave", "home_return"] and ev[0]["t_ns"] < ev[1]["t_ns"]
    assert meta["quality"] == "PASS" and meta["hw_event_in_episode"] is False   # protocol marks are not hardware events
    ep = RawEpisode.load(sdir / meta["episode_dir"])
    assert ep.protocol_events("home_leave") == [ev[0]["t_ns"]]


def test_calibration_versioning_and_uncalibrated_is_explicit(tmp_path):
    c = SideCalibration("left", cal_dir=tmp_path)
    assert not c.tcp_calibrated and c.versions == {"fisheye": None, "camera_imu": None, "camera_tcp": None}
    T = se3.make_T(Rotation.from_euler("x", 0.1).as_matrix(), [0.0, 0.0, 0.07])
    p1 = write_camera_tcp("left", T, method="pivot", cal_dir=tmp_path); p2 = write_camera_tcp("left", T, method="pivot", cal_dir=tmp_path)
    assert [p.name for p in versions("camera_tcp_left", tmp_path)] == ["camera_tcp_left_v001.yaml", "camera_tcp_left_v002.yaml"] and p1 != p2
    c2 = SideCalibration("left", cal_dir=tmp_path); assert c2.versions["camera_tcp"] == "camera_tcp_left_v002" and np.allclose(c2.T_camera_tcp, T)
    c3 = SideCalibration("left", cal_dir=tmp_path, which={"camera_tcp": "camera_tcp_left_v001"}); assert c3.versions["camera_tcp"] == "camera_tcp_left_v001"
    with pytest.raises(ValueError): write_camera_tcp("left", np.ones((4, 4)), cal_dir=tmp_path)


FIDUCIAL = re.compile(r"apriltag|aruco|charuco|aprilgrid|fiducial_reloc|pupil_apriltags|cv2\.aruco", re.I)


def test_no_fiducial_dependency_in_handumi_collector():
    """Design rule: no AprilTag / ArUco / ChArUco / AprilGrid anywhere in the production package (docstrings that state the rule excepted)."""
    import subprocess, sys
    code = ("import sys, handumi_collector.pose, handumi_collector.pose.run, handumi_collector.pose.backends.opencv_vo, handumi_collector.pose.live, "
            "handumi_collector.collector.recorder, handumi_collector.collector.session, handumi_collector.tools.pose_process, handumi_collector.tools.pose_benchmark; "
            "bad = [m for m in sys.modules if ('apriltag' in m or 'aruco' in m or m.startswith('ego_collector.tracking')) and not m.startswith('cv2.')]   # cv2.aruco is loaded by opencv-contrib itself; source scan below forbids using it; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(PKG.parent))   # isolated interpreter: legacy tests may import them here
    assert r.returncode == 0, r.stdout + r.stderr
    offenders = []
    for py in PKG.rglob("*.py"):
        for ln, line in enumerate(py.read_text().splitlines(), 1):
            if FIDUCIAL.search(line) and not re.search(r"^\s*(#|\"\"\"|No AprilTag|.*never|.*not |.*fiducial-free)", line, re.I) and "import" in line:
                offenders.append(f"{py.relative_to(PKG)}:{ln}: {line.strip()}")
    assert not offenders, offenders
    assert not list(PKG.rglob("*apriltag*")) and not list(PKG.rglob("*aruco*")) and not list(PKG.rglob("*charuco*"))


def test_the_relative_path_does_not_import_the_absolute_verdicts():
    """`pose.qa` judges an absolute trajectory -- how much of it solved, and how far it drifted returning HOME. The
    relative pipeline consumes inv(T(t)) T(t+k) and object-relative geometry and never builds a path those describe,
    so its modules must not reach for them.

    This is a real mistake that was made: `opencv_vo` was first rejected on valid ratio 0.29 and 48 m of HOME drift,
    neither of which says anything about short-window relative motion. It was rejected again on the right evidence
    and the verdict held -- but a relative estimator judged by absolute criteria can just as easily be kept for the
    wrong reason."""
    from handumi_collector.pose import qa
    assert qa.JUDGES == "absolute_trajectory"
    relative_modules = ["handumi_collector/pose/cube_anchor.py", "handumi_collector/tools/pnp_feasibility.py"]
    for rel in relative_modules:
        src = (PKG.parent / rel).read_text()
        for line in src.splitlines():
            code = line.split("#", 1)[0]
            assert "pose.qa" not in code and "from .qa" not in code and "import qa" not in code, f"{rel}: {line.strip()}"
            for m in qa.ABSOLUTE_ONLY_METRICS:
                assert f".{m}" not in code and f"[\"{m}\"]" not in code, f"{rel} uses the absolute metric {m!r}: {line.strip()}"
