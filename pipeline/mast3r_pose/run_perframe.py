#!/usr/bin/env python3
"""[2026-09-23] trackA_mast3r_pose_v1, step 2 (GPU, m3slam env): official MASt3R-SLAM with PER-FRAME poses.

The official main.py saves only KEYFRAME poses (evaluate.save_traj). This wrapper runs the same loop (imports the
official modules; the repo is not modified) and records, per processed frame, the reference keyframe and the
tracked pose, then re-expresses every tracked frame on the FINAL optimised keyframe pose:

    T_WC_final(f) = T_WK_final(k) * T_WK_at_track(k)^-1 * T_WC_at_track(f)        (Sim3, lietorch)

Validity: INIT frame and successfully tracked frames are valid; a frame whose tracking failed (low match fraction /
Cholesky -> RELOC) or that was processed in RELOC mode is INVALID (its pose is only the propagated guess), unless
it later became a keyframe through relocalisation, in which case it takes the final keyframe pose.
single_thread = True (deterministic ordering with the backend). Output CSV uses the ORB-SLAM3 export columns so the
same QA runs on both, plus kf_ref, sim3_scale, match_frac, mode. Pose = T_WC (camera -> world), t in world units,
quaternion xyzw -- verified by test_convention.py, not assumed.
"""
import argparse, json, os, pathlib, sys, time, hashlib, types
ROOT = pathlib.Path(os.environ.get("M3_REPO", pathlib.Path.home() / "mast3r_slam_official"))
sys.path.insert(0, str(ROOT))
import numpy as np, torch, lietorch, yaml
# torch >= 2.6 defaults torch.load(weights_only=True); the official naver checkpoints pickle an argparse.Namespace.
# They are trusted (md5 in checkpoints/MD5SUMS), so restore the pre-2.6 default here -- module top level, so the
# spawned backend process gets it too. Recorded in the run provenance.
import functools as _ft
torch.load = _ft.partial(torch.load, weights_only=False)
import torch.multiprocessing as mp
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker
# main.py imports the in3d/imgui viewer at module level; we never open it (--no-viz equivalent), so a stub
# module stands in for mast3r_slam.visualization instead of patching the official repo.
_viz = types.ModuleType("mast3r_slam.visualization")
class _WindowMsg:
    is_terminated = False; is_paused = False; next = False; C_conf_threshold = 1.5
_viz.WindowMsg = _WindowMsg; _viz.run_visualization = lambda *a, **k: None
sys.modules["mast3r_slam.visualization"] = _viz
import main as M                                   # official run_backend / relocalization

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=None); ap.add_argument("--calib", default=None)
ap.add_argument("--video", default=None, help="C8: fisheye raw_video.mp4, undistorted in memory with the SAME fisheye_to_pinhole_v1 as prep_undistort.py (no PNG on disk)")
ap.add_argument("--setting", default=None, help="C8: orbslam_setting.yaml next to --video (KannalaBrandt8 intrinsics)")
ap.add_argument("--subsample", type=int, default=1); ap.add_argument("--out", required=True)
ap.add_argument("--config", default=str(ROOT / "config/base.yaml")); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    load_config(a.config)
    config["single_thread"] = True; config["dataset"]["subsample"] = a.subsample
    device = "cuda:0"
    manager = mp.Manager()
    if a.video:
        import re as _re, cv2 as _cv2
        from mast3r_slam.dataloader import MonocularDataset
        _t = pathlib.Path(a.setting).read_text(); _g = lambda k: float(_re.search(rf"^{k}:\s*([-\d.eE]+)", _t, _re.M).group(1))
        _K = np.array([[_g("Camera1.fx"), 0, _g("Camera1.cx")], [0, _g("Camera1.fy"), _g("Camera1.cy")], [0, 0, 1.0]])
        _D = np.array([_g(f"Camera1.k{i}") for i in (1, 2, 3, 4)]).reshape(4, 1); _W, _H = int(_g("Camera.width")), int(_g("Camera.height"))
        _P = _cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(_K, _D, (_W, _H), np.eye(3), balance=0.0)
        _mx, _my = _cv2.fisheye.initUndistortRectifyMap(_K, _D, np.eye(3), _P, (_W, _H), _cv2.CV_32FC1)
        _cap = _cv2.VideoCapture(a.video); _frames = []
        while True:
            _ok, _f = _cap.read()
            if not _ok: break
            _frames.append(_cv2.remap(_f, _mx, _my, _cv2.INTER_LINEAR))        # identical to the PNG written by prep_undistort.py
        _cap.release()
        class _VideoUndistorted(MonocularDataset):
            def __init__(self):
                super().__init__(); self.rgb_files = list(range(len(_frames))); self.timestamps = list(np.arange(len(_frames), dtype=np.float32) / 30.0)
            def read_img(self, idx):
                return _cv2.cvtColor(_frames[self.rgb_files[idx]], _cv2.COLOR_BGR2RGB)
        dataset = _VideoUndistorted(); dataset.subsample(a.subsample)
        a.calib = str(pathlib.Path(a.out).with_suffix(".calib.yaml")); pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.calib).write_text("width: %d\nheight: %d\ncalibration: [%.6f, %.6f, %.6f, %.6f]\n" % (_W, _H, _P[0, 0], _P[1, 1], _P[0, 2], _P[1, 2]))
    else:
        dataset = load_dataset(a.frames); dataset.subsample(a.subsample)
    h, w = dataset.get_img_shape()[0]
    intr = yaml.safe_load(open(a.calib))
    config["use_calib"] = True; dataset.use_calibration = True
    dataset.camera_intrinsics = Intrinsics.from_calib(dataset.img_size, intr["width"], intr["height"], intr["calibration"])
    keyframes = SharedKeyframes(manager, h, w); states = SharedStates(manager, h, w)
    model = load_mast3r(device=device); model.share_memory()
    K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32); keyframes.set_intrinsics(K)
    tracker = FrameTracker(model, keyframes, device)
    backend = mp.Process(target=M.run_backend, args=(config, model, states, keyframes, K)); backend.start()

    log = []; t0 = time.time()
    for i in range(len(dataset)):
        mode = states.get_mode()
        timestamp, img = dataset[i]
        T_WC = lietorch.Sim3.Identity(1, device=device) if i == 0 else states.get_frame().T_WC
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        rec = dict(i=i, src=i * a.subsample, mode=mode.name, valid=False, kf_ref=-1, is_kf=False, match_frac=float("nan"))
        add_new_kf = False
        if mode == Mode.INIT:
            X, C = mast3r_inference_mono(model, frame); frame.update_pointmap(X, C)
            keyframes.append(frame); states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING); states.set_frame(frame)
            rec.update(valid=True, is_kf=True, kf_ref=0, T=frame.T_WC.data.cpu().numpy().ravel().tolist(),
                       T_kf=keyframes.T_WC[0].cpu().numpy().ravel().tolist())
            log.append(rec); continue
        if mode == Mode.TRACKING:
            kref = len(keyframes) - 1
            T_kf_now = keyframes.T_WC[kref].detach().cpu().numpy().ravel().tolist()
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc: states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            rec.update(valid=not try_reloc, kf_ref=kref, T=frame.T_WC.data.detach().cpu().numpy().ravel().tolist(), T_kf=T_kf_now)
            if isinstance(match_info, (list, tuple)) and len(match_info):
                try: rec["match_frac"] = float(match_info[0])
                except Exception: pass
        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame); frame.update_pointmap(X, C)
            states.set_frame(frame); states.queue_reloc()
            while True:
                with states.lock:
                    if states.reloc_sem.value == 0: break
                time.sleep(0.01)
        if add_new_kf:
            keyframes.append(frame); states.queue_global_optimization(len(keyframes) - 1)
            rec["is_kf"] = True
            while True:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0: break
                time.sleep(0.01)
        log.append(rec)
    states.set_mode(Mode.TERMINATED); backend.join()
    elapsed = time.time() - t0

    # final keyframe poses and their frame ids
    nkf = len(keyframes)
    kf_final = {k: keyframes.T_WC[k].detach().cpu() for k in range(nkf)}
    kf_frame = {k: int(keyframes[k].frame_id) for k in range(nkf)}
    frame_kf = {f: k for k, f in kf_frame.items()}
    rows = []
    for r in log:
        T = None
        if r["i"] in frame_kf:                                        # became a keyframe (incl. via relocalisation)
            T = lietorch.Sim3(kf_final[frame_kf[r["i"]]]); r["valid"] = True; r["is_kf"] = True
        elif r["valid"]:
            k = r["kf_ref"]
            T = lietorch.Sim3(kf_final[k]) * lietorch.Sim3(torch.tensor(r["T_kf"]).view(1, 8)).inv() * lietorch.Sim3(torch.tensor(r["T"]).view(1, 8))
        d = T.data.view(-1).numpy() if T is not None else np.zeros(8)
        rows.append((r["src"], r["src"] / 30.0, r["mode"], r["valid"], r["is_kf"], d, r["kf_ref"], r["match_frac"]))
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w,map_updated,kf_ref,sim3_scale,match_frac\n")
        for src, ts, mode, valid, iskf, d, kref, mf in rows:
            f.write(f"{src},{ts:.6f},{mode},{'false' if valid else 'true'},{'true' if iskf else 'false'},"
                    f"{d[0]:.7f},{d[1]:.7f},{d[2]:.7f},{d[3]:.7f},{d[4]:.7f},{d[5]:.7f},{d[6]:.7f},0,{kref},{d[7]:.6f},{mf:.4f}\n")
    ck = sorted((ROOT / "checkpoints").glob("*"))
    json.dump(dict(pose_source="mast3r_slam", variant="rgb_only_calibrated", repo=str(ROOT), build_patch="build_compat_5090.patch (compile-only)",
                   torch_load_weights_only=False,
                   git_commit=(ROOT / "INSTALLED_COMMIT").read_text().strip(), checkpoints={c.name: c.stat().st_size for c in ck},
                   calibrated=True, calib=intr, intrinsic_hash=hashlib.md5(json.dumps(intr["calibration"]).encode()).hexdigest(),
                   subsample=a.subsample, source_fps=30, seed=a.seed, single_thread=True, config=a.config,
                   pose_convention="T_WC camera->world, Sim3 re-expressed on final keyframes, t xyz + q xyzw (test_convention.py)",
                   scale="monocular Sim3: world units are MASt3R's metric-prior scale, NOT guaranteed metres (checked vs ArUco)",
                   n_frames=len(rows), n_valid=int(sum(r[3] for r in rows)), n_keyframes=nkf, seconds=elapsed),
              open(out.with_suffix(".json"), "w"), indent=1)
    print(f"done {out}  frames {len(rows)} valid {sum(r[3] for r in rows)} kf {nkf}  {elapsed:.0f}s")
