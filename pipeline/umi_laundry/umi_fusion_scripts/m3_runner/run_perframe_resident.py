#!/usr/bin/env python3
"""[2026-09-24] Resident variant of run_perframe.py (C8 execution infrastructure only; algorithm and config unchanged).

run_perframe.py pays ~108 s of fixed cost per side (python/torch/CUDA start + 2.75 GB checkpoint load + ...). This runner
loads the MASt3R model ONCE (read-only weights) and processes many sides in one process. Everything episode-specific is
rebuilt from scratch for every side, exactly as a fresh run_perframe.py process would build it:
    config (load_config again), RNG state (restored to the post-model-load state of a fresh process), mp.Manager, SharedKeyframes, SharedStates, FrameTracker,
    and the BACKEND PROCESS (spawned per side: its retrieval database / global optimiser live and die with it).
After each side the per-side objects are deleted and torch.cuda.empty_cache() is called; VRAM is logged (process-tree
peak during the side, and allocated / reserved / GPU-used after cleanup) so a leak across sides is visible.
The per-side body below is run_perframe.py lines 57-165 verbatim, except: `a.*` -> function arguments, the model is
passed in, `runner="resident_v1"` is added to the provenance json. Validation vs run_perframe.py outputs: c8 contract §21.
Usage: run_perframe_resident.py --list <tags> --in-root <dir> --out-dir <dir> [--name ss1] [--mem-stop-mib 15872]
Input dir per tag = <in-root>/<tag> (or <in-root>/<tag with first '_' -> '/'>), video-direct mode only.
"""
import argparse, json, os, pathlib, subprocess, sys, threading, time, hashlib, types
ROOT = pathlib.Path(os.environ.get("M3_REPO", pathlib.Path.home() / "mast3r_slam_official"))
sys.path.insert(0, str(ROOT))
import numpy as np, torch, lietorch, yaml
import functools as _ft
torch.load = _ft.partial(torch.load, weights_only=False)
import torch.multiprocessing as mp
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker
_viz = types.ModuleType("mast3r_slam.visualization")
class _WindowMsg:
    is_terminated = False; is_paused = False; next = False; C_conf_threshold = 1.5
_viz.WindowMsg = _WindowMsg; _viz.run_visualization = lambda *a, **k: None
sys.modules["mast3r_slam.visualization"] = _viz
import main as M

ap = argparse.ArgumentParser()
ap.add_argument("--list", required=True); ap.add_argument("--in-root", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--name", default="ss1"); ap.add_argument("--subsample", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--config", default=str(ROOT / "config/base.yaml")); ap.add_argument("--mem-stop-mib", type=int, default=15872)
a = ap.parse_args()


def tree_mib():
    """sum of nvidia-smi used_memory over this process and all its descendants"""
    pids, todo = {os.getpid()}, [os.getpid()]
    while todo:
        p = todo.pop(); r = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True)
        for c in r.stdout.split(): c = int(c); pids.add(c); todo.append(c)
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return sum(int(u) for p, u in (l.split(", ") for l in q.strip().splitlines() if l) if int(p) in pids)


def gpu_used_mib():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def run_side(model, video, setting, out, subsample, seed, config_path, device, rng0):
    # RNG = the state a fresh run_perframe.py process has when its loop starts: seeded BEFORE load_mast3r (model
    # instantiation consumes RNG), so the post-load state captured once in main is restored here, not a re-seed.
    torch.set_rng_state(rng0[0]); torch.cuda.set_rng_state_all(rng0[1]); np.random.set_state(rng0[2])
    load_config(config_path)
    config["single_thread"] = True; config["dataset"]["subsample"] = subsample
    manager = mp.Manager()
    import re as _re, cv2 as _cv2
    from mast3r_slam.dataloader import MonocularDataset
    _t = pathlib.Path(setting).read_text(); _g = lambda k: float(_re.search(rf"^{k}:\s*([-\d.eE]+)", _t, _re.M).group(1))
    _K = np.array([[_g("Camera1.fx"), 0, _g("Camera1.cx")], [0, _g("Camera1.fy"), _g("Camera1.cy")], [0, 0, 1.0]])
    _D = np.array([_g(f"Camera1.k{i}") for i in (1, 2, 3, 4)]).reshape(4, 1); _W, _H = int(_g("Camera.width")), int(_g("Camera.height"))
    _P = _cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(_K, _D, (_W, _H), np.eye(3), balance=0.0)
    _mx, _my = _cv2.fisheye.initUndistortRectifyMap(_K, _D, np.eye(3), _P, (_W, _H), _cv2.CV_32FC1)
    _cap = _cv2.VideoCapture(video); _frames = []
    while True:
        _ok, _f = _cap.read()
        if not _ok: break
        _frames.append(_cv2.remap(_f, _mx, _my, _cv2.INTER_LINEAR))
    _cap.release()
    class _VideoUndistorted(MonocularDataset):
        def __init__(self):
            super().__init__(); self.rgb_files = list(range(len(_frames))); self.timestamps = list(np.arange(len(_frames), dtype=np.float32) / 30.0)
        def read_img(self, idx):
            return _cv2.cvtColor(_frames[self.rgb_files[idx]], _cv2.COLOR_BGR2RGB)
    dataset = _VideoUndistorted(); dataset.subsample(subsample)
    calib = str(pathlib.Path(out).with_suffix(".calib.yaml")); pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(calib).write_text("width: %d\nheight: %d\ncalibration: [%.6f, %.6f, %.6f, %.6f]\n" % (_W, _H, _P[0, 0], _P[1, 1], _P[0, 2], _P[1, 2]))
    h, w = dataset.get_img_shape()[0]
    intr = yaml.safe_load(open(calib))
    config["use_calib"] = True; dataset.use_calibration = True
    dataset.camera_intrinsics = Intrinsics.from_calib(dataset.img_size, intr["width"], intr["height"], intr["calibration"])
    keyframes = SharedKeyframes(manager, h, w); states = SharedStates(manager, h, w)
    K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32); keyframes.set_intrinsics(K)
    tracker = FrameTracker(model, keyframes, device)
    backend = mp.Process(target=M.run_backend, args=(config, model, states, keyframes, K)); backend.start()

    log = []; t0 = time.time()
    for i in range(len(dataset)):
        mode = states.get_mode()
        timestamp, img = dataset[i]
        T_WC = lietorch.Sim3.Identity(1, device=device) if i == 0 else states.get_frame().T_WC
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        rec = dict(i=i, src=i * subsample, mode=mode.name, valid=False, kf_ref=-1, is_kf=False, match_frac=float("nan"))
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

    nkf = len(keyframes)
    kf_final = {k: keyframes.T_WC[k].detach().cpu() for k in range(nkf)}
    kf_frame = {k: int(keyframes[k].frame_id) for k in range(nkf)}
    frame_kf = {f: k for k, f in kf_frame.items()}
    rows = []
    for r in log:
        T = None
        if r["i"] in frame_kf:
            T = lietorch.Sim3(kf_final[frame_kf[r["i"]]]); r["valid"] = True; r["is_kf"] = True
        elif r["valid"]:
            k = r["kf_ref"]
            T = lietorch.Sim3(kf_final[k]) * lietorch.Sim3(torch.tensor(r["T_kf"]).view(1, 8)).inv() * lietorch.Sim3(torch.tensor(r["T"]).view(1, 8))
        d = T.data.view(-1).numpy() if T is not None else np.zeros(8)
        rows.append((r["src"], r["src"] / 30.0, r["mode"], r["valid"], r["is_kf"], d, r["kf_ref"], r["match_frac"]))
    out = pathlib.Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w,map_updated,kf_ref,sim3_scale,match_frac\n")
        for src, ts, mode, valid, iskf, d, kref, mf in rows:
            f.write(f"{src},{ts:.6f},{mode},{'false' if valid else 'true'},{'true' if iskf else 'false'},"
                    f"{d[0]:.7f},{d[1]:.7f},{d[2]:.7f},{d[3]:.7f},{d[4]:.7f},{d[5]:.7f},{d[6]:.7f},0,{kref},{d[7]:.6f},{mf:.4f}\n")
    ck = sorted((ROOT / "checkpoints").glob("*"))
    json.dump(dict(pose_source="mast3r_slam", variant="rgb_only_calibrated", repo=str(ROOT), build_patch="build_compat_5090.patch (compile-only)",
                   torch_load_weights_only=False, runner="resident_v1",
                   git_commit=(ROOT / "INSTALLED_COMMIT").read_text().strip(), checkpoints={c.name: c.stat().st_size for c in ck},
                   calibrated=True, calib=intr, intrinsic_hash=hashlib.md5(json.dumps(intr["calibration"]).encode()).hexdigest(),
                   subsample=subsample, source_fps=30, seed=seed, single_thread=True, config=config_path,
                   pose_convention="T_WC camera->world, Sim3 re-expressed on final keyframes, t xyz + q xyzw (test_convention.py)",
                   scale="monocular Sim3: world units are MASt3R's metric-prior scale, NOT guaranteed metres (checked vs ArUco)",
                   n_frames=len(rows), n_valid=int(sum(r[3] for r in rows)), n_keyframes=nkf, seconds=elapsed),
              open(out.with_suffix(".json"), "w"), indent=1)
    msg = f"done {out}  frames {len(rows)} valid {sum(r[3] for r in rows)} kf {nkf}  {elapsed:.0f}s"
    del tracker, keyframes, states, backend, dataset, _frames, K, manager, log, kf_final
    return msg


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = "cuda:0"
    torch.manual_seed(a.seed); np.random.seed(a.seed)                          # as run_perframe.py, before the model load
    t_load = time.time(); model = load_mast3r(device=device); model.share_memory(); t_load = time.time() - t_load
    rng0 = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state())
    print(f"RESIDENT model loaded once in {t_load:.1f}s  gpu used {gpu_used_mib()} MiB", flush=True)
    IN, OUT = pathlib.Path(a.in_root), pathlib.Path(a.out_dir); n = 0; OUT.mkdir(parents=True, exist_ok=True)
    for tag in open(a.list).read().split():
        out = OUT / tag / f"{a.name}.csv"
        # SKIP only a COMPLETE resident output, and trust a fresh READDIR of the out dir, not a path stat: on 2026-09-25 the
        # 5080 NFS client kept stale dentries for 56 files that had been moved to an archive on the server, so a path
        # stat saw them and 56 sides were skipped (contract §21). listdir forces a server round trip.
        sfx = "" if a.name == "ss1" else f".{a.name}"; listed = set(os.listdir(OUT))
        js = OUT / tag / f"{a.name}.json"
        if tag in listed and f"{tag}{sfx}.vram" in listed and a.name + ".csv" in os.listdir(OUT / tag) and js.is_file() \
                and json.load(open(js)).get("runner") == "resident_v1":
            print(f"SKIP {tag}", flush=True); continue
        d = IN / tag if (IN / tag).is_dir() else IN / tag.replace("_", "/", 1)
        peak = [0]; stop = threading.Event()
        def sampler():
            while not stop.is_set():
                try: peak[0] = max(peak[0], tree_mib())
                except Exception: pass
                time.sleep(0.5)
        th = threading.Thread(target=sampler, daemon=True); th.start(); t0 = time.time()
        # per-side log file <out-dir>/<tag>[.<name>].log, like the per-process runner: fds 1/2 point at it while the side
        # runs, so the spawned backend (inherits them at spawn) logs there too. Logging only; no effect on computation.
        suffix = "" if a.name == "ss1" else f".{a.name}"
        sys.stdout.flush(); sys.stderr.flush(); lf = open(OUT / f"{tag}{suffix}.log", "w"); so, se = os.dup(1), os.dup(2); os.dup2(lf.fileno(), 1); os.dup2(lf.fileno(), 2)
        try:
            msg = run_side(model, str(d / "raw_video.mp4"), str(d / "orbslam_setting.yaml"), str(out), a.subsample, a.seed, a.config, device, rng0); rc = 0
        except Exception as ex:
            import traceback; traceback.print_exc(); msg = f"ERROR {type(ex).__name__}: {ex}"; rc = 1
        print(msg); sys.stdout.flush(); sys.stderr.flush(); os.dup2(so, 1); os.dup2(se, 2); os.close(so); os.close(se); lf.close()
        stop.set(); th.join(); import gc; gc.collect(); torch.cuda.empty_cache()
        wall = time.time() - t0
        post = dict(alloc_MiB=torch.cuda.memory_allocated() // 2**20, reserved_MiB=torch.cuda.memory_reserved() // 2**20, gpu_used_MiB=gpu_used_mib(), tree_MiB=tree_mib())
        (OUT / f"{tag}{suffix}.vram").write_text(f"tree_peak_MiB {peak[0]} post_tree_MiB {post['tree_MiB']} post_alloc_MiB {post['alloc_MiB']} "
                                                f"post_reserved_MiB {post['reserved_MiB']} post_gpu_MiB {post['gpu_used_MiB']} rc {rc} sec {int(wall)}\n")
        print(f"{'OK' if rc == 0 else 'FAIL'} {tag} {msg}  vram tree peak {peak[0]} | after cleanup tree {post['tree_MiB']} alloc {post['alloc_MiB']} "
              f"reserved {post['reserved_MiB']} gpu {post['gpu_used_MiB']} MiB  {wall:.0f}s", flush=True)
        if rc or peak[0] > a.mem_stop_mib or "out of memory" in (OUT / f"{tag}{suffix}.log").read_text().lower(): print(f"STOP {tag} rc {rc} peak {peak[0]}", flush=True); sys.exit(4)
        n += 1
    print(f"RESIDENT_DONE {n}", flush=True)
