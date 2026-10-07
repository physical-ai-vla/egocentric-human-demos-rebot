#!/usr/bin/env python3
"""XVLA-Soft-Fold (Facebear/XVLA-Soft-Fold, Agilex Aloha, hdf5) -> raw episodes in the ego_cart20 raw format.
Source adapter ONLY: poses / gripper / timestamps / 3 camera videos; no labels are made here.

Facts measured 2026-10-07 on 0929_11am_new/episode_105 and 0702_.../episode_77 (scratchpad sf_tools/):
  eef_quaternion   per arm [xyz m | quat xyzw | gripper m]; eef_6d rot = R[:, :2].reshape(6) (columns, interleaved), err 0.0
  clocks           eef_{left,right}_time per arm (~35 ms median, p99 70 ms); images on time_stamp (same clock origin)
  gripper          follower jaw width in m (= qpos gripper), 0 .. ~0.0614  -> g = clip(w / W_OPEN_M, 0, 1), 0 = closed
  tool frame       Agilex tool +z = approach (pre-grasp motion 0.8-0.99 along +z), +x = toward the jaws (wrist-image
                   down, flow corr -0.6), +y = wrist-image left.  reBot dataset TCP: x = approach, y = cam x, z = cam y
                   (umi_bridge/track_c/reports/open1).  So T_rebot = T_agilex @ X, X columns = [+z, -y, +x].
Cameras: cam_high -> head, cam_left_wrist -> left_wrist, cam_right_wrist -> right_wrist, re-encoded at 320x240 (4:3 kept;
the policy pads to 224 like the reBot cameras), frame k <-> time_stamp[k].

usage: softfold_export.py --out <raw_root> <hdf5> [<hdf5> ...] [--workers N] [--delete-source]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

W_OPEN_M = 0.0615
X = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])       # columns = reBot x, y, z in Agilex tool coordinates
CAMS = (("cam_high", "head"), ("cam_left_wrist", "left_wrist"), ("cam_right_wrist", "right_wrist"))
ARM_SLICE = {"left": 0, "right": 8}                              # eef_quaternion offsets
VIDEO_WH = (320, 240)


def episode_id(h5path):
    p = pathlib.Path(h5path)
    return f"sf_{p.parent.name}_{p.stem.replace('episode_', 'ep')}"


def encode_video(frames_bytes, out_mp4):
    import av, cv2
    c = av.open(str(out_mp4), "w"); s = c.add_stream("libx264", rate=30); s.width, s.height = VIDEO_WH; s.pix_fmt = "yuv420p"
    s.options = {"crf": "18", "preset": "veryfast"}
    for b in frames_bytes:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(cv2.resize(img, VIDEO_WH, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for pk in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()


def export_one(h5path, out_root, delete_source=False):
    import h5py
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from ego_cart20.geometry.rotation6d import matrix_to_quaternion
    from ego_cart20.io.raw_episode_loader import save_raw_episode
    e = episode_id(h5path); out = pathlib.Path(out_root) / e
    if (out / "raw_episode.json").exists(): return dict(episode=e, status="exists")
    if out.exists(): import shutil; shutil.rmtree(out)           # half-written by a killed run (json is written last)
    tmp = pathlib.Path(out_root) / f".{e}.videos"; tmp.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5path, "r") as h:
        ob = h["observations"]; q = ob["eef_quaternion"][()].astype(np.float64); arrays = {}; info = {}
        for a, o in ARM_SLICE.items():
            t = ob[f"eef_{a}_time"][()].astype(np.float64).ravel()
            T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, :3] = Rotation.from_quat(q[:, o + 3:o + 7]).as_matrix() @ X; T[:, :3, 3] = q[:, o:o + 3]
            w = q[:, o + 7]; g = np.clip(w / W_OPEN_M, 0.0, 1.0)
            ok = np.isfinite(T).all((1, 2)) & np.isfinite(g) & np.concatenate([[True], np.diff(t) > 0])
            arrays.update({f"{a}_t_ns": np.round(t * 1e9).astype(np.int64), f"{a}_position": T[:, :3, 3],
                           f"{a}_quaternion": matrix_to_quaternion(T[:, :3, :3]), f"{a}_valid": ok, f"{a}_gripper": g,
                           f"{a}_width_m_raw": w})
            info[a] = dict(n=int(len(t)), dt_ms_p50=float(np.median(np.diff(t)) * 1e3), dt_ms_max=float(np.diff(t).max() * 1e3),
                           width_max_m=float(w.max()), width_min_m=float(w.min()), clip_hi_frac=float((w > W_OPEN_M).mean()))
        ts = h["time_stamp"][()].astype(np.float64).ravel(); videos = {}
        for src, cam in CAMS:
            mp4 = tmp / f"{cam}.mp4"; encode_video(ob["images"][src], mp4); videos[cam] = mp4
            arrays[f"cam_{cam}_t_ns"] = np.round(ts * 1e9).astype(np.int64); arrays[f"cam_{cam}_frame"] = np.arange(len(ts), dtype=np.int64)
        instr = h["language_instruction"][()]; instr = instr.decode() if isinstance(instr, bytes) else str(instr)
    meta = dict(schema="ego_cart20_raw_episode/v1", episode_id=e, source="robot_agilex_aloha_softfold", era=pathlib.Path(h5path).parent.name,
                task="cloth_folding", instruction=instr, stack_order="fold", clock="softfold_episode_seconds",
                videos={c: str(out / f"{c}.mp4") for _, c in CAMS},
                tool_frame=dict(name="agilex_eef @ X (reBot dataset-TCP axes: x approach, y cam x, z cam y)", X=X.tolist()),
                gripper_calibration=dict(version="softfold_width_v1", convention="0 = closed, 1 = open, width_m / W_OPEN_M", W_OPEN_M=W_OPEN_M),
                provenance=dict(hdf5=str(pathlib.Path(h5path).resolve()), dataset="Facebear/XVLA-Soft-Fold", arms=info,
                                exporter=str(pathlib.Path(__file__).resolve()), exported=time.strftime("%Y-%m-%d %H:%M:%S")))
    save_raw_episode(out, meta, arrays)
    for _, c in CAMS: videos[c].rename(out / f"{c}.mp4")
    tmp.rmdir()
    if delete_source: pathlib.Path(h5path).unlink()
    return dict(episode=e, status="ok", n=info["left"]["n"], dur_s=round(float(ts[-1] - ts[0]), 1))


def _worker(args):
    h5, out, dele = args
    try: return export_one(h5, out, dele)
    except Exception as ex: return dict(episode=episode_id(h5), status="error", reason=f"{type(ex).__name__}: {ex}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("hdf5", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--delete-source", action="store_true"); a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True); jobs = [(h, a.out, a.delete_source) for h in a.hdf5]
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex: res = []; [res.append(r) or print(json.dumps(r), flush=True) for r in ex.map(_worker, jobs)]
    else:
        res = []; [res.append(_worker(j)) or print(json.dumps(res[-1]), flush=True) for j in jobs]
    import collections; print("DONE", dict(collections.Counter(r["status"] for r in res)), flush=True)
