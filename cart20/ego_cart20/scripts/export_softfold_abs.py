#!/usr/bin/env python3
"""Soft-Fold raw episodes (sources/softfold_export.py) -> LeRobot v3 with the X-VLA AIR-AGILEX-HQ contract (user 2026-10-07:
"same as X-VLA").  Re-implements 2toinf/X-VLA datasets/domain_handler/real_world.py AIRAgilexHQHandler + base.iter_episode:

  per arm  eef_6d = [xyz | R[:, :2].reshape(6) (columns, interleaved) | grip]   in the Agilex base frame, Agilex tool axes
           (the raw stores T_agilex @ X, so T_agilex = T_raw @ X^T; width_m_raw is the untouched follower jaw width)
  grip     binarised BEFORE interpolation: 1.0 if width * 50 < 1.0 (closed), else 0.0
  interp   scipy interp1d (linear, also on rot6d) per arm on its own clock, clamped to the end values
  rows     every raw index idx in range(0, T - 60); ref = (lt + rt) / 2; q = linspace(ref[idx], min(ref[idx] + 2.0, ref.max()), 31)
           static rows (both arms |seq[1] - seq[0]| < 1e-5) are skipped
  label    observation.state = [L(q0) | R(q0)] (20), action = [L(q1..30) | R(q1..30)] (30, 20)  -- absolute, NOT normalised
  images   raw frame idx of cam_high / left / right wrist (320x240, 4:3; the policy pads to 224)
fps is declared 30 (rows are raw frames, ~28.6 Hz); timestamps are only used to index the videos.
usage: export_softfold_abs.py <raw_root> <out_root> --name softfold_abs30_v1 [--val-folders F ...] [--workers N]
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
AGG = pathlib.Path.home() / "c8/rel16ego"
X = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])        # sources/softfold_export.py tool frame (T_raw = T_agilex @ X)
NUM_ACTIONS, QDUR, MARGIN = 30, 2.0, 60
CAMS = (("head", "observation.images.global"), ("left_wrist", "observation.images.left_wrist"), ("right_wrist", "observation.images.right_wrist"))
DEFAULT_VAL = ["0706_17pm_stage_1_stage2new_new_cam_very_slow", "0808_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve"]


def features():
    v = lambda: {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
    f = lambda *s: {"dtype": "float32", "shape": s, "names": None}
    return {"observation.images.global": v(), "observation.images.left_wrist": v(), "observation.images.right_wrist": v(),
            "observation.state": f(20), "action": f(NUM_ACTIONS, 20), "aux.src_index": f(1), "aux.ref_time_s": f(1)}


def arm_eef6d(z, a):
    from ego_cart20.geometry.rotation6d import quaternion_to_matrix
    R = quaternion_to_matrix(z[f"{a}_quaternion"]) @ X.T                 # back to the Agilex tool frame
    grip = (z[f"{a}_width_m_raw"] * 50 < 1.0).astype(np.float64)
    return np.concatenate([z[f"{a}_position"], R[:, :, :2].reshape(-1, 6), grip[:, None]], 1), z[f"{a}_t_ns"] / 1e9


def labels(raw_dir):
    """X-VLA iter_episode, deterministic (training-time shuffling is the sampler's job)"""
    from scipy.interpolate import interp1d
    z = np.load(pathlib.Path(raw_dir) / "raw_episode.npz")
    left, lt = arm_eef6d(z, "left"); right, rt = arm_eef6d(z, "right")
    assert len(left) == len(right) == len(z["cam_head_frame"]), (len(left), len(right))
    L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
    R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
    ref = (lt + rt) / 2.0; rows, S, A = [], [], []
    for idx in range(0, max(0, len(left) - MARGIN)):
        cur = ref[idx]; q = np.linspace(cur, min(cur + QDUR, float(ref.max())), NUM_ACTIONS + 1, dtype=np.float32)
        ls, rs = L(q), R(q)
        if np.abs(ls[1] - ls[0]).max() < 1e-5 and np.abs(rs[1] - rs[0]).max() < 1e-5: continue
        seq = np.concatenate([ls, rs], -1).astype(np.float32); rows.append(idx); S.append(seq[0]); A.append(seq[1:])
    return np.asarray(rows, np.int64), np.stack(S), np.stack(A), ref


def decode_rows(path, want):
    import av
    out = {}; want = set(int(x) for x in want)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want: out[i] = fr.to_ndarray(format="rgb24"); assert out[i].shape == (240, 320, 3), out[i].shape
            if i >= max(want): break
    return out


def write_shard(args):
    raw_dir, shard_root = map(pathlib.Path, args)
    mk = shard_root.with_suffix(".complete")
    if mk.exists(): return json.load(open(mk))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    shutil.rmtree(shard_root, ignore_errors=True); shard_root.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    meta = json.load(open(raw_dir / "raw_episode.json")); rows, S, A, ref = labels(raw_dir)
    imgs = {cam: decode_rows(raw_dir / f"{cam}.mp4", rows) for cam, _ in CAMS}
    ds = LeRobotDataset.create(repo_id=f"local/{shard_root.name}", fps=30, features=features(), root=str(shard_root), robot_type="agilex_aloha",
                               use_videos=True, image_writer_threads=4)
    for j, r in enumerate(rows):
        fr = {key: imgs[cam][int(r)] for cam, key in CAMS}
        fr.update({"observation.state": S[j], "action": A[j], "aux.src_index": np.array([r], np.float32),
                   "aux.ref_time_s": np.array([ref[r] - ref[0]], np.float32), "task": meta["instruction"]})
        ds.add_frame(fr)
    ds.save_episode(parallel_encoding=False); ds.finalize()
    rec = dict(episode_id=meta["episode_id"], era=meta["era"], frames=int(len(rows)), raw_frames=int(len(ref)),
               grip_closed_frac=float(A[:, 0, [9, 19]].mean()), seconds=round(time.time() - t0, 1), written=time.strftime("%F %T"))
    json.dump(rec, open(mk, "w")); return rec


def fix_stats(root):
    import glob, pyarrow as pa, pyarrow.parquet as pq
    from ego_cart20.scripts.export_lerobot import stats
    col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
    S, A = [], []
    for f in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["observation.state", "action"])
        S.append(np.asarray(col(t, "observation.state").to_pylist(), np.float32)); A.append(np.asarray(col(t, "action").to_pylist(), np.float32))
    S, A = np.concatenate(S), np.concatenate(A)
    assert A.shape[1:] == (NUM_ACTIONS, 20) and np.isfinite(S).all() and np.isfinite(A).all()
    st = json.load(open(root / "meta/stats.json"))
    st["observation.state"] = {k: v.tolist() for k, v in stats(S).items()}; st["action"] = {k: v.tolist() for k, v in stats(A.reshape(-1, 20)).items()}
    json.dump(st, open(root / "meta/stats.json", "w"), indent=4)
    return dict(rows=int(len(S)), grip_closed_frac=[round(float(A[..., 9].mean()), 3), round(float(A[..., 19].mean()), 3)],
                pos_range_m=[[round(float(x), 3) for x in A.reshape(-1, 20)[:, i].min(0, keepdims=True).tolist() + A.reshape(-1, 20)[:, i].max(0, keepdims=True).tolist()] for i in (0, 1, 2)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("raw_root"); ap.add_argument("out"); ap.add_argument("--name", default="softfold_abs30_v1")
    ap.add_argument("--val-folders", nargs="*", default=DEFAULT_VAL); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    RAW, OUT = pathlib.Path(a.raw_root), pathlib.Path(a.out); SH = OUT / f"{a.name}_shards"; report = {}
    import os
    if os.environ.get("AGG_PATCH", "1") == "1": sys.path.insert(0, str(AGG)); import agg_patch; agg_patch.install()
    from lerobot.datasets.aggregate import aggregate_datasets
    from concurrent.futures import ProcessPoolExecutor
    raws = sorted(p for p in RAW.iterdir() if (p / "raw_episode.json").exists())
    if a.limit: raws = raws[:a.limit]
    era = {p.name: json.load(open(p / "raw_episode.json"))["era"] for p in raws}
    split = {s: [p for p in raws if (era[p.name] in set(a.val_folders)) == (s == "val")] for s in ("train", "val")}
    for s, eps in split.items():
        if not eps: print(f"{s}: empty, skipped"); continue
        jobs = [(str(p), str(SH / s / p.name)) for p in eps]; recs = []
        with ProcessPoolExecutor(a.workers) as ex:
            for rec in ex.map(write_shard, jobs): recs.append(rec); print(s, json.dumps(rec), flush=True)
        jobs = [(r, sr) for (r, sr), rec in zip(jobs, recs) if rec["frames"] > 0]
        dst = OUT / f"{a.name}_{s}"; tmp = OUT / f"{a.name}_{s}.tmp"; shutil.rmtree(tmp, ignore_errors=True)
        aggregate_datasets(repo_ids=[f"local/{pathlib.Path(sr).name}" for _, sr in jobs], aggr_repo_id=f"local/{a.name}_{s}",
                           roots=[pathlib.Path(sr) for _, sr in jobs], aggr_root=tmp)
        report[s] = dict(episodes=len(jobs), **fix_stats(tmp))
        assert not dst.exists(), f"{dst} exists"; tmp.rename(dst); print(s, report[s], flush=True)
        json.dump(dict(schema="softfold_xvla_abs30/v1", raw_root=str(RAW.resolve()), split=s, val_folders=a.val_folders, report=report[s],
                       exporter_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
                       contract="X-VLA AIR-AGILEX-HQ: abs eef_6d (Agilex base frame, rot6d = R[:, :2] interleaved), grip = width*50<1 "
                                "binarised then linear interp, 30 actions over 2.0 s, state = q0; no normalisation"),
                  open(dst / "EXPORT.json", "w"), indent=1)
    print("EXPORT DONE", json.dumps(report))
