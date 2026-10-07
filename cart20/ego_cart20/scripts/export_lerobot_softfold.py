#!/usr/bin/env python3
"""COPY of export_lerobot.py for Soft-Fold (2026-10-07).  Only change: images are stored 4:3 at 320x240 (the raw softfold
videos' size) instead of squashed 224x224, so the X-VLA processor pads them to 224 like the reBot 640x480 cameras.
Chunk 32 (user 2026-10-07): action (32, 32).  Everything else (columns, stats, shards, aggregation) is identical.
processed ego_cart20_v2 -> LeRobot v3 datasets <out>/<name>_{train,val} for the X-VLA REL-only D6 recipe
(umi_bridge/RELCART20_RELONLY_D6_RECIPE.md section 9).

Per LeRobot episode = one processed ego episode; frames = its TRAINING rows only (rows with all 16 real targets and a head
frame), in row order.  Columns:
  observation.images.{global,left_wrist,right_wrist}   head / left wrist / right wrist, full frame INTER_AREA 224x224
                                                        (the existing ego REL16 exporter policy); a missing wrist frame is black
  observation.state (20)       state.npy = RELCART20 task anchor [L9 | R9 | gL gR]       (default; NEVER state_prevrel)
  action (16, 32)              action.npy = CART20 | AUX12 = 0
  state_prev_valid (1)
  aux.q_t (12)                 NaN: required by preflight_rel16v2 for 32-dim actions, unused by rel16_aux_relonly; NaN (not a
                               pseudo q) so any dq / FK plugin pointed at this data fails loudly instead of training on it
  aux.has_dq_supervision (1)   0
  aux.state_prevrel (20)       diagnostic copy (aux.* keys are not policy features)
  aux.stage_id (1), aux.row (1), aux.time_s (1), aux.camera_valid (3)
meta/stats.json: observation.state / action recomputed over all exported rows (action over N*16 rows per channel); AUX12
channels mean 0 / std 1 (the gcal1 placeholder convention, never std 0); aux.q_t stats zeroed.
Resumable: one shard per episode (+ .complete marker), then lerobot aggregate_datasets with the pyarrow agg_patch.
usage (env ~/xvla-mac/bin/python): export_lerobot.py <processed_root> <out_root> [--name ego_cart20_v2] [--workers 6]
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
CAMS = (("head", "observation.images.global"), ("left_wrist", "observation.images.left_wrist"), ("right_wrist", "observation.images.right_wrist"))
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def features():
    v = lambda: {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
    f = lambda *s: {"dtype": "float32", "shape": s, "names": None}
    return {"observation.images.global": v(), "observation.images.left_wrist": v(), "observation.images.right_wrist": v(),
            "observation.state": f(20), "action": f(32, 32), "state_prev_valid": f(1), "aux.q_t": f(12), "aux.has_dq_supervision": f(1),
            "aux.state_prevrel": f(20), "aux.stage_id": f(1), "aux.row": f(1), "aux.time_s": f(1), "aux.camera_valid": f(3)}


def decode224(path, want):
    import av, cv2
    out = {}; want = set(int(x) for x in want if x >= 0)
    if not want: return out
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want: out[i] = fr.to_ndarray(format="rgb24"); assert out[i].shape == (240, 320, 3), out[i].shape
            if i >= max(want): break
    return out


def write_shard(args):
    ep_dir, shard_root = map(pathlib.Path, args)
    mk = shard_root.with_suffix(".complete")
    if mk.exists(): return json.load(open(mk))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ego_cart20.io.processed_writer import load_episode
    shutil.rmtree(shard_root, ignore_errors=True); shard_root.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    ep = load_episode(ep_dir, mmap=False); m = ep["metadata"]; rows = ep["train_rows"]
    assert np.count_nonzero(ep["action"][..., 20:]) == 0 and np.isfinite(ep["state"][rows]).all() and np.isfinite(ep["action"]).all()
    imgs, fidx = {}, {}
    for cam, _ in CAMS:
        if cam in ep["cameras"]:
            fidx[cam] = ep["cameras"][cam]["frame_index"][rows]; imgs[cam] = decode224(ep["cameras"][cam]["video"], fidx[cam])
    blank = np.zeros((240, 320, 3), np.uint8)
    ds = LeRobotDataset.create(repo_id=f"local/{shard_root.name}", fps=15, features=features(), root=str(shard_root), robot_type="agilex_aloha",
                               use_videos=True, image_writer_threads=4)
    nan12 = np.full(12, np.nan, np.float32)
    for j, r in enumerate(rows):
        fr = {}
        for cam, key in CAMS:
            f = int(fidx[cam][j]) if cam in fidx else -1
            fr[key] = imgs[cam][f] if f >= 0 else blank
        assert fr["observation.images.global"] is not blank, "training row without head frame"
        fr.update({"observation.state": ep["state"][r].astype(np.float32), "action": ep["action"][j].astype(np.float32),
                   "state_prev_valid": np.array([ep["state_prev_valid"][r]], np.float32), "aux.q_t": nan12,
                   "aux.has_dq_supervision": np.zeros(1, np.float32), "aux.state_prevrel": ep["state_prevrel"][r].astype(np.float32),
                   "aux.stage_id": np.array([ep["stage_id"][r]], np.float32), "aux.row": np.array([r], np.float32),
                   "aux.time_s": np.array([ep["timestamps"][r] - ep["timestamps"][0]], np.float32),
                   "aux.camera_valid": ep["camera_valid"][r].astype(np.float32), "task": m["instruction"]})
        ds.add_frame(fr)
    ds.save_episode(parallel_encoding=False); ds.finalize()
    rec = dict(episode_id=m["episode_id"], frames=int(len(rows)), seconds=round(time.time() - t0, 1), written=time.strftime("%F %T"))
    json.dump(rec, open(mk, "w")); return rec


def stats(X):
    X = np.asarray(X, np.float64)
    d = dict(min=X.min(0), max=X.max(0), mean=X.mean(0), std=X.std(0), count=np.array([len(X)]))
    for q in (1, 10, 50, 90, 99): d[f"q{q:02d}"] = np.percentile(X, q, axis=0)
    return d


def fix_stats(root):
    """recompute state / action stats from the written parquet; AUX12 mean 0 std 1; aux.q_t zeros"""
    import glob, pyarrow as pa, pyarrow.parquet as pq
    col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
    S, A = [], []
    for f in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["observation.state", "action"])
        S.append(np.asarray(col(t, "observation.state").to_pylist(), np.float32)); A.append(np.asarray(col(t, "action").to_pylist(), np.float32))
    S, A = np.concatenate(S), np.concatenate(A)
    assert A.shape[1:] == (32, 32) and np.count_nonzero(A[..., 20:]) == 0 and np.isfinite(S).all() and np.isfinite(A).all()
    st = json.load(open(root / "meta/stats.json"))
    ss = stats(S); sa = stats(A.reshape(-1, 32)); sa["mean"][20:] = 0.0; sa["std"][20:] = 1.0
    st["observation.state"] = {k: v.tolist() for k, v in ss.items()}; st["action"] = {k: v.tolist() for k, v in sa.items()}
    if "aux.q_t" in st: st["aux.q_t"] = {k: ([0.0] * 12 if k != "count" else st["aux.q_t"]["count"]) for k in st["aux.q_t"]}
    json.dump(st, open(root / "meta/stats.json", "w"), indent=4)
    return dict(rows=int(len(S)), action_std_0_20=[round(float(x), 5) for x in sa["std"][:20]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("processed"); ap.add_argument("out"); ap.add_argument("--name", default="ego_cart20_v2")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--splits", nargs="*", default=["train", "val"]); a = ap.parse_args()
    P, OUT = pathlib.Path(a.processed), pathlib.Path(a.out); SH = OUT / f"{a.name}_shards"; report = {}
    import os
    if os.environ.get("AGG_PATCH", "1") == "1": sys.path.insert(0, str(AGG)); import agg_patch; agg_patch.install()   # Mac lerobot 0.5.2 only
    from lerobot.datasets.aggregate import aggregate_datasets
    from concurrent.futures import ProcessPoolExecutor
    for s in a.splits:
        man = [json.loads(l) for l in open(P / f"{s}_manifest.jsonl")]
        if not man: print(f"{s}: empty, skipped"); continue
        jobs = [(str(P / r["path"]), str(SH / s / r["episode_id"])) for r in man]
        with ProcessPoolExecutor(a.workers) as ex:
            for rec in ex.map(write_shard, jobs): print(s, json.dumps(rec), flush=True)
        dst = OUT / f"{a.name}_{s}"; tmp = OUT / f"{a.name}_{s}.tmp"; shutil.rmtree(tmp, ignore_errors=True)
        aggregate_datasets(repo_ids=[f"local/{pathlib.Path(sr).name}" for _, sr in jobs], aggr_repo_id=f"local/{a.name}_{s}",
                           roots=[pathlib.Path(sr) for _, sr in jobs], aggr_root=tmp)
        report[s] = dict(episodes=len(jobs), **fix_stats(tmp))
        assert not dst.exists(), f"{dst} exists"; tmp.rename(dst); print(s, report[s], flush=True)
        json.dump(dict(schema="softfold_cart20_h32_lerobot/v1", processed=str(P.resolve()), split=s, report=report[s],
                       processed_metadata_sha256=hashlib.sha256((P / "metadata.json").read_bytes()).hexdigest(),
                       exporter_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(), agg_patch_sha256=hashlib.sha256((AGG / "agg_patch.py").read_bytes()).hexdigest(),
                       recipe="umi_bridge/RELCART20_RELONLY_D6_RECIPE.md", state="RELCART20 task anchor", aux12="zero", aux_q_t="NaN placeholder"),
                  open(dst / "EXPORT.json", "w"), indent=1)
    print("EXPORT DONE", json.dumps(report))
