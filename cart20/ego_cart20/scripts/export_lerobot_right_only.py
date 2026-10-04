#!/usr/bin/env python3
"""processed right-only tree (ego_cart20.right_only) -> LeRobot v3 for the REL-only + loss-mask trainer.

Same features as export_lerobot.py (so the model / processor / launcher contract is unchanged) plus
    aux.loss_mask  (20)  1 = supervised action dim (right pose 10:19, right gripper 19 only if it moves), 0 = dummy LEFT
    aux.task_id    (1)   integer task label; the string lives in EXPORT.json (TASK_IDS)
Right wrist image: 100 % robotized with robot100's export_lerobot_robotized.robotize (KB8 -> C922-like pinhole, R = I, so the
labels stay valid; 320x240 + blur/JPEG/photometric, -> 224).  Head: unchanged.  Left wrist: blank (no camera).
Stats: like export_lerobot.fix_stats, and additionally every state / action dim whose std is ~0 (the dummy left arm, a
constant gripper) gets mean = its value, std = 1, so normalisation never divides by ~0. AUX12 mean 0 / std 1 as before.

usage: export_lerobot_right_only.py <processed> <out> [--name ego_hra_red_rightonly_v1] [--workers 6]
"""
import argparse, hashlib, json, pathlib, shutil, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from ego_cart20.scripts import export_lerobot as EL          # noqa: E402  (decode224, stats, AGG, CAMS, agg plumbing)
from ego_cart20.right_only import TASK_ID                    # noqa: E402
from ego_cart20.scripts import export_lerobot_robotized as RX  # noqa: E402  (robot100's robotize(): KB8 -> C922-like pinhole, R = I)
RX.ROBOTIZE_P = 1.0                                           # user 2026-10-03: this task's right wrist is 100 % robotized

TASK_IDS = {TASK_ID: 1}            # 0 is reserved for "bimanual ego v2 / robot" (no column there = all ones mask)


def features():
    f = EL.features(); f["aux.loss_mask"] = {"dtype": "float32", "shape": (20,), "names": None}
    f["aux.task_id"] = {"dtype": "float32", "shape": (1,), "names": None}; return f


def write_shard(args):
    ep_dir, shard_root = map(pathlib.Path, args)
    mk = shard_root.with_suffix(".complete")
    if mk.exists(): return json.load(open(mk))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ego_cart20.io.processed_writer import load_episode
    shutil.rmtree(shard_root, ignore_errors=True); shard_root.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    ep = load_episode(ep_dir, mmap=False); m = ep["metadata"]; rows = ep["train_rows"]; lm = np.load(ep_dir / "loss_mask.npy")
    assert lm.shape == (len(rows), 20) and np.count_nonzero(ep["action"][..., 20:]) == 0 and np.isfinite(ep["state"][rows]).all()
    assert np.isfinite(ep["action"]).all() and m.get("task_id") == TASK_ID
    imgs, fidx = {}, {}
    for cam, _ in EL.CAMS:
        if cam in ep["cameras"]:
            fidx[cam] = ep["cameras"][cam]["frame_index"][rows]
            imgs[cam] = (RX.decode224_wrist(ep["cameras"][cam]["video"], fidx[cam], "right", m["episode_id"], cam)[0] if cam == "right_wrist"
                         else EL.decode224(ep["cameras"][cam]["video"], fidx[cam]))
    blank = np.zeros((224, 224, 3), np.uint8)
    ds = LeRobotDataset.create(repo_id=f"local/{shard_root.name}", fps=15, features=features(), root=str(shard_root), robot_type="rebot_b601",
                               use_videos=True, image_writer_threads=4)
    nan12 = np.full(12, np.nan, np.float32); tid = np.array([TASK_IDS[TASK_ID]], np.float32)
    for j, r in enumerate(rows):
        fr = {}
        for cam, key in EL.CAMS:
            f = int(fidx[cam][j]) if cam in fidx else -1
            fr[key] = imgs[cam][f] if f >= 0 else blank        # left_wrist: no camera on this rig -> blank (as v2 for a missing view)
        assert fr["observation.images.global"] is not blank, "training row without head frame"
        fr.update({"observation.state": ep["state"][r].astype(np.float32), "action": ep["action"][j].astype(np.float32),
                   "state_prev_valid": np.array([ep["state_prev_valid"][r]], np.float32), "aux.q_t": nan12,
                   "aux.has_dq_supervision": np.zeros(1, np.float32), "aux.state_prevrel": ep["state_prevrel"][r].astype(np.float32),
                   "aux.stage_id": np.array([ep["stage_id"][r]], np.float32), "aux.row": np.array([r], np.float32),
                   "aux.time_s": np.array([ep["timestamps"][r] - ep["timestamps"][0]], np.float32),
                   "aux.camera_valid": ep["camera_valid"][r].astype(np.float32), "aux.loss_mask": lm[j].astype(np.float32),
                   "aux.task_id": tid, "task": m["instruction"]})
        ds.add_frame(fr)
    ds.save_episode(parallel_encoding=False); ds.finalize()
    rec = dict(episode_id=m["episode_id"], frames=int(len(rows)), seconds=round(time.time() - t0, 1), written=time.strftime("%F %T"))
    json.dump(rec, open(mk, "w")); return rec


def safe_stats(X, const_tol=1e-6):
    d = EL.stats(X); const = d["std"] < const_tol
    d["std"] = np.where(const, 1.0, d["std"]); return d, const


def fix_stats(root):
    import glob, pyarrow as pa, pyarrow.parquet as pq
    col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
    S, A = [], []
    for f in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["observation.state", "action"])
        S.append(np.asarray(col(t, "observation.state").to_pylist(), np.float32)); A.append(np.asarray(col(t, "action").to_pylist(), np.float32))
    S, A = np.concatenate(S), np.concatenate(A)
    assert A.shape[1:] == (16, 32) and np.count_nonzero(A[..., 20:]) == 0 and np.isfinite(S).all() and np.isfinite(A).all()
    st = json.load(open(root / "meta/stats.json"))
    ss, cs = safe_stats(S); sa, ca = safe_stats(A.reshape(-1, 32)); sa["mean"][20:] = 0.0; sa["std"][20:] = 1.0
    st["observation.state"] = {k: v.tolist() for k, v in ss.items()}; st["action"] = {k: v.tolist() for k, v in sa.items()}
    if "aux.q_t" in st: st["aux.q_t"] = {k: ([0.0] * 12 if k != "count" else st["aux.q_t"]["count"]) for k in st["aux.q_t"]}
    json.dump(st, open(root / "meta/stats.json", "w"), indent=4)
    return dict(rows=int(len(S)), const_state_dims=np.flatnonzero(cs).tolist(), const_action_dims=np.flatnonzero(ca[:20]).tolist(),
                action_std_10_20=[round(float(x), 5) for x in sa["std"][10:20]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("processed"); ap.add_argument("out"); ap.add_argument("--name", default="ego_hra_red_rightonly_v1")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--splits", nargs="*", default=["train", "val"]); a = ap.parse_args()
    P, OUT = pathlib.Path(a.processed), pathlib.Path(a.out); SH = OUT / f"{a.name}_shards"; report = {}
    sys.path.insert(0, str(EL.AGG)); import agg_patch; agg_patch.install()
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
        json.dump(dict(schema="ego_cart20_right_only_lerobot/v1", task_ids=TASK_IDS, processed=str(P.resolve()), split=s, report=report[s],
                       processed_metadata_sha256=hashlib.sha256((P / "metadata.json").read_bytes()).hexdigest(),
                       exporter_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
                       trainer="umi_bridge/rel16_audit/relonly/train_rel16_relonly_lossmask.py (aux.loss_mask honoured)",
                       state="RELCART20 task anchor, LEFT = dummy identity", aux12="zero", aux_q_t="NaN placeholder",
                       wrist_robotize=dict(p=RX.ROBOTIZE_P, cameras=["right_wrist"], model="export_lerobot_robotized.robotize (robot100 recipe)",
                                           head="unchanged full-frame 224", left_wrist="none (blank)"),
                       warning="train ONLY with the loss-mask trainer: the plain REL-only trainer would supervise the dummy left arm"),
                  open(dst / "EXPORT.json", "w"), indent=1)
