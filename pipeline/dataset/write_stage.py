"""[2026-09-25] C-old dataset WRITE stage (no JAX in this process): cached rows (c8old_rows/*.npz) -> LeRobot c8old_train / c8old_val.
GRIPPER [2026-09-26, gripper contract gate, ~/c8/c8old_runs/gripper_gate]: mapping A = the cached V3 values unchanged:
state raw = -270 * open_fraction, action = raw / -6 = 45 * open_fraction. R150 wrist video (both arms, ep 10/20/81/100) shows
raw 0 = fingers touching (rest, CLOSED), -270 = fully OPEN, cube held at -85..-151, and cmd >= 27 = OPENING command, so -270 * f
(f = 1 open) has the B1 physical semantics. The 25 Sep "B1 convention" raw = -270 * (1 - f) inverted it (it read the
humanik_delta comment "1 = closed" instead of the video) and is REMOVED.

[2026-09-26] RESUMABLE. LeRobot v3 concatenates all episodes into shared file-NNN parquet/mp4 whose parquet footer is only written at
finalize(), so a killed single-stream write leaves the whole split unreadable (the 25 Sep run: 54 episodes, both parquets footerless).
Now each source npz is written as its own small finalized LeRobot dataset (shard) under c8old_shards/<split>/<stem>/, verified
(parquet readable, rows == 65 * segments, every episode 65 rows, each camera's packet count == rows) and only then marked with
<stem>.complete. On restart a shard with a valid marker is SKIP_COMPLETE, anything else is deleted and REBUILT. When every shard is
complete the shards are aggregated (remux / concat only, no re-encode) into <split>.tmp, spot-checked frame-exact against the shards,
and atomically renamed to <split>. Codec / encoder settings unchanged."""
import json, pathlib, shutil, time
import numpy as np, cv2, av
H = pathlib.Path.home(); C8 = H / "c8"; OUT = C8 / "c8old_data"; ROWS = C8 / "c8old_rows"; SHARDS = C8 / "c8old_shards"
GRIP = [6, 13]; GRIP_MAPPING = "A: state raw = -270 * open_fraction, action = raw / -6 (R150 video-verified 2026-09-26)"
CAMS = ("observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist")


def decode(path, want):
    out = {}; want = set(int(x) for x in want)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want: out[i] = cv2.resize(fr.to_ndarray(format="rgb24"), (640, 480), interpolation=cv2.INTER_AREA)
            if i > max(want): break
    return out


def features():
    v = lambda: {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    return {"observation.images.global": v(), "observation.images.left_wrist": v(), "observation.images.right_wrist": v(),
            "observation.state": {"dtype": "float32", "shape": (14,), "names": None}, "action": {"dtype": "float32", "shape": (14,), "names": None},
            "observation.ee.tcp_tgt": {"dtype": "float32", "shape": (32,), "names": None}}


def video_packets(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]; return sum(1 for p in c.demux(s) if p.size)


def verify_root(root, n_seg):
    """Integrity of one finalized LeRobot v3 root holding n_seg 65-frame episodes. Returns None if OK, else the reason."""
    import pyarrow.parquet as pq
    info = json.load(open(root / "meta/info.json"))
    if info["total_episodes"] != n_seg or info["total_frames"] != 65 * n_seg: return f"info {info['total_episodes']} ep / {info['total_frames']} frames"
    try:
        dt = pq.read_table(sorted((root / "data").rglob("*.parquet"))); mt = pq.read_table(sorted((root / "meta/episodes").rglob("*.parquet")))
    except Exception as ex: return f"parquet unreadable: {ex}"
    if dt.num_rows != 65 * n_seg: return f"data rows {dt.num_rows}"
    lens = np.bincount(dt["episode_index"].to_numpy())
    if len(lens) != n_seg or set(lens.tolist()) != {65}: return f"episode lengths {sorted(set(lens.tolist()))}"
    if mt.num_rows != n_seg: return f"meta episodes {mt.num_rows}"
    for k in CAMS:
        files = sorted((root / "videos" / k).rglob("*.mp4"))
        if not files or any(f.stat().st_size == 0 for f in files): return f"{k} video missing/empty"
        n = sum(video_packets(f) for f in files)
        if n != 65 * n_seg: return f"{k} packets {n} != {65 * n_seg}"
    return None


def write_shard(f, split, root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    z = np.load(f, allow_pickle=False)
    ds = LeRobotDataset.create(repo_id=f"local/{split}_{f.stem}", fps=30, features=features(), root=str(root), robot_type="rebot_b601", use_videos=True)
    epd = pathlib.Path(str(z["epd"])); instr = str(z["instr"])
    imL = decode(epd / "left_wrist.mp4", z["vfL"].ravel()); imR = decode(epd / "right_wrist.mp4", z["vfR"].ravel()); imH = decode(epd / "head.mp4", z["vfH"].ravel())
    st_all = z["st"]; ac_all = z["ac"]
    assert np.abs(ac_all[..., GRIP] - st_all[..., GRIP] / -6.0).max() < 1e-4, "cached gripper action != state / -6"
    for j in range(len(z["starts"])):
        for t in range(65):
            ds.add_frame({"observation.images.global": imH[int(z["vfH"][j, t])], "observation.images.left_wrist": imL[int(z["vfL"][j, t])],
                          "observation.images.right_wrist": imR[int(z["vfR"][j, t])], "observation.state": st_all[j, t], "action": ac_all[j, t],
                          "observation.ee.tcp_tgt": z["tcp_tgt"][j, t], "task": instr})
        ds.save_episode()
    ds.finalize()
    return len(z["starts"])


def spot_check_frames(aggr_root, split, shard_list, n=12):
    """Decode random (episode, frame) pairs from the aggregated split and from its source shard; they must be pixel-identical
    (aggregation is a remux, so any difference means a timestamp / episode-offset bug)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    agg = LeRobotDataset(f"local/{split}", root=str(aggr_root), video_backend="pyav")
    offs = np.cumsum([0] + [k for _, _, k in shard_list]); rng = np.random.default_rng(0); worst = 0.0
    for g in rng.choice(offs[-1], min(n, offs[-1]), replace=False):
        s = int(np.searchsorted(offs, g, side="right") - 1); stem, root, _ = shard_list[s]; e_local = int(g - offs[s]); t = int(rng.integers(65))
        sh = LeRobotDataset(f"local/{split}_{stem}", root=str(root), video_backend="pyav")
        a = agg[int(g) * 65 + t]; b = sh[e_local * 65 + t]
        if int(a["episode_index"]) != int(g) or int(b["episode_index"]) != e_local: return f"episode index mismatch at global ep {g}"
        for k in CAMS + ("observation.state", "action", "observation.ee.tcp_tgt"): worst = max(worst, float((a[k] - b[k]).abs().max()))
    return None if worst == 0.0 else f"aggregated vs shard max|d| {worst}"


def main():
    from lerobot.datasets.aggregate import aggregate_datasets
    idx = json.load(open(ROWS / "index.json")); report = {}
    # the pre-resume single-stream output (footerless parquet) is kept aside, never trusted
    for split in ("c8old_train", "c8old_val"):
        old = OUT / split
        if old.exists() and not (OUT / f"{split}.complete").exists():
            dst = OUT / f"{split}.interrupted_{time.strftime('%Y%m%d_%H%M%S')}"; old.rename(dst); print(f"MOVED_ASIDE {old.name} -> {dst.name}", flush=True)
    for split in ("c8old_train", "c8old_val"):
        sdir = SHARDS / split; sdir.mkdir(parents=True, exist_ok=True); shard_list = []; n_skip = n_build = 0
        for f in sorted(ROWS.glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            if bool(z["val"]) != (split == "c8old_val"): continue
            n_seg = len(z["starts"]); root = sdir / f.stem; marker = sdir / f"{f.stem}.complete"
            if marker.exists() and root.exists() and json.load(open(marker)).get("segments") == n_seg and json.load(open(marker)).get("gripper") == GRIP_MAPPING and verify_root(root, n_seg) is None:
                n_skip += 1; print(f"SKIP_COMPLETE {split} {f.stem} ({n_seg} segments)", flush=True)
            else:
                marker.unlink(missing_ok=True); shutil.rmtree(root, ignore_errors=True); t0 = time.time()
                write_shard(f, split, root); why = verify_root(root, n_seg)
                if why: raise SystemExit(f"SHARD_VERIFY_FAIL {split} {f.stem}: {why}")
                json.dump(dict(segments=n_seg, rows=65 * n_seg, npz=f.name, gripper=GRIP_MAPPING, written=time.strftime("%F %T")), open(marker.with_suffix(".tmp"), "w"))
                marker.with_suffix(".tmp").rename(marker); n_build += 1
                print(f"REBUILD {split} {f.stem}: {n_seg} segments in {time.time() - t0:.1f}s", flush=True)
            shard_list.append((f.stem, root, n_seg))
        n = sum(k for _, _, k in shard_list)
        print(f"  {split}: {len(shard_list)} shards complete ({n_skip} skipped, {n_build} built), {n} segments -> aggregating", flush=True)
        tmp = OUT / f"{split}.tmp"; shutil.rmtree(tmp, ignore_errors=True)
        aggregate_datasets(repo_ids=[f"local/{split}_{s}" for s, _, _ in shard_list], aggr_repo_id=f"local/{split}", roots=[r for _, r, _ in shard_list], aggr_root=tmp)
        why = verify_root(tmp, n) or spot_check_frames(tmp, split, shard_list)
        if why: raise SystemExit(f"AGGREGATE_VERIFY_FAIL {split}: {why}")
        shutil.rmtree(OUT / split, ignore_errors=True); tmp.rename(OUT / split)
        json.dump(dict(segments=n, shards=len(shard_list), gripper=GRIP_MAPPING, finished=time.strftime("%F %T")), open(OUT / f"{split}.complete", "w"))
        print(f"  {split}: aggregated + verified ({n} episodes, {65 * n} rows)", flush=True)
        report[split] = dict(segments=n, rows=65 * n, chunk_starts=31 * n)
    json.dump(dict(schema="track_c_c_old_dataset/v1", contract="TRACK_C_PSEUDO_JOINT_CONTRACT.md §22b / §24", report=report, val_episodes=idx["val_episodes"],
                   main_target="stored Phase-3 pseudo q", cd_target="observation.ee.tcp_tgt = retargeted measured TCP (no IK) @ diag(C^T, 1)",
                   gripper=GRIP_MAPPING, quarantine="imu_camera_rotation_inconsistent_v1",
                   writer="resumable shard writer (per-npz finalized shards + .complete markers, lerobot aggregate_datasets, frame-exact spot check)"),
              open(OUT / "c8old_provenance.json", "w"), indent=1)
    print(report); return 0


if __name__ == "__main__":
    raise SystemExit(main())
