#!/usr/bin/env python3
"""raw ego episodes -> processed ego_cart20_v2 tree (spec 23): episodes/<id>/..., metadata.json, {train,val,test}_manifest.jsonl.
usage: convert_dataset.py <raw_root> <out_root> [--split frozen_handumi|hash] [--workers N]"""
import argparse, collections, hashlib, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.config import CONTRACT, Cart20Config
from ego_cart20.convert import convert_episode
from ego_cart20.dataset.splits import frozen_handumi_split, hash_split
from ego_cart20.io.processed_writer import write_episode
from ego_cart20.io.raw_episode_loader import load_raw_episode


def one(args):
    raw_dir, out_root, cfg = args
    try:
        raw = load_raw_episode(raw_dir); ep = convert_episode(raw, cfg)
        write_episode(pathlib.Path(out_root) / "episodes" / raw.episode_id, ep); m = ep["metadata"]
        return dict(episode_id=raw.episode_id, status="ok", stack_order=m["stack_order"], instruction=m["instruction"],
                    frames_rows=m["frames_rows"], rows_valid=m["rows_valid"], train_rows=m["train_rows"], era=raw.meta.get("era"),
                    jump_events=m.get("jump_events"))
    except Exception as ex:
        return dict(episode_id=pathlib.Path(raw_dir).name, status="error", reason=f"{type(ex).__name__}: {ex}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("raw_root"); ap.add_argument("out_root")
    ap.add_argument("--split", default="frozen_handumi"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--discontinuity-filter", action="store_true", help="v2b: drop raw MASt3R silent jumps (pose_filter.discontinuity_filter)")
    ap.add_argument("--jump-window", default="frame", choices=["frame", "row"]); ap.add_argument("--jump-isolated-step-m", type=float, default=None); a = ap.parse_args()
    CFG = Cart20Config(discontinuity_filter=a.discontinuity_filter, jump_window=a.jump_window, jump_isolated_step_m=a.jump_isolated_step_m)
    out = pathlib.Path(a.out_root); assert not out.exists(), f"{out} exists (processed trees are write-once)"; (out / "episodes").mkdir(parents=True)
    raws = sorted(p for p in pathlib.Path(a.raw_root).iterdir() if (p / "raw_episode.json").exists())
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex: res = list(ex.map(one, [(str(r), str(out), CFG) for r in raws]))
    ok = [r for r in res if r["status"] == "ok" and r["train_rows"] > 0]
    sp, rule = (frozen_handumi_split if a.split == "frozen_handumi" else hash_split)([r["episode_id"] for r in ok])
    for s in ("train", "val", "test"):
        with open(out / f"{s}_manifest.jsonl", "w") as f:
            for r in ok:
                if sp[r["episode_id"]] == s: f.write(json.dumps(dict(r, split=s, path=f"episodes/{r['episode_id']}")) + "\n")
    cnt = {s: dict(episodes=sum(sp[r["episode_id"]] == s for r in ok), train_rows=sum(r["train_rows"] for r in ok if sp[r["episode_id"]] == s),
                   orders=dict(collections.Counter(r["stack_order"] for r in ok if sp[r["episode_id"]] == s))) for s in ("train", "val", "test")}
    meta = dict(schema="ego_cart20_v2/v1", created=time.strftime("%F %T"), raw_root=str(pathlib.Path(a.raw_root).resolve()), contract=CONTRACT,
                config=CFG.to_dict(), split=rule, counts=cnt, conversion=res,
                code_sha256={str(p.relative_to(pathlib.Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                             for p in sorted(pathlib.Path(__file__).resolve().parents[1].rglob("*.py"))})
    json.dump(meta, open(out / "metadata.json", "w"), indent=1)
    print("DONE", json.dumps(cnt), collections.Counter(r["status"] for r in res))
