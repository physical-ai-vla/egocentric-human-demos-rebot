#!/usr/bin/env python3
"""Soft-Fold raw episodes -> processed CART20 tree.  Same pipeline as convert_dataset.py; only the source-specific config
differs: max_raw_gap_s 0.10 (Agilex logs ~35 ms with dropped samples up to ~70 ms; ego default 0.05 would cut rows) and a
FOLDER-level split (val = whole recording sessions, never frames/episodes of a train session).
usage: convert_dataset_softfold.py <raw_root> <out_root> [--val-folders F ...] [--workers N]"""
import argparse, collections, hashlib, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import ego_cart20.softfold_h32  # noqa: F401  (chunk 32; must precede the label imports)
from ego_cart20.config import CONTRACT, Cart20Config
from ego_cart20.scripts.convert_dataset import one

DEFAULT_VAL = ["0706_17pm_stage_1_stage2new_new_cam_very_slow", "0808_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("raw_root"); ap.add_argument("out_root")
    ap.add_argument("--val-folders", nargs="*", default=DEFAULT_VAL); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    CFG = Cart20Config(max_raw_gap_s=0.10, horizon=32)
    out = pathlib.Path(a.out_root); assert not out.exists(), f"{out} exists (processed trees are write-once)"; (out / "episodes").mkdir(parents=True)
    raws = sorted(p for p in pathlib.Path(a.raw_root).iterdir() if (p / "raw_episode.json").exists())
    era = {p.name: json.load(open(p / "raw_episode.json"))["era"] for p in raws}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex: res = list(ex.map(one, [(str(r), str(out), CFG) for r in raws]))
    ok = [r for r in res if r["status"] == "ok" and r["train_rows"] > 0]
    sp = {r["episode_id"]: ("val" if era[r["episode_id"]] in set(a.val_folders) else "train") for r in ok}
    for s in ("train", "val", "test"):
        with open(out / f"{s}_manifest.jsonl", "w") as f:
            for r in ok:
                if sp[r["episode_id"]] == s: f.write(json.dumps(dict(r, split=s, path=f"episodes/{r['episode_id']}")) + "\n")
    cnt = {s: dict(episodes=sum(sp[r["episode_id"]] == s for r in ok), train_rows=sum(r["train_rows"] for r in ok if sp[r["episode_id"]] == s))
           for s in ("train", "val", "test")}
    meta = dict(schema="softfold_cart20_h32/v1", created=time.strftime("%F %T"), raw_root=str(pathlib.Path(a.raw_root).resolve()), contract=CONTRACT,
                config=CFG.to_dict(), split=dict(rule="folder-level", val_folders=a.val_folders), counts=cnt, conversion=res,
                code_sha256={str(p.relative_to(pathlib.Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                             for p in sorted(pathlib.Path(__file__).resolve().parents[1].rglob("*.py"))})
    json.dump(meta, open(out / "metadata.json", "w"), indent=1)
    print("DONE", json.dumps(cnt), collections.Counter(r["status"] for r in res))
