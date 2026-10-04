"""Episode-level splits (spec 22).  An episode belongs to exactly one split; frames are never split.

``frozen_handumi_split``: the split already shared by every ego dataset (C-old v1/v2, REL16 ego, gcal1):
    val = the frozen C-old v1 26 old259 source episodes  +  the HRL80 round-block holdout ep49-54 (hrl80_split_frozen.json)
    train = every other included episode;  test = empty (no third split was ever frozen; adding one would change train)
``hash_split``: deterministic sha1(episode_id) buckets for other sources.
"""
import hashlib
import json
import pathlib

H = pathlib.Path.home()


def frozen_handumi_split(episode_ids):
    v1 = set(json.load(open(H / "c8/c8old_rows/index.json"))["val_episodes"])
    hrl = json.load(open(H / "umi_bridge/track_c/v2k/hrl80_split_frozen.json"))
    val = v1 | set(hrl["val_source_episodes"])
    sp = {e: ("val" if e in val else "train") for e in episode_ids}
    return sp, dict(rule="frozen: val = C-old v1 26 (c8old_rows/index.json) + HRL80 round-block ep49-54; test empty",
                    hrl80_split_sha256=hrl.get("sha256"))


def hash_split(episode_ids, val_frac=0.1, test_frac=0.0):
    sp = {}
    for e in episode_ids:
        u = int(hashlib.sha1(e.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        sp[e] = "test" if u < test_frac else ("val" if u < test_frac + val_frac else "train")
    return sp, dict(rule=f"sha1 bucket val {val_frac} test {test_frac}")


def check_disjoint(split):
    seen = {}
    for e, s in split.items():
        assert seen.setdefault(e, s) == s
    return True
