"""Processed episode writer (spec 23).  One directory per episode; arrays + metadata.json; camera folders hold the
row -> source-frame map (and optionally a 224x224 row-rate mp4)."""
import hashlib
import json
import pathlib

import numpy as np

ARRAYS = ("timestamps", "row_valid", "train_rows", "state", "state_prevrel", "state_prev_valid", "cart20", "action",
          "left_pose", "right_pose", "left_gripper", "right_gripper", "stage_id", "camera_valid")


def write_episode(out_dir, ep, overwrite=False):
    out = pathlib.Path(out_dir)
    if out.exists() and not overwrite: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); sha = {}
    for k in ARRAYS:
        np.save(out / f"{k}.npy", ep[k]); sha[f"{k}.npy"] = hashlib.sha256((out / f"{k}.npy").read_bytes()).hexdigest()[:16]
    for cam, info in ep["cameras"].items():
        d = out / cam; d.mkdir(exist_ok=True)
        np.save(d / "frame_index.npy", info["frame_index"])
        json.dump({k: v for k, v in info.items() if k != "frame_index"}, open(d / "source.json", "w"), indent=1)
    meta = dict(ep["metadata"]); meta["array_sha256_16"] = sha
    json.dump(meta, open(out / "metadata.json", "w"), indent=1)
    return out


def load_episode(ep_dir, mmap=True):
    ep_dir = pathlib.Path(ep_dir); mode = "r" if mmap else None
    d = {k: np.load(ep_dir / f"{k}.npy", mmap_mode=mode) for k in ARRAYS}
    d["metadata"] = json.load(open(ep_dir / "metadata.json")); d["cameras"] = {}
    for cam in d["metadata"].get("cameras", []):
        d["cameras"][cam] = dict(frame_index=np.load(ep_dir / cam / "frame_index.npy"), **json.load(open(ep_dir / cam / "source.json")))
    return d


def encode_row_video(src_mp4, frame_index, dst_mp4, size=224, fps=15):
    """decode the source mp4 sequentially and write the selected frames (row order) as a size x size H.264 mp4;
    rows whose frame_index is -1 get a black frame (marked invalid in camera_valid)"""
    import av
    import numpy as _np
    want = {}
    for r, f in enumerate(frame_index):
        if f >= 0: want.setdefault(int(f), []).append(r)
    rows = [None] * len(frame_index)
    with av.open(str(src_mp4)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                img = fr.to_image().resize((size, size)); a = _np.asarray(img)
                for r in want[i]: rows[r] = a
            if i > max(want, default=-1): break
    blank = _np.zeros((size, size, 3), _np.uint8)
    with av.open(str(dst_mp4), "w") as o:
        s = o.add_stream("libx264", rate=int(fps)); s.width = s.height = size; s.pix_fmt = "yuv420p"; s.options = {"crf": "18"}
        for a in rows:
            for p in s.encode(av.VideoFrame.from_ndarray(blank if a is None else a, format="rgb24")): o.mux(p)
        for p in s.encode(): o.mux(p)
