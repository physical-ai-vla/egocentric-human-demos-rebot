"""Native (non-LeRobot) reader of the processed ego_cart20_v2 tree.  One sample = one TRAINING row.
Returns state (20, RELCART20) / action (16, 32) / cart20 (16, 20) / state_prevrel / prev_valid / instruction / order / stage,
and optionally images (decoded from the source mp4 at the row's frame index, full frame resized to `image_size`).
Shape asserts of spec 27 run on every item."""
import json
import pathlib

import numpy as np

from ..io.processed_writer import load_episode
from .sampler import sample_weights


class XVLAEgoDataset:
    def __init__(self, root, split="train", load_images=False, image_size=224):
        self.root = pathlib.Path(root); self.load_images = load_images; self.size = image_size
        self.man = [json.loads(l) for l in open(self.root / f"{split}_manifest.jsonl")]
        self.eps = [load_episode(self.root / r["path"]) for r in self.man]
        self.index = [(e, j) for e, ep in enumerate(self.eps) for j in range(len(ep["train_rows"]))]

    def __len__(self):
        return len(self.index)

    def weights(self, order_balanced=True, stage_weights=(1.0, 1.0, 1.0, 2.0)):
        orders = [self.eps[e]["metadata"]["stack_order"] for e, _ in self.index]
        stages = [int(self.eps[e]["stage_id"][self.eps[e]["train_rows"][j]]) for e, j in self.index]
        return sample_weights(orders, stages, order_balanced, stage_weights)

    def __getitem__(self, i):
        e, j = self.index[i]; ep = self.eps[e]; r = int(ep["train_rows"][j]); m = ep["metadata"]
        x = dict(state=np.asarray(ep["state"][r], np.float32), action=np.asarray(ep["action"][j], np.float32),
                 cart20=np.asarray(ep["cart20"][j], np.float32), state_prevrel=np.asarray(ep["state_prevrel"][r], np.float32),
                 state_prev_valid=np.float32(ep["state_prev_valid"][r]), instruction=m["instruction"], stack_order=m["stack_order"],
                 stage_id=int(ep["stage_id"][r]), camera_valid=np.asarray(ep["camera_valid"][r]), episode_id=m["episode_id"], row=r)
        assert x["state"].shape == (20,) and x["cart20"].shape == (16, 20) and x["action"].shape == (16, 32)
        assert np.array_equal(x["action"][:, :20], x["cart20"]) and np.count_nonzero(x["action"][:, 20:]) == 0
        if self.load_images:
            x["images"] = {c: self._frame(ep["cameras"][c], r) for c in m["camera_order"] if c in ep["cameras"]}
        return x

    def _frame(self, cam, r):
        import av, cv2
        f = int(cam["frame_index"][r])
        if f < 0: return None
        with av.open(cam["video"]) as c:
            for k, fr in enumerate(c.decode(video=0)):
                if k == f: return cv2.resize(fr.to_ndarray(format="rgb24"), (self.size, self.size), interpolation=cv2.INTER_AREA)
