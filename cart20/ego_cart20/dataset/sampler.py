"""Sample weights (spec 19/21): order balancing x stage weighting, as weights only -- files are never duplicated and labels
never changed.  stage -1 (unknown) gets weight 1.0.
NOTE: the X-VLA REL-only recipe trains through lerobot_train (uniform sampling over LeRobot frames); these weights are for
the native dataset class / a custom sampler (experiment D) and are not used by the default pretrain."""
import collections

import numpy as np


def sample_weights(orders, stages, order_balanced=True, stage_weights=(1.0, 1.0, 1.0, 2.0)):
    orders = np.asarray(orders); stages = np.asarray(stages, int); w = np.ones(len(orders))
    if order_balanced:
        c = collections.Counter(orders.tolist()); w *= np.array([len(orders) / (len(c) * c[o]) for o in orders])
    sw = np.array([stage_weights[s] if 0 <= s < len(stage_weights) else 1.0 for s in stages])
    w *= sw
    return w / w.sum()


class WeightedSampler:
    """infinite with-replacement index stream (numpy rng; torch-free).  torch: torch.utils.data.WeightedRandomSampler(w, n)"""
    def __init__(self, weights, seed=0):
        self.w = np.asarray(weights); self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            yield from self.rng.choice(len(self.w), size=4096, p=self.w).tolist()
