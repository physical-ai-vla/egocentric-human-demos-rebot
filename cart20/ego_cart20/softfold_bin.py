"""Soft-Fold binary gripper (user 2026-10-08: REL32 + binary gripper + BCE).  Call install() BEFORE convert_dataset.one runs.

X-VLA's AIRAgilexHQHandler rule: jaw width * 50 < 1 (width < 2 cm) -> closed, binarised on the RAW samples before interpolation
(so rows / targets between two raw samples across a switch get the interpolated value in (0, 1), as in X-VLA).
Our polarity is kept (config D2): 0 = closed, 1 = open.  The raw tree stores g = width_m / W_OPEN_M (softfold_export.py).
The state grippers (dims 18, 19) and the action grippers (dims 9, 19) both come from this binarised track."""
import numpy as np

from . import config as _c

W_OPEN_M = 0.0615            # = sources/softfold_export.W_OPEN_M (not imported: that module needs h5py)
CLOSED_BELOW_M = 0.02        # X-VLA: width * 50 < 1


def binarize(g):
    g = np.asarray(g, np.float64)
    return np.where(np.isfinite(g), (g * W_OPEN_M >= CLOSED_BELOW_M).astype(np.float64), g)


def install():
    import ego_cart20.scripts.convert_dataset as CD
    if getattr(CD, "_softfold_bin", False):
        return
    orig = CD.load_raw_episode

    def load(ep_dir):
        raw = orig(ep_dir)
        for arm in raw.arms.values():
            arm.gripper = binarize(arm.gripper)
        raw.meta["gripper_calibration"] = dict(raw.meta.get("gripper_calibration") or {}, binary=dict(
            rule="open = width_m >= 0.02 (X-VLA width*50 < 1 -> closed), binarised on raw samples before interpolation",
            W_OPEN_M=W_OPEN_M, CLOSED_BELOW_M=CLOSED_BELOW_M))
        return raw
    CD.load_raw_episode = load
    CD._softfold_bin = True
    _c.CONTRACT.update(gripper_representation="binary 0 = closed / 1 = open (width >= 2 cm, X-VLA rule) on raw samples, "
                       "linearly interpolated to rows/targets; trained with BCE on action dims 9, 19")
