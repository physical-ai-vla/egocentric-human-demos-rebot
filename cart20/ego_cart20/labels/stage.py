"""Stacking-stage labels (spec 20).  stage 0 = approach / before first placement, 1..3 = placement phase 1..3, -1 = unknown.

No automatic labeler is applied.  On HandUMI the human gripper rests CLOSED (g ~ 0), opens to ~0.7 for the approach and
closes only to ~0.6 on a cube, so grasp vs. release differ by ~5-8 mm of aperture: a gripper-only stage detector would
be noise, and a wrong stage label silently biases the stage-weighted sampler.  Stages therefore come only from an explicit
annotation file (per episode: a sorted list of stage start times in seconds on the episode clock, 4 entries for stages 0..3),
and rows without one are -1 (weight 1.0 in the sampler).
"""
import json
import pathlib

import numpy as np

ANNOTATION_ENV = "EGO_CART20_STAGE_ANNOTATIONS"


def stage_ids(times, g_left, g_right, row_valid, starts=None):
    """-> (stage_id int8 [N], info).  starts: list of 4 stage start times (s, same clock as times) or None."""
    if starts is None:
        return np.full(len(times), -1, np.int8), dict(source="none", reason="no stage annotation; gripper-only detection rejected (see labels/stage.py)")
    starts = np.asarray(starts, np.float64); assert len(starts) == 4 and np.all(np.diff(starts) >= 0)
    st = (np.searchsorted(starts, times, side="right") - 1).astype(np.int8); st[times < starts[0]] = -1
    return st, dict(source="annotation", starts_s=starts.tolist())


def load_annotations(path):
    p = pathlib.Path(path)
    return json.load(open(p)) if p.exists() else {}
