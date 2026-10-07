"""Soft-Fold FT contract override (user 2026-10-07: chunk 32).  Import BEFORE any other ego_cart20 module: the label /
validation modules copy config.HORIZON at import time, so patching it here makes them build and check [32, 20] CART20 /
[32, 32] action chunks (k = 1..32, 32 * UMI_DT = 1.60 s).  The ego / reBot REL16 contract (config.py) is NOT changed."""
from . import config as _c

assert "ego_cart20.labels.build_cart20" not in __import__("sys").modules, "import softfold_h32 before the label modules"
_c.HORIZON = 32
_c.CONTRACT.update(action_horizon=32, horizon_span_s=32 * _c.UMI_DT,
                   action_contract="current_anchor_relative: A_k = inv(T(t)) @ T(t + k*UMI_DT), k = 1..32, raw-trajectory interpolation")
HORIZON = 32
