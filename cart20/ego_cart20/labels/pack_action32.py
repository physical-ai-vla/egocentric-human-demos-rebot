"""CART20 [16,20] -> X-VLA v4 action tensor [16,32]: channels 0:20 = CART20, 20:32 = AUX12 = 0 (ego-only)."""
import numpy as np
from ..config import AUX, CART_DIM, HORIZON, MODEL_ACTION_DIM


def pack_cart20_to_action32(cart20):
    cart20 = np.asarray(cart20, np.float32)
    assert cart20.shape[-2:] == (HORIZON, CART_DIM), f"CART20 must be [..., {HORIZON}, {CART_DIM}], got {cart20.shape}"
    action = np.zeros(cart20.shape[:-1] + (MODEL_ACTION_DIM,), np.float32)
    action[..., :CART_DIM] = cart20
    return action


def check_action32(action, cart20):
    """hard check (spec 27 / 30): never silently fixes a violation"""
    assert action.shape[-2:] == (HORIZON, MODEL_ACTION_DIM), action.shape
    assert np.array_equal(action[..., :CART_DIM], cart20), "action[:, :20] != CART20"
    nz = int(np.count_nonzero(action[..., AUX]))
    if nz: raise ValueError(f"AUX12 must be exactly zero for ego-only data: {nz} non-zero entries")
