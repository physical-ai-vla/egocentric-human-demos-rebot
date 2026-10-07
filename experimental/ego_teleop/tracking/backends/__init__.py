"""Extra pose backends for ego_teleop, registered into handumi_collector.pose.backends at import time (that registry is a
plain dict keyed by pose.yaml `backend:` names; we add without editing the handumi_collector package)."""
from __future__ import annotations
from handumi_collector.pose import backends as _hb

EXTRA = {
    # regression / reference only. Spec section 2: OpenVINS is not the production tracking backend any more.
    "openvins": "ego_teleop.tracking.backends.openvins:OpenVinsBackend",
    # the live wrist visual frontend (spec section 10): MASt3R-Fusion driven over a socket, nothing vendored.
    "mast3r_live": "ego_teleop.tracking.backends.mast3r_live:Mast3rLiveBackend",
}


def register() -> None:
    for k, v in EXTRA.items(): _hb._REGISTRY.setdefault(k, v)


register()
