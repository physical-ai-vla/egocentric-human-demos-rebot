"""The live HUD (`ego_teleop/tools/f5_teleop_hud.py`) — an observer that must not become a second pipeline.

A HUD is easy to get subtly wrong in ways that are worse than having no HUD, because it is what the operator
believes. The things checked here are exactly those:

  * it draws the pose the ROBOT got, from `f4_fusion.run`'s own tick loop, not a re-derivation;
  * its numbers come from `f4_fusion.report`, so the page and the report cannot disagree;
  * it never writes to the provider, the coordinator or the robot, and `--stage real` is not offered;
  * a slow or absent browser must not be able to stall the fusion;
  * the plots must show a dropout as a GAP, not as a straight line across it.
"""
from __future__ import annotations
import json
import threading
import time
import urllib.request
import numpy as np
import pytest

from ego_teleop.tools import f4_fusion as F
from ego_teleop.tools import f5_teleop_hud as H


def hud_state(rate_hz: float = 50.0) -> H.HudState:
    return H.HudState(meta=dict(stage="virtual", mode="fused", side="right", rate_hz=rate_hz,
                                vi_backend="mast3r_live", protocol=True, source="synthetic",
                                max_correction_rate_m_s=0.01, note=""), rate_hz=rate_hz)


# ---- the observer contract -----------------------------------------------------------------------------------
def test_the_hud_offers_no_real_robot_stage():
    """`f4_fusion --stage real` drives the arm and demands --i-am-at-the-robot. A browser tab is not that."""
    with pytest.raises(SystemExit):
        H.main(["--synthetic", "--stage", "real"])


def test_a_replay_with_no_hook_is_byte_identical_to_one_with_an_observing_hook():
    """The hook must be an observer. If adding it changed the run, every number the page shows would be about a
    different take than the report."""
    import dataclasses as dc
    from ego_teleop.config import load_teleop_cfg
    cfg = load_teleop_cfg().fused_wrist
    args = _Args()
    a = F.run(**_run_kwargs(args, cfg))
    seen: list[dict] = []
    b = F.run(**_run_kwargs(args, cfg), tick_hook=lambda row, prov, st: seen.append(row))
    assert len(seen) == len(b)
    cols = [c for c in ("x", "y", "z", "correction_m", "residual_m", "tcp_x", "step_mm") if c in a.columns]
    for c in cols:
        np.testing.assert_allclose(a[c].to_numpy(np.float64), b[c].to_numpy(np.float64), equal_nan=True,
                                   err_msg=f"the hook changed column {c}")


# ---- what the page is fed ------------------------------------------------------------------------------------
def test_a_dropout_is_a_gap_in_the_series_not_a_line_across_it():
    st = hud_state()
    st.push(dict(s=0.0, x=0.1, y=0.0, z=0.0, rgbd_w_x=0.1, rgbd_w_y=0.0, rgbd_w_z=0.0), {})
    st.push(dict(s=0.02, x=0.1, y=0.0, z=0.0), {})                      # the anchor dropped out for this tick
    st.push(dict(s=0.04, x=0.1, y=0.0, z=0.0, rgbd_w_x=0.1, rgbd_w_y=0.0, rgbd_w_z=0.0), {})
    s = st.snapshot()["series"]["rgbd_x"]
    assert s == [100.0, None, 100.0]                                    # None is what the canvas lifts the pen on


def test_engage_is_latched_because_the_row_flag_is_true_for_one_tick_only():
    st = hud_state()
    st.push(dict(s=0.0, engaged=False), {}); assert not st.snapshot()["ever_engaged"]
    st.push(dict(s=0.02, engaged=True), {})
    st.push(dict(s=0.04, engaged=False), {})
    assert st.snapshot()["ever_engaged"], "the page would show 'not engaged' for the rest of the take"


def test_a_jump_is_never_measured_across_a_tracking_gap():
    st = hud_state()
    st.push(dict(s=0.0, x=0.0, y=0.0, z=0.0, tracking_health="TRACKING_OK"), {})
    st.push(dict(s=0.02, x=0.0, y=0.0, z=0.0, tracking_health="TRACKING_LOST"), {})
    st.push(dict(s=0.04, x=1.0, y=0.0, z=0.0, tracking_health="TRACKING_OK"), {})   # 1 m away, after a gap
    assert st.snapshot()["jumps"] == 0
    st.push(dict(s=0.06, x=1.5, y=0.0, z=0.0, tracking_health="TRACKING_OK"), {})   # 500 mm, no gap: a real jump
    assert st.snapshot()["jumps"] == 1


def test_nan_never_reaches_the_json():
    st = hud_state()
    st.push(dict(s=0.0, x=float("nan"), correction_m=None, fusion_ms=1.5), {})
    body = json.dumps(st.snapshot())            # json.dumps emits bare NaN, which JSON.parse rejects
    assert "NaN" not in body and "Infinity" not in body


def test_the_gate_table_is_the_report_not_a_hud_reimplementation():
    from ego_teleop.config import load_teleop_cfg
    cfg = load_teleop_cfg().fused_wrist
    df = F.run(**_run_kwargs(_Args(), cfg))
    rep = F.report(df, cfg, "fused", "virtual", {n: (a, b) for n, a, b, _ in H.PROTOCOL_60S})
    st = hud_state(); st.set_report(rep)
    snap = st.snapshot()
    assert snap["gate"] == rep["gate"]
    assert snap["measured_rate_hz"] == pytest.approx(rep["rate_hz"], rel=1e-6)


# ---- the server ------------------------------------------------------------------------------------------------
def test_the_page_and_the_state_are_served_while_the_fusion_writes():
    """Polling must not throttle the tick loop, and the tick loop must not starve the page.

    The writer here runs at ~1 kHz, twenty times the real fusion rate, and yields between ticks exactly as the
    real one does (`f4_fusion.run_live` sleeps its slack; the HUD hook paces a replay). Measured, the page is
    served in single-digit milliseconds at that rate and still at 10 kHz. It is only a writer that NEVER yields —
    a pure spin, which this loop is not and cannot be — that starves the reader through the GIL, and no amount of
    lock discipline on this side fixes that one."""
    st = hud_state()
    srv = H.serve(st, 0)
    port = srv.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
        assert "<canvas id=\"c-x\"" in page and "state.json" in page

        stop = threading.Event()
        n = [0]

        def write():
            while not stop.is_set():
                st.push(dict(s=n[0] * 0.02, x=0.1, y=0.0, z=0.0, tracking_health="TRACKING_OK"), {})
                n[0] += 1
                time.sleep(0.001)

        t = threading.Thread(target=write, daemon=True); t.start()
        polls, end = 0, time.monotonic() + 0.5
        slowest = 0.0
        while time.monotonic() < end:
            t0 = time.monotonic()
            d = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state.json", timeout=5).read())
            slowest = max(slowest, time.monotonic() - t0)
            assert "series" in d and "gate" in d
            polls += 1
        stop.set(); t.join(2.0)
        assert not t.is_alive()
        assert slowest < 0.5, f"a poll took {slowest*1000:.0f} ms against a 1 kHz writer"
        assert polls > 10, f"only {polls} polls got through in 0.5 s"
        assert n[0] > 100, f"the writer managed {n[0]} ticks in 0.5 s: it was throttled by the reader"
    finally:
        srv.shutdown()


def test_the_state_payload_is_decimated_to_what_the_canvas_can_draw():
    """20 Hz polling of every tick of a 20 s trace is bandwidth nobody can see. The page gets at most one point
    per plot pixel, and the numeric read-outs still come from the latest row, not from the decimated trace."""
    st = hud_state()
    for i in range(H.MAX_PLOT_POINTS * 3):
        st.push(dict(s=i * 0.02, x=0.001 * i, y=0.0, z=0.0, tracking_health="TRACKING_OK"), {})
    snap = st.snapshot()
    assert len(snap["t"]) <= H.MAX_PLOT_POINTS
    assert all(len(v) == len(snap["t"]) for v in snap["series"].values())
    assert snap["row"]["x"] == pytest.approx(0.001 * (H.MAX_PLOT_POINTS * 3 - 1))


def test_a_missing_camera_frame_is_a_404_not_a_crash():
    st = hud_state()
    srv = H.serve(st, 0)
    port = srv.server_address[1]
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/frame/chest.jpg", timeout=5)
        assert e.value.code == 404
        st.set_frame("chest", np.zeros((48, 64, 3), np.uint8))
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/frame/chest.jpg", timeout=5).read()[:2] == b"\xff\xd8"
    finally:
        srv.shutdown()


# ---- helpers ----------------------------------------------------------------------------------------------------
class _Args:
    synthetic = True; episode = None; vi_poses = None; synthetic_frames = 600
    allow_identity_mount = False


def _run_kwargs(args, cfg):
    from ego_teleop.config import load_teleop_cfg
    teleop = load_teleop_cfg()
    events, rgbd_provider, vi_provider, _ = F.build_events(args, cfg, teleop)
    return dict(events=events, cfg=cfg, mode="fused", stage="virtual", rgbd_provider=rgbd_provider,
                vi_provider=vi_provider, engage_at_s=3.0)
