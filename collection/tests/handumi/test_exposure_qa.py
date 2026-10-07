"""Exposure is chosen by feature retention under FAST motion, not by brightness.

The failure this guards against is drawing a confident conclusion from a take that never moved: a 15.6 ms exposure
looks perfect on a static scene and smears at speed, so a sweep evaluated on static frames would pick exactly the wrong
setting."""
import numpy as np
from handumi_collector.tools.wrist_exposure_qa import BUCKETS, bucket_of, report


def _rows(flows, survived, *, brightness=200.0, blur=120.0, corners=800):
    out = [dict(t_ns=0, brightness=brightness, blur_var=blur, corners=corners,
                survived=np.nan, flow_px=np.nan, bucket="n/a")]
    for i, (f, s) in enumerate(zip(flows, survived), 1):
        out.append(dict(t_ns=i * 33_000_000, brightness=brightness, blur_var=blur, corners=corners,
                        survived=s, flow_px=f, bucket=bucket_of(f)))
    return out


def test_motion_buckets_cover_the_range_without_gaps():
    assert [b[2] for b in BUCKETS] == ["static", "slow", "normal", "fast"]
    assert bucket_of(0.0) == "static" and bucket_of(0.99) == "static"
    assert bucket_of(1.0) == "slow" and bucket_of(4.99) == "slow"
    assert bucket_of(5.0) == "normal" and bucket_of(14.99) == "normal"
    assert bucket_of(15.0) == "fast" and bucket_of(1e6) == "fast"


def test_report_refuses_to_conclude_from_a_static_take():
    res = dict(episode="e", stream="left_wrist", n_frames=6, rows=_rows([0.1] * 5, [1.0] * 5))
    txt = report(res)
    assert "no fast-motion frames" in txt and "cannot tell you anything about blur" in txt


def test_report_separates_fast_motion_retention_from_the_easy_frames():
    """A setting can hold every static frame and lose most features at speed. The per-bucket split is the whole point."""
    flows = [0.2] * 20 + [30.0] * 10
    surv = [1.0] * 20 + [0.25] * 10
    txt = report(dict(episode="e", stream="left_wrist", n_frames=31, rows=_rows(flows, surv)))
    lines = {ln.split()[0]: ln for ln in txt.splitlines() if ln.startswith("  ") and ln.split()}
    assert "1.000" in lines["static"] and "0.250" in lines["fast"]
    assert "fast bucket: 10 frames, median retention 0.250" in txt
