from handumi_collector.collector.timeline import ClockAnchor, GapDetector, index_gaps


def test_anchor_and_gaps():
    a = ClockAnchor(monotonic_ns=1_000, wall_ns=5_000, perf_counter_ns=0)
    assert a.to_wall_ns(1_500) == 5_500
    g = GapDetector(max_gap_ms=25)
    assert g.update(0) is None and g.update(2_500_000) is None            # 2.5 ms fine
    assert g.update(2_500_000 + 40_000_000) == 40.0 and g.count == 1      # 40 ms gap flagged
    assert g.update(1_000_000) is not None and g.count == 2               # backwards time flagged
    assert index_gaps(None, 0) == 0 and index_gaps(5, 6) == 0 and index_gaps(5, 9) == 3
