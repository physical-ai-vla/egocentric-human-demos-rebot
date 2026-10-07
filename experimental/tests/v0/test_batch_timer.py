"""BatchTimer state machine: rec -> rest -> rec ... -> done, discard retakes the same index."""

from ego_collector.recording.recorder import BatchTimer


def test_batch_cycle():
    b = BatchTimer(episodes=2, record_s=10, rest_s=5)
    assert b.phase(0)[0] == "armed"
    b.start(0)
    assert b.phase(3) == ("rec", 7, 0)
    assert b.tick(9.9) is None
    assert b.tick(10.1) == "stop"          # ep1 saved -> rest
    assert b.phase(12)[0] == "rest"
    assert b.tick(15.2) == "start"         # ep2 starts
    assert b.phase(16)[0] == "rec"
    assert b.tick(25.3) == "stop"          # ep2 saved -> done
    assert b.phase(26)[0] == "done" and b.count == 2


def test_discard_retakes_same_episode():
    b = BatchTimer(episodes=3, record_s=10, rest_s=5)
    b.start(0)
    b.discard_current(4)                    # user pressed D mid-recording
    assert b.phase(5)[0] == "rest" and b.count == 0
    assert b.tick(9.1) == "start"           # same episode index retried
    assert b.tick(19.2) == "stop" and b.count == 1


def test_order_cycle_and_display(tmp_path):
    from ego_collector.camera.capture import CaptureConfig
    from ego_collector.recording.recorder import Recorder

    r = Recorder(raw_root=tmp_path, capture=CaptureConfig(), task="cube_stack", instruction="x",
                 batch=BatchTimer(20, 10, 10), batch_orders=["RBP", "RPB", "BRP", "BPR", "PRB", "PBR"],
                 colors={"R": "red", "B": "blue", "P": "purple"})
    assert r.order_for(0) == "RBP" and r.order_for(6) == "RBP" and r.order_for(7) == "RPB"
    assert "red" in r.order_display("RBP") and "purple" in r.order_display("RBP")
    r._apply_batch_order(1)
    assert r.order == "RPB" and "purple in the middle" in r.instruction
    # 20 eps over 6 orders -> 4/4/3/3/3/3 balance
    from collections import Counter
    c = Counter(r.order_for(i) for i in range(20))
    assert sorted(c.values(), reverse=True) == [4, 4, 3, 3, 3, 3]
