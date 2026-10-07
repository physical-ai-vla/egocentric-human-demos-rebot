"""G0-0 UI server: NaN-safe JSON, episode listing, path guard, stdout segment parsing."""
import json
import threading
import urllib.error
import urllib.request

from ego_teleop.tools import g0_ui as U


def test_nan_to_none_is_recursive():
    assert U._nan_to_none({"a": float("nan"), "b": [1.0, float("nan")], "c": {"d": float("nan")}}) == {"a": None, "b": [1.0, None], "c": {"d": None}}


def test_segment_line_pattern_matches_record_output():
    m = U.SEG_RE.match("  still_open     5.0s  손 펴고 정지")
    assert m and m.group(1) == "still_open" and float(m.group(2)) == 5.0 and m.group(3) == "손 펴고 정지"
    assert U.EP_RE.match("episode: /x/G0MOUNT_1/episode_000001 streams: {}").group(1) == "/x/G0MOUNT_1/episode_000001"


def test_server_lists_episodes_serves_nan_report_and_guards_paths(tmp_path):
    ep = tmp_path / "G0MOUNT_20261002_000000" / "episode_000001"; (ep / "derived" / "g0_mount").mkdir(parents=True)
    (ep / "episode_meta.json").write_text(json.dumps(dict(notes="n", duration_s=1.0, t_start_wall_iso="2026-10-02T00:00:00")))
    (ep / "derived" / "g0_mount" / "g0_report.json").write_text('{"verdict": "SHARE: x", "v": NaN}')
    app = U.App(root=tmp_path, hardware="g0_mount_right", mock=True)
    srv = U.ThreadingHTTPServer(("127.0.0.1", 0), U.make_handler(app)); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        st = json.load(urllib.request.urlopen(base + "/state"))
        assert st["episodes"][0]["verdict"] == "SHARE: x" and len(st["protocol"]) == len(U.G.PROTOCOL)
        rep = json.loads(urllib.request.urlopen(base + f"/report?ep={ep}").read())     # must be strict JSON
        assert rep["v"] is None
        try: urllib.request.urlopen(base + "/report?ep=/etc"); assert False
        except urllib.error.HTTPError as e: assert e.code == 403
        r = urllib.request.Request(base + "/analyze", data=json.dumps({"episode": str(tmp_path)}).encode(), method="POST")
        assert "not an episode" in json.load(urllib.request.urlopen(r))["msg"]
    finally:
        srv.shutdown()
