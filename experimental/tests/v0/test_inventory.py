import json
from pathlib import Path

from ego_collector.commands.inventory import main


def _mk(root: Path, eid: int, layout: str, order: str):
    d = root / f"episode_{eid:06d}"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({
        "episode_id": eid, "layout": layout, "order": order,
        "frame_count": 300, "duration_s": 10.0, "dataset_tier": "action_labelled",
        "task": "cube_stack", "operator_id": "op01"}))


def test_coverage_flags_missing_and_dups(tmp_path, capsys):
    for i, o in enumerate(("RBP", "RBP", "RPB", "BRP")):
        _mk(tmp_path, i, "L99", o)
    main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "MISSING ['BPR', 'PRB', 'PBR']" in out
    assert "'RBP': 2" in out
    assert "--layout L99 --batch 3" in out and "--orders BPR,PRB,PBR" in out


def test_complete_layout_ok(tmp_path, capsys):
    for i, o in enumerate(("RBP", "RPB", "BRP", "BPR", "PRB", "PBR")):
        _mk(tmp_path, i, "L42", o)
    main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "OK 6/6" in out and "MISSING" not in out


def test_excluded_ids_ignored(tmp_path, capsys):
    _mk(tmp_path, 45, "L01", "RBP")
    _mk(tmp_path, 1, "L01", "RPB")
    main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "episode_000045" not in out and "episode_000001" in out
