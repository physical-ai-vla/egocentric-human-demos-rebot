"""[2026-10-14] Episode end trim (user: drop the operator's reach to the SPACE key at the end of a take).

<episode>/trim.json {"end_ns": <capture monotonic ns>, ...} marks the last usable instant; frames after it are not
training data. A sidecar, not episode_meta.json: the collector may still rewrite the last episode's meta. Raw files
are never cut. No trim.json -> the whole take is used.
"""
import json, pathlib


def trim_end_ns(ep):
    p = pathlib.Path(ep) / "trim.json"
    if not p.exists(): return None
    return int(json.loads(p.read_text())["end_ns"])


def trim_info(ep):
    p = pathlib.Path(ep) / "trim.json"
    return json.loads(p.read_text()) if p.exists() else None
