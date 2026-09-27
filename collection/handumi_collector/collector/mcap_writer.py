"""sensors.mcap writer: JSON-schema channels, log_time = host monotonic ns (anchor in /system/sync).
Kept deliberately simple and self-describing so Foxglove / `mcap` readers work without ROS."""
from __future__ import annotations
import json
from pathlib import Path
from mcap.writer import Writer
from ..devices.base import GripperSample, ImuSample

SCHEMAS: dict[str, dict] = {
    "handumi.FrameMeta": {"type": "object", "properties": {
        "stream": {"type": "string"}, "frame_index": {"type": "integer"}, "video_frame": {"type": "integer"},
        "capture_ns": {"type": "integer"}, "skipped_before": {"type": "integer"}}},
    "handumi.Imu": {"type": "object", "properties": {
        "side": {"type": "string"}, "seq": {"type": "integer"}, "device_timestamp_us": {"type": "integer"},
        "host_receive_ns": {"type": "integer"}, "ax": {"type": "number"}, "ay": {"type": "number"}, "az": {"type": "number"},
        "gx": {"type": "number"}, "gy": {"type": "number"}, "gz": {"type": "number"}, "temperature_c": {"type": "number"}}},
    "handumi.Gripper": {"type": "object", "properties": {
        "side": {"type": "string"}, "seq": {"type": "integer"}, "sample_ns": {"type": "integer"},
        "raw_position": {"type": "integer"}, "raw_position_mod": {"type": "integer"}, "normalized": {"type": "number"},
        "telemetry": {"type": ["object", "null"]}}},
    "handumi.Sync": {"type": "object"},
    "handumi.Event": {"type": "object", "properties": {"t_ns": {"type": "integer"}, "kind": {"type": "string"},
                                                        "device": {"type": "string"}, "detail": {"type": "object"}}},
}
TOPIC_SCHEMA = {"/system/sync": "handumi.Sync", "/system/events": "handumi.Event"}
for _s in ("left", "right"):
    TOPIC_SCHEMA[f"/{_s}/imu"] = "handumi.Imu"; TOPIC_SCHEMA[f"/{_s}/gripper"] = "handumi.Gripper"


def _clean(v):
    if isinstance(v, float) and v != v:   # NaN -> null (JSON)
        return None
    return v


class SensorMcapWriter:
    def __init__(self, path: Path, *, streams: list[str]) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "wb")
        self._w = Writer(self._fh)
        self._w.start(profile="handumi", library="handumi_collector")
        self._schema_ids = {n: self._w.register_schema(n, "jsonschema", json.dumps(s).encode()) for n, s in SCHEMAS.items()}
        self._chan: dict[str, int] = {}
        for t, s in TOPIC_SCHEMA.items():
            self._chan[t] = self._w.register_channel(t, "json", self._schema_ids[s])
        for st in streams:
            self._chan[f"/{st}/frame_meta"] = self._w.register_channel(f"/{st}/frame_meta", "json", self._schema_ids["handumi.FrameMeta"])
        self._seq: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self._closed = False

    def _add(self, topic: str, t_ns: int, obj: dict) -> None:
        n = self._seq.get(topic, 0)
        self._w.add_message(self._chan[topic], log_time=t_ns, publish_time=t_ns, sequence=n,
                            data=json.dumps({k: _clean(v) for k, v in obj.items()}, separators=(",", ":")).encode())
        self._seq[topic] = n + 1
        self.counts[topic] = n + 1

    def frame_meta(self, stream: str, frame_index: int, video_frame: int, capture_ns: int, skipped_before: int) -> None:
        self._add(f"/{stream}/frame_meta", capture_ns, dict(stream=stream, frame_index=frame_index, video_frame=video_frame,
                                                          capture_ns=capture_ns, skipped_before=skipped_before))

    def imu(self, s: ImuSample) -> None:
        self._add(f"/{s.side}/imu", s.host_receive_ns, s.__dict__)

    def gripper(self, s: GripperSample) -> None:
        self._add(f"/{s.side}/gripper", s.sample_ns, s.__dict__)

    def sync(self, t_ns: int, obj: dict) -> None:
        self._add("/system/sync", t_ns, obj)

    def event(self, t_ns: int, kind: str, device: str, detail: dict | None = None) -> None:
        self._add("/system/events", t_ns, dict(t_ns=t_ns, kind=kind, device=device, detail=detail or {}))

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        self._w.finish(); self._fh.close()


def read_messages(path: Path):
    """Yield (topic, log_time, dict) for every message — used by tests/tools."""
    from mcap.reader import make_reader
    with open(path, "rb") as fh:
        r = make_reader(fh)
        for schema, channel, msg in r.iter_messages():
            yield channel.topic, msg.log_time, json.loads(msg.data)
