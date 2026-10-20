"""Which Orbbec colour/depth resolution pairs deliver aligned frames? (2026-10-15: 1280x720 colour gave 0 fps in the hub.)
    sudo .venv/bin/python -m visual_teleop.tools.orbbec_res_probe        (stop the sensor hub first -- the Orbbec is exclusive)
For each pair, 3 s of: frameset received -> align ok -> colour+depth both present -> JPEG decoded; plus the decoded/depth sizes."""
from __future__ import annotations
import time
import numpy as np


def probe(cw, ch, dw, dh, fps=30, secs=3.0) -> str:
    import cv2
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, AlignFilter, OBStreamType
    pipe, cfg = Pipeline(), Config()
    try:
        cfg.enable_stream(pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(dw, dh, OBFormat.UNKNOWN_FORMAT, fps))
        cfg.enable_stream(pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(cw, ch, OBFormat.MJPG, fps))
        pipe.start(cfg)
    except Exception as e:
        return f"열기 실패: {type(e).__name__}: {str(e)[:120]}"
    al = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM); n = dict(fs=0, align=0, both=0, decoded=0); sizes = set(); t0 = time.time()
    try:
        while time.time() - t0 < secs:
            fs = pipe.wait_for_frames(300)
            if fs is None: continue
            n["fs"] += 1; a = al.process(fs)
            if a is None: continue
            a = a.as_frame_set() if hasattr(a, "as_frame_set") else a
            if a is None: continue
            n["align"] += 1; cf, df = a.get_color_frame(), a.get_depth_frame()
            if not (cf and df): continue
            n["both"] += 1; img = cv2.imdecode(np.frombuffer(cf.get_data(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None: n["decoded"] += 1; sizes.add((img.shape[1], img.shape[0], df.get_width(), df.get_height()))
    finally:
        pipe.stop()
    return f"{n}  (컬러 w,h / 깊이 w,h: {sorted(sizes)})"


if __name__ == "__main__":
    for c, d in [((848, 480), (848, 480)), ((1280, 720), (848, 480)), ((1280, 720), (1280, 720)), ((1280, 800), (1280, 800)), ((1280, 720), (640, 400))]:
        print(f"컬러 {c[0]}x{c[1]} + 깊이 {d[0]}x{d[1]}:", probe(*c, *d), flush=True)
