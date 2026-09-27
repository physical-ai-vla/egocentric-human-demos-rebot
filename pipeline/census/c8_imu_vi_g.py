#!/usr/bin/env python3
"""[2026-09-25] imu_vi_g: cp6_scale.imu_vi with the SAME source (exec of the frozen text), whose return additionally carries
the gravity vector g_world = xs[1:4] (in that side's MASt3R world; model R_wb f_b - a_lever = s a_cam - g, so at rest
g = -R_wb f_b = physical gravity, pointing DOWN). The only textual change is the return line; s / g_norm / n / resid are
verified bit-identical to cp6's imu_vi before use (run this file as a script: verification on given tags).
Needs M3_BASE / M3_VARIANT set to a staging dir like c8_phase3.py's (stage_p3)."""
import os, pathlib
TA = pathlib.Path.home() / "umi_bridge/trackA_mast3r_pose_v1"
RET = 'return dict(s=float(xs[0]), g_norm=float(np.linalg.norm(xs[1:4])), n=int(len(y) // 3),'


def load():
    src = open(TA / "cp6_scale.py").read(); pre = src[:src.index("tags = [")]
    assert pre.count(RET) == 1, "cp6_scale.imu_vi return line changed -- re-verify imu_vi_g"
    ns_ref, ns_g = {}, {}
    exec(pre, ns_ref); exec(pre.replace(RET, RET.replace("dict(", "dict(g_world=xs[1:4].tolist(), ")), ns_g)
    ns_g["imu_vi_ref"] = ns_ref["imu_vi"]
    return ns_g


if __name__ == "__main__":
    import sys, json, numpy as np
    os.environ.setdefault("M3_BASE", str(pathlib.Path.home() / "c8/stage_p3")); os.environ.setdefault("M3_VARIANT", "ss1")
    ns = load(); bad = 0
    for t in sys.argv[1:]:
        a, b = ns["imu_vi_ref"](t, 15, False), ns["imu_vi"](t, 15, False)
        same = (a is None and b is None) or all(a[k] == b[k] for k in a)
        g = np.array(b["g_world"]) if b else None
        print(t, "IDENTICAL" if same else "DIFFERENT", f"s {b['s']:.6f} |g| {b['g_norm']:.4f} ‖g_world‖ {np.linalg.norm(g):.4f}" if b else "none")
        bad += not same
    print("IMU_VI_G_VERIFIED" if bad == 0 else f"IMU_VI_G_MISMATCH {bad}")
