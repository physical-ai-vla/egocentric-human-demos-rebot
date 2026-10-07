"""[2026-10-07] right_only export for HRA_A100:
  1. occlusion-aware cube blob (cube_pnp_occl): the fingers split the cube near the end of the approach;
  2. metric-scale FALLBACK: when the cube-PnP scale is invalid but the IMU-VI scale of the same SLAM trajectory is valid
     (the origin -> start move excites the IMU; e.g. 210219/001: s_imu 0.131 vs s_pnp 0.147), use s_imu. Every fallback episode is
     listed in <out_raw>/scale_fallback.jsonl with both numbers; its export-log entry carries scale_fallback=imu_vi.
Everything else = ego_cart20.right_only export_session. usage: export_a100.py <session> <out_raw>"""
import pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import cube_pnp_occl; cube_pnp_occl.install()
from ego_cart20 import right_only as RO, cube_pnp as CP
from ego_cart20.sources import handumi_export as HX
_last = {}
_orig_init = HX._init
def _init():                                      # export_session calls HX._init(), which RELOADS RLC: re-wrap side_nogrip every time
    _orig_init(); orig = HX.RLC.side_nogrip
    if getattr(orig, "_a100", False): return
    def side_nogrip(tag, epd, side):
        d = orig(tag, epd, side); _last.update(tag=tag, s=float(d["s"]), valid=bool(d["scale_valid"])); return d
    side_nogrip._a100 = True; HX.RLC.side_nogrip = side_nogrip
HX._init = _init
_orig_pnp = CP.episode_pnp_scale
_ORIGIN = json.load(open(pathlib.Path.home() / "c8/hra_a100/origin_scale_kf.json")) if (pathlib.Path.home() / "c8/hra_a100/origin_scale_kf.json").exists() else {}
OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
def episode_pnp_scale(video, setting, csv):
    pn = _orig_pnp(video, setting, csv)
    # [2026-10-07] ORIGIN scale fallback (MASt3R keyframe-0 pointmap, origin_scale_kf.py): used when the cube-PnP scale is invalid.
    # Validated against cube PnP (n=9): ratio median 1.06, |log ratio| p50 0.08. Only new-protocol episodes have an origin hold.
    if not pn["valid"] and _ORIGIN.get(_last.get("tag", "")[5:], {}).get("valid"):
        o = _ORIGIN[_last["tag"][5:]]
        rec = dict(tag=_last["tag"], s_origin=o["s"], s_pnp_invalid=pn.get("s_pnp"), pnp_reason=pn.get("reason"), n_valid_pnp=pn.get("n_valid_pnp"))
        with open(OUT / "scale_fallback.jsonl", "a") as f: f.write(json.dumps(rec, default=float) + "\n")
        return dict(pn, valid=True, s_pnp=o["s"], reason="", scale_fallback="origin_plane", pnp_reason_original=rec["pnp_reason"])
    # DISABLED 2026-10-07: s_imu was not reproducible (0.131 / 0.117 / 0.080 on the same episode vs s_pnp 0.15). A100_IMU_FALLBACK=1 re-enables.
    import os
    if os.environ.get("A100_IMU_FALLBACK") == "1" and not pn["valid"] and _last.get("valid") and np.isfinite(_last.get("s", np.nan)):
        rec = dict(tag=_last["tag"], s_imu=_last["s"], s_pnp_invalid=pn.get("s_pnp"), pnp_reason=pn.get("reason"), n_valid_pnp=pn.get("n_valid_pnp"))
        with open(OUT / "scale_fallback.jsonl", "a") as f: f.write(json.dumps(rec, default=float) + "\n")
        pn = dict(pn, valid=True, s_pnp=_last["s"], reason="", scale_fallback="imu_vi", pnp_reason_original=rec["pnp_reason"])
    return pn
CP.episode_pnp_scale = episode_pnp_scale
res = RO.export_session(pathlib.Path(sys.argv[1]), OUT)
json.dump(res, open(OUT / f"export_log_{pathlib.Path(sys.argv[1]).name}.json", "w"), indent=1, default=float)   # for the sanity gate
print("DONE", {s: sum(r.get("status") == s for r in res) for s in {r.get("status") for r in res}}, "fallback", {k: sum(r.get("scale_fallback") == k for r in res) for k in ("origin_plane", "imu_vi")})
