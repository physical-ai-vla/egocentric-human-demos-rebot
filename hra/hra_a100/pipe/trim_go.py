"""[2026-10-07] HRA_A100: training rows start at the AUTO "go" event (origin hold + prep stay in the raw video / SLAM but are
marked invalid in raw_episode.npz). Old-protocol takes (no auto 'go') use go_cue. Writes trim_go_report.json."""
import json, pathlib, sys, numpy as np
H = pathlib.Path.home(); RAW = pathlib.Path(sys.argv[1]); SRC = H / "ego_collector/datasets/human_handumi_raw/HRA_A100"
rep = {}
for d in sorted(p for p in RAW.iterdir() if (p / "raw_episode.npz").exists()):
    eid = d.name; sess, ep = eid.rsplit("_", 1)
    ev = json.load(open(SRC / sess / f"episode_{ep}" / "events.json"))
    g = [e for e in ev if e.get("kind") == "auto_loop" and e.get("detail", {}).get("cue") == "go"] or [e for e in ev if e.get("kind") == "go_cue"]
    t_go = int(g[0]["t_ns"]); src = "auto_go" if g[0].get("kind") == "auto_loop" else "go_cue"
    z = dict(np.load(d / "raw_episode.npz", allow_pickle=True)); t = z["right_t_ns"].astype(np.int64)
    before = t < t_go; z["right_valid"] = z["right_valid"].astype(bool) & ~before
    np.savez(d / "raw_episode.npz", **z)
    rep[eid] = dict(source=src, rows_trimmed=int(before.sum()), rows_valid_after=int(z["right_valid"].sum()), t_go_rel_s=round((t_go - int(t[0])) / 1e9, 2))
json.dump(rep, open(RAW / "trim_go_report.json", "w"), indent=1)
v = [r["rows_valid_after"] for r in rep.values()]; print("trimmed", len(rep), "episodes; valid rows after go p10/50/90", np.percentile(v, [10, 50, 90]).round(0).tolist(),
      "| sources", {s: sum(r["source"] == s for r in rep.values()) for s in ("auto_go", "go_cue")})
