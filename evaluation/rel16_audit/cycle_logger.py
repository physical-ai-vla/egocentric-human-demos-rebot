"""Read-only: poll :8025/status, one CSV row per new cycle (executed waypoint cmd vs actual, tracking, latency)."""
import json, sys, time, urllib.request, numpy as np
out = open(sys.argv[1], "a"); seen = None; idle = 0
out.write("t,cycles,arm,wp,cmd_mm,actual_mm,ratio,reached,timeout,cause,repeats,dwell_ms,infer_ms,travel_L,travel_R\n"); out.flush()
while True:
    try:
        d = json.load(urllib.request.urlopen("http://localhost:8025/status", timeout=5))
    except Exception:
        time.sleep(1); continue
    u, l = d["ui"], d.get("last") or {}
    if u.get("running"):
        idle = 0
    else:
        idle += 1
        if idle > 1200: break          # 10 min idle -> quit
    c = u.get("cycles")
    if l and c != seen and u.get("running"):
        seen = c; tr = (l.get("tracking") or [{}])[-1]
        for a in ("left", "right"):
            for w in (l.get("axes") or {}).get(a, []):
                cm = float(np.linalg.norm(w["cmd_world_dxyz_mm"])); ac = float(np.linalg.norm(w["actual_world_dxyz_mm"] or [0, 0, 0]))
                out.write(f"{time.time():.1f},{c},{a},{w['wp']},{cm:.2f},{ac:.2f},{ac/max(cm,1e-6):.3f},{tr.get('reached')},{tr.get('timeout')},"
                          f"{tr.get('cause')},{tr.get('repeats')},{tr.get('dwell_ms')},{(l.get('latency') or {}).get('infer_ms')},"
                          f"{u['travel_mm'][0]:.1f},{u['travel_mm'][1]:.1f}\n")
        out.flush()
    time.sleep(0.5)
