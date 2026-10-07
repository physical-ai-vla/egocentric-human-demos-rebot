"""[2026-09-28] LIVE global-camera orientation with a hand-placed RED cube. No leader arm, no robot command: /observe
is only READ. The operator puts one red cube at four spots on the table and presses Enter at each:
    L  on the robot's LEFT-arm side      R  on the robot's RIGHT-arm side      (same distance from the robot)
    F  FAR from the robot (forward, +x)   N  NEAR the robot                     (same left/right position)
Training convention (HEAD180 motion signs 2026-09-28: robot-left +y -> du < 0, forward +x -> dv < 0), in the deployed
preprocessing as-is (rot0):   u_L < u_R   and   v_F < v_N.
    both hold      -> V4_GLOBAL_ROT180=0        both reversed -> V4_GLOBAL_ROT180=1        one reversed -> AMBIGUOUS
The red mask is HSV-thresholded on the 224x224 deployed image; each spot needs a clear single blob.
usage: live_cube_orient.py
"""
import base64, json, os, time
import numpy as np, cv2, requests

HERE = os.path.expanduser("~/umi_bridge/rel16_audit/r380")


def grab():
    o = requests.get("http://localhost:8020/observe", timeout=15).json()
    im = cv2.imdecode(np.frombuffer(base64.b64decode(o["images"]["middle"]), np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)       # deployed geometry, rot0


def red_centroid(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0, 110, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 110, 60), (180, 255, 255)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return tuple(float(x) for x in cen[k]), int(st[k, cv2.CC_STAT_AREA])


pts, shots = {}, []
for key, desc in (("L", "robot's LEFT-arm side"), ("R", "robot's RIGHT-arm side (same distance as L)"),
                  ("F", "FAR from the robot"), ("N", "NEAR the robot (same left/right as F)")):
    while True:
        input(f"place the RED cube {desc}, hands out of view, then press Enter ")
        time.sleep(0.5); im = grab(); c, area = red_centroid(im)
        if c and area >= 20:
            pts[key] = c; shots.append(im.copy()); cv2.circle(shots[-1], (int(c[0]), int(c[1])), 6, (0, 255, 0), 2)
            print(f"  {key}: u={c[0]:.1f} v={c[1]:.1f} (area {area})"); break
        print("  no clear red blob -- is the red cube in view and the only red thing? retry")
x_ok = pts["L"][0] < pts["R"][0]; y_ok = pts["F"][1] < pts["N"][1]
print(f"u_L {pts['L'][0]:.1f} {'<' if x_ok else '>'} u_R {pts['R'][0]:.1f}   (training: <)")
print(f"v_F {pts['F'][1]:.1f} {'<' if y_ok else '>'} v_N {pts['N'][1]:.1f}   (training: <)")
verdict = "rot0" if (x_ok and y_ok) else "rot180" if (not x_ok and not y_ok) else "ambiguous"
print("VERDICT:", {"rot0": "both match training -> V4_GLOBAL_ROT180=0",
                   "rot180": "both reversed -> V4_GLOBAL_ROT180=1",
                   "ambiguous": "one axis reversed -> AMBIGUOUS, do not freeze"}[verdict])
cv2.imwrite(f"{HERE}/live_cube_orient.png", np.hstack(shots))
json.dump(dict(points=pts, x_ok=bool(x_ok), y_ok=bool(y_ok), verdict=verdict, t=time.strftime("%F %T")),
          open(f"{HERE}/live_cube_orient.json", "w"), indent=1)
