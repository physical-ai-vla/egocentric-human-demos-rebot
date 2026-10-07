"""[2026-09-28] LIVE global-camera orientation, measured, no model and no robot command: /observe is only READ.

The operator moves ONE leader arm (the follower copies it) while this records ~5 Hz observations. For frames where only
that arm moves (TCP speed from FK of the observed joints), the horizontal centroid of |global(t+1)-global(t)| is taken.
Training reference (R380/R312 HEAD, audit 2026-09-28): LEFT-arm-only motion appears on the IMAGE LEFT in 180/180
episodes (median centroid 0.25), right arm on the image right (0.72). With the deployed preprocessing as-is (rot0):
    live left-arm centroid < 0.5  -> V4_GLOBAL_ROT180=0 correct
    live left-arm centroid > 0.5  -> the live camera is rotated vs training -> deploy with V4_GLOBAL_ROT180=1
usage: live_side_test.py [seconds=25]   (move the LEFT leader arm left/right/forward for the whole window)
"""
import base64, os, sys, time
import numpy as np, cv2, requests
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
import infer_core_v4 as IC

T = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
kin = IC.eef_kin.Kin()
frames, tcps = [], []
t0 = time.time()
print(f"recording {T:.0f}s -- move the LEFT leader arm now", flush=True)
while time.time() - t0 < T:
    o = requests.get("http://localhost:8020/observe", timeout=10).json()
    g = cv2.imdecode(np.frombuffer(base64.b64decode(o["images"]["middle"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    q = np.asarray(o["joints_rad"], np.float64)
    L7, R7 = kin.fk_pose7(q[IC.ARM_IDX])
    frames.append(cv2.resize(g, (224, 224)).astype(np.float32)); tcps.append(np.r_[np.asarray(L7)[:3], np.asarray(R7)[:3]])
    time.sleep(0.15)
tcps = np.array(tcps); cx = {"L": [], "R": []}
for i in range(len(frames) - 1):
    vL = np.linalg.norm(tcps[i + 1, :3] - tcps[i, :3]); vR = np.linalg.norm(tcps[i + 1, 3:] - tcps[i, 3:])
    d = np.abs(frames[i + 1] - frames[i]); d[d < 20] = 0
    if d.sum() < 1e3:
        continue
    x = (np.arange(224)[None, :] * d).sum() / d.sum() / 224
    if vL > 0.004 and vR < 0.002:
        cx["L"].append(x)
    elif vR > 0.004 and vL < 0.002:
        cx["R"].append(x)
for a in "LR":
    if cx[a]:
        print(f"{a}-arm-only frames {len(cx[a])}: motion centroid median {np.median(cx[a]):.2f} (0 = image left)")
if cx["L"]:
    v = np.median(cx["L"])
    print("VERDICT:", "left arm appears on the IMAGE LEFT as in training -> V4_GLOBAL_ROT180=0 correct" if v < 0.5 else
          "left arm appears on the IMAGE RIGHT, opposite to training -> live camera rotated: deploy with V4_GLOBAL_ROT180=1")
else:
    print("no left-arm-only motion captured; move only the left leader arm and rerun")
