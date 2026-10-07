"""[2026-10-06 user] Start-frame view comparison, model-input resolution (224x224):
training HRA episodes (robotized right wrist + head 'global', first row of each episode) vs robot HRA run starts (cycle 0 frames).
Red cube = largest red HSV blob; visible / centroid / area fraction per view."""
import base64, glob, json, os, pathlib
import cv2, numpy as np
H = pathlib.Path.home(); OUT = H / "c8/hra_red/start_view"
SH = H / "c8/hra_red/lerobot/ego_hra_red_rightonly_v1_shards"


def red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); a = st[i, cv2.CC_STAT_AREA]
    if a < 12: return None
    return dict(u=cen[i][0] / img.shape[1], v=cen[i][1] / img.shape[0], area=a / (img.shape[0] * img.shape[1]))


def first_frame(mp4):
    c = cv2.VideoCapture(str(mp4)); ok, f = c.read(); c.release(); return f if ok else None


train = {"wrist": [], "global": []}; timgs = {"wrist": [], "global": []}
for d in sorted(glob.glob(str(SH / "*" / "HRA_red_*"))):
    if d.endswith(".complete"): continue
    for view, key in (("wrist", "right_wrist"), ("global", "global")):
        f = first_frame(next(pathlib.Path(d).glob(f"videos/observation.images.{key}/chunk-000/*.mp4")))
        if f is None: continue
        f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA); train[view].append(red(f)); timgs[view].append(f)
robot = {"wrist": [], "global": []}; rimgs = {"wrist": [], "global": []}
for fn in sorted(glob.glob(os.path.expanduser("~/v4_cycle_frames/*.json")))[-8000:]:
    try: d = json.load(open(fn))
    except Exception: continue
    if "HRA" not in str(d.get("ckpt", "")) or d.get("cycle") != 0: continue
    im = d["obs"][-1]["images"]
    for view, key in (("wrist", "right"), ("global", "middle")):
        f = cv2.imdecode(np.frombuffer(base64.b64decode(im[key]), np.uint8), 1)
        f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA); robot[view].append(red(f)); rimgs[view].append(f)


def summ(name, L):
    vis = [x for x in L if x]; n = len(L)
    if not vis: return f"{name:22s} n {n:3d}  cube visible 0%"
    U = np.array([x["u"] for x in vis]); V = np.array([x["v"] for x in vis]); A = np.array([x["area"] for x in vis]) * 100
    return (f"{name:22s} n {n:3d}  cube visible {100*len(vis)/n:5.1f}%  u(left->right) p10/50/90 {np.percentile(U,10):.2f}/{np.median(U):.2f}/{np.percentile(U,90):.2f}"
            f"  v(top->bottom) {np.percentile(V,10):.2f}/{np.median(V):.2f}/{np.percentile(V,90):.2f}  area% {np.percentile(A,10):.2f}/{np.median(A):.2f}/{np.percentile(A,90):.2f}")


for view in ("wrist", "global"):
    print(summ(f"train {view}", train[view])); print(summ(f"robot {view}", robot[view]))
rng = np.random.default_rng(0)
def grid(imgs, k=8):
    idx = np.linspace(0, len(imgs) - 1, k).astype(int); return np.hstack([imgs[i] for i in idx])
for view in ("wrist", "global"):
    cv2.imwrite(str(OUT / f"start_{view}.jpg"), np.vstack([grid(timgs[view]), grid(rimgs[view])]))
json.dump({k: {"train": train[k], "robot": robot[k]} for k in train}, open(OUT / "start_views.json", "w"))
