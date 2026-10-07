"""[2026-09-28] Training RGB <-> deployment RGB parity (read-only; the robot is only OBSERVED, never commanded).

 P1 preprocessing: a recorded HEAD source frame (R150 headview mp4, the camera the live rig uses) is JPEG-encoded the
    way robot_service does (q80), run through the deployed infer_core_v4 V4Inferencer._img (GLOBAL_ROT180 as deployed)
    and compared with the stored training frame (same episode/frame). rot0 must match; rot180 must be far worse.
 P2 live camera: one /observe global frame from robot_service, edge-map correlated with the mean canonical training
    global views (HEAD180 and the rot180-corrected FRONT from the training set) as-is and rotated 180. The deployed
    setting (V4_GLOBAL_ROT180=0) is right only if the as-is orientation wins.
"""
import base64, glob, os, sys, json
import numpy as np, cv2, torch, requests, av
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ.setdefault("V4_STATE_MODE", "umi94")
import infer_core_v4 as IC
from lerobot.datasets.lerobot_dataset import LeRobotDataset

L = os.path.expanduser("~/holobrain-data/lerobot")
ds = LeRobotDataset("rebot/r380v2B", root=f"{L}/r380_umi94_rel16_v2B", video_backend="pyav")      # HEAD180 = same frames as R312c
import pandas as pd
srcm = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{L}/src_rebot_3stack_R150_headview/meta/episodes/**/*.parquet", recursive=True)])
print(f"deployed V4_GLOBAL_ROT180 = {IC.GLOBAL_ROT180}, V4_GLOBAL_MIRROR = {IC.GLOBAL_MIRROR}")

# ---- P1
def src_frame(ep, fi):
    row = srcm[srcm.episode_index == ep].iloc[0]
    k = "observation.images.global"
    path = f"{L}/src_rebot_3stack_R150_headview/videos/{k}/chunk-{int(row[f'videos/{k}/chunk_index']):03d}/file-{int(row[f'videos/{k}/file_index']):03d}.mp4"
    t = float(row[f"videos/{k}/from_timestamp"]) + fi / 30.0
    with av.open(path) as c:
        st = c.streams.video[0]; c.seek(int(max(t - 0.5, 0) / float(st.time_base)), stream=st)
        for f in c.decode(st):
            if float(f.pts * st.time_base) >= t - 1 / 60:
                return f.to_ndarray(format="bgr24")


res = []
for ep in (3, 41, 88, 120, 149):
    j = int(ds.meta.episodes[ep]["dataset_from_index"]) + 30
    x = ds[j]; fi = 2 * int(x["frame_index"]) + 2
    bgr = src_frame(ep, fi)
    b64 = base64.b64encode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()).decode()
    run0 = IC.V4Inferencer._img(b64, rot180=False)            # the deployed path with GLOBAL_ROT180=0
    run180 = IC.V4Inferencer._img(b64, rot180=True)
    tr = x["observation.images.global"]
    res.append((float((run0 - tr).abs().mean()), float((run180 - tr).abs().mean())))
    print(f"  P1 ep{ep} f{fi}: mean|runtime rot0 - training| {res[-1][0]:.4f}   rot180 {res[-1][1]:.4f}")
p1 = all(a < 0.03 and b > 4 * a for a, b in res)
print(("PASS" if p1 else "FAIL") + "  P1 deployed preprocessing (JPEG q80 -> _img, rot0) reproduces the training frame; rot180 far worse")

# ---- P2
def edge(img_rgb01):
    g = cv2.cvtColor((img_rgb01.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    e = cv2.GaussianBlur(cv2.resize(cv2.Canny(g, 60, 150).astype(np.float32), (48, 48)), (5, 5), 0).ravel()
    return (e - e.mean()) / (e.std() + 1e-6)


H = np.mean([edge(ds[int(ds.meta.episodes[e]["dataset_from_index"])]["observation.images.global"]) for e in range(0, 180, 6)], 0)
F = np.mean([edge(ds[int(ds.meta.episodes[e]["dataset_from_index"])]["observation.images.global"]) for e in range(180, 380, 7)], 0)
H, F = (H - H.mean()) / H.std(), (F - F.mean()) / F.std()
o = requests.get("http://localhost:8020/observe", timeout=30).json()
live = IC.V4Inferencer._img(o["images"]["middle"], rot180=False)
live180 = torch.flip(live, dims=[1, 2])
lv, lv180 = edge(live), edge(live180)
c = {k: float(np.dot(v, ref) / v.size) for k, v, ref in (("HEAD rot0", lv, H), ("HEAD rot180", lv180, H), ("FRONTcanon rot0", lv, F), ("FRONTcanon rot180", lv180, F))}
for k, v in c.items():
    print(f"  P2 live global vs {k:18s}: corr {v:+.3f}")
p2 = c["HEAD rot0"] > c["HEAD rot180"] + 0.1
print(("PASS" if p2 else "FAIL") + f"  P2 live camera matches the canonical training view as-is (rot0) -> V4_GLOBAL_ROT180=0 is correct")
cv2.imwrite(os.path.expanduser("~/umi_bridge/rel16_audit/r380/live_global_now.png"), (live.permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8))
json.dump(dict(p1=res, p1_pass=p1, p2=c, p2_pass=p2), open(os.path.expanduser("~/umi_bridge/rel16_audit/r380/image_parity.json"), "w"), indent=1)
