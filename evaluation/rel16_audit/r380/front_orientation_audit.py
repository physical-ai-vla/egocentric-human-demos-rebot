"""[2026-09-28] FRONT200 observation audit before training REL16-v2B (read-only).

For every R380 episode (HEAD 180 + FRONT 200) on the TRAINING images (global view after the converter's rot180 for FRONT):
 A orientation signature: 32x32 grey of the first global frame, correlated with its pool's mean signature and with
   rot180 of it. An episode that matches rot180(mean) better than mean is flagged (missing / double rotation).
 B action<->vision side test: frames where only ONE arm moves (TCP speed from state94 TCP18); the horizontal centroid of
   |global(t+3) - global(t)| should fall on the same image side for the same arm in every episode and in both pools
   (a missing rot180 or a mirror would put the left arm's motion on the other side). Vertical: when the moving arm
   rises (dz > 0), the motion centroid's vertical shift sign should be consistent too.
 C provenance table: episode | pool | source ep | set_id | set key (session, pan) | order | first-target | orientation
   score | L-side fraction | in R280 FRONT100 -- plus the layout->order 1-NN split by group.
Outputs in r380/: front_orientation_table.csv, front_contact_sheet_{1,2}.png, HEAD sheet, summary printed.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, cv2
sys.path.insert(0, os.path.expanduser("~/umi_bridge/umi76"))
import derive_v2 as DV
from lerobot.datasets.lerobot_dataset import LeRobotDataset

L = os.path.expanduser("~/holobrain-data/lerobot"); N = "/home/bh-aiteam/holobrain-data/lerobot"
DSR = f"{L}/r380_umi94_rel16_v2B"; HERE = os.path.expanduser("~/umi_bridge/rel16_audit/r380")
DV.REMAP.update({f"{N}/rebot_3stack_R150_headview": f"{L}/src_rebot_3stack_R150_headview",
                 f"{N}/rebot_3stack_R30_day4_headview": f"{L}/src_rebot_3stack_R30_day4_headview",
                 f"{N}/rebot_3stack_center675_s96": f"{L}/src_rebot_3stack_center675_s96"})
src_of = DV.row_sources(DSR, f"{HERE}/r675_rbp.json")
sm = json.load(open(f"{HERE}/set_map.json")); SET = sm["episode_sets"]; KEYS = sm["set_keys"]
sel280 = set()
try:
    sel = json.load(open(os.path.expanduser("~/umi_bridge/umi76/r280_front100_v1.json")))
    sel280 = {int(x) for x in sel["zarr_episodes"]}
except OSError:
    pass
sel380 = json.load(open(f"{HERE}/r380_front200_v1.json"))["zarr_episodes"]
ds = LeRobotDataset("rebot/r380v2B", root=DSR, video_backend="pyav")


def col(t, c):
    a = t.column(c).combine_chunks(); return a.storage if isinstance(a, pa.ExtensionArray) else a


tabs = [pq.read_table(f, columns=["observation.state", "episode_index", "frame_index", "task_index"]) for f in sorted(glob.glob(f"{DSR}/data/chunk-*/*.parquet"))]
S = np.concatenate([np.asarray(col(t, "observation.state").to_pylist()) for t in tabs])
EP = np.concatenate([np.asarray(col(t, "episode_index").to_numpy()) for t in tabs])
TS = {int(v): k for k, v in pd.read_parquet(f"{DSR}/meta/tasks.parquet")["task_index"].items()}
TK = np.concatenate([np.asarray(col(t, "task_index").to_numpy()) for t in tabs])


def grey(x):
    im = (x["observation.images.global"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)


rows, thumbs = [], {}
rng = np.random.default_rng(0)
for e in range(380):
    s0 = int(ds.meta.episodes[e]["dataset_from_index"]); n = int(ds.meta.episodes[e]["length"])
    x0 = ds[s0]; g0 = grey(x0)
    thumbs[e] = cv2.resize((x0["observation.images.global"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), (112, 112))
    sig = cv2.resize(g0, (32, 32)).ravel(); sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    # B: single-arm-motion frames
    m = np.arange(s0, s0 + n - 3)
    vL = np.linalg.norm(S[m + 3, 76:79] - S[m, 76:79], axis=1); vR = np.linalg.norm(S[m + 3, 85:88] - S[m, 85:88], axis=1)
    sideL, sideR, vert = [], [], []
    for arm, v, o in (("L", vL - vR, vR), ("R", vR - vL, vL)):
        cand = np.flatnonzero((v > 0.02) & (o < 0.004))
        for j in rng.choice(cand, min(6, len(cand)), replace=False) if len(cand) else []:
            a, b = grey(ds[int(m[j])]), grey(ds[int(m[j]) + 3])
            dif = np.abs(b - a); dif[dif < 20] = 0
            if dif.sum() < 1e3:
                continue
            ys, xs = np.mgrid[0:dif.shape[0], 0:dif.shape[1]]
            cx = (xs * dif).sum() / dif.sum() / dif.shape[1]
            (sideL if arm == "L" else sideR).append(cx)
            dz = S[m[j] + 3, 78 if arm == "L" else 87] - S[m[j], 78 if arm == "L" else 87]
            if abs(dz) > 0.015:
                # vertical: centroid of positive (new) vs negative (vanished) change -> motion direction in image y
                pos = np.clip(b - a, 0, None); neg = np.clip(a - b, 0, None)
                if pos.sum() > 1e3 and neg.sum() > 1e3:
                    vy = (ys * pos).sum() / pos.sum() - (ys * neg).sum() / neg.sum()
                    vert.append(np.sign(dz) * np.sign(vy))
    r, se, zp, zep = src_of[e]
    pool = "HEAD" if "headview" in r else "FRONT"
    setid = SET[se] if pool == "FRONT" else None
    order = "".join(c[0].upper() for c in re.findall(r"(\w+) cube", TS[int(TK[EP == e][0])]))
    rows.append(dict(ep=e, pool=pool, src_ep=se, zarr_ep=zep, set_id=setid, set_key=(KEYS.get(str(setid)) if setid is not None else "HEAD"),
                     order=order, first=order[0], sig=sig, L_cx=np.mean(sideL) if sideL else np.nan, R_cx=np.mean(sideR) if sideR else np.nan,
                     nL=len(sideL), nR=len(sideR), vert=np.mean(vert) if vert else np.nan, nV=len(vert),
                     in_R280=(pool == "FRONT" and zep in sel280)))
    if e % 40 == 0:
        print(f"ep {e}", flush=True)
df = pd.DataFrame(rows)
for p in ("HEAD", "FRONT"):
    mk = df.pool == p; M = np.stack(df.sig[mk]).mean(0); M = (M - M.mean()) / M.std()
    Mr = M.reshape(32, 32)[::-1, ::-1].ravel()
    df.loc[mk, "orient_score"] = [float(np.dot(s, M) - np.dot(s, Mr)) / 1024 for s in df.sig[mk]]
H = np.stack(df.sig[df.pool == "HEAD"]).mean(0); F = np.stack(df.sig[df.pool == "FRONT"]).mean(0)
Hn, Fn = (H - H.mean()) / H.std(), (F - F.mean()) / F.std()
print(f"\n[A] corr(HEAD mean, FRONT mean) = {np.dot(Hn, Fn)/1024:+.3f}; vs rot180(FRONT mean) {np.dot(Hn, Fn.reshape(32,32)[::-1,::-1].ravel())/1024:+.3f}")
for p in ("HEAD", "FRONT"):
    o = df.orient_score[df.pool == p]
    print(f"    {p:5s} orientation score (corr mean - corr rot180 mean): min {o.min():+.3f} p5 {o.quantile(.05):+.3f} med {o.median():+.3f};  flagged (<0) {int((o < 0).sum())}")
print("\n[B] single-arm motion: horizontal image centroid (0 = image left, 1 = right)")
for p in ("HEAD", "FRONT"):
    d = df[df.pool == p]
    print(f"    {p:5s} L-arm-only med {d.L_cx.median():.2f} (episodes with L on the right half: {int((d.L_cx > 0.5).sum())}/{int(d.L_cx.notna().sum())})  "
          f"R-arm-only med {d.R_cx.median():.2f} (R on the left half: {int((d.R_cx < 0.5).sum())}/{int(d.R_cx.notna().sum())})  "
          f"vertical sign agreement {np.nanmean(d.vert):+.2f} (n eps {int(d.vert.notna().sum())})")
df.drop(columns=["sig"]).to_csv(f"{HERE}/front_orientation_table.csv", index=False)
# contact sheets: FRONT by recording order (source episode), labelled with set key
F_ = df[df.pool == "FRONT"].sort_values("src_ep")
for part, chunk in enumerate((F_.iloc[:100], F_.iloc[100:])):
    tiles = []
    for _, r in chunk.iterrows():
        t = thumbs[r.ep][:, :, ::-1].copy()
        cv2.putText(t, f"{r.src_ep} s{r.set_id}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(t, f"{r.order}", (2, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{HERE}/front_contact_sheet_{part+1}.png", np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
Hh = df[df.pool == "HEAD"].iloc[::3]
tiles = [thumbs[e][:, :, ::-1].copy() for e in Hh.ep][:60]
cv2.imwrite(f"{HERE}/head_contact_sheet.png", np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
print("\n[C] FRONT by set key (session / pan):")
print(df[df.pool == "FRONT"].groupby("set_key").agg(n=("ep", "size"), in_R280=("in_R280", "sum"), orient_min=("orient_score", "min"),
      L_cx=("L_cx", "median"), R_cx=("R_cx", "median")).sort_values("n", ascending=False).round(2).to_string())
