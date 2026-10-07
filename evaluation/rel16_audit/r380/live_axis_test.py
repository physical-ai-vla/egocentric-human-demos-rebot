"""[2026-09-28] Global-camera orientation by MOTION SIGN on two axes (+ z), measured, no model, no robot command.

Motion direction in the image = mean Farneback optical flow (du, dv) over pixels that changed, between two frames.
Per axis the statistic is corr(TCP displacement component, image flow component) over frames where only the LEFT arm
moves:  robot-left/right  corr(dy, du)   forward/back  corr(dx, dv)   up/down  corr(dz, dv) (reported, not required).
  ref   : training HEAD180 global frames (t, t+3 at 15 Hz = 0.2 s), TCP from state94 TCP18  -> the canonical signs
  live  : robot_service /observe (only READ) while the operator moves the LEFT leader arm, deployed preprocessing as-is
Verdict: rot0 = live signs on BOTH x and y equal the reference; rot180 = both opposite (a 180 rotation negates du and
dv); anything else = ambiguous -> do not freeze, re-check mounting.
usage: live_axis_test.py ref             (compute + save the training reference; done once)
       live_axis_test.py dataset <dir> <lo> <hi>   (converted vs raw signs for episodes [lo, hi), e.g. FRONT132 180 312)
       live_axis_test.py live [seconds]  (operator moves the LEFT leader arm: left/right, forward/back, up/down)
"""
import base64, json, os, sys, time
import numpy as np, cv2
HERE = os.path.expanduser("~/umi_bridge/rel16_audit/r380"); REF = f"{HERE}/live_axis_ref.json"


def flow(a, b):
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    m = np.abs(b.astype(np.float32) - a.astype(np.float32)) > 20
    if m.sum() < 200:
        return None
    return float(f[..., 0][m].mean()), float(f[..., 1][m].mean())


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 5 and x.std() > 0 and y.std() > 0 else float("nan")


def stats(rows):
    r = np.array(rows)          # dx dy dz du dv
    return dict(n=len(r), lr=corr(r[:, 1], r[:, 3]), fb=corr(r[:, 0], r[:, 4]), ud=corr(r[:, 2], r[:, 4]))


mode = sys.argv[1] if len(sys.argv) > 1 else "live"
if mode == "ref":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq, pyarrow as pa, glob
    D = os.path.expanduser("~/holobrain-data/lerobot/r380_umi94_rel16_v2B")
    ds = LeRobotDataset("rebot/r380v2B", root=D, video_backend="pyav")
    tabs = [pq.read_table(f, columns=["observation.state"]) for f in sorted(glob.glob(f"{D}/data/chunk-*/*.parquet"))]
    c = lambda t: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column("observation.state").combine_chunks())
    S = np.concatenate([np.asarray(c(t).to_pylist()) for t in tabs])
    g = lambda j: cv2.cvtColor((ds[int(j)]["observation.images.global"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    rng = np.random.default_rng(0); rows = []
    for e in range(0, 180, 3):
        s0 = int(ds.meta.episodes[e]["dataset_from_index"]); n = int(ds.meta.episodes[e]["length"])
        j = np.arange(s0, s0 + n - 3)
        dL = S[j + 3, 76:79] - S[j, 76:79]; dR = np.linalg.norm(S[j + 3, 85:88] - S[j, 85:88], axis=1)
        cand = np.flatnonzero((np.linalg.norm(dL, axis=1) > 0.015) & (dR < 0.004))
        for k in rng.choice(cand, min(6, len(cand)), replace=False) if len(cand) else []:
            fl = flow(g(j[k]), g(j[k] + 3))
            if fl:
                rows.append([*dL[k], *fl])
    r = stats(rows); json.dump(r, open(REF, "w"), indent=1)
    print(f"training HEAD reference (left arm only, n {r['n']}): corr(dy,du) {r['lr']:+.2f}  corr(dx,dv) {r['fb']:+.2f}  corr(dz,dv) {r['ud']:+.2f}")
    sys.exit(0)

if mode == "dataset":
    # dataset-side check: signs on the CONVERTED frames of episodes [lo, hi) and on the RAW orientation (converted
    # rotated back by 180, i.e. the source frame -- gate G4 verifies converted == rot180(source)). Only the left arm.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq, pyarrow as pa, glob
    D, lo, hi = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ds = LeRobotDataset("rebot/x", root=D, video_backend="pyav")
    tabs = [pq.read_table(f, columns=["observation.state"]) for f in sorted(glob.glob(f"{D}/data/chunk-*/*.parquet"))]
    c = lambda t: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column("observation.state").combine_chunks())
    S = np.concatenate([np.asarray(c(t).to_pylist()) for t in tabs])
    g = lambda j: cv2.cvtColor((ds[int(j)]["observation.images.global"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    rng = np.random.default_rng(0); conv, raw = [], []
    for e in range(lo, hi, max(1, (hi - lo) // 60)):
        s0 = int(ds.meta.episodes[e]["dataset_from_index"]); n = int(ds.meta.episodes[e]["length"])
        j = np.arange(s0, s0 + n - 3)
        dL = S[j + 3, 76:79] - S[j, 76:79]; dR = np.linalg.norm(S[j + 3, 85:88] - S[j, 85:88], axis=1)
        cand = np.flatnonzero((np.linalg.norm(dL, axis=1) > 0.015) & (dR < 0.004))
        for k in rng.choice(cand, min(6, len(cand)), replace=False) if len(cand) else []:
            a, b = g(j[k]), g(j[k] + 3)
            f1 = flow(a, b); f2 = flow(cv2.rotate(a, cv2.ROTATE_180), cv2.rotate(b, cv2.ROTATE_180))
            if f1: conv.append([*dL[k], *f1])
            if f2: raw.append([*dL[k], *f2])
    ref = json.load(open(REF)); rc, rr = stats(conv), stats(raw)
    print(f"HEAD ref         : corr(dy,du) {ref['lr']:+.2f}  corr(dx,dv) {ref['fb']:+.2f}  corr(dz,dv) {ref['ud']:+.2f}")
    print(f"converted [{lo},{hi}): corr(dy,du) {rc['lr']:+.2f}  corr(dx,dv) {rc['fb']:+.2f}  corr(dz,dv) {rc['ud']:+.2f}  (n {rc['n']})")
    print(f"raw (unrotated)  : corr(dy,du) {rr['lr']:+.2f}  corr(dx,dv) {rr['fb']:+.2f}  corr(dz,dv) {rr['ud']:+.2f}")
    ok = all(np.sign(rc[k]) == np.sign(ref[k]) and abs(rc[k]) > 0.3 for k in ("lr", "fb"))
    print("PASS  converted FRONT signs == HEAD reference on x and y" if ok else "FAIL  converted FRONT signs differ from HEAD / too weak")
    json.dump(dict(ref=ref, converted=rc, raw=rr, pass_=bool(ok)), open(f"{HERE}/dataset_axis_{os.path.basename(D.rstrip('/'))}_{lo}_{hi}.json", "w"), indent=1)
    sys.exit(0 if ok else 1)

import requests
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model")); import infer_core_v4 as IC
ref = json.load(open(REF)); T = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
kin = IC.eef_kin.Kin(); obs = []
print(f"recording {T:.0f}s -- move ONLY the LEFT leader arm: left/right, forward/back, up/down", flush=True)
t0 = time.time()
while time.time() - t0 < T:
    o = requests.get("http://localhost:8020/observe", timeout=10).json()
    im = cv2.imdecode(np.frombuffer(base64.b64decode(o["images"]["middle"]), np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)   # deployed geometry, rot0
    q = np.asarray(o["joints_rad"], np.float64); m, _ = IC.V4Inferencer._tcp_mat(type("K", (), {"kin": kin})(), q)
    obs.append((g, m[0][:3, 3].copy(), m[1][:3, 3].copy()))
rows = []
for (ga, La, Ra), (gb, Lb, Rb) in zip(obs[:-1], obs[1:]):
    if np.linalg.norm(Lb - La) > 0.006 and np.linalg.norm(Rb - Ra) < 0.003:
        fl = flow(ga, gb)
        if fl:
            rows.append([*(Lb - La), *fl])
if len(rows) < 8:
    print(f"only {len(rows)} usable left-arm-only frame pairs; move the left arm more (and only it) and rerun"); sys.exit(1)
lv = stats(rows)
print(f"training ref : corr(dy,du) {ref['lr']:+.2f}  corr(dx,dv) {ref['fb']:+.2f}  corr(dz,dv) {ref['ud']:+.2f}")
print(f"live (rot0)  : corr(dy,du) {lv['lr']:+.2f}  corr(dx,dv) {lv['fb']:+.2f}  corr(dz,dv) {lv['ud']:+.2f}   (n {lv['n']})")
same = [np.sign(lv[k]) == np.sign(ref[k]) and abs(lv[k]) > 0.3 for k in ("lr", "fb")]
opp = [np.sign(lv[k]) == -np.sign(ref[k]) and abs(lv[k]) > 0.3 for k in ("lr", "fb")]
print("VERDICT:", "rot0 -- both axes match training -> V4_GLOBAL_ROT180=0" if all(same) else
      "rot180 -- both axes opposite to training -> V4_GLOBAL_ROT180=1" if all(opp) else
      "AMBIGUOUS -- axes disagree or too weak (|corr| <= 0.3): do not freeze; re-check mounting / move more")
json.dump(dict(ref=ref, live=lv, verdict=("rot0" if all(same) else "rot180" if all(opp) else "ambiguous"), t=time.strftime("%F %T")),
          open(f"{HERE}/live_axis_result.json", "w"), indent=1)
