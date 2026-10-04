"""[2026-10-02] PHYSICAL axis check of the ego (HandUMI) CART20 tool frame against the RAW wrist videos, with the robot as reference.
For pairs (t, t+~0.27 s): wrist-camera optical flow (Farneback on 160x90 grey) -> mean u (right +), mean v (down +), divergence
(looming, + = camera moving toward the scene); and the TCP-frame relative motion inv(T_t) T_t+dt (translation mm, rotation deg).
Report corr(flow stat, each TCP translation/rotation axis). If the ego tool frame matches the robot's, the sign/axis pattern must match."""
import glob, json, os, sys, numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
H = os.path.expanduser("~")
def flow_stats(a, b):
    g0 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    g0 = cv2.resize(g0, (160, 90)); g1 = cv2.resize(g1, (160, 90))
    f = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = f[..., 0], f[..., 1]; du = np.gradient(u, axis=1); dv = np.gradient(v, axis=0)
    c = np.s_[10:80, 15:145]                                   # skip borders
    return float(u[c].mean()), float(v[c].mean()), float((du + dv)[c].mean())
def rel(T0, T1):
    A = np.linalg.inv(T0) @ T1; return A[:3, 3] * 1000, Rot.from_matrix(A[:3, :3]).as_rotvec(degrees=True)
def report(name, F, M):
    F = np.array(F); M = np.array(M); lab = ["tx", "ty", "tz", "rx", "ry", "rz"]
    print(f"== {name}: {len(F)} pairs | |t| p50 {np.median(np.linalg.norm(M[:, :3], axis=1)):.1f} mm, |r| p50 {np.median(np.linalg.norm(M[:, 3:], axis=1)):.1f} deg")
    for j, fn in enumerate(("mean_u", "mean_v", "diverg")):
        cc = [np.corrcoef(F[:, j], M[:, k])[0, 1] for k in range(6)]
        print(f"   {fn:7s} " + "  ".join(f"{l} {c:+.2f}" for l, c in zip(lab, cc)))
def to_T(p, q):   # wxyz
    T = np.eye(4); T[:3, :3] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = p; return T
# ---- ego RAW: raw export poses (SLAM * IMU scale * camera->TCP * tool X) + raw wrist mp4
eps = sorted(os.listdir(f"{H}/c8/ego_cart20_v2_raw")); rng = np.random.default_rng(0); eps = list(rng.choice(eps, 14, replace=False))
for side in ("left", "right"):
    F, M = [], []
    for e in eps:
        d = f"{H}/c8/ego_cart20_v2_raw/{e}"; Z = np.load(f"{d}/raw_episode.npz"); J = json.load(open(f"{d}/raw_episode.json"))
        p, q, v, vf = Z[f"{side}_position"], Z[f"{side}_quaternion"], Z[f"{side}_valid"], Z[f"{side}_video_frame"]
        cap = cv2.VideoCapture(J["videos"][f"{side}_wrist"]); frames = {}; want = set(int(x) for x in vf[::4]) | set(int(x) for x in vf[8::4])
        k = -1
        while True:
            ok, fr = cap.read(); k += 1
            if not ok or k > max(want): break
            if k in want: frames[k] = cv2.resize(fr, (320, 180))
        for i in range(0, len(vf) - 8, 4):
            j = i + 8
            if not (v[i] and v[j]) or int(vf[i]) not in frames or int(vf[j]) not in frames: continue
            t, r = rel(to_T(p[i], q[i]), to_T(p[j], q[j]))
            if np.linalg.norm(t) < 5 and np.linalg.norm(r) < 2: continue
            F.append(flow_stats(frames[int(vf[i])], frames[int(vf[j])])); M.append(np.r_[t, r])
    report(f"EGO raw UMI {side} wrist ({len(eps)} episodes)", F, M)
# ---- robot reference: HEAD180 wrist images + FK TCP from aux.q_t (15 Hz, pairs 4 rows apart)
os.environ.update(V4_STATE_MODE="relcart20", V4_ACTION_MODE="umi")
sys.path.insert(0, f"{H}/holobrain-mac-model"); import infer_core_v4 as IC
import pyarrow.parquet as pq, pyarrow as pa
from lerobot.datasets.lerobot_dataset import LeRobotDataset
col = lambda t, c: (lambda a: a.storage if isinstance(a, pa.ExtensionArray) else a)(t.column(c).combine_chunks())
FEED = f"{H}/holobrain-data/lerobot/r180_relcart20_rel16_v3d"; ROOT = f"{H}/holobrain-data/lerobot/r180_umi76_rel16_v3d"
ft = [pq.read_table(f, columns=["index", "episode_index", "aux.q_t"]) for f in sorted(glob.glob(f"{FEED}/data/chunk-000/*.parquet"))]
IX = np.concatenate([col(t, "index").to_numpy() for t in ft]); EP = np.concatenate([col(t, "episode_index").to_numpy() for t in ft])
Q = np.concatenate([np.asarray(col(t, "aux.q_t").to_pylist()) for t in ft])
ds = LeRobotDataset("rebot/r180", root=ROOT, video_backend="pyav")
class K_: kin = IC.eef_kin.Kin(); _tcp_mat = IC.V4Inferencer._tcp_mat
kk = K_(); pos = {int(x): n for n, x in enumerate(IX)}
cand = [n for n in range(0, len(IX) - 4, 7) if EP[n] == EP[n + 4]]; cand = list(np.random.default_rng(1).choice(cand, 700, replace=False))
for side, a in (("left", 0), ("right", 1)):
    F, M = [], []
    for n in cand:
        x0 = ds[int(IX[n])]; x1 = ds[int(IX[n + 4])]
        im = lambda x: cv2.cvtColor((x[f"observation.images.{side}_wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        q0 = np.zeros(14); q0[IC.ARM_IDX] = Q[n]; q1 = np.zeros(14); q1[IC.ARM_IDX] = Q[n + 4]
        t, r = rel(kk._tcp_mat(q0)[0][a], kk._tcp_mat(q1)[0][a])
        if np.linalg.norm(t) < 5 and np.linalg.norm(r) < 2: continue
        F.append(flow_stats(cv2.resize(im(x0), (320, 180)), cv2.resize(im(x1), (320, 180)))); M.append(np.r_[t, r])
    report(f"ROBOT R312c-HEAD180 {side} wrist", F, M)
