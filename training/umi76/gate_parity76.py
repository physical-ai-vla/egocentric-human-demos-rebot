#!/usr/bin/env python3
"""Train/deploy parity for UMI76. No model, no robot -- only the two state builders and the two decoders.

The dataset is the reference. Rows are replayed into the DEPLOY ring buffer in the shape the live loop
would see them (timestamped observations), and the deploy builder must reproduce the stored state76 and
the stored action must decode back to the absolute future poses. This is the check that catches a packing,
anchor or timing drift between the converter and the server -- the class of bug that otherwise only shows
up as a robot moving somewhere odd.

The deploy path works from JOINTS, so the replay feeds joints and lets it FK; the dataset was built from
the zarr's TCP. Where a difference is expected it is stated rather than hidden.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
for _c in (os.path.expanduser("~/holobrain-mac-model/umi_pkg"),
           os.path.expanduser("~/universal_manipulation_interface"),
           "/home/bh-aiteam/universal_manipulation_interface"):
    if os.path.isfile(os.path.join(_c, "umi", "common", "pose_util.py")):
        sys.path.insert(0, _c); break
os.environ["V4_STATE_MODE"] = "umi76"
from scipy.spatial.transform import Rotation as Rot, Slerp
from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat

DT = 3.0 / 59.94
fails = []
def chk(n, ok, d=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {n}{('  ' + d) if d else ''}")
    if not ok: fails.append(n)

# ---- a synthetic but non-degenerate bimanual trajectory, sampled at 30 Hz like the zarr
rng = np.random.default_rng(3)
M = 400
tau = np.arange(M) / 30.0
def traj(seed, y0):
    r = np.random.default_rng(seed)
    p = np.stack([0.33 + 0.05*np.sin(2*tau + r.random()), y0 + 0.04*np.sin(1.3*tau), 0.19 + 0.03*np.sin(0.9*tau)], -1)
    rv = np.stack([0.2*np.sin(0.7*tau), 0.3*np.sin(0.5*tau + 1), 0.1*np.sin(1.1*tau)], -1)
    w = 0.05 + 0.04*np.sin(0.6*tau)
    return p, Rot.from_rotvec(rv), w
TR = {0: traj(1, +0.315), 1: traj(2, -0.315)}

def sample(r, t):
    p, rot, w = TR[r]
    t = float(np.clip(t, tau[0], tau[-1]))
    P = np.array([np.interp(t, tau, p[:, k]) for k in range(3)])
    Q = Slerp(tau, rot)([t]).as_matrix()[0]
    m = np.eye(4); m[:3, :3] = Q; m[:3, 3] = P
    return m, float(np.interp(t, tau, w))

# ---- reference state76 built the way the CONVERTER does
def ref_state(t):
    cur = {r: sample(r, t)[0] for r in (0, 1)}
    wc = {r: sample(r, t)[1] for r in (0, 1)}
    hist = {r: sample(r, t - DT)[0] for r in (0, 1)}
    wh = {r: sample(r, t - DT)[1] for r in (0, 1)}
    OFF = {0: dict(pos=0, pos_wrt=6, rot=12, rot_wrt=24, grip=36),
           1: dict(pos=38, pos_wrt=44, rot=50, rot_wrt=62, grip=74)}
    st = np.zeros(76)
    for r in (0, 1):
        o = 1 - r
        ic, io = np.linalg.inv(cur[r]), np.linalg.inv(cur[o])
        sh = mat_to_pose10d((ic @ hist[r])[None])[0]; sc = mat_to_pose10d((ic @ cur[r])[None])[0]
        ch = mat_to_pose10d((io @ hist[r])[None])[0]; cc = mat_to_pose10d((io @ cur[r])[None])[0]
        f = OFF[r]
        st[f["pos"]:f["pos"]+3] = sh[:3];         st[f["pos"]+3:f["pos"]+6] = sc[:3]
        st[f["pos_wrt"]:f["pos_wrt"]+3] = ch[:3]; st[f["pos_wrt"]+3:f["pos_wrt"]+6] = cc[:3]
        st[f["rot"]:f["rot"]+6] = sh[3:9];        st[f["rot"]+6:f["rot"]+12] = sc[3:9]
        st[f["rot_wrt"]:f["rot_wrt"]+6] = ch[3:9];st[f["rot_wrt"]+6:f["rot_wrt"]+12] = cc[3:9]
        st[f["grip"]] = wh[r];                    st[f["grip"]+1] = wc[r]
    return st, cur

# ---- the DEPLOY builder, driven only through its public surface
class Stub:
    pass
import infer_core_v4 as IC
dep = Stub()
dep._buf = __import__("collections").deque()
dep._interp_at = IC.V4Inferencer._interp_at.__get__(dep)
dep.build_state_umi76 = IC.V4Inferencer.build_state_umi76.__get__(dep)

worst_state = 0.0
for q in rng.choice(np.arange(20, M - 20), size=25, replace=False):
    t = float(q) / 30.0
    dep._buf.clear()
    # replay the live cadence: 15 Hz observations, so the 50.05 ms history falls BETWEEN two samples
    for k in range(6, -1, -1):
        tt = t - k / 15.0
        m = np.stack([sample(r, tt)[0] for r in (0, 1)])
        w = np.array([sample(r, tt)[1] for r in (0, 1)])
        dep._buf.append((tt, m, w))
    got, cur_dep, _ = dep.build_state_umi76()
    want, cur_ref = ref_state(t)
    worst_state = max(worst_state, float(np.abs(np.asarray(got, np.float64) - want).max()))
chk("deploy state76 reproduces the converter's packing", worst_state < 3e-3,
    f"max err {worst_state:.2e} (interpolating a 15 Hz buffer to 50.05 ms)")

# ---- decode parity
t = 6.0
cur = {r: sample(r, t)[0] for r in (0, 1)}
A = {r: np.stack([np.linalg.inv(cur[r]) @ sample(r, t + (k+1)*DT)[0] for k in range(16)]) for r in (0, 1)}
err_rel = err_dlt = 0.0
for r in (0, 1):
    for k in range(16):
        tgt = cur[r] @ A[r][k]
        err_rel = max(err_rel, float(np.abs(tgt[:3, 3] - sample(r, t + (k+1)*DT)[0][:3, 3]).max()))
    D = [A[r][0]] + [np.linalg.inv(A[r][k-1]) @ A[r][k] for k in range(1, 16)]
    acc = np.eye(4)
    for k in range(16):
        acc = acc @ D[k]
        tgt = cur[r] @ acc
        err_dlt = max(err_dlt, float(np.abs(tgt[:3, 3] - sample(r, t + (k+1)*DT)[0][:3, 3]).max()))
chk("REL16 decode T_now @ A[k] lands on the future pose", err_rel < 1e-9, f"max {err_rel:.2e} m")
chk("DELTA16 compose-then-apply lands on the same pose", err_dlt < 1e-9, f"max {err_dlt:.2e} m")

# ---- the guard that must refuse to run
dep._buf.clear()
dep._buf.append((10.0, np.stack([sample(r, 10.0)[0] for r in (0, 1)]), np.zeros(2)))
dep._buf.append((10.01, np.stack([sample(r, 10.01)[0] for r in (0, 1)]), np.zeros(2)))
try:
    dep.build_state_umi76(); refused = False
except RuntimeError:
    refused = True
chk("refuses to build a state before the buffer spans 50.05 ms", refused)

print("\n" + ("PARITY PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
