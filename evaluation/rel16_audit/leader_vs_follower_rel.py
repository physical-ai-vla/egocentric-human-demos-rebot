#!/usr/bin/env python3
"""[2026-09-29] REL16 label audit: the stored label is built from the FOLLOWER's measured future TCP; C-old learns the LEADER
command. Same rows, same anchor (follower TCP at t, as deployed), same FK (rebot_fk_torch = dataset TCP), same timing as
derive_v3d (row f = source frame 2f+2, target k = +1.5015*(k+1) source frames, linear interpolation):
  REL_f(k) = inv(T_f(t)) T_f(t+k)   (== the stored label; checked)
  REL_l(k) = inv(T_f(t)) T_l(t+k)   (leader command; the source action is already in the follower frame, fit a=1.00)
Reports per arm: |REL| p50 at k1/4/8/16, ratio leader/follower, direction cos, the static leader-follower offset at t,
the best time lag (follower(t+k) ~ leader(t+k-lag)), and the same on rest_go rows (arm still, then moves).
Read-only on the Mac copies of the source datasets.
"""
import glob, json, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
os.environ.setdefault("REBOT_URDF", os.path.expanduser("~/holobrain-mac-model/reBot_B601_DM_dualarm.urdf"))
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model")); sys.path.insert(0, os.path.expanduser("~/c8/c8old"))
import rebot_fk_torch

R = os.path.expanduser("~/holobrain-data/lerobot/")
MAC = {"rebot_3stack_R150_headview": R + "src_rebot_3stack_R150_headview", "rebot_3stack_R30_day4_headview": R + "src_rebot_3stack_R30_day4_headview"}
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; FLIP = [0, 1, 5, 7, 8, 12]; DT = 30.0 * 3.0 / 59.94
col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
V = R + "r180_umi76_rel16_v3d"
prov = json.load(open(V + "/v4_provenance.json"))
srcmap, e = {}, 0
for s in prov["sources"]:
    root = MAC[os.path.basename(s["lerobot"].rstrip("/"))]
    assert s.get("select_zarr_episodes") is None, "episode selection not handled here"
    n = json.load(open(root + "/meta/info.json"))["total_episodes"]
    for z in range(n):
        srcmap[e] = (root, z); e += 1
src = {}
for root in set(r for r, _ in srcmap.values()):
    ts = [pq.read_table(f, columns=["observation.state", "action", "episode_index"]) for f in sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))]
    St = np.concatenate([np.asarray(col(t, "observation.state").to_pylist(), np.float64) for t in ts])
    Ac = np.concatenate([np.asarray(col(t, "action").to_pylist(), np.float64) for t in ts])
    Ep = np.concatenate([col(t, "episode_index").to_numpy() for t in ts])
    src[root] = {int(z): (St[Ep == z], Ac[Ep == z]) for z in np.unique(Ep)}
ts = [pq.read_table(f, columns=["episode_index", "frame_index", "action", "observation.state"]) for f in sorted(glob.glob(V + "/data/chunk-*/*.parquet"))]
E = np.concatenate([col(t, "episode_index").to_numpy() for t in ts]); F = np.concatenate([col(t, "frame_index").to_numpy() for t in ts])
A = np.concatenate([np.asarray(col(t, "action").flatten().flatten().to_numpy(), np.float32).reshape(-1, 16, 32) for t in ts])
S76 = np.concatenate([np.asarray(col(t, "observation.state").to_pylist()) for t in ts])
rows = np.arange(0, len(E), 3)                                                        # every 3rd row (~34k)
KS = np.arange(-6, 17)                                                                 # offsets for the lag search (k<0 = past)
qf = np.zeros((len(rows), len(KS), 12)); ql = np.zeros((len(rows), len(KS), 12))
for n, i in enumerate(rows):
    root, z = srcmap[int(E[i])]; St, Ac = src[root][z]; m = len(St); i0 = 2 * int(F[i]) + 2
    tk = np.clip(i0 + DT * KS, 0, m - 1)
    lead = Ac.copy()          # [2026-09-29] this source's action is ALREADY in the follower frame (per-joint fit a=1.00, b~0): no FLIP
    qf[n] = np.radians(np.stack([np.interp(tk, np.arange(m), St[:, j]) for j in ARM], -1))
    ql[n] = np.radians(np.stack([np.interp(tk, np.arange(m), lead[:, j]) for j in ARM], -1))
fk = rebot_fk_torch.ReBotFKTorch(dtype=torch.float64)
with torch.no_grad():
    Tf = fk.tcp(torch.tensor(qf.reshape(-1, 12))).numpy().reshape(len(rows), len(KS), 2, 4, 4)
    Tl = fk.tcp(torch.tensor(ql.reshape(-1, 12))).numpy().reshape(len(rows), len(KS), 2, 4, 4)
i0k = int(np.flatnonzero(KS == 0)[0])
inv0 = np.linalg.inv(Tf[:, i0k])                                                       # (n,2,4,4) follower at t
pf = np.einsum("naij,nkaj->nkai", inv0[..., :3, :3], Tf[..., :3, 3] - Tf[:, i0k, None, :, :3, 3]) * 1000
pl = np.einsum("naij,nkaj->nkai", inv0[..., :3, :3], Tl[..., :3, 3] - Tf[:, i0k, None, :, :3, 3]) * 1000
# check: follower REL at k = label
lab = np.stack([A[rows][:, :, 0:3], A[rows][:, :, 10:13]], 2) * 1000                  # (n,16,2,3)
kk = [int(np.flatnonzero(KS == k + 1)[0]) for k in range(16)]
print(f"rows {len(rows)} | follower-REL vs stored label: max |diff| {np.abs(pf[:, kk] - lab).max():.3f} mm (label = follower future)")
hist = np.stack([np.linalg.norm(S76[rows][:, o:o + 3], axis=1) * 1000 for o in (0, 38)], 1)
g16 = np.linalg.norm(lab[:, 15], axis=-1)
for ai, arm in ((0, "L"), (1, "R")):
    mv = g16[:, ai] > 20; rg = (hist[:, ai] < 0.5) & (g16[:, ai] > 30)
    print(f"\n[{arm}] moving rows {mv.sum()}, rest_go rows {rg.sum()}")
    off = np.linalg.norm(pl[:, i0k, ai], axis=-1)
    print(f"  static leader-follower TCP offset at t: p50 {np.median(off):.1f} p90 {np.percentile(off, 90):.1f} mm; mean vector {np.round(pl[mv][:, i0k, ai].mean(0), 1)} mm (T_f(t) frame)")
    for name, msk in (("moving", mv), ("rest_go", rg)):
        out = []
        for k in (1, 4, 8, 16):
            j = int(np.flatnonzero(KS == k)[0])
            nf = np.linalg.norm(pf[msk, j, ai], axis=-1); nl = np.linalg.norm(pl[msk, j, ai], axis=-1)
            c = np.sum(pf[msk, j, ai] * pl[msk, j, ai], -1) / np.maximum(nf * nl, 1e-9)
            out.append(f"k{k}: |F| {np.median(nf):5.1f} |L| {np.median(nl):5.1f} L/F {np.median(nl / np.maximum(nf, 1e-6)):4.2f} cos {np.median(c):.2f}")
        print(f"  {name:8s} " + " | ".join(out))
    # lag: shift s (in label steps) minimising |follower(t+k) - leader(t+k-s)| over k=4..16 on moving rows
    best = []
    for s_ in range(0, 7):
        err = []
        for k in range(4, 17):
            jf = int(np.flatnonzero(KS == k)[0]); jl = int(np.flatnonzero(KS == k - s_)[0])
            err.append(np.linalg.norm(pf[mv, jf, ai] - pl[mv, jl, ai], axis=-1))
        best.append(np.median(np.concatenate(err)))
    s_best = int(np.argmin(best))
    print(f"  best lag follower-behind-leader: {s_best} steps = {s_best * 50.05:.0f} ms (median residual {best[s_best]:.1f} mm; lag 0 residual {best[0]:.1f} mm)")
