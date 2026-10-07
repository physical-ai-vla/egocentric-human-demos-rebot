#!/usr/bin/env python3
"""[2026-09-29] RELCART20 QA on the Mac (real dataset), before any training:
  QA1 anchor identity: every episode's frame_index-0 row == [0,0,0, 1,0,0,0,1,0] per arm
  QA2 independent reconstruction: the deploy path (eef_kin/pyroki FK + V4_FRAME_FIX via infer_core_v4._tcp_mat, NOT rebot_fk_torch)
      on aux.q_t, anchored at frame 0 through infer_core_v4.relcart20_state (the function the robot UI uses) vs the stored state
  QA3 non-state columns vs the parent v3d (Mac copy) byte-equal; video files are the parent's (same size + mtime on the node, hard links)
  Distribution report (real only; the ego side is built by the ego agent): per-arm rel xyz p1/p50/p99/mean/std, rel rotation
  geodesic p50/p90/p99, gripper mean / p(open) / transition rates.
usage: relcart20_qa.py
"""
import glob, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
import infer_core_v4 as IC, eef_kin

R = os.path.expanduser("~/holobrain-data/lerobot/")
col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
def tables(root):
    return [pq.read_table(f) for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))]
P, N = tables(R + "r180_umi76_rel16_v3d"), tables(R + "r180_relcart20_rel16_v3d")
arr = lambda ts, c: np.concatenate([np.asarray(col(t, c).to_pylist()) for t in ts])
S = arr(N, "observation.state").astype(np.float64); Q = arr(P, "aux.q_t"); S76 = arr(P, "observation.state")
E = np.concatenate([col(t, "episode_index").to_numpy() for t in P]); F = np.concatenate([col(t, "frame_index").to_numpy() for t in P])
print(f"rows {len(S)} episodes {len(np.unique(E))}")

ID = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float64)
z = F == 0
q1 = np.abs(S[z][:, :18] - np.tile(ID, 2)).max()
print(f"QA1 anchor identity: {z.sum()} anchor rows, max |state - identity| {q1:.2e}  -> {'PASS' if q1 < 1e-6 else 'FAIL'}")


class K:
    kin = eef_kin.Kin()


k = K(); M = np.zeros((len(Q), 2, 4, 4))
for i in range(len(Q)):
    q14 = np.zeros(14); q14[IC.ARM_IDX] = Q[i]
    M[i] = np.stack(IC.V4Inferencer._tcp_mat(k, q14)[0])
anc = {int(e): M[np.flatnonzero((E == e) & z)[0]] for e in np.unique(E)}
W = S76[:, [37, 75]]
S2 = np.stack([IC.relcart20_state(anc[int(E[i])], M[i], W[i]) for i in range(len(Q))]).astype(np.float64)
dp = np.abs(S2[:, [0, 1, 2, 9, 10, 11]] - S[:, [0, 1, 2, 9, 10, 11]]).max(); dr = np.abs(S2[:, list(range(3, 9)) + list(range(12, 18))] - S[:, list(range(3, 9)) + list(range(12, 18))]).max()
dg = np.abs(S2[:, 18:] - S[:, 18:]).max()
print(f"QA2 independent reconstruction (deploy FK + relcart20_state): pos max {dp * 1000:.4f} mm, rot6d max {dr:.2e}, grip max {dg:.2e}  -> "
      f"{'PASS' if dp < 1e-4 and dr < 1e-4 and dg < 1e-5 else 'FAIL'}")

bad = [n for tp, tn in zip(P, N) for n in tp.column_names if n != "observation.state" and not tp.column(n).equals(tn.column(n))]
print(f"QA3 non-state columns byte-identical to v3d: {'PASS' if not bad else 'FAIL ' + str(sorted(set(bad)))}  (columns {[c for c in P[0].column_names if c != 'observation.state']})")

print("\n[distribution, REAL] (for the ego<->real comparison once the ego dataset exists)")
for ai, arm in ((0, "L"), (1, "R")):
    xyz = S[:, ai * 9:ai * 9 + 3] * 100
    r6 = S[:, ai * 9 + 3:ai * 9 + 9]
    Rm = np.stack([IC.pose10d_to_mat(np.r_[np.zeros(3), x][None])[0][:3, :3] for x in r6])
    ang = np.degrees(np.arccos(np.clip((np.trace(Rm, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    print(f"  {arm} xyz cm: p1 {np.percentile(xyz, 1, 0).round(1)} p50 {np.percentile(xyz, 50, 0).round(1)} p99 {np.percentile(xyz, 99, 0).round(1)} "
          f"mean {xyz.mean(0).round(1)} std {xyz.std(0).round(1)} | |xyz| p50 {np.median(np.linalg.norm(xyz, axis=1)):.1f} p99 {np.percentile(np.linalg.norm(xyz, axis=1), 99):.1f}")
    print(f"  {arm} rel rotation deg: p50 {np.percentile(ang, 50):.1f} p90 {np.percentile(ang, 90):.1f} p99 {np.percentile(ang, 99):.1f}")
    g = S[:, 18 + ai]; op = g >= 0.6
    tr = [(E[i] == E[i - 1]) and (op[i] != op[i - 1]) for i in range(1, len(g))]
    oc = sum(1 for i in range(1, len(g)) if E[i] == E[i - 1] and op[i - 1] and not op[i]); co = sum(1 for i in range(1, len(g)) if E[i] == E[i - 1] and not op[i - 1] and op[i])
    print(f"  {arm} gripper: mean {g.mean():.3f} p(open >= 0.6) {op.mean():.3f} | transitions/episode {sum(tr) / len(np.unique(E)):.2f} (OPEN->CLOSE {oc}, CLOSE->OPEN {co})")
