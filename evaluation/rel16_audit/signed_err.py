"""[2026-09-29] Per-arm signed Cartesian error of the teacher-forced one-chunk prediction, in the robot BASE frame.
    e_k = R_t (p̂_REL,k − p_REL,k)   (mm; + = robot x forward / y LEFT / z up), k = 16, moving samples (|GT| > 20 mm)
R_t = dataset-frame TCP rotation at t from FK(aux.q_t) + V4_FRAME_FIX (the deployment's own _tcp_mat), so v3 (state76, no TCP18)
gets the same base-frame view that R312C-B got from TCP18. Diagnostic reference (R312C-B 150k, a CONFOUNDED run -- scrambled-encoder init): L y +5.2 mm, 71 % of samples > 0.
usage: signed_err.py <v2eval npz> <dataset root>
"""
import glob, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
import infer_core_v4 as IC

npz, root = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True); n = int(d["done"])
pick, gt, pr = d["pick"][:n], d["gt"][:n], d["pred_real"][:n].mean(1)
col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
tabs = [pq.read_table(f, columns=["aux.q_t", "index"]) for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))]
Q = np.concatenate([np.asarray(col(t, "aux.q_t").to_pylist()) for t in tabs]); IX = np.concatenate([np.asarray(col(t, "index").to_numpy()) for t in tabs])
pos = {int(j): i for i, j in enumerate(IX)}


class K:
    kin = IC.eef_kin.Kin(); _tcp_mat = IC.V4Inferencer._tcp_mat


k = K()
Rt = []
for i in pick:
    q14 = np.zeros(14); q14[IC.ARM_IDX] = Q[pos[int(i)]]
    m, _ = k._tcp_mat(q14); Rt.append(np.stack([m[0][:3, :3], m[1][:3, :3]]))
Rt = np.stack(Rt)
print(f"[signed error] {os.path.basename(npz)}  (A16, base frame mm; + = x forward / y LEFT / z up; moving |GT| > 20 mm)")
for arm, r in (("L", 0), ("R", 1)):
    g = np.einsum("nij,nj->ni", Rt[:, r], gt[:, 15, r * 10:r * 10 + 3]) * 1000
    p = np.einsum("nij,nj->ni", Rt[:, r], pr[:, 15, r * 10:r * 10 + 3]) * 1000
    mv = np.linalg.norm(g, axis=1) > 20; e = p - g
    c = np.sum(p * g, 1) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(g, axis=1), 1e-9)
    print(f"  {arm}: n {mv.sum():2d} | median x {np.median(e[mv,0]):+6.1f} y {np.median(e[mv,1]):+6.1f} z {np.median(e[mv,2]):+6.1f} mm | "
          f"frac y>0 {np.mean(e[mv,1] > 0):.2f} | p90 |y err| {np.percentile(np.abs(e[mv,1]), 90):.1f} mm | |pred|/|GT| {np.median(np.linalg.norm(p[mv],axis=1)/np.linalg.norm(g[mv],axis=1)):.2f} cos {np.median(c[mv]):.2f}")
