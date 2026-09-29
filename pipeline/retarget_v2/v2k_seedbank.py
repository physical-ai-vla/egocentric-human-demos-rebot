#!/usr/bin/env python3
"""[2026-09-28] C-old v2-K seed bank: paired bimanual (qL, qR) configurations generated from the reBot's OWN kinematics.

No teleop episode, trajectory or episode identity is used as a seed. R150 enters only as an ENVELOPE (calibration of
bounds), never as a sample: per-joint p1-p99 range, the same R150 workspace NN test that Phase 3 gates targets with,
the table floor (TCP z p0.1 minus 10 mm) and a manipulability floor (p1 of R150's position-Jacobian sigma_min).
FROZEN before any v2 retarget result exists (user, 2026-09-28): rule + params + output hash in seedbank.json.

  1. Sobol 2^SOBOL_M points in 12-D (6 joints per arm) inside the R150 p1-p99 joint box
  2. per arm: FK TCP inside the R150 workspace (pos NN < 50 mm AND ori NN-50 < 20 deg == c8_phase3.ws_in)
  3. per arm: every link / finger point above the table floor
  4. per arm: position-Jacobian sigma_min >= R150 p1 (no near-singular pose)
  5. bimanual clearance d_min >= 0.075 m (c8_collision_v1, contract D_HARD)
  6. greedy dedup in 12-D joint space: keep a sample only if >= DEDUP_RAD from every kept one
"""
import hashlib, json, pathlib, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import qmc

H = pathlib.Path.home(); sys.path.insert(0, str(H / "c8")); sys.path.insert(0, str(H / "umi_bridge/track_c"))
import pseudo_joint_pipeline as P                                                   # noqa: E402
from c8_collision_v1 import Clearance                                              # noqa: E402
OUT = pathlib.Path(__file__).resolve().parent
SOBOL_M, SEED, DEDUP_RAD, FLOOR_MARGIN_M, WS_POS_M, WS_ORI_DEG, D_HARD = 17, 20260928, 0.25, 0.010, 0.05, 20.0, 0.075
LINKS = ["link3", "link4", "link5", "link6", "gripper_link", "tcp", "gripper_left", "gripper_right"]


def main():
    # --ref r30 (2026-09-28 contract): EVERY R150-derived quantity below comes from the frozen R30 calibration episodes only
    # (r30_episodes.json = nested set_id < 5); the other 120 R150 episodes are never read. Default = legacy full R150.
    ref_name = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "r150full"
    t0 = time.time(); kin = P.eef_kin.Kin(dict(ori_weight=20.0, max_joint_delta=0.35)); clr = Clearance(kin)
    d = np.load(H / "umi_bridge/track_c/data/r150_q_tcp.npz")
    keep_rows = np.isin(d["ep"], json.load(open(OUT / "r30_episodes.json"))["episodes"]) if ref_name == "r30" else np.ones(len(d["ep"]), bool)
    d = {k: d[k][keep_rows] for k in ("J", "tcp")}
    J = np.radians(np.c_[d["J"][:, :6], d["J"][:, 7:13]])
    lo, hi = np.percentile(J, 1, axis=0), np.percentile(J, 99, axis=0)
    # envelope references (R150 frames: never used as samples)
    ix = np.random.default_rng(0).choice(len(J), 6000, replace=False)                  # == c8_phase3.init()
    ref = {0: [], 1: []}
    for i in ix:
        for a, p7 in enumerate(kin.fk_pose7(J[i])): ref[a].append(p7)
    ref = {a: np.array(v) for a, v in ref.items()}; tree = {a: cKDTree(ref[a][:, :3]) for a in (0, 1)}
    WR = {a: Rot.from_quat(ref[a][:, 3:]).as_matrix() for a in (0, 1)}
    floor = [float(np.percentile(d["tcp"][:, 2], 0.1)) - FLOOR_MARGIN_M, float(np.percentile(d["tcp"][:, 10], 0.1)) - FLOOR_MARGIN_M]

    def fk_pos(q12):
        l7, r7 = kin.fk_pose7(q12); return np.array([l7[:3], r7[:3]])

    def sigma_min(q12):
        """smallest singular value of each arm's 3x6 position Jacobian (finite differences, 1e-4 rad)"""
        p0 = fk_pos(q12); out = []
        for a in (0, 1):
            Jp = np.zeros((3, 6))
            for k in range(6):
                dq = np.zeros(12); dq[6 * a + k] = 1e-4; Jp[:, k] = (fk_pos(q12 + dq)[a] - p0[a]) / 1e-4
            out.append(float(np.linalg.svd(Jp, compute_uv=False)[-1]))
        return out

    sm_ref = np.array([sigma_min(J[i]) for i in ix[:1500]]); sm_floor = np.percentile(sm_ref, 1, axis=0)

    def ws_ok(p7, a):
        dp, ii = tree[a].query(p7[:3], k=50)
        R = Rot.from_quat(p7[3:]).as_matrix(); tr = np.einsum("nij,ij->n", WR[a][ii], R)          # trace(R_ref^T R)
        ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min()
        return dp[0] < WS_POS_M and ang < WS_ORI_DEG

    S = qmc.Sobol(12, scramble=True, seed=SEED).random_base2(SOBOL_M); Q = lo + S * (hi - lo)
    funnel = dict(sobol=len(Q)); keep = []
    for q in Q:
        l7, r7 = kin.fk_pose7(q)
        if not (ws_ok(l7, 0) and ws_ok(r7, 1)): continue
        keep.append(q)
    funnel["workspace"] = len(keep); k2 = []
    for q in keep:
        pts = clr.points(q)
        if all(pts[f"{s}_{n}"][2] >= floor[a] for a, s in enumerate(("left", "right")) for n in LINKS): k2.append(q)
    funnel["table_floor"] = len(k2); k3 = [q for q in k2 if all(s >= f for s, f in zip(sigma_min(q), sm_floor))]
    funnel["manipulability"] = len(k3); k4 = [q for q in k3 if clr.dmin(q)[0] >= D_HARD]
    funnel["collision"] = len(k4)
    bank = []
    for q in k4:                                                                      # greedy dedup, Sobol order
        if not bank or np.min(np.linalg.norm(np.array(bank) - q, axis=1)) >= DEDUP_RAD: bank.append(q)
    bank = np.array(bank, np.float64); funnel["dedup"] = len(bank)
    tag = "" if ref_name == "r150full" else f"_{ref_name.upper()}"
    np.save(OUT / f"seedbank{tag}_q12.npy", bank)
    h = hashlib.sha256(bank.tobytes()).hexdigest()
    meta = dict(schema="c_old_v2k_seedbank/v1", frozen="2026-09-28 (before any v2 retarget result)", rule=__doc__.strip(),
                reference=ref_name, reference_frames=int(len(J)),
                params=dict(sobol_m=SOBOL_M, sobol_seed=SEED, dedup_rad=DEDUP_RAD, floor_margin_m=FLOOR_MARGIN_M, ws_pos_m=WS_POS_M,
                            ws_ori_deg=WS_ORI_DEG, d_hard=D_HARD, floor_links=LINKS),
                envelope=dict(joint_lo_deg=np.degrees(lo).round(2).tolist(), joint_hi_deg=np.degrees(hi).round(2).tolist(),
                              table_floor_m=[round(f, 4) for f in floor], sigma_min_floor=sm_floor.round(5).tolist()),
                funnel=funnel, n=len(bank), sha256_q12=h, seconds=round(time.time() - t0, 1),
                columns="q12 rad: left j1..j6, right j1..j6 (URDF / solver frame), paired same configuration")
    (OUT / f"seedbank{tag}.json").write_text(json.dumps(meta, indent=1)); print(json.dumps({k: meta[k] for k in ("funnel", "n", "sha256_q12", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
