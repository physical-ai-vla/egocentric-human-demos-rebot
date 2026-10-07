#!/usr/bin/env python3
"""Role 2 -- fit the SLAM->table transform once, from the mapping passes, and never again.

UMI's 04_detect_aruco.py produces tx_slam_tag so that every demo lands in one physical table frame. The
point of doing it HERE, on the mapping passes only, is that demonstrations must not have to see a marker:
the markers are removed for demos, and the 2026-09-22 ablation showed relocalisation does not need them
anyway. So the transform is computed once against the frozen atlas and stored beside it; a demo that
relocalises into that atlas is then carried into the table frame by a constant.

CONVENTION, fixed here and not to be re-derived downstream:

    T_table_slam      maps a point expressed in the SLAM/atlas frame INTO the table frame
                      p_table = T_table_slam @ p_slam
                      the camera pose becomes  T_table_cam = T_table_slam @ T_slam_cam

Only this direction is stored. An inverse taken silently somewhere downstream is exactly how the Track B
Ry(90 deg) deployment mismatch happened, so the name carries the direction and the JSON repeats it.

Table frame:
    origin  centroid of the four marker centres -- symmetric, and better conditioned than pinning it to
            one marker whose own pose carries the worst viewing angles
    +z      table-plane normal, signed to point towards the cameras (up out of the table)
    +x      M0 -> M1 projected into the plane
    +y      z cross x

Method. Each marker is STATIC in the SLAM frame, so T_slam_tag_i is a constant to be estimated rather than
something to be re-derived per frame. Every tracked frame that sees marker i contributes one observation

    T_slam_tag_i(f) = T_slam_cam(f) @ T_cam_tag_i(f)

and those are aggregated per marker with a Huber-weighted mean (translation) and a Huber-weighted chordal
SO(3) mean (rotation). Aggregating per marker first, then building the frame from the four aggregates, is
what lets a badly observed marker degrade its own estimate without dragging the frame with it -- the
2026-09-22 unit test already showed M0-M2 and M0-M3 carry ~3 cm scatter against 0.4-1.4 cm elsewhere.

The tag corners are fisheye-undistorted with the same KannalaBrandt8 k1..k4 the SLAM run uses, then given
to solvePnP with ZERO distortion -- matching what KannalaBrandt8::ReconstructWithTwoViewsAndTags does, and
zero is what that code SHOULD have passed (it built an uninitialised cv::Mat1f(5,1); fixed in
orb_slam3:prod6b_dcfix).
"""
import argparse, csv, json, pathlib, re
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot

HUBER_T, HUBER_R = 0.02, np.radians(5.0)      # metres, radians


def se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def huber_w(r, c):
    r = np.asarray(r, float); return np.where(r <= c, 1.0, c / np.maximum(r, 1e-12))


def mean_rot(Rs, w):
    """Huber-weighted chordal SO(3) mean: SVD of the weighted sum, projected back onto SO(3)."""
    M = np.einsum("i,ijk->jk", w, Rs)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def robust_centre(cs, area, cosi, iters=15):
    """Huber-IRLS weighted centre of one static marker, prior-weighted by viewing quality.

    A marker's centre is far better conditioned than its orientation, and the conditioning still varies:
    a marker filling 5000 px seen face-on constrains the centre far more tightly than one covering 200 px
    at 70 degrees. sqrt(area) scales with the linear extent the corner fit has to work with, and cos of
    the incidence angle falls off as the square shortens into a sliver, so the prior is their product.
    """
    cs = np.asarray(cs, float)
    prior = np.sqrt(np.maximum(area, 1.0)) * np.maximum(cosi, 0.05)
    prior = prior / prior.max()
    c = np.average(cs, axis=0, weights=prior)
    for _ in range(iters):
        d = np.linalg.norm(cs - c, axis=1)
        w = prior * huber_w(d, HUBER_T)
        c = (w[:, None] * cs).sum(0) / w.sum()
    return c, np.linalg.norm(cs - c, axis=1), prior


def load_traj(p):
    out = {}
    for r in csv.DictReader(open(p)):
        if r["is_lost"] == "true":
            continue
        q = [float(r["q_x"]), float(r["q_y"]), float(r["q_z"]), float(r["q_w"])]
        if abs(np.linalg.norm(q) - 1.0) > 1e-3:
            continue
        out[int(r["frame_idx"])] = se3(Rot.from_quat(q).as_matrix(),
                                       np.array([float(r["x"]), float(r["y"]), float(r["z"])]))
    return out


def observe(video, setting, traj, size_m):
    txt = pathlib.Path(setting).read_text()
    g = lambda k: float(re.search(rf"^{k}:\s*([-\d.eE]+)", txt, re.M).group(1))
    K = np.array([[g("Camera1.fx"), 0, g("Camera1.cx")],
                  [0, g("Camera1.fy"), g("Camera1.cy")], [0, 0, 1]], np.float32)
    D = np.array([[g(f"Camera1.k{i}")] for i in (1, 2, 3, 4)], np.float32)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    h = size_m / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    Z5 = np.zeros((5, 1), np.float32)
    v = cv2.VideoCapture(str(video)); i = 0
    obs = {}
    while True:
        ok, f = v.read()
        if not ok:
            break
        if i in traj:
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                for c_, mid in zip(corners, ids.ravel()):
                    und = cv2.fisheye.undistortPoints(c_.reshape(1, 4, 2).astype(np.float32), K, D,
                                                      R=np.eye(3), P=K).reshape(4, 2)
                    ok2, rv, tv = cv2.solvePnP(obj, und, K, Z5, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok2:
                        T_cam_tag = se3(cv2.Rodrigues(rv)[0], tv.ravel())
                        # CENTRE ONLY. A square planar marker has a four-fold orientation ambiguity and
                        # flips between the 90-degree branches: the 2026-09-22 right-pass fit showed M1
                        # with rotation residual p95 152.8 deg while its centre stayed sound. The table
                        # frame is built from centres and the plane through them, so no single marker's
                        # orientation is ever trusted.
                        c = (traj[i] @ T_cam_tag)[:3, 3]
                        # Viewing quality, so a marker seen small and edge-on does not weigh the same as
                        # one seen large and face-on. area: pixels^2, conditioning of the corner fit.
                        # cos_inc: the marker normal in camera frame against the viewing ray.
                        area = abs(cv2.contourArea(c_.reshape(4, 2).astype(np.float32)))
                        cosi = abs(float(T_cam_tag[:3, 2] @ (T_cam_tag[:3, 3] / max(np.linalg.norm(T_cam_tag[:3, 3]), 1e-9))))
                        obs.setdefault(int(mid), []).append((i, c, area, cosi))
        i += 1
    v.release()
    return obs


def table_frame(centres, cams):
    """origin = centroid of the marker centres; +z = plane normal toward the cameras; +x = M0->M1 in-plane.

    The origin is the centroid rather than one nominated marker so that no single marker's error becomes
    the frame's error, and so a leave-one-out fit is a meaningful perturbation rather than a redefinition.
    """
    ids = sorted(centres)
    P = np.array([centres[k] for k in ids])
    o = P.mean(0)
    n = np.linalg.svd(P - o)[2][2]
    if n @ (cams.mean(0) - o) < 0:
        n = -n
    a, b = ids[0], ids[1]
    x = centres[b] - centres[a]
    x = x - (x @ n) * n
    x /= np.linalg.norm(x)
    return se3(np.column_stack([x, np.cross(n, x), n]), o)      # T_slam_table


def fit(obs, cams, ids):
    """Robust centres -> table frame, for a chosen subset of marker ids."""
    cen, resid, prior = {}, {}, {}
    for k in ids:
        c, d, w = robust_centre([o[1] for o in obs[k]], [o[2] for o in obs[k]], [o[3] for o in obs[k]])
        cen[k], resid[k], prior[k] = c, d, w
    T_slam_table = table_frame(cen, cams)
    return T_slam_table, cen, resid, prior


def layout(T_slam_table, cen):
    T = np.linalg.inv(T_slam_table)
    return {k: (T @ np.r_[cen[k], 1.0])[:3] for k in cen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="passes", action="append", required=True,
                    metavar="NAME:VIDEO:SETTING:TRAJ", help="repeatable; one per mapping pass")
    ap.add_argument("--marker-size", type=float, default=0.08, help="BLACK SQUARE side, metres")
    # DICT_4X4_50 holds 50 codes and the scene contains only four, so anything else is a false positive
    # off clutter or a blurred edge. Unfiltered they arrive with 1-10 observations and metre-scale poses,
    # and because the frame is built from the centroid a single spurious id drags the whole frame.
    ap.add_argument("--ids", default="0,1,2,3", help="marker ids actually laid on the table")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keep = [int(x) for x in a.ids.split(",")]

    results = {}
    for spec in a.passes:
        name, video, setting, traj_p = spec.split(":")
        traj = load_traj(traj_p)
        obs = {k: v for k, v in observe(video, setting, traj, a.marker_size).items() if k in keep}
        ids = sorted(obs)
        print(f"\n=== {name} ===  tracked frames {len(traj)}   "
              + "  ".join(f"M{k}:{len(obs[k])}" for k in ids))
        if len(ids) < 3:
            print("   <3 markers -- a plane cannot be defined"); continue

        cams = np.array([T[:3, 3] for T in traj.values()])
        T_st, cen, resid, prior = fit(obs, cams, ids)
        lay = layout(T_st, cen)

        print(f"   {'marker':<8} {'n':>6} {'centre resid p50/p95 mm':>26} {'mean quality w':>15}")
        for k in ids:
            print(f"   M{k:<7} {len(resid[k]):>6} {np.percentile(resid[k],50)*1000:>13.1f} /"
                  f"{np.percentile(resid[k],95)*1000:>11.1f} {prior[k].mean():>15.3f}")
        print(f"   marker z in table frame (co-planarity): "
              + "  ".join(f"M{k} {lay[k][2]*1000:+.1f}" for k in ids) + " mm")
        print(f"   pairwise distances (mm): "
              + "  ".join(f"M{i}-M{j} {np.linalg.norm(lay[i]-lay[j])*1000:.0f}"
                          for n_, i in enumerate(ids) for j in ids[n_+1:]))

        # LEAVE-ONE-MARKER-OUT.
        # Measured in the SLAM frame, which is common to every subset, NOT by comparing table frames.
        # Comparing table frames directly is meaningless here: the origin IS the centroid of whichever
        # markers were used and +x IS defined by the two lowest ids, so dropping a marker redefines the
        # frame rather than perturbing it. A first version of this check did exactly that and reported
        # 146 deg and 86 deg "instability" for M0 and M1 against 0.08 deg for M2 and M3 -- which is just
        # the +x definition changing hands, not the fit moving.
        # What is convention-free: how much the fitted PLANE tilts, and how far each retained marker's
        # estimated centre moves, when one marker is removed.
        print(f"   LEAVE-ONE-OUT, measured in the SLAM frame (convention-free):")
        n_all = T_st[:3, 2]
        loo = {}
        for drop in ids:
            sub = [k for k in ids if k != drop]
            if len(sub) < 3: continue
            T2, c2, _, _ = fit(obs, cams, sub)
            tilt = np.degrees(np.arccos(np.clip(abs(n_all @ T2[:3, 2]), -1, 1)))
            shift = max(np.linalg.norm(c2[k] - cen[k]) for k in sub) * 1000
            loo[drop] = (tilt, shift)
            print(f"      without M{drop}:  plane tilts {tilt:>6.3f} deg   worst retained centre moves {shift:>6.2f} mm")
        if loo:
            worst = max(loo, key=lambda k: loo[k][0])
            print(f"      -> the plane leans on M{worst} most ({loo[worst][0]:.3f} deg)")

        results[name] = dict(T_slam_table=T_st, layout=lay, ids=ids,
                             resid_p50_mm=float(np.median(np.concatenate([resid[k] for k in ids]))*1000),
                             resid_p95_mm=float(np.percentile(np.concatenate([resid[k] for k in ids]),95)*1000),
                             n_obs=int(sum(len(obs[k]) for k in ids)),
                             loo={f"without_M{k}": {"plane_tilt_deg": round(v[0], 4),
                                                   "worst_centre_shift_mm": round(v[1], 3)}
                                  for k, v in loo.items()})

    if len(results) == 2:
        (na, ra), (nb, rb) = results.items()
        print(f"\n=== L/R AGREEMENT via the TABLE frame: {na} vs {nb} ===")
        print(f"   The two passes are SEPARATE atlases with unrelated origins, so their T_slam_table")
        print(f"   cannot be compared directly. What must agree is the physical board they both measure:")
        print(f"   the same four markers, expressed in each pass's own table frame.")
        common = [k for k in ra['ids'] if k in rb['ids']]
        for k in common:
            d = np.linalg.norm(ra['layout'][k] - rb['layout'][k]) * 1000
            print(f"      M{k} position differs by {d:>7.1f} mm")
        print(f"   pairwise distance agreement (frame-free, the strongest check):")
        for n_, i in enumerate(common):
            for j in common[n_+1:]:
                da = np.linalg.norm(ra['layout'][i]-ra['layout'][j])*1000
                db = np.linalg.norm(rb['layout'][i]-rb['layout'][j])*1000
                print(f"      M{i}-M{j}   {na} {da:>7.1f}   {nb} {db:>7.1f}   diff {abs(da-db):>6.1f} mm")

    if a.out:
        for name, r in results.items():
            T = r["T_slam_table"]; Tinv = np.linalg.inv(T)
            o = pathlib.Path(a.out).with_name(pathlib.Path(a.out).stem + f"_{name}.json")
            o.write_text(json.dumps({
                "convention": "T_table_slam",
                "meaning": "p_table = T_table_slam @ p_slam ; T_table_cam = T_table_slam @ T_slam_cam",
                "warning": "only this direction is stored; do not infer an inverse without renaming it",
                "frame": {"origin": "centroid of marker centres",
                          "+z": "table-plane normal, toward the cameras",
                          "+x": "lowest-id -> next-lowest-id marker, projected into the plane",
                          "+y": "z cross x"},
                "translation_m": Tinv[:3, 3].tolist(),
                "quaternion_xyzw": Rot.from_matrix(Tinv[:3, :3]).as_quat().tolist(),
                "matrix_4x4": Tinv.tolist(),
                "marker_size_m": a.marker_size, "markers_used": r["ids"],
                "marker_layout_table_frame_m": {f"M{k}": v.tolist() for k, v in r["layout"].items()},
                "orientation_note": "marker ORIENTATIONS are deliberately unused: a square planar marker "
                                    "has a four-fold ambiguity and was measured flipping (M1 rot p95 152.8 deg)",
                "estimated_from_pass": name, "num_observations": r["n_obs"],
                "centre_residual_p50_mm": round(r["resid_p50_mm"], 2),
                "centre_residual_p95_mm": round(r["resid_p95_mm"], 2),
                "leave_one_out": r["loo"],
                "leave_one_out_note": "measured in the SLAM frame; comparing table frames across subsets "
                                      "would report the +x/origin convention changing, not the fit moving",
            }, indent=1))
            print(f"-> {o}")


if __name__ == "__main__":
    main()
