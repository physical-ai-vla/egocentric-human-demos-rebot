"""Synthetic RGB-D pilot with ground truth — the Stage-A self-test that needs no hardware and no CAD.

Ray-casts a rigid body along a KNOWN 6DoF trajectory (still -> X/Y/Z translation -> roll/pitch/yaw -> combined SE(3) ->
fast motion -> partial occlusion -> still) into an Orbbec-shaped RGB-D stream, and writes a flat episode:

    <out>/color/frame_%06d.png   <out>/depth/frame_%06d.png (uint16 mm)   intrinsics.json
    <out>/events.json            static_begin / static_end marks for the two still windows
    <out>/ground_truth.json      poses7_T_depthcam_body + the test-sequence segment table + the init bbox

Tracking it and comparing against ground_truth.json is what makes the QA numbers meaningful BEFORE a real pilot exists:
it separates "the pipeline is wrong" from "the sensor/scene is hard". It does NOT replace the real pilot — synthetic
depth has no sensor noise, no multipath and no material effects.

    python -m handumi_collector.tools.synth_rgbd_pilot /tmp/synth_left --seconds 8
    python -m handumi_collector.tools.synth_rgbd_pilot /tmp/synth_left --mesh assets/handumi/left_tracking_body.obj
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..pose.se3 import Ts_to_pose7, make_T
from ..pose.rgbd_io import CameraIntrinsics, project

# Orbbec Gemini 336 colour stream as the collector records it (848x480, depth aligned to colour)
W, H = 848, 480
INTR = CameraIntrinsics(fx=461.1717, fy=461.4451, cx=424.8777, cy=240.9170, width=W, height=H,
                        depth_unit_m=1e-3, aligned_to_rgb=True, source="synthetic (Gemini 336 colour intrinsics)")


def default_body_mesh():
    """A deliberately ASYMMETRIC rigid body of HandUMI size (~0.14 m): rotation about every axis is observable in depth,
    so a wrong orientation cannot hide behind a symmetric silhouette."""
    import trimesh
    body = trimesh.creation.box((0.070, 0.045, 0.120))                          # main body, centred on the origin
    arm = trimesh.creation.box((0.026, 0.026, 0.075))                           # camera support sticking forward
    arm.apply_translation([0.0, 0.035, 0.032])
    cyl = trimesh.creation.cylinder(radius=0.020, height=0.050, sections=32)    # servo housing on one side
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    cyl.apply_translation([0.048, -0.010, -0.030])
    out = trimesh.util.concatenate([body, arm, cyl])
    return out


def trajectory(n: int, fps: float, *, z0: float = 0.55) -> tuple[np.ndarray, list[dict]]:
    """The §14 test sequence as one continuous path. Returns (N,4,4) T_depthcam_body and the segment table."""
    segs = [("static", 1.2), ("x_translation", 1.0), ("y_translation", 1.0), ("z_translation", 1.0),
            ("roll", 0.8), ("pitch", 0.8), ("yaw", 0.8), ("combined_se3", 1.2), ("fast_motion", 0.8),
            ("partial_occlusion", 1.0), ("static", 1.2)]
    total = sum(d for _n, d in segs)
    scale = (n / fps) / total
    Ts, table, i = [], [], 0
    p = np.array([0.0, 0.0, z0])
    R = Rotation.from_euler("xyz", [15, 20, 5], degrees=True).as_matrix()
    for name, dur in segs:
        k = max(int(round(dur * scale * fps)), 1)
        i0 = i
        for j in range(k):
            u = (j + 1) / k
            s = 0.5 - 0.5 * np.cos(np.pi * u)                       # smooth in/out, no teleports
            dp, drot = np.zeros(3), np.zeros(3)
            if name == "x_translation":   dp = np.array([0.12 * s, 0, 0])
            elif name == "y_translation": dp = np.array([0, 0.08 * s, 0])
            elif name == "z_translation": dp = np.array([0, 0, 0.10 * s])
            elif name == "roll":          drot = np.array([35 * s, 0, 0])
            elif name == "pitch":         drot = np.array([0, 35 * s, 0])
            elif name == "yaw":           drot = np.array([0, 0, 35 * s])
            elif name == "combined_se3":  dp, drot = np.array([0.06, -0.05, 0.04]) * s, np.array([20, -15, 25]) * s
            elif name == "fast_motion":   dp, drot = np.array([0.10, 0.06, 0]) * s, np.array([0, 0, 30]) * s
            Ts.append(make_T(Rotation.from_euler("xyz", drot, degrees=True).as_matrix() @ R, p + dp))
            i += 1
        p = Ts[-1][:3, 3].copy()
        R = Ts[-1][:3, :3].copy()
        table.append(dict(name=name, i0=i0, i1=i - 1))
    while len(Ts) < n:
        Ts.append(Ts[-1].copy())
        table[-1]["i1"] = len(Ts) - 1
    Ts = Ts[:n]
    table = [dict(s, i0=min(s["i0"], n - 1), i1=min(s["i1"], n - 1)) for s in table if s["i0"] <= n - 1]
    return np.asarray(Ts), table


def render(mesh, Ts, table, *, occluder: bool = True, background_z: float = 1.25, n_points: int = 400_000,
           seed: int = 0):
    """Depth (metres) + a shaded BGR image per frame.

    A numpy z-buffer over densely sampled surface points, not a raycaster: Open3D 0.18's `RaycastingScene.cast_rays`
    segfaults on this macOS build, and a splat z-buffer has no native dependency at all. Sampling is sub-pixel dense at
    the working distance, and the remaining pinholes are closed morphologically."""
    import cv2
    import trimesh
    K = INTR.K
    pts, fid = trimesh.sample.sample_surface(mesh, int(n_points), seed=int(seed))
    P = np.asarray(pts, np.float64)
    N = np.asarray(mesh.face_normals[fid], np.float64)
    kern = np.ones((3, 3), np.uint8)
    occ_range = next((s for s in table if s["name"] == "partial_occlusion"), None) if occluder else None

    for i, T in enumerate(Ts):
        Pc = (T[:3, :3] @ P.T + T[:3, 3:4]).T
        Nc = (T[:3, :3] @ N.T).T
        depth = np.full((H, W), float(background_z), np.float32)
        is_body = np.zeros((H, W), bool)
        shade = np.full((H, W), 0.55, np.float32)
        uv = project(Pc, K)
        ok = np.isfinite(uv).all(axis=1) & (Pc[:, 2] > 1e-3)
        u = np.round(uv[ok, 0]).astype(np.int64); v = np.round(uv[ok, 1]).astype(np.int64)
        z = Pc[ok, 2]; nz = np.abs(Nc[ok, 2])
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < background_z)
        u, v, z, nz = u[inb], v[inb], z[inb], nz[inb]
        order = np.argsort(-z)                       # far first, so the nearest point wins the last write
        u, v, z, nz = u[order], v[order], z[order], nz[order]
        depth[v, u] = z.astype(np.float32)
        shade[v, u] = np.clip(nz, 0.15, 1.0).astype(np.float32)
        is_body[v, u] = True
        closed = cv2.morphologyEx(is_body.astype(np.uint8), cv2.MORPH_CLOSE, kern).astype(bool)
        holes = closed & ~is_body
        if holes.any():                              # fill pinholes with the nearest surrounding surface depth
            near = cv2.erode(np.where(is_body, depth, np.float32(background_z)), kern)
            depth[holes] = near[holes]
            shade[holes] = cv2.dilate(shade, kern)[holes]
            is_body = closed
        if occ_range and occ_range["i0"] <= i <= occ_range["i1"]:
            uu = (i - occ_range["i0"]) / max(occ_range["i1"] - occ_range["i0"], 1)
            z_occ = float(T[2, 3]) - 0.12
            corners = np.array([[-0.16 + 0.34 * uu, -0.07, z_occ], [-0.16 + 0.34 * uu, 0.07, z_occ],
                                [0.00 + 0.34 * uu, 0.07, z_occ], [0.00 + 0.34 * uu, -0.07, z_occ]])
            poly = project(corners, K)
            if np.isfinite(poly).all():
                occ_mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(occ_mask, [np.round(poly).astype(np.int32)], 1)
                om = (occ_mask > 0) & (depth > z_occ)
                depth[om] = z_occ
                is_body[om] = False
                shade[om] = 0.8
        rgbimg = np.zeros((H, W, 3), np.uint8)
        rgbimg[..., 0] = (shade * np.where(is_body, 210, 90)).astype(np.uint8)
        rgbimg[..., 1] = (shade * np.where(is_body, 190, 110)).astype(np.uint8)
        rgbimg[..., 2] = (shade * np.where(is_body, 90, 130)).astype(np.uint8)
        yield depth, rgbimg, is_body


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--mesh", help="tracking mesh to render (default: a synthetic HandUMI-sized asymmetric body)")
    ap.add_argument("--side", default="left")
    ap.add_argument("--no-occluder", action="store_true")
    ap.add_argument("--depth-noise-mm", type=float, default=0.0, help="gaussian noise added to depth (0 = perfect sensor)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from ..pose.tracking_mesh import load_tracking_mesh
    mesh = load_tracking_mesh(a.mesh, side=a.side).mesh if a.mesh else default_body_mesh()
    print(f"mesh: {len(mesh.faces)} triangles, extent {np.round(mesh.extents, 4)} m", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    mesh.export(str(Path(a.out) / "body_mesh.obj"))          # the exact mesh that was rendered -> what the tracker gets
    n = int(round(a.seconds * a.fps))
    Ts, table = trajectory(n, a.fps)
    out = Path(a.out)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    bbox = None
    for i, (depth, rgb, is_body) in enumerate(render(mesh, Ts, table, occluder=not a.no_occluder)):
        if a.depth_noise_mm:
            depth = np.where(depth > 0, depth + rng.normal(0, a.depth_noise_mm * 1e-3, depth.shape), 0).astype(np.float32)
        cv2.imwrite(str(out / "color" / f"frame_{i:06d}.png"), rgb)
        cv2.imwrite(str(out / "depth" / f"frame_{i:06d}.png"), np.round(depth * 1e3).astype(np.uint16))
        if i == 0:
            v, u = np.nonzero(is_body)
            bbox = [int(u.min()), int(v.min()), int(u.max() - u.min() + 1), int(v.max() - v.min() + 1)]
            cv2.imwrite(str(out / "init_mask.png"), (is_body.astype(np.uint8) * 255))
        if (i + 1) % 30 == 0:
            print(f"  rendered {i+1}/{n}", flush=True)

    t_ns = (np.arange(n, dtype=np.int64) * int(1e9 / a.fps))
    (out / "timestamps.csv").write_text("index,t_ns\n" + "\n".join(f"{i},{t}" for i, t in enumerate(t_ns)) + "\n")
    (out / "intrinsics.json").write_text(json.dumps(INTR.to_dict(), indent=1))
    still = [s for s in table if s["name"] == "static"]
    events = []
    for s in still:
        events += [dict(t_ns=int(t_ns[s["i0"]]), kind="static_begin", device="synthetic", detail={}),
                   dict(t_ns=int(t_ns[s["i1"]]), kind="static_end", device="synthetic", detail={})]
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "ground_truth.json").write_text(json.dumps(dict(
        schema="handumi_synth_rgbd_gt/v1", side=a.side, fps=a.fps, n_frames=n,
        intrinsics=INTR.to_dict(), segments=table, init_bbox=bbox, init_mask="init_mask.png",
        mesh="body_mesh.obj", mesh_source=(str(a.mesh) if a.mesh else "synthetic_default_body"),
        poses7_T_depthcam_body=Ts_to_pose7(Ts).tolist(), t_ns=t_ns.tolist()), indent=1))
    (out / "packet_meta.json").write_text(json.dumps(dict(schema="handumi_rgbd_packet/v1", synthetic=True, side=a.side,
                                                          init_bbox=bbox, init_mask="init_mask.png", mesh="body_mesh.obj"), indent=1))
    print(f"wrote {n} frames to {out}   init_bbox={bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
