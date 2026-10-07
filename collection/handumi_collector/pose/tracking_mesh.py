"""The tracking-only HandUMI rigid mesh (§3): load, validate, assemble.

A 6DoF model-based tracker assumes the model is RIGID. The HandUMI CAD contains moving parts (thumb/finger links, crank,
connecting links) whose pose changes with the jaw — including them makes the tracker fight the grip. The mesh handed to a
tracker must therefore contain ONLY parts that are fixed with respect to the controller body:

    include   fisheye_camera_main_support, <side>_controller_support, main_support_cover_plate, camera_mount,
              hand_support_base, <side>_servo_controller_support, servo_controller_cover
    exclude   <side>_thumb_link, <side>_index_middle_finger_link, crank_mechanism_plate, connecting_link_1/2, tips

Body frame B (must match the TCP convention, §4/§5):
    origin  a repeatable CAD datum on the rigid body (documented per side in the assembly file)
    +x      gripper forward / approach axis      +y  gripper left      +z  gripper up

IMPORTANT — the per-part STLs in handumi-hw/hardware/STL are each exported at their OWN local origin (every part's bbox is
centred near zero), so they CANNOT simply be unioned: the assembly poses are not in the files. Either export one assembled
rigid body from CAD (Shapr3D), or fill the `pose` of every included part in an assembly YAML (tools.make_tracking_mesh
writes a template). This module refuses to invent a pose."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from .se3 import pose7_to_T

# Part-name classification for the HandUMI V1 print package (handumi-hw @ e58de33).
RIGID_PART_STEMS = ("fisheye_camera_main_support", "controller_support", "main_support_cover_plate", "camera_mount",
                    "hand_support_base", "servo_controller_support", "servo_controller_cover")
MOVING_PART_STEMS = ("thumb_link", "index_middle_finger_link", "crank_mechanism_plate", "connecting_link")

# Plausible size of the HandUMI rigid body (metres, longest bbox edge). A mesh outside this is almost always a unit error.
EXPECTED_EXTENT_M = (0.05, 0.40)


def classify_part(name: str) -> str:
    stem = Path(name).stem.lower()
    if any(k in stem for k in MOVING_PART_STEMS):
        return "moving"
    if any(k in stem for k in RIGID_PART_STEMS):
        return "rigid"
    return "unknown"


@dataclass
class TrackingMesh:
    """A trimesh.Trimesh in METRES, expressed in the body frame B, plus its provenance.

    trimesh (not open3d) is the mesh layer: open3d 0.18 — the wheel available for this Mac — segfaults on every numpy->
    Eigen conversion under numpy 2 (translate/scale/transform/registration_icp), and trimesh is also what FoundationPose
    itself consumes, so one mesh object serves both backends."""
    mesh: object                       # trimesh.Trimesh
    path: Path | None
    side: str | None
    units_in: str
    scale_applied: float
    meta: dict

    @property
    def vertices(self) -> np.ndarray:
        return np.asarray(self.mesh.vertices, np.float64)

    @property
    def triangles(self) -> np.ndarray:
        return np.asarray(self.mesh.faces, np.int64)

    @property
    def extent_m(self) -> np.ndarray:
        v = self.vertices
        return v.max(axis=0) - v.min(axis=0)

    @property
    def centroid_m(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def bbox_corners(self) -> np.ndarray:
        v = self.vertices
        lo, hi = v.min(axis=0), v.max(axis=0)
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    def sample_points(self, n: int = 8000, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) surface points + (N,3) outward normals in the body frame, for ICP / render-and-compare.
        Area-weighted surface sampling, so a large flat face is not under-represented relative to a dense small one."""
        import trimesh
        rs = np.random.default_rng(seed)
        pts, fid = trimesh.sample.sample_surface(self.mesh, int(n), seed=int(rs.integers(1 << 31)))
        return np.asarray(pts, np.float64), np.asarray(self.mesh.face_normals[fid], np.float64)

    def diagnostics(self) -> dict:
        v = self.vertices
        d = dict(path=str(self.path) if self.path else None, side=self.side, units_in=self.units_in,
                 scale_applied=self.scale_applied, n_vertices=int(len(v)), n_triangles=int(len(self.triangles)),
                 extent_m=[round(float(x), 5) for x in self.extent_m],
                 longest_edge_m=round(float(self.extent_m.max()), 5),
                 centroid_m=[round(float(x), 5) for x in self.centroid_m],
                 origin_inside_bbox=bool(np.all(v.min(axis=0) <= 0) and np.all(v.max(axis=0) >= 0)),
                 origin_to_centroid_mm=round(float(np.linalg.norm(self.centroid_m) * 1e3), 2),
                 is_watertight=bool(self.mesh.is_watertight), problems=[])
        lo, hi = EXPECTED_EXTENT_M
        if not (lo <= d["longest_edge_m"] <= hi):
            d["problems"].append(f"longest bbox edge {d['longest_edge_m']:.3f} m outside the plausible HandUMI range "
                                 f"{lo}-{hi} m — check `units` (mm vs m) in the mesh/assembly file")
        if d["n_triangles"] == 0:
            d["problems"].append("mesh has no triangles")
        if not d["origin_inside_bbox"]:
            d["problems"].append(f"body-frame origin is outside the mesh bbox ({d['origin_to_centroid_mm']:.0f} mm from the "
                                 "centroid) — the mesh is probably still in its CAD/print coordinates, not the body frame B")
        return d


_UNIT_SCALE = {"m": 1.0, "meter": 1.0, "metre": 1.0, "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3, "cm": 1e-2, "inch": 0.0254}


def _scale_for(units: str | None, vertices: np.ndarray) -> tuple[str, float]:
    if units:
        u = units.lower()
        if u not in _UNIT_SCALE:
            raise ValueError(f"unknown mesh units {units!r}; known: {sorted(_UNIT_SCALE)}")
        return u, _UNIT_SCALE[u]
    # autodetect: a HandUMI body is ~0.1-0.2 m; anything with a >1 longest edge is millimetres
    v = np.asarray(vertices, np.float64)
    if len(v) == 0:
        return "m", 1.0
    longest = float((v.max(axis=0) - v.min(axis=0)).max())
    return ("mm", 1e-3) if longest > 1.0 else ("m", 1.0)


def _read_mesh(path: Path):
    import trimesh
    m = trimesh.load_mesh(str(path), process=False, force="mesh")
    if not hasattr(m, "faces"):
        raise ValueError(f"{path}: not a single triangle mesh (got {type(m).__name__})")
    return m


def load_mesh_file(path: str | Path, *, units: str | None = None, T_body_mesh: np.ndarray | None = None,
                   side: str | None = None) -> TrackingMesh:
    """Load one mesh file (.obj/.stl/.ply) and express it in metres in the body frame.
    `T_body_mesh` re-expresses a mesh exported in some other frame into B; None = the file is already in B."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    m = _read_mesh(path)
    if len(m.vertices) == 0:
        raise ValueError(f"{path}: no vertices (unsupported or empty mesh file)")
    u, s = _scale_for(units, m.vertices)
    if s != 1.0:
        m.vertices = np.asarray(m.vertices, np.float64) * s
    if T_body_mesh is not None:
        T = np.asarray(T_body_mesh, np.float64)
        m.vertices = (T[:3, :3] @ np.asarray(m.vertices, np.float64).T + T[:3, 3:4]).T
    meta_p = path.with_suffix(path.suffix + ".meta.yaml")
    meta = yaml.safe_load(meta_p.read_text()) if meta_p.exists() else {}
    return TrackingMesh(m, path, side or meta.get("side"), u, s, meta)


def load_assembly(path: str | Path) -> TrackingMesh:
    """Union the included parts of an assembly YAML (see tools.make_tracking_mesh --template) into one rigid body mesh.
    Every included part needs an explicit pose; a missing pose is an error listing exactly which parts are unresolved."""
    import trimesh
    path = Path(path)
    spec = yaml.safe_load(path.read_text()) or {}
    units = spec.get("units")
    root = path.parent
    parts, missing, included, excluded = [], [], [], []
    for part in spec.get("parts", []):
        name = part["file"]
        if not part.get("include", True):
            excluded.append(name)
            continue
        pose = part.get("pose")
        if pose is None:
            missing.append(name)
            continue
        p = Path(name)
        p = p if p.is_absolute() else (root / p)
        T = pose7_to_T(np.concatenate([np.asarray(pose["translation_m"], np.float64).reshape(3),
                                       np.asarray(pose["quaternion_xyzw"], np.float64).reshape(4)]))
        tm = load_mesh_file(p, units=part.get("units", units))
        mm = tm.mesh.copy()
        mm.vertices = (T[:3, :3] @ np.asarray(mm.vertices, np.float64).T + T[:3, 3:4]).T
        parts.append(mm)
        included.append(name)
    if missing:
        raise ValueError(f"{path}: {len(missing)} included part(s) have no `pose` — fill them in (the per-part STLs are each "
                         f"at their own local origin, so the assembly poses must come from CAD): {missing}")
    if not parts:
        raise ValueError(f"{path}: assembly produced an empty mesh (no included parts)")
    out = trimesh.util.concatenate(parts)
    meta = dict(assembly=str(path), included=included, excluded=excluded, body_frame=spec.get("body_frame", {}),
                side=spec.get("side"), source=spec.get("source", ""))
    return TrackingMesh(out, path, spec.get("side"), units or "auto", 1.0, meta)


def load_tracking_mesh(path: str | Path, *, units: str | None = None, side: str | None = None) -> TrackingMesh:
    """Dispatch on extension: .yaml/.yml = assembly file, anything else = a single already-assembled mesh."""
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        tm = load_assembly(p)
        return tm if side is None else TrackingMesh(tm.mesh, tm.path, side, tm.units_in, tm.scale_applied, tm.meta)
    return load_mesh_file(p, units=units, side=side)
