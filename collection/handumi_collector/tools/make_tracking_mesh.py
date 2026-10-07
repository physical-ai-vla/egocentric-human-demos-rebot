"""Build / validate the tracking-only HandUMI rigid mesh (§3).

Three modes:

  --template   scan a directory of per-part meshes, classify each part rigid vs moving, and write an assembly YAML with
               the moving parts already excluded and every included part's `pose` left blank for you to fill from CAD
  --build      union an assembly YAML into one mesh (metres, body frame B) and export .obj + a .meta.yaml provenance file
  --validate   print the diagnostics of an existing mesh (units, extent, origin, watertightness) and its problems

Why `pose` cannot be filled in automatically: the per-part STL/STEP exports in handumi-hw are each written at their OWN
local origin (every part's bbox is centred near zero), so the assembly relationship is simply not in the files. Either
export ONE assembled rigid body from CAD — the short path — or measure each part's pose and put it here. This tool will
not guess: a missing pose is an error, because a silently wrong assembly becomes a silently wrong TCP.

    python -m handumi_collector.tools.make_tracking_mesh --template left --parts-dir ~/handumi-hw/hardware/STL/left_handumi
    python -m handumi_collector.tools.make_tracking_mesh --build assets/handumi/left_tracking_body.yaml
    python -m handumi_collector.tools.make_tracking_mesh --validate assets/handumi/left_tracking_body.obj
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import yaml
from ..config import REPO_ROOT
from ..pose.tracking_mesh import MOVING_PART_STEMS, RIGID_PART_STEMS, classify_part, load_tracking_mesh

MESH_EXTS = (".stl", ".obj", ".ply", ".off", ".glb")
BODY_FRAME_DOC = {
    "convention": "origin at a repeatable CAD datum on the rigid body; +x = gripper forward (approach), "
                  "+y = gripper left, +z = gripper up — the SAME axes as the TCP frame, so body->TCP is a pure offset",
    "datum": "FILL IN: name the CAD feature the origin sits on (e.g. 'centre of the controller-support ring face')",
    "note": "the tracker reports T_depthcam_body in THIS frame; configs/calibration/handumi_body_tcp_<side>_vNNN.yaml "
            "then carries body->TCP (Stage B / M2C)",
}


def write_template(side: str, parts_dir: Path, out: Path) -> Path:
    files = sorted(p for p in parts_dir.iterdir() if p.suffix.lower() in MESH_EXTS)
    if not files:
        raise SystemExit(f"{parts_dir}: no mesh files ({', '.join(MESH_EXTS)})")
    parts = []
    for p in files:
        kind = classify_part(p.name)
        entry = {"file": str(p), "classified": kind, "include": kind == "rigid"}
        if kind == "rigid":
            entry["pose"] = None          # <- fill in: translation_m [x,y,z] + quaternion_xyzw [x,y,z,w]
        else:
            entry["reason"] = ("moves with the jaw — a rigid tracker must not see it" if kind == "moving"
                               else "UNCLASSIFIED: decide whether this part is fixed to the body, then set include")
        parts.append(entry)
    spec = {
        "schema": "handumi_tracking_mesh_assembly/v1",
        "side": side,
        "units": "mm",
        "source": f"{parts_dir}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "body_frame": dict(BODY_FRAME_DOC),
        "how_to_fill": [
            "Preferred: export ONE assembled rigid body from CAD in the body frame and skip this file entirely -",
            "  point depth_pose.yaml `mesh.<side>` straight at that .obj/.stl.",
            "Otherwise: for each included part set pose.translation_m and pose.quaternion_xyzw = the part's pose in B.",
            "Never include a moving part; never scale a part to make an assembly 'fit'.",
        ],
        "parts": parts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--template", choices=("left", "right"), help="write an assembly template for this side")
    g.add_argument("--build", help="assembly YAML to union and export")
    g.add_argument("--validate", help="mesh or assembly to check")
    ap.add_argument("--parts-dir", type=Path, help="directory of per-part meshes (with --template)")
    ap.add_argument("--out", type=Path, help="output path")
    ap.add_argument("--units", help="units of the input mesh (m|mm|cm|inch); default autodetect")
    ap.add_argument("--decimate", type=int, help="target triangle count on export (needs fast_simplification)")
    a = ap.parse_args(argv)

    if a.template:
        parts_dir = a.parts_dir or Path.home() / "handumi-hw" / "hardware" / "STL" / f"{a.template}_handumi"
        out = a.out or REPO_ROOT / "assets" / "handumi" / f"{a.template}_tracking_body.yaml"
        p = write_template(a.template, Path(parts_dir).expanduser(), out)
        spec = yaml.safe_load(p.read_text())
        inc = [x["file"] for x in spec["parts"] if x["include"]]
        exc = [x["file"] for x in spec["parts"] if not x["include"]]
        print(f"wrote {p}")
        print(f"  rigid (included, pose TO FILL IN): {len(inc)}")
        for f in inc:
            print(f"    {Path(f).name}")
        print(f"  excluded (moving / unclassified):  {len(exc)}")
        for f in exc:
            print(f"    {Path(f).name}")
        print("\nNext: fill every `pose`, or (preferred) export one assembled rigid body from CAD and point "
              "configs/handumi/depth_pose.yaml `mesh` at it.")
        return 0

    src = Path(a.build or a.validate)
    tm = load_tracking_mesh(src, units=a.units)
    d = tm.diagnostics()
    print(f"{src}")
    for k in ("side", "units_in", "scale_applied", "n_vertices", "n_triangles", "extent_m", "longest_edge_m",
              "centroid_m", "origin_inside_bbox", "origin_to_centroid_mm", "is_watertight"):
        print(f"  {k:24s} {d[k]}")
    if tm.meta.get("included"):
        print(f"  {'included parts':24s} {len(tm.meta['included'])}")
        print(f"  {'excluded parts':24s} {len(tm.meta.get('excluded', []))}")
    for p in d["problems"]:
        print(f"  PROBLEM: {p}")

    if a.validate:
        return 1 if d["problems"] else 0

    mesh = tm.mesh
    if a.decimate and len(mesh.faces) > a.decimate:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=int(a.decimate))
            print(f"  decimated to {len(mesh.faces)} triangles")
        except Exception as exc:
            print(f"  decimation unavailable ({exc}) — exporting the full mesh")
    out = a.out or src.with_suffix(".obj")
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    meta = dict(schema="handumi_tracking_mesh/v1", side=tm.side, units="m", source=str(src),
                body_frame=tm.meta.get("body_frame", BODY_FRAME_DOC), included=tm.meta.get("included", []),
                excluded=tm.meta.get("excluded", []), diagnostics={k: v for k, v in d.items() if k != "problems"},
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    Path(str(out) + ".meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {out}  (+ {out.name}.meta.yaml)")
    return 1 if d["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
