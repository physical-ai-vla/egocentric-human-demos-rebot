"""[2026-10-07] processed_mix = HRA_red processed_robotcam (166, robotcam_v2 labels) + HRA_A100 processed_robotcam: episode dirs
symlinked, manifests concatenated per split (each source keeps its own deterministic split)."""
import json, pathlib, sys
srcs = [pathlib.Path(p) for p in sys.argv[1:-1]]; out = pathlib.Path(sys.argv[-1]); (out / "episodes").mkdir(parents=True)
for s in srcs:
    for e in (s / "episodes").iterdir(): (out / "episodes" / e.name).symlink_to(e.resolve())
for sp in ("train", "val", "test"):
    with open(out / f"{sp}_manifest.jsonl", "w") as f:
        for s in srcs:
            p = s / f"{sp}_manifest.jsonl"
            if p.exists(): f.write(p.read_text())
m = {sp: sum(1 for _ in open(out / f"{sp}_manifest.jsonl")) for sp in ("train", "val")}
(out / "metadata.json").write_text(json.dumps(dict(sources=[str(s) for s in srcs], counts=m), indent=1)); print("mix", m)
