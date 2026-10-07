# hardware/ MANIFEST

Assembled 2026-10-07. The sources were only read, never modified.

| source | dest | notes |
|---|---|---|
| `~/handumi-print/README.md` | `print/README.md` | Plate overview for the whole package |
| `~/handumi-print/left/` | `print/left/` | 4 × 3MF, README, `stl/` (12 STL) |
| `~/handumi-print/right/` | `print/right/` | 4 × 3MF, README, `stl/` (12 STL) |
| `~/handumi-print/tips/` | `print/tips/` | 1 × 3MF, `stl/` (2 STL, the temporary AgileX-Piper tip) |
| (new) | `README.md` | Build summary, BOM pointers, upstream attribution |
| (new) | `MANIFEST.md` | This file |

## Excluded

| item | reason |
|---|---|
| `~/handumi-print/bambu_cloud_access_code.py`, `bambu_cloud_access_code.py.bak`, `bambu_lan_auth_test.py`, `bambu_status.py`, `bambu_upload.py` | Printer credential and access-code helpers (Bambu cloud/LAN). These must not go into the repo. |
| `~/handumi-print/__pycache__/`, `.DS_Store` | Build and OS cruft |

## Size notes

The STL, 3MF and STEP files are allowed above 5 MB. Two STLs are larger than 5 MB:

- `print/left/stl/left_controller_support.stl` (6.9 MB)
- `print/right/stl/right_controller_support.stl` (6.9 MB)

The 3MF plate projects (0.7–3.4 MB each) embed sliced G-code for the X1C. The 3MF metadata was checked and holds no
printer serial or access code.

The upstream STEP sources are not copied. They live in `robonet-ai/handumi-hw` at `e58de33`, under `hardware/STEP/`;
see `../upstream/README.md`.
