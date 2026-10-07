# Our modifications to the upstream HandUMI parts

Three parts were changed in Shapr3D (September 2026). The upstream originals are in `../cad/STEP/` and `../print/`; the files
here are what we actually use or tried.

| part | date | what changed | files | status |
|---|---|---|---|---|
| `left_controller_support` → **connector** | 2026-09-08 | New connector derived from the controller support (85.7 × 21.4 × 42.1 mm, two bodies: a C-channel clamp plate + the connector block). Replaces the upstream controller support for our rig (no VR controller). | `left_controller_support_connector/connector_only_shapr3d.stl` (Shapr3D export), `connector_only_brep.step` (connector block as a clean 70-face B-rep) | printed (×2) |
| connector **plate, y +10 mm** | 2026-09-14 | The clamp plate's C-channel arms extended 10 mm along y (depth 21.4 → 31.4 mm); the connector block unchanged. A 15 mm version and three z +30 mm variants were tried and rejected. | `plate_ystretch10.step` (plate + connector, 2 solids), `plate_ystretch10_print.stl` (moved to z ≥ 0 for slicing), `plate_ystretch10_X1C.3mf` (X1C, 0.4 nozzle, 0.20 mm, PLA, tree support; ~35 min, ~17.5 g) | sliced; current version |
| `camera_mount` | 2026-09-08 | A second rectangular window cut into the plate (above the existing one). Outline and hinge tabs unchanged. | `camera_mount/camera_mount_mod.stl` | — |
| `servo_controller_cover` | 2026-09-11 | The cable opening widened to almost the full width and the inside of the cover reshaped. Outline unchanged. | `servo_controller_cover/servo_controller_cover_mod.stl` | — |

Images: `images/controller_support_connector.png`, `images/camera_mount_servo_cover_vs_upstream.png` (upstream left, ours right).

Notes:

- The editable sources of these parts live in the Shapr3D project (not exportable from the command line). The STL exports
  here are meshes; the STEP files in `left_controller_support_connector/` were made from them (`plate_ystretch10.step` is a
  meshed STEP, ~15 MB; the connector block alone is a true B-rep). To change offsets freely, edit in Shapr3D and re-export.
- Only a left-hand connector was made. The camera mount and servo cover are the same part on both hands.
- Shapr3D STL exports are in the original model coordinates; some have negative z, which the Bambu CLI rejects (translate
  to z ≥ 0 first, as in `plate_ystretch10_print.stl`).
