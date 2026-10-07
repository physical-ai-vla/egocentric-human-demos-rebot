# HandUMI Right — print package (generated 2026-08-26)

Source: https://github.com/robonet-ai/handumi-hw  (commit e58de33, 2026-08-20)
`stl/` = hardware/STL/right_handumi/*.stl, geometry untouched, only translated so
all coords are positive (original files had negative Z; Bambu CLI refused them).

## Plates (Bambu 0.20mm Standard, PLA, 2 walls, 20% infill, NO supports, auto-orient)
Estimates from BambuStudio 2.08 CLI with X1C preset; H2D will be similar.

| 3mf | parts | time | PLA |
|---|---|---|---|
| A_fitcheck    | right_thumb_link, right_index_middle_finger_link, hand_support_base | 1h06m | 31 g |
| B_main_support| fisheye_camera_main_support (114x140x83) | 1h17m | 33 g |
| C_mounts      | right_controller_support, right_servo_controller_support, servo_controller_cover, main_support_cover_plate | 1h07m | 37 g |
| D_small_parts | camera_mount, crank_mechanism_plate, connecting_link_1, connecting_link_2 | 0h37m | 9 g |
| **total right hand** | 12 parts | **~4h** | **~110 g** |

Per-part: camera_mount 26m/4.3g · links 5m/1.1g each · crank 7m/2.7g · main_support 77m/33g ·
hand_support_base 22m/10g · controller_support 26m/15g · index_middle_link 22m/10g ·
servo_controller_support 25m/10g · thumb_link 23m/11g · main_cover_plate 10m/5g · servo_cover 8m/6g

## Order
1. Print **A_fitcheck first** — thumb/index-middle cradles are designed from the author's hand scan.
   If they don't fit, edit STEP/right_handumi/right_thumb_link.step / right_index_middle_finger_link.step.
2. Then B, C, D.

## Orientation / support notes (check in GUI before sending)
- Finger links: cradle up, LM4UU bores (8 mm) vertical — keep this; bore accuracy matters.
- controller_support: ring flat; the screw tab is cantilevered -> enable supports (tree) for that part.
- main_support: flat plate down, camera post up. Check the end box overhang; add supports if >45°.
- Small pins/holes: bearings MR63ZZ 6 mm OD, LM4UU 8 mm OD, shaft 4 mm. Measure after first print;
  if tight, XY hole compensation +0.05~0.1 mm rather than scaling.

## Open in BambuStudio
Files are X1C projects; switching printer to H2D in the GUI is fine (re-slice).
