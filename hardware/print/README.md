# HandUMI full print package  (2026-08-26, from robonet-ai/handumi-hw @ e58de33)

PLA, Bambu 0.20mm Standard, 2 walls / 20% infill. All 3mf are X1C projects (open in Bambu Studio;
for H2D just switch printer and re-slice). STLs are the originals translated to positive coords.

| folder | plate | parts | time | PLA |
|---|---|---|---|---|
| left/  | A_fitcheck      | thumb + index/middle cradle, hand_support_base | 1h06m | 31 g |
| left/  | B_main_support  | fisheye_camera_main_support | 1h17m | 33 g |
| left/  | C_mounts        | controller_support, servo_controller_support, servo cover, main cover plate | 1h07m | 37 g |
| left/  | D_small_parts   | camera_mount, crank plate, connecting links x2 | 0h28m | 9 g |
| right/ | A_fitcheck      | (mirror) | 1h06m | 31 g |
| right/ | B_main_support  | (mirror) | 1h17m | 33 g |
| right/ | C_mounts        | (mirror) | 1h06m | 37 g |
| right/ | D_small_parts   | (mirror) | 0h37m | 9 g |
| tips/  | E_piper_generic | AgileX-Piper L+R tip — TEMPORARY generic parallel-jaw tip for mechanism/pinch testing; the reBot tip will replace it | 0h59m | 28 g |
| **total** | 9 plates, 26 parts | | **~9h** | **~250 g** |

## Order
1. left/A_fitcheck and right/A_fitcheck FIRST (finger cradles are from the author's hand scan; check fit
   before printing the rest). If they don't fit, stop and tell us — cradle STEP files need editing.
2. Then B, C, D for both hands, then tips/E.

## Print notes
- Supports: OFF everywhere except `*_controller_support` (ring with cantilevered screw tab -> tree support).
  Check `fisheye_camera_main_support` end box in preview; add support only if the slicer flags >45° overhang.
- Finger links: cradle facing up, the two 8 mm LM4UU bearing bores vertical (as arranged in the 3mf).
- Fits: MR63ZZ bearing 6 mm OD, LM4UU 8 mm OD, 4 mm shaft. Check after the first plate; if tight use
  XY hole compensation (+0.05~0.1 mm), don't scale parts.
- Any PLA color is fine. No TPU needed for these plates.
