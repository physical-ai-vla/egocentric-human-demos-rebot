# HandUMI hardware (reBot skill-capture gripper)

HandUMI is a hand-worn variant of UMI. We use it as the human-side "skill capture gripper" for the reBot B601
instead of designing our own. The base design is **[robonet-ai/handumi-hw](https://github.com/robonet-ai/handumi-hw)**
(Apache-2.0, released 2026-08-20). Our print package was generated from commit `e58de33` of that repo. We have made no
geometry changes to the upstream parts; the only thing left to design is a reBot-specific gripper tip (see below).
License text and attribution are in `../upstream/` (`LICENSE-handumi-hw-apache-2.0.txt`, `NOTICE.md`).

## What is here

| path | content |
|---|---|
| `print/README.md` | Plate overview, print order and fit notes for the full package (both hands plus the tip) |
| `print/left/`, `print/right/` | Four Bambu Studio 3MF projects per hand (`A_fitcheck`, `B_main_support`, `C_mounts`, `D_small_parts`), `stl/` (12 parts per hand), and a per-hand README with per-part time and PLA |
| `print/tips/` | `E_piper_generic` 3MF plus the L/R AgileX-Piper tip STLs. This is a **temporary** generic parallel-jaw tip for mechanism and pinch tests. A reBot tip will replace it. |

The STLs are the upstream `hardware/STL/{left,right}_handumi` and `gripper_tips` meshes. The geometry is unchanged; each
mesh is only translated so all coordinates are positive, because the originals have negative Z and the Bambu CLI
refuses them ("no object fully inside"). Edit the upstream STEP sources (`handumi-hw/hardware/STEP/...`), not these STLs.

## Build summary

- **Print**: PLA, Bambu 0.20 mm Standard, 2 walls, 20 % infill. There are 9 plates and 26 parts in total, about 9 h and
  about 250 g PLA (about 4 h and 110 g per hand). Supports are off everywhere except `*_controller_support` (tree
  support under the cantilevered screw tab).
- **Print `A_fitcheck` first, for both hands.** The thumb and index/middle cradles come from the upstream author's hand
  scan. If they don't fit, edit the cradle STEP files before printing anything else.
- **Fits**: MR63ZZ bearing (6 mm OD), LM4UU linear bearing (8 mm OD), 4 mm shaft. If holes are tight, use XY hole
  compensation of +0.05 to 0.1 mm instead of scaling the part. Print the finger links cradle-up with the LM4UU bores
  vertical, as arranged in the 3MF.
- **Slicing gotchas** (Bambu Studio 2.08):
  - All 3MFs are X1C projects. For the H2D, switch printer in the GUI and re-slice.
  - The H2D cannot be sliced from the CLI. The filament-to-extruder map is a per-project key inside the 3MF, not a
    preset key, so the CLI cannot set it.
  - CLI time and PLA estimates ignore supports.
- **CAD editing**: we edit the STEP sources in Shapr3D. Fusion 360 and FreeCAD were tried and rejected.

## Electronics / BOM

The upstream BOM is `handumi-hw/bom/README.md` (also as PDF/DOCX), roughly US$110 per unit:

- Mechanical: PLA, 4x MR63ZZ, 4x LM4UU, 4 mm axle (135 mm rod), M2/M3 hardware, velcro straps.
- Jaw encoder: Feetech **STS3215** servo plus a Waveshare serial bus servo adapter. The adapter is a CH343, 1 Mbaud, with
  servo id 1 on our units. It needs its 12 V supply before the servo answers.
- Wrist camera: **Arducam 160° fisheye USB** (1080p UVC).

Our capture rig deviates from upstream in three ways:

- **No VR tracking.** The upstream design gets wrist SE(3) from a PICO/Quest controller. Our collector gets wrist
  motion from the fisheye camera plus an IMU. The Quest-tracker variant lives in our `handumi-sw` fork; see
  `../upstream/`.
- **Added wrist IMU per unit**: a Teensy 4.1 with an ICM-42688-P on SPI. Firmware is in `../collector/firmware/`
  (`teensy_imu/`, plus a `xiao_imu/` variant and the `bringup/` probes).
- **Head camera**: a Logitech C922 at 640x480@30 is the shared human/robot policy view.

Device identities (USB serials, product names) and the bring-up procedure are in `../collector/README.md` and
`../collector/docs/IMU_BRINGUP.md`.

## Open items

- A reBot B601 gripper tip does not exist yet. The base for it is upstream `hardware/STEP/gripper_tips`.
- The finger cradles fit the upstream author's hand. Check the fit for each operator.
