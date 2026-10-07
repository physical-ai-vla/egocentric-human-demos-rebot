# REL16 400k real-robot serving provenance (frozen 2026-09-28)

Source: live UI :8025 /status (model block), launcher umi_bridge/load_ssd_ckpt_to_ui.sh, v4_smoke_ui_8025.log.

| item | value | how verified |
|---|---|---|
| checkpoint | ckpt_B180H-UMI76-REL16-D600K_400k (model.safetensors md5 e89c332b2580952b6b026aa76c0863b4) | /status model.ckpt |
| state_mode | umi76 (width 76) | /status |
| action_mode | umi (current-anchor REL, T_now @ A_k) | /status |
| chunk | 16 | /status |
| exec_k / n_action | 8 / 1 (execute waypoint index 7, then replan) | /status |
| global rot180 / mirror | false / false | /status |
| camera map | middle->global->image, left->left_wrist->image2, right->right_wrist->image3 | infer_core_v4 CAM_MAP/RENAME |
| V4_FRAME_FIX | 1 (env unset; default "1") -> USE_FRAME_FIX=True | launcher env + code default |
| history_dt | 50.05 ms (3/59.94) | /status history_dt_ms |
| clamp | 0 mm / 0 deg (off) | startup line |
| gripper | predict (width -> raw -> cmd = raw/-6, clip 0..45) | /status |
| history-ready rejections | 0: a rejection raises -> loop ends "error: inference failed"; all 3 loops ended "stopped by user" / "rest" | 8025 log |
| noise patch | applied in xvla-mac lerobot (predict_action_chunk forwards noise; 1 call sites pass noise=noise) | grep modeling_xvla.py |
| /observe gripper unit | raw count (arms rad) -> matches raw_to_width | robot_service.py:97-101 |

Runs logged on 8025: 28 cycles [47, 30] mm; 21 cycles [23, 18] mm; 246 cycles [1346, 117] mm (L approaches first cube,
jaw ends fully open: last pred width L 110.5 mm -> cmd 43.5, pred k8 motion 1.3 mm).

Deployment observation cadence: stop-and-go. Each cycle observes twice 66.7 ms apart AFTER the previous waypoint's dwell
(+0.2 s settle), so the state76 self-history velocity is ~0 at every inference by construction.

## 2026-09-28 deployment vision orientation (REL16-v2B contract)
Live geometric cube-placement calibration (`r380/live_cube_orient.py`: u_L 55 < u_R 182, v_F 42 < v_N 214) matched the
training image convention on both horizontal and depth axes; deployment is frozen with `V4_GLOBAL_ROT180=0`.
Runtime preprocessing parity P1 PASS (JPEG q80 -> _img reproduces training frames, rot0 0.005-0.007 vs rot180 0.08-0.12).
Failure attribution for the v1 live runs stays separate: wrong cube = prompt/conditioning/visual shortcut; no grasp =
follower-width gripper label. Orientation is not a factor.
Baseline contract: R312-center-v2B + frozen launcher + 4090 physical GPU1 + seed 1000 (5090 not used for the baseline).

## 2026-09-28 R312-center build validation note
gate_r312c.py was fixed twice (G3 read a nonexistent `r663_episodes` key; G3 compared full task strings with short
order codes). Gate implementation bugs only; no dataset content changed; chain resumed from validation without
reconversion.

## 2026-09-28 R312-center training orientation validation (recorded as-is, not reclassified)
- original frozen full-motion sign gate (HEAD180 vs converted FRONT132, left arm only, x/y same sign and |corr| > 0.3):
  **FAIL on the forward/back magnitude criterion** -- HEAD (-0.39, -0.40), FRONT converted (-0.53, -0.17); dz->dv HEAD -0.25, FRONT +0.25
- diagnosis: cross-view dz->dv confound (FRONT faces the robot from the front; its vertical image motion for +z is opposite to HEAD)
- planar sensitivity analysis (|dz| < 0.3|dxy|, same method on both pools, post hoc): PASS -- HEAD (-0.43, -0.43), FRONT (-0.44, -0.51)
- conclusion: rot180 / table-plane orientation validated (also G4 pixel parity, live cube test ROT180=0); a known HEAD<->FRONT
  **VIEW_DOMAIN_GAP** in vertical image-motion geometry remains. Not an ORIENTATION_ERROR; no transform is added (a vertical
  flip would break the table plane), FRONT132 is not re-selected, the dataset is unchanged.
- consequence for the ladder: an R312 vs R180 difference may partly come from this viewpoint gap, not only from data quantity.

## 2026-09-28 pre-training contract hardening (R312-center-v2B)
Pre-training contract hardening performed before any R312-center-v2B result existed; no training semantics changed, only a
missing critical dependency was added to the freeze/assertion coverage. The node's uncommitted datasets/factory.py patch
(sha 3998f9b7...) keeps the stored (16,20) action chunk from being re-expanded; a negative test with the unpatched git-HEAD
file reproduced the silent (16,16,20) nesting. New `preflight_rel16v2.py` (sha f11692fd...) runs lerobot_train's own
code path on CPU; the launcher `train_umi_v2c.sh` (now 6f7e54699460a420, supersedes d9087bbb kept as .frozen_d9087bbb,
recipe arguments identical) hard-fails on a factory/preflight hash mismatch or a preflight failure. Addendum:
~/umi_bridge/rel16v2b_r312c_freeze/INIT_FREEZE_ADDENDUM.json. End-to-end guard test: preflight PASS, then refused on busy GPU1.

## 2026-09-29 REL16-v3 deployment contract + R312C-B reinterpretation
- Gripper execution threshold FROZEN at V4_GRIP_THRESH=0.6 (= cmd 27/45, the validated binary boundary); OPEN -> 42, CLOSE -> 0.
  Loader `umi_bridge/load_v3_ckpt_to_ui.sh` pins the whole v3 contract (umi76, REL chunk 16, dims 0..19 only, exec_k 16, n_action 1,
  dwell 1.5 s, clamp off, ROT180 0, mirror 0) and asserts the checkpoint is 32-action / 76-state.
- R312C-REL16V2-B (and REL16 v1) started from a widened base whose action encoder was scrambled (widen_proprio.py /
  widen_append94.py reshaped input-major weights as output-major; cosine to the true pretrained rows +0.001). Its Stage-1 failure is
  therefore confounded by a broken initialization path and must not be read as evidence against state94 / FRONT / REL16 as such.
  The FRONT view-domain gap remains a separate, real finding. REL16-v3 uses widen_v3.py (functional equivalence max diff 0).
