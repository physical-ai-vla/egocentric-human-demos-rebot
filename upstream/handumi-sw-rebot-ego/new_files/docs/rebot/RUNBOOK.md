# Egocentric Bimanual Data Collector MVP — head-cam AprilTag EEF → robot-agnostic raw → QA → retarget → VLA

HandUMI(`murobotics-ai/handumi-sw`) fork, branch `rebot-ego`. **고정 global cam / Quest / T265 / VIO / SLAM 없음.**
Head B0202(160° fisheye) 하나가 (a) VLA head view, (b) world tag로 매 프레임 head 로컬라이제이션, (c) L/R UMI tag 트래킹을 담당한다.

```
Head B0202 ─┬─ world tags 100..103 ─→ T_cam_world (per frame)
            └─ UMI tags 10/11, 20/21 ─→ T_cam_anchor
                 T_world_TCP = inv(T_cam_world) · T_cam_anchor · T_anchor_TCP
Wrist B0202 ×2 (observation only) + Feetech jaw telemetry (pos/goal/speed/load/current/V/T)
      → 30 Hz synchronizer → robot-agnostic raw (LeRobot v3 + raw tag/motor columns)
      → QA → derived → convert-eef (14D) / convert --robot rebot_b601 (IK) → replay → VLA → reBot
```

## 0. 환경 (완료)

```bash
cd ~/handumi-sw && uv sync --extra sim   # py3.12; vosk(음성) darwin 제외 → record 시 --no-voice-control
uv run pytest tests/test_apriltag.py tests/test_apriltag_provider.py tests/test_tag_mask.py tests/test_feetech_telemetry.py \
              tests/test_derived.py tests/test_eef_actions.py tests/test_tracker_eval.py tests/test_cube_stack.py -q
```

### 제안 repo 구조 ↔ fork 모듈

| 제안 (`bimanual_ego_collector/`) | fork | 상태 |
|---|---|---|
| `cameras/*` | upstream `handumi/cameras/` (OpenCV UVC, ring buffer, monotonic ts); rig `cameras.head/left_wrist/right_wrist` | upstream(+`head` 이름 허용) |
| `tracking/apriltag_detector, world_tag_map, umi_tag_bundle, head_pose_solver, eef_pose_solver, tracking_quality` | `handumi/tracking/apriltag.py` (`AprilTagDetector`, `WorldMap`, `BundleConfig`, `solve_bundle`, `process_frame`, `build_pair_sample`, `AprilTagTrackingProvider`, `CameraGeometry` fisheye→rectified) | **신규** |
| `gripper/motor_driver, jaw_calibration, contact_estimator` | `feetech/telemetry.py` + `bus.read_status_block`(addr 42, 29 B) / upstream `calibrate grippers` / `dataset/derived.py` | 신규 / upstream / 신규 |
| `calibration/camera_intrinsics, world_map` | `scripts/setup/calibrate_head_camera.py` (`intrinsics` fisheye, `world-map` ChArUco 기준) | **신규** |
| `calibration/tag_to_tcp` | upstream pivot (`calibrate tcp pivot`), anchor tag = "controller" | upstream |
| `sync/*` | upstream `synchronization.py` (target −40 ms nearest sample + skew) | upstream |
| `recorder/*` | upstream `handumi record --device apriltag` (+ `observation.apriltag.*`, telemetry 컬럼) | 패치 |
| `processing/tag_mask` | `processing/tag_mask.py` + `handumi dataset mask-tags` → `observation.images.head_clean` | **신규** |
| `processing/relative_pose, relative_action, bimanual_features, contact_features` | `dataset/derived.py`(head-relative TCP 포함), `dataset/eef_actions.py` | **신규** |
| `qa/*` | upstream `handumi validate` + `configs/quality_mvp.yaml` | upstream+설정 |
| `recorder/metadata` | `handumi dataset label` (outcome/recovery/operator/cube plan sidecar) | **신규** |
| `retarget/*`, `exporters/lerobot` | upstream IK + `configs/robots/rebot_b601.yaml`; writer(`convert`, `convert-eef`) | 프로필 신규 |

## 1. 하드웨어

- B0202 ×3 (head 1920×1080@30 fisheye, wrist ×2 640×480@30). rig.yaml `cameras.head` 인덱스 확인(`handumi setup ports`).
- HandUMI L/R + STS3215 jaw servo, 2-face UMI tag bundle (L 10 anchor+11, R 20 anchor+21), world tag 100–103 (테이블 네 귀, 중앙 비움).
- ChArUco 5×7 보드(world frame 정의용).

```bash
uv run handumi tracking apriltag print --out outputs/tags --world-size 0.08   # 100 % 출력
```

**태그 크기 — 합성 테스트에서 확인된 한계:** 160° fisheye 1080p는 f≈560 px. 50 mm 태그가 0.7 m에서 ≈40 px(4 px/module)이면
검출이 빠지기 시작한다(`test_fisheye_small_hand_tags_drop_out`). **손 태그 ≥70 mm, world 태그 ≥80 mm** 권장; 그때 합성 정확도
≈2 cm/3°(pinhole 1100 px 기준 <8 mm/1.5°). 5–10 mm repeatability가 필요하면 head 카메라 해상도(4K) 또는 협시야 렌즈를 검토.

## 2. Calibration (녹화 전 5가지)

```bash
# ① 카메라 intrinsics
uv run handumi calibrate head-camera intrinsics --views 20                 # fisheye → spatial.yaml cameras.head
uv run handumi calibrate spatial intrinsics --camera left_wrist            # (upstream, wrist는 observation 전용)
uv run handumi calibrate spatial intrinsics --camera right_wrist

# ② World tag map: ChArUco 보드를 테이블 원점에 두고(+X 우, +Y 전방, +Z 위) head를 움직이며 40프레임
uv run handumi calibrate head-camera world-map --tag-ids 100,101,102,103 --tag-size 0.08 --frames 40
#    → outputs/calibration/world_map.yaml (T_world_tag ×4, 잔차 RMS). 보드 없이: --anchor-tag 100 (world = tag100 프레임)

# ③ UMI tag → TCP
uv run handumi tracking apriltag bundle-calib --side left  --frames 40    # 2번째 면 T_anchor_tag (bundles.yaml 갱신)
uv run handumi tracking apriltag bundle-calib --side right --frames 40
uv run handumi tracking apriltag preview                                   # world / L / R 오버레이 + head pose 출력 (M1–M3 gate)
LEFT=outputs/pivot_left
uv run handumi record --device apriltag --output-dir $LEFT --skip-feetech --no-voice-control --task pivot-left
uv run handumi calibrate tcp pivot --side left --dataset $LEFT --device apriltag   # RMS<5 mm, max<10 mm; right 동일
#    → 대칭화 후 configs/calibration/controller_tcp/apriltag_piper_tip_temp.yaml (rebot_b601.yaml이 가리킴)

# ④ Encoder → jaw width
uv run handumi setup ports && uv run handumi calibrate grippers

# ⑤ Motor current baseline: 빈손 open/close 반복 에피소드 1개 녹화 → `dataset derive`의 current_baseline(20 % 백분위)로 사용
```

## 3. Tracker acceptance (M4 gate)

```bash
CAL="--device apriltag --robot rebot_b601"
uv run handumi tracking eval capture $CAL --seconds 30 --no-marks --output outputs/eval/static.csv   # HandUMI 고정, head는 착용 상태로 정지
uv run handumi tracking eval jitter outputs/eval/static.csv                                           # trans RMS <5 mm, rot <1°
uv run handumi tracking eval capture $CAL --output outputs/eval/return.csv                            # A점 Enter ×20
uv run handumi tracking eval return outputs/eval/return.csv --side left                               # <10 mm
uv run handumi tracking eval table-z outputs/eval/return.csv                                          # world z≈0 (<15 mm)
uv run handumi tracking eval capture $CAL --output outputs/eval/agree.csv                             # 양 tip 같은 점 Enter ×10
uv run handumi tracking eval agreement outputs/eval/agree.csv                                         # <10 mm
```

Head가 움직이는 상태의 jitter는 world-tag 로컬라이제이션 오차를 포함한다 — 정지 head와 움직이는 head 두 조건을 모두 기록해 비교.

## 4. 녹화 (M5–M8): **먼저 10 에피소드**

```bash
uv run handumi task cube plan --regime A --episodes 60 --seed 1 --output outputs/plans/A.json
EP=0
uv run handumi record --device apriltag --robot rebot_b601 --no-voice-control \
    --output-dir outputs/cubes_A \
    --task "$(uv run handumi task cube show outputs/plans/A.json --episode $EP --task-only)"
uv run handumi dataset label outputs/cubes_A --episode $EP --outcome success --operator op01 --plan outputs/plans/A.json
```

row(30 Hz)에 저장: `observation.images.{head,left_wrist,right_wrist}`, `observation.state`(world-frame anchor pose7 ×2 + width),
`observation.tracking.*` (`workspace_from_device_pose` = **head pose in world**, `{side}_device_controller_pose` = **head-relative** anchor),
`observation.apriltag.{world,left,right}_{num_tags,reproj_px,tag_ids,corners_px,camera_pose}` + `corner_space`(1 = rectified 픽셀),
`observation.feetech.{side}_{ticks,width_mm,normalized,goal_position,speed,load,current_ma,voltage_v,temperature_c,telemetry_ok}`,
`observation.camera.*` / `observation.sync.*`(skew). world anchor가 안 보이면 두 손 `tracked=0` → 1 s 이상 지속 시 에피소드 discard(upstream gate).

## 5. QA + head_clean (M9)

```bash
uv run handumi validate outputs/cubes_A --quality-config configs/quality_mvp.yaml   # visibility>99 %, skew<10 ms, jump 0
uv run handumi dataset analyze outputs/cubes_A
uv run handumi dataset mask-tags outputs/cubes_A --camera head --method fill        # → observation.images.head_clean (VLA 입력)
uv run handumi replay outputs/cubes_A --robot rebot_b601 ...                         # pose/gripper/RGB 정렬 시각 확인
```

## 6. Derived / Action / Retarget (M10–M11)

```bash
uv run handumi dataset derive outputs/cubes_A --robot rebot_b601
#   derived/episode_XXX.parquet: {side}.delta.*, bimanual.left_from_right.*, hand_distance(_rate),
#   {side}.gripper.{closing,opening,grasp_start,release,grasped}, {side}.motor.{current_baseline,current_delta,contact_estimate,possible_slip},
#   head.pose_world.*, {side}.tcp_head.*  (head-relative TCP)
uv run handumi convert-eef outputs/cubes_A --robot rebot_b601 --output outputs/cubes_A_eef        # state16 + 14D delta action (world frame)
uv run handumi convert     outputs/cubes_A --robot rebot_b601 --output outputs/cubes_A_rebot \
    --robot-table-calibration configs/calibration/table/rebot_b601.yaml                             # IK → reBot 14D joint
```

VLA 입력: `head_clean` + wrist L/R + instruction + state; action = 14D delta chunk.

## 7. 남은 것

- [ ] B0202 ×3 장착·인덱스, 태그 출력(손 ≥70 mm), ①–⑤ 캘리브레이션, §3 gate, 10-에피소드 M9 replay
- [ ] Feetech telemetry 실기 확인(addr 42..70 블록; 실패 시 position-only fallback)
- [ ] raw 100 Hz motor / 30 Hz tag 스트림 별도 파일(`episode/raw/*.parquet`) — 현재 30 Hz row에 nearest sample + skew
- [ ] 짧은 world-anchor 손실 구간 offline interpolation (현재 frame_valid=False만)
- [ ] `handumi doctor` / `calibrate verify` apriltag 지원
- [ ] `robot_from_table`·gripper 폭 실측, reBot 메쉬 vendoring(viser), reBot LeRobot gripper raw↔width 어댑터
