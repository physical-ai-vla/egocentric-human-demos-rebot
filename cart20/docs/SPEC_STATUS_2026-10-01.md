# Ego CART20 v2 pretrain: 명세 + 진행 현황 (2026-10-01)

X-VLA v4 ego-only pretrain 파이프라인. 목표 순서: X-VLA base → **ego CART20 pretrain** → reBot R312c RELCART20 REL-only D6 fine-tune → 실로봇.
코드 `~/ego_cart20/` (패키지 `ego_cart20/`), 데이터 `~/c8/ego_cart20_v2*`.

---

## 1. 확정 contract (spec v2 + 사용자 결정 3개 반영)

spec v2에서 실제 v4 contract와 어긋난 세 가지를 사용자가 결정했다(D1–D3). 나머지는 spec v2 그대로다.

| 항목 | 최종 값 | 비고 |
|---|---|---|
| row 저장 주기 | ~15 Hz (canonical grid `t_s + n/15`) | |
| **target 시간 간격 (D1)** | **UMI_DT = 3/59.94 s = 50.05 ms** | spec v2의 "k-step = 66.7 ms"는 폐기. 실제 v4/R312c 라벨과 같다. REL16 = 16 × 50.05 ms = **0.80 s** |
| action | `A_k = inv(T(t)) @ T(t + k·UMI_DT)`, k = 1..16 | 현재 자세 기준 상대값. 순차 delta 아님. 미래 자세는 **raw 30 Hz 궤적을 보간**한다(위치 lerp, 회전 SLERP). 15 Hz row를 가져다 쓰지 않는다 |
| CART20 | `[L xyz3 rot6d6 g | R xyz3 rot6d6 g]` (그리퍼 9, 19) | LEFT 먼저 |
| model action | **[16, 32]** = CART20 (0:20) + **AUX12 = 0** (20:32) | 가짜 pseudo-q나 Δq를 만들지 않는다 |
| **state (D3)** | **RELCART20 task anchor**, 팔별 `inv(T(t_task_start)) @ T(t)`, layout `[L pose9 | R pose9 | gL | gR]` (그리퍼 18, 19) | reBot FT 데이터(R312c RELCART20)와 의미·채널 배치가 같다 |
| state_prevrel | spec v2의 `inv(T(t)) @ T(t − UMI_DT)`, layout `[L9 g | R9 g]` | **진단·ablation 전용**. 기본 export에 쓰지 않는다. `state_prev_valid` 마스크를 함께 둔다 |
| **gripper (D2)** | 연속값 **0 = 닫힘, 1 = 열림**, g = caliper mm / 80 mm | v4 극성과 같다. binary로 바꾸지 않는다. action k에는 `g(t + k·UMI_DT)`를 보간해서 넣는다 |
| rot6d | 회전 행렬의 첫 두 **행** (umi `mat_to_rot6d`), identity = [1,0,0,0,1,0] | v4와 같다 |
| tail | 16개 target이 모두 실제 데이터 안에 있는 row만 쓴다. padding 없음 | 보간 구간에 invalid raw sample이나 50 ms를 넘는 공백이 있으면 그 row를 버린다 |
| canonical frame | 팔별 task-start frame (`per_arm_task_start_v1`) | 왼쪽에 곱하는 frame 변환은 라벨에서 상쇄되므로 action/state 값에는 영향이 없다(아래 4.1) |

---

## 2. 입력 데이터

- **Source:** HandUMI 사람 egocentric 데모.
  - old259 (Hpilot): census first_fail ∈ {None, no_segment}이고 양손 IMU-VI scale이 유효한 273 ep.
  - HRL80: frozen split의 60 ep.
- **포즈:** 손목별 MASt3R-SLAM + IMU-VI metric scale + `handumi_camera_tcp_v2` → 사람 TCP.
  - 여기에 고정 tool frame `X = F @ C @ CT4`를 오른쪽에 곱해서 reBot dataset-TCP 축 규약에 맞춘다(gcal1과 같은 conjugation).
  - **좌·우는 각자 자기 시계에서 따로 보간한다.** 기존 gcal1은 오른손을 왼손의 가장 가까운 frame에 짝지었다.
- **Gripper:** gcal1 caliper calibration을 그대로 쓴다(`umi_aperture_cal_v1`, `rel16ego_derive_gcal1.G` 함수를 import).
- **카메라:** head → `global`, left_wrist, right_wrist.
  - capture_ns 기준 20 ms 안의 가장 가까운 frame을 쓴다. head가 없는 row는 학습에서 뺀다.
  - 원본 1920×1080을 INTER_AREA로 224×224로 줄인다. 기존 ego exporter와 같은 방식이다.
- **선택 규칙:** workspace·IK·pseudo-q로 거르지 않는다. 이전 TR 세트(996 segments)는 로봇 IK가 가능한지로 골랐기 때문에 이번에는 쓰지 않았다.
- **Export 결과:** 330 ep OK. 3 ep(HRL80 21/29/48)은 한쪽 손의 IMU-VI scale이 유효하지 않아 제외했다.

## 3. 데이터셋 수치

| | episodes | 학습 row |
|---|---|---|
| train | 298 | 51,612 |
| val | 32 | 5,671 |
| test | 0 | 0 (freeze된 test split이 없다. 새로 만들면 train이 바뀐다) |

- 순서별 train episode: RBP 55 / RPB 53 / BRP 53 / PRB 48 / BPR 47 / PBR 42.
- Split은 기존 ego 세트와 같다: val = C-old v1 26 + HRL80 round-block ep49–54.
- 15 Hz row 중 유효(학습) row는 약 50%다. 나머지는 tracking lost, sync 실패, 16-step tail 때문에 빠졌다.
- 이전 gcal1 TR 세트(17,024 row)보다 약 3배 많다.

## 4. 검증 결과 (전부 PASS)

- **합성 단위 테스트 10/10** (`tests/test_contract.py`):
  - current-anchor (A_k.x = 0.01k)
  - 시간 간격 = UMI_DT (66.7 ms 아님)
  - state_prevrel 방향 (−0.01)
  - RELCART20 layout, anchor = identity
  - 미래 gripper 보간
  - pack/AUX12 = 0. 0이 아니거나 NaN이면 hard error
  - tail/gap에서 padding 없음
  - rot6d·quaternion 왕복, SLERP
- **Integrity report** (`~/c8/ego_cart20_v2_integrity_report.json`):
  - NaN 0, Inf 0, AUX12 nonzero 0
  - gripper ∈ [0, 1]
  - 1.95M 회전 중 invalid 0 (orthogonality·det 오차 1e-15)
  - task-start anchor 오차 5.6e-16
  - row dt = 66.67 ms 고정
- **Horizon 크기:** 이동량이 k에 따라 단조 증가한다. sequential-delta 버그가 없다는 sanity check다.

  | k | translation p50 / p95 | rotation p50 / p95 |
  |---|---|---|
  | 1 | 0.9 mm / 1.5 cm | 0.3° / 4.4° |
  | 4 | 2.4 mm / 5.4 cm | 0.9° / 15° |
  | 8 | 4.0 mm / 9.6 cm | 1.5° / 29° |
  | 16 | 7.4 mm / 14.6 cm | 2.9° / 52° |

- **기존 frozen gcal1과의 parity** (같은 시각에서 비교):
  - action pos p50 0.3–0.5 mm, p95 ~6 mm, max 12 mm. 원인은 gcal1이 오른손을 왼손의 가장 가까운 frame(≤ 20 ms)에 짝지은 것이다. 새 파이프라인은 각자 시계에서 보간한다.
  - rot p50 0.1°, gripper ≤ 0.017. state pos p50 0.2 mm.

### 4.1 참고 사항
- 왼쪽에 곱하는 canonical 변환 W는 라벨에서 정확히 상쇄된다: `inv(W T_t)(W T_s) = inv(T_t) T_s`. 그래서 "어떤 canonical frame을 쓰느냐"는 action/state에 영향이 없다. 영향을 주는 것은 오른쪽에 곱하는 **tool frame** X뿐이다.
- HandUMI 좌·우 손목은 각자 다른 SLAM map이다. **좌·우 공유 world frame이 없다.** 그래서 팔별 상대 표현만 가능하다. 현재 contract는 팔별 상대값이라 문제없다.
- **stage_id는 전부 −1 (unknown).**
  - 사람 gripper는 쉴 때 닫혀 있다(≈ 0). 접근할 때 ~0.7까지 열고, 큐브를 잡아도 ~0.6까지만 닫힌다. grasp와 release의 차이가 5–8 mm라서 gripper만으로 stage를 판정하면 잡음이다.
  - stage 가중 sampler(stage 3 ×2)는 구현돼 있지만 annotation 파일이 있어야 동작한다. 실험 D는 annotation이 생기면 진행한다.

## 5. 학습 (recipe: `~/umi_bridge/RELCART20_RELONLY_D6_RECIPE.md`)

- **REL-only D6 recipe를 그대로 쓴다.**
  - base `xvla_base_cart20v3_d6` (domain_id **6**, R312c FT와 같은 slot)
  - plugin, entry, preflight, factory sha 모두 동일
  - seed 1000, bs 4, fp32, lr 1e-4, warmup 1000, cosine
  - 손실 `MSE(pred[..., :20])`. 20:32는 입력·출력 모두 hard-zero
- **Schedule (사용자 지시): STEPS = DECAY = 100,000.** save 10k. LeRobot의 LR auto-scale 함정을 피하려고 STEPS와 DECAY를 같게 둔다. 51.6k row × bs 4 기준 약 7.8 epoch이다.
- **LeRobot export** (`ego_cart20_v2_train/val`):
  - state = RELCART20, action = [16,32] (20:32 = 0)
  - `aux.q_t` = **NaN placeholder**. preflight는 32-dim action일 때 이 열이 있는지만 본다. REL-only plugin은 쓰지 않는다. dq/FK plugin을 잘못 붙이면 즉시 NaN이 나서 바로 드러난다.
  - stats: AUX12 채널은 mean 0 / std 1. FT 때는 lerobot_train이 R312c stats로 덮어쓴다.
- **런처** `train_ego_cart20v2_relonly_d6.sh EGO <gpu>`는 D6 런처(sha 6ecdd805)를 복사한 것이다. 바뀐 것은 다음뿐이다.
  - 데이터셋
  - run 이름 (`EGO-CART20V2-*`)
  - schedule gate (STEPS == DECAY)
  - 로봇 gripper gate 대신 ego EXPORT.json contract gate와 SHA256SUMS 검사
- **장비:** 4090 GPU1. GPU0은 R384 run 몫이다. /home 여유가 85 G라서 아카이버(`trackb_archive_v2_relonly.sh`)를 반드시 돌린다.
- **진행 상태:** LeRobot export 진행 중 → node 복사 + SHA256SUMS → 런처 preflight → `ray job submit` (node 100.64.0.2 고정, GPU1). 시작할 때 domain 6, Auto-scaling 없음, preflight PASS를 확인한다.

## 6. GPT와 상의하면 좋을 열린 질문

1. **domain_id:** ego pretrain을 FT와 같은 6으로 할지, 사람 데이터용 별도 slot으로 할지. 현재는 6이다(FT 연속성 우선).
2. **사람 vs 로봇 gripper 분포:** 사람은 쉴 때 닫힘(0), 잡을 때 ~0.6이다. 로봇은 g 평균 0.2, g ≥ 0.6 비율 14–20%다. normalized 의미는 같지만 grasp 시점의 값 분포가 다르다. FT에서 다시 배워야 할 가능성이 있다.
3. **100k 체크포인트 평가 항목:** 이전 pre-registered 목록이 있다.
   - train/val REL loss
   - moving k4/k8/k16 오차, direction cosine
   - 크기 분포
   - gripper loss는 따로 보고
   - prompt sensitivity, representation collapse
   - 짧은 R312c FT를 scratch와 비교
4. **stage annotation:** 3번째 큐브 placement 가중(실험 D)에 필요하다. 사람 데이터 annotation을 어떻게 만들지(물체 이동 기반 HumanIK grip detector 재사용 등).
5. **test split:** 지금은 없다. 순서 일반화 같은 평가를 하려면 새 holdout이 필요한지.
6. **유효 row가 ~50%:** tracking lost와 sync 때문이다. 보간 허용 공백(50 ms)이나 카메라 허용 오차(20 ms)를 완화할지. 지금은 보수적으로 둔다.

## 7. 파일

| 무엇 | 경로 |
|---|---|
| 코드 | `~/ego_cart20/ego_cart20/` (config, geometry, preprocessing, labels, io, dataset, validation, scripts, sources/handumi_export.py) |
| 테스트 | `~/ego_cart20/tests/test_contract.py` |
| raw episodes | `~/c8/ego_cart20_v2_raw/` (330 + export log) |
| processed | `~/c8/ego_cart20_v2/` (episodes/, manifests, metadata.json) |
| integrity report | `~/c8/ego_cart20_v2_integrity_report.json` |
| 3D 샘플 시각화 | `~/c8/ego_cart20_v2_report/trajectory_samples.png` |
| LeRobot | `~/c8/ego_cart20_v2_lerobot/ego_cart20_v2_{train,val}` |
| 런처 | `~/ego_cart20/launch/train_ego_cart20v2_relonly_d6.sh` |
