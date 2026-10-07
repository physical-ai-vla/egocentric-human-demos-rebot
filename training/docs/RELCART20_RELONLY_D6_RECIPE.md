# reBot X-VLA RELCART20 REL-only (domain 6) 학습 레시피

작성일 2026-10-01. 지금 4090에서 돌리는 **REL-only, domain 6, 600k** 학습을 다른 에이전트가 같은 코드로 재현하거나 pretrain에 쓸 수 있게 정리했다.
모든 값은 실제 run의 `train_config.json`, 런처, 시작 로그에서 직접 확인했다. 이 문서의 run은 두 개다.

| run | 데이터 | 상태 |
|---|---|---|
| `R312C-RELCART20-RELONLY-D6-S600K` | r312c_relcart20_rel16_v4 (312 ep) | 30k에서 멈춤 (사용자). 030000은 training_state까지 SSD에 보관 |
| `R384-RELCART20-RELONLY-D6-S600K` | r384_relcart20_rel16_v4 (384 ep) | 4090 GPU0에서 시작 (2026-10-01, 빌드 완료 후 자동 제출) |

---

## 1. 요약

| 항목 | 값 |
|---|---|
| 모델 | X-VLA 0.9B (Florence-2). 초기값 `lerobot/xvla-base` (snapshot `cdb7964e4fe842935d671bfab5a5ebe00a96648c`)를 넓힌 `xvla_base_cart20v3_d6` |
| domain_id | **6** (robotwin2 = AgileX 양팔 슬롯). 베이스의 `policy_preprocessor.json`에 박혀 있다 |
| 입력 | 카메라 3대 (global/head, left_wrist, right_wrist) 224×224 + state 20 (RELCART20) + 언어 지시문 |
| 출력 | action chunk **16 스텝 × 32차원** (REL16 20 + Δq 12). **REL-only라 20:32는 항상 0** |
| 손실 | `MSE(pred[..., :20], target[..., :20])` 하나 (Δq 손실, FK 손실 없음) |
| 정밀도 | **fp32 가중치**, AMP 없음 |
| 배치 / seed | 4 / 1000, num_workers 4 |
| 학습 길이 | **steps 600000, cosine decay 600000** (둘이 같아야 한다, 6장 참고) |
| 저장 | 10000 스텝마다 |
| 장비 | RTX 4090 24 GB 한 장, 약 0.30 s/step (600k ≈ 50시간) |

---

## 2. 코드와 환경 (4090 `100.64.0.2`, 사용자 `bh-aiteam`)

| 항목 | 경로 | sha256 |
|---|---|---|
| python | `/home/bh-aiteam/miniforge3/envs/holobrain/bin/python` (torch 2.6) | |
| LeRobot | `/home/bh-aiteam/lerobot-seeed/src` (0.4.4 fork) | `lerobot/datasets/factory.py` = `3998f9b70422548bc3ea7a622d027be91e97a1d403f2503219091bdf763bb3a9` (REL16 chunk 패치) |
| 학습 엔트리 | `/home/bh-aiteam/umi_bridge/umi76/train_rel16_relonly.py` | `1f0b486133b4590a7685be667fd0c87d5c6459c323b450e4adc5a14d4a41df35` |
| REL-only 플러그인 | `/home/bh-aiteam/umi_bridge/umi76/rel16_aux_relonly.py` | `3511f81aec809103ddc97c7440c3f111c08c90d3d7083b577fc9f5fc004a4bf3` |
| preflight | `/home/bh-aiteam/umi_bridge/umi76/preflight_rel16v2.py` | `4c20d0f3819653556ff0611b601b0b4891cd1214078e4883a5c224bc07743b89` |
| 런처 (R312c, twin F) | `/home/bh-aiteam/train_relcart20_v4_relonly_d6.sh` | `6ecdd805a218c752584c98e24be4279d8d12e922a644d20d03d447066e2f766f` |
| 런처 (R384, twin G) | `/home/bh-aiteam/train_relcart20_r384_relonly_d6.sh` | `cb928a01c7a9d8c44381004064364afb29ec6d345c3c353bd25739f425381a56` |

런처는 factory/preflight/플러그인/엔트리 sha가 다르면 **시작을 거부**한다. 코드를 고치려면 새 파일로 복사하고 런처의 sha도 함께 바꾼다. 같은 레시피라고 부르려면 위 sha가 같아야 한다.

5090(sm_120)에서는 torch 2.6이 안 돈다. 5090용 venv는 `/srv/data/johann/relonly/venv` (torch 2.7.1+cu128, 나머지는 4090 freeze와 같은 버전)이고, 코드 복사본은 `/srv/data/johann/relonly/code/`다(플러그인·엔트리 sha 동일). GPU/torch가 다르면 완전히 같은 twin은 아니다.

---

## 3. 베이스 모델 계보

```
lerobot/xvla-base (snapshot cdb7964e..., action 20, proprio 20)
  └ widen_v3.py      → /home/bh-aiteam/xvla_base_rel16v3   action 20→32 (20:32 새 N(0,sd)), proprio 20→76 (새 N(0,sd)), decoder 20→32
                        INPUT-major로 넓힘 (DomainAwareLinear: Embedding(nd, in*out).view(nd, in, out)). 기능 게이트: 원래 encoder/decoder와 일치
  └ widen_cart20.py  → /home/bh-aiteam/xvla_base_cart20v3  proprio 76→20 (rows 0:20 유지), 나머지 비트 동일. model md5 5dfd99f2611d503bba52dc57c5412925
  └ (복사)           → /home/bh-aiteam/xvla_base_cart20v3_d6  모든 파일 hard link, policy_preprocessor.json 의 domain_id 만 0 → 6
```

- domain_id는 CLI 플래그가 없다. `lerobot_train`이 `--policy.path`의 `policy_preprocessor.json`에서 processor를 그대로 불러오기 때문에 **베이스 json을 바꾸는 것이 유일한 방법**이다. 저장된 체크포인트에도 6이 들어가므로 서빙 쪽은 자동으로 맞는다.
- 확인 방법: 런처가 시작할 때 `[domain] ... processor domain_id 6`을 찍는다. 다르면 시작을 거부한다(`EXPECT_DOMAIN`).
- 예전 run(v3, v4, 5090 REL-only)은 모두 domain_id 0(Bridge 슬롯)이다.

---

## 4. 데이터

### 4.1 정의 (`derive_relcart20.py`)
- **state (20)** = `[L_rel pos3 (m) | L_rel rot6d | R_rel pos3 | R_rel rot6d | g_L | g_R]`
  - `T_rel(t) = inv(T_anchor) @ T(t)`, 팔별. `T = rebot_fk_torch.tcp(aux.q_t)` (로봇 base frame, 데이터셋 TCP와 같음)
  - **anchor = 에피소드의 첫 행** (frame_index 0)
  - rot6d = 회전 행렬의 첫 두 **행** (umi pose_util). identity = [1,0,0,0,1,0]
  - g = follower 집게 열림 정도 = width / W_OPEN, W_OPEN = 270·0.05/118 m, [0,1] 클립. **0 = 닫힘, 1 = 열림**
- **action (16, 32)** = REL16 20 + Δq 12, 16 스텝 = `t + (k+1)·50.05 ms` (k = 0..15)
  - 0–19: 팔별 `inv(T_t) @ T_{t+k}`의 pos3 + rot6d + 연속 그리퍼 (`g = clip(leader cmd / 45, 0, 1)`), **측정된 follower TCP** 기준
  - 20–31: Δq = `q(t+(k+1)dt) − q(t)` (follower frame, rad). REL-only에서는 학습·추론 모두 0으로 고정
- 이미지: `observation.images.global / left_wrist / right_wrist` → 모델 이름 `image / image2 / image3` (`--rename_map`)
- 에피소드 끝: 16개 미래 목표가 모두 에피소드 안에 있는 행만 남긴다(패딩 없음). X-VLA는 `action_is_pad`를 안 읽는다.

### 4.2 데이터셋 구성
| 데이터셋 | 구성 |
|---|---|
| r312c_relcart20_rel16_v4 | HEAD180 (R150 headview 150 + R30 day4 30) + FRONT132 (예전 R675, pan=center, QA A, 순서당 22, **rot180**) = 312 ep, 174,012 frames |
| r384_relcart20_rel16_v4 | R312c + FRONT far 21 + phase1 51 (QA A; 제외 src_ep 81 30 34 292 525 541 556) = 384 ep |

선택 목록: `~/umi_bridge/umi76/r312c_front132_v1.json` (sha `b961cd0e…`), `r384_front204_v1.json` (sha `22306b5a…`, r675rbp zarr 번호).

### 4.3 빌드 명령 (R384, `~/build_chain_r384.sh`, sha `0a6a6427…`)
```bash
export PYTHONPATH=/home/bh-aiteam/lerobot-seeed/src:/home/bh-aiteam/universal_manipulation_interface UMI_ROOT=/home/bh-aiteam/universal_manipulation_interface
cd ~/umi_bridge/umi76
# 1) 변환 (HEAD 그대로, FRONT rot180)
umi76_to_lerobot.py \
  --source zarr=$B/r150_umi.zarr,lerobot=$D/rebot_3stack_R150_headview \
  --source zarr=$B/r30_umi.zarr,lerobot=$D/rebot_3stack_R30_day4_headview \
  --source zarr=$B/r675rbp_umi.zarr,lerobot=$D/rebot_3stack_center675_s96,global_transform=rot180,episodes_json=$B/r675_rbp.json,select=$B/umi76/r384_front204_v1.json \
  --out $D/r384_umi76_rel16_v1 --repo-id rebot/r384_umi76_rel16_v1 --action-mode umi --state-mode slim20 --gripper-target next
# 2) 연속 그리퍼 (variant C), 게이트
run_derive_v2_ro_parent.py --parent $D/r384_umi76_rel16_v1 --out $D/r384_umi76_rel16_v3c --variant C
GRIP=continuous gate_v2.py $D/r384_umi76_rel16_v1 $D/r384_umi76_rel16_v3c
# 3) Δq12 + aux.q_t (FK 일관성 게이트 내장: < 1 mm)
derive_v3d.py --parent $D/r384_umi76_rel16_v3c --out $D/r384_umi76_rel16_v3d
# 4) state 76 → RELCART20 (anchor 게이트 내장)
derive_relcart20.py --parent $D/r384_umi76_rel16_v3d --out $D/r384_relcart20_rel16_v4
```
(`B=/home/bh-aiteam/umi_bridge`, `D=/home/bh-aiteam/holobrain-data/lerobot`. 스크립트 sha: umi76_to_lerobot `85c7d496…`, derive_v2 `51a3edf9…`, run_derive_v2_ro_parent `c15b1276…`, gate_v2 `4b616782…`, derive_v3d `933a2d03…`, derive_relcart20 `36ecf859…`.)
R312c는 `~/build_r312c_umi76.sh` + `~/chain_r312c_relcart20_v4.sh`로 같은 단계를 밟았다.

---

## 5. 손실 (`rel16_aux_relonly.py`)
- `rel_loss = MSE(pred[..., :20], target[..., :20])` (그리퍼 9/19 포함). Δq 손실, FK 손실, aux.q_t 사용 **없음**.
- **20:32 채널 hard-zero**, 학습과 추론의 모든 denoising 단계에서:
  - 입력 `action_with_noise[..., 20:32] = 0` → 정답 Δq가 네트워크에 새지 않는다
  - 출력 `pred[..., 20:32] = 0` → decoder 20:32 행은 task gradient 0
- X-VLA flow-matching: 정답에 잡음을 섞어(`noisy = noise·t + action·(1−t)`) **깨끗한 action을 직접 예측**하고 위 MSE를 계산한다. 추론은 10 denoising step.
- `RELONLY_PREFLIGHT=3`이면 첫 3 배치에서 자가 점검을 찍는다. 시작 로그에 다음이 모두 PASS여야 한다.
  `PASS loss invariant to GT dq randomisation | PASS pred 20:32 == 0 | input 20:32 non-zero 0 | decoder grad rows 20:32 max 0`
- v3/v4 (REL-only 아님)는 `rel16_aux.py` + `train_rel16aux.py`: rel + λq·dq(λq=1.0) + λfk·FK(λfk=20).

---

## 6. 최적화기와 스케줄 (체크포인트 `train_config.json` 기준)

| 항목 | 값 |
|---|---|
| optimizer | `xvla-adamw`: lr 1e-4, betas (0.9, 0.95), eps 1e-8, weight decay 1e-4, grad clip 10. VLM 그룹은 LR ×0.1 (XVLAAdamW 기본) |
| soft prompt | lr scale 1.0, warmup lr scale 없음 |
| scheduler | `cosine_decay_with_warmup`: warmup 1000 → peak 1e-4 → **600000에서 2.5e-6** |
| 정규화 | STATE IDENTITY, **ACTION MEAN_STD**, VISUAL IDENTITY |
| 기타 | chunk 16, n_action_steps 16, max_state_dim 20, max_action_dim 32, action_mode auto, use_proprio, num_image_views 3, resize 224 패딩, tokenizer bart-large max_length 50, 이미지 증강 꺼짐, VLM·언어 인코더 학습 |

> ⚠️ **LeRobot은 `--steps`가 `scheduler_decay_steps`보다 작으면 decay를 자동으로 줄인다** (`Auto-scaling LR scheduler: ... decay: 600000 → 300000`).
> 2026-10-01에 STEPS=300000/DECAY=600000으로 돌린 REL-only 두 run(5090 R312C-RELCART20-RELONLY-V4, 4090 첫 D6)이 이 때문에 실제로는 decay 300k였다. 지금 run은 **STEPS = DECAY = 600000**이고 시작 로그에 Auto-scaling 메시지가 없는 것을 확인했다.
> 짧게 멈추고 싶으면 STEPS는 600000으로 두고 바깥에서 멈춘다.

---

## 7. 학습 명령

GPU 작업은 반드시 `ray job submit`으로 노드를 고정해서 제출한다 (ssh + nohup 금지). 4090 GPU0은 다른 팀(tobey)이 Ray 밖에서 쓰기도 하니 `nvidia-smi`로 먼저 비었는지 본다. 런처는 1 GB 넘게 쓰인 GPU에서는 시작을 거부한다.

```bash
export RAY_ADDRESS=http://100.64.0.1:8265
ray job submit --no-wait --submission-id r384-relonly-d6-s600k-<HHMM> \
  --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  -- bash -c 'STEPS=600000 DECAY_STEPS=600000 SAVE_FREQ=10000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_relcart20_r384_relonly_d6.sh G 0'
```
- 마지막 인자는 **물리 GPU 번호**다 (Ray의 CUDA_VISIBLE_DEVICES는 믿지 않는다).
- 런처가 하는 일: gripper contract 게이트 (`~/umi_bridge/gripper_contract_v2.json` passed) → steps/decay/save 게이트 → sha 게이트 → domain 게이트 → CPU preflight → 디스크 확인 (MIN_START_GB) → 디스크 가드 스레드 (GUARD_GB 50 아래로 떨어지면 학습을 멈춘다) → 학습.
- 실제 학습 인자 (런처 안):
```
--dataset.repo_id=rebot/<DS> --dataset.root=<DATA> --policy.path=/home/bh-aiteam/xvla_base_cart20v3_d6 --policy.device=cuda
--policy.push_to_hub=false --policy.dtype=float32 --policy.chunk_size=16 --policy.n_action_steps=16 --policy.max_state_dim=20
--policy.max_action_dim=32 --policy.action_mode=auto --policy.use_proprio=true --policy.freeze_vision_encoder=false
--policy.freeze_language_encoder=false --policy.train_policy_transformer=true --policy.train_soft_prompts=true
--policy.scheduler_decay_steps=600000 --rename_map='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'
--seed=1000 --steps=600000 --batch_size=4 --num_workers=4 --eval_freq=0 --save_freq=10000 --wandb.enable=false --output_dir=/home/bh-aiteam/holobrain-data/trainB/<RUN>
env: CUDA_VISIBLE_DEVICES=<gpu> XVLA_STRICT_STATE_DIM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
- 이어 학습: `RESUME=1 ... bash <launcher> G 0` (`checkpoints/last`의 train_config + optimizer state 필요, 저장된 config가 600000/600000/seed 1000인지 검사).
- 출력: `/home/bh-aiteam/holobrain-data/trainB/<RUN>/checkpoints/<step>/{pretrained_model,training_state}`, 로그 `trainB/<RUN>.log`.

### 7.1 시작할 때 확인할 것
1. `[domain] /home/bh-aiteam/xvla_base_cart20v3_d6 processor domain_id 6`
2. `cfg.steps=600000 (600K)` 이고 **Auto-scaling 메시지 없음**
3. `[relonly-preflight] batch 1..3: PASS ...` 전부
4. 첫 step 로그: `step:200 ... loss ~0.9 ... lr:1.0e-06` (warmup 1000이라 200 스텝에서 1e-6, VLM 그룹 기준)
5. 로그는 `\r`로 덮어쓴다. `tr '\r' '\n' < trainB/<RUN>.log | grep 'step:'`로 본다.

### 7.2 디스크와 아카이브
- 4090 /home은 여유가 70~85 GB뿐이다. 체크포인트 하나가 weights 3.5 GB + optimizer state 6.6 GB.
- Mac에서 `~/umi_bridge/trackb_archive_v2_relonly.sh <RUN>`을 돌린다 (sha256 확인 후 SSD `/Volumes/PortableSSD/rebot_ckpts_archive/trackb_<RUN>/`로 옮기고 노드에서 지움, training_state는 25k 배수만). 아카이버가 안 돌면 디스크 가드가 학습을 멈춘다.
- 학습을 끝낼 때는 마지막 체크포인트를 `STATE_STEPS=<step> ONESHOT=1`로 아카이브해야 SSD에서 재개할 수 있다.

---

## 8. 추론 / 서빙 (Mac)

- 로더: `~/umi_bridge/load_relonly_d6_ckpt_to_ui.sh <step> <port>` (R312C D6, 4090에서 받아옴). R384용은 같은 방식으로 RUN 이름과 identity 검사만 바꾸면 된다.
- 필수: **`V4_RELONLY=1`** (20:32 zero mask를 추론 쪽에도 설치), IK는 pink 또는 diffik (pinkdq/policydq는 Δq 헤드가 학습되지 않아 거부), domain_id 6은 체크포인트 processor에서 자동.
- state: `V4_STATE_MODE=relcart20` (anchor = `/run` 후 첫 관측, 새 에피소드마다 RESET ANCHOR), action `V4_ACTION_MODE=umi` (목표 = 관측 자세 @ A_k, base frame 절대 목표).
- 2026-10-01 현재 UI 설정 (:8044/45/46): 스트리밍 sync (`V4_STREAM=1 V4_STREAM_SYNC=1 V4_STREAM_ROWS=2 V4_STREAM_SPEED=0.4`), fp16, Pink + `V4_PINK_LOCK=joint5`, 연속 그리퍼, `V4_JAW_TORQUE=0`, 로봇 `V4_ROBOT=http://localhost:8021` (MIT 브리지 `~/robot-cockpit/robot_service_mit.py`, integral law, 100 Hz, 45°/s, 180°/s², 그리퍼 90°/s).

---

## 9. 이 코드로 pretrain 하려면 (예: ego 데이터)

같은 레시피를 쓰려면 **데이터가 4장과 같은 contract**여야 한다.
1. LeRobot v3 데이터셋, 15 Hz 저장, 3 카메라 키 (`observation.images.global/left_wrist/right_wrist`), 224 패딩 리사이즈 전 원본.
2. `observation.state` (20) = RELCART20 (anchor = 에피소드 첫 행), `action` (16, 32) = REL16 20 + Δq 12. Δq가 없는 데이터(사람 데이터 등)는 20:32을 0으로 두면 되고, **REL-only라 학습에 영향이 없다** (입력·출력 모두 0으로 고정).
3. `meta/stats.json`의 action 통계가 실제 값과 맞아야 한다 (ACTION MEAN_STD 정규화).
4. domain_id: 로봇 fine-tune과 같은 6으로 할지, 사람 데이터용으로 다른 슬롯을 쓸지 정한다. 다른 슬롯을 쓰려면 베이스를 복사해 `policy_preprocessor.json`의 domain_id만 바꾸고 런처의 `EXPECT_DOMAIN`을 맞춘다.
5. 런처를 복사해 새 twin (DS, BASE)을 추가하고, run 이름 거부 목록을 확인한 뒤, **STEPS = DECAY**로 제출한다.
6. pretrain 후 로봇 fine-tune은 그 체크포인트를 `--policy.path`로 두고 같은 런처 구조로 돌린다 (processor의 domain_id가 그대로 따라간다).

---

## 10. 알려진 함정

- **LR auto-scale** (6장). steps < decay이면 decay가 몰래 줄어든다.
- **zsh**: `set -- $x`가 문자열을 나누지 않고, `echo ====`가 명령 오류가 된다. 반복 작업은 bash 스크립트로 쓴다. ssh 안의 `while read` 루프는 `< /dev/null`.
- **이미 있는 output_dir**이면 LeRobot이 에러를 낸다. run 이름을 새로 정한다.
- **training_state 보관**: 아카이버는 25k 배수만 state를 옮긴다. 재개할 지점은 `STATE_STEPS`로 따로 지정한다.
- **5090은 torch 2.6 불가**, 전용 venv 사용. GPU/torch가 다르면 결과를 1:1로 비교하지 않는다.
- REL-only 체크포인트를 `V4_RELONLY=1` 없이 서빙하면 20:32에 학습 안 된 값이 다음 denoising step으로 들어간다.
