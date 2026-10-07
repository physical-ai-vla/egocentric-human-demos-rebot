# reBot X-VLA 미세조정 레시피 (사전학습 없음 / scratch = B1-old)

작성일 2026-09-29. `lerobot/xvla-base`에서 **사람 데이터 사전학습 없이 바로** reBot B601 실제 텔레옵 데이터(R150 또는 그 부분집합)로
미세조정하는 방법이다. 실제 로봇에서 큐브 3개 쌓기가 동작한 B1-old 레시피이며, C-old 실험의 비교 기준(scratch)도 이 설정을 그대로 쓴다.
아래 수치는 모두 실행 중인 학습의 시작 로그와 코드에서 직접 확인했다.

참고 문서: `~/humanik_retarget/R150_SPEC.md` (서빙 설정 원본), `~/umi_bridge/track_c/TRACK_C_PSEUDO_JOINT_CONTRACT.md` (결정 기록).

---

## 1. 요약

| 항목 | 값 |
|---|---|
| 모델 | X-VLA 0.9B (Florence-2 기반). 초기 가중치 **`lerobot/xvla-base`** (HF snapshot `cdb7964e4fe842935d671bfab5a5ebe00a96648c`, `model.safetensors` md5 `0bed971480d94ddfe002560bde13a59a`) |
| 데이터 | `rebot/rebot_3stack_R150_headview` (LeRobot v3 형식, 150 에피소드, 173,818 프레임, 30 fps, 쌓는 순서 6가지 × 25개) 또는 그 부분집합 |
| 학습 방식 | fp32 전체 미세조정(비전·언어 인코더 포함), 배치 4, seed 1000 |
| 학습 길이 | `--steps=600000`, cosine 감쇠 400k. 실제로는 **250k(비교 기준점) / 300k(종료)**에서 밖에서 멈춘다 |
| 저장 주기 | 10k마다 (디스크가 부족하면 50k) |
| 장비 | RTX 4090 24 GB 한 장: 메모리 최대 약 21.5 GB, 초당 약 3.45 스텝(샘플 13.8개) → 250k까지 약 20시간 |

---

## 2. 모델 입력과 출력

### 입력
- **카메라 3대**. 데이터셋 이름 → 모델 입력 이름(`--rename_map`):
  - `observation.images.global`(머리 위치 C922) → `image`
  - `observation.images.left_wrist` → `image2`
  - `observation.images.right_wrist` → `image3`
  - 각각 224×224로 비율을 유지해 패딩 후 줄인다.
- **상태 14차원**. 데이터셋에는 [왼팔 관절 6개(도), 왼쪽 그리퍼 원시값, 오른팔 관절 6개(도), 오른쪽 그리퍼 원시값]으로 들어 있다.
  - 모델에 넣기 직전에 `humanik_delta.py`가 관절은 **라디안**으로 바꾸고, 그리퍼는 **0/1**(원시값 ≤ −135이면 1)로 바꾼다.
- **언어 지시문**. 로봇용 문장 형식이며 쌓는 순서마다 하나씩, 모두 6가지다.
- 정규화는 상태·행동·영상 모두 **IDENTITY**(정규화 안 함)다. 행동 정규화를 켜면 코드가 즉시 에러를 내고 멈추도록 되어 있다.

### 출력(행동)
- **LEAD 5, chunk 30**: 현재 시점 t 기준 t+5 … t+34, 30개 시점을 한 번에 예측한다.
- 모델 출력은 **20차원**이다 (`max_action_dim=20`). 실제 행동 14차원과 보조 6차원으로 나뉜다.
  - **0–13번(실제 행동)**:
    - 팔 12개 = 리더 팔 **명령값**(q_cmd)과 현재 관절값의 **누적 차이 Δq**(라디안). 설정은 `HUMANIK_TARGET=cmd`다.
    - 그리퍼 2개 = 명령값을 0/1로 바꾼 것(명령 원시값 ≥ 27이면 1).
  - **14–19번(보조, C)**: 아래 3장에서 설명하는 TCP 상대 이동량이다. 추론 때는 쓰지 않고 버린다.
- **그리퍼 의미**: follower 원시값은 **0 = 닫힘, −270 = 열림**이다. 따라서 label 1은 **열림**이다.
  - 코드 주석의 "1 = closed"는 틀린 표기다. R150 손목 카메라 영상으로 확인했다.
  - 어느 쪽 이름을 쓰든 학습과 추론이 같은 변환을 쓰므로 동작에는 문제가 없다. 데이터를 새로 만들 때만 주의하면 된다.

---

## 3. 손실 함수 (주 손실 + C 보조 + D FK 일관성)

X-VLA는 flow-matching 방식이다.
- **학습**: 정답 행동에 잡음을 섞는다(`noisy = 잡음·t + 정답·(1−t)`, t는 무작위). 모델은 이것에서 **깨끗한 행동을 직접 예측**하고, 아래 손실을 계산한다.
- **추론**: 잡음에서 시작해 10단계로 복원한다.

전체 손실은 다음과 같다.

```
L = L_main(팔 + 그리퍼)  +  L_C (보조 TCP, 가중치 2.0)  +  L_D (FK 일관성, 가중치 20)
```

### 3.1 주 손실 L_main — 팔 Δq와 그리퍼
- 0–13번 차원의 **평균제곱오차(MSE)**다. 그리퍼도 0/1 값에 대한 MSE다(BCE가 아니다). 그래서 추론 때 sigmoid가 필요 없다.
- 유효 마스크를 곱한 뒤, 마스크 합으로 나눈다. 로그에는 `joints_loss`와 `gripper_loss`로 나뉘어 찍히고, 둘의 합이 주 손실이다.
- 유효 마스크:
  - 에피소드 끝을 넘어가는 chunk는 가린다.
  - 표본을 뽑는 단계(sampler)에서도 **끝에서 34프레임 안쪽의 시작점은 아예 뽑지 않는다**. 그래서 R150 데이터에서는 마스크가 사실상 모두 1이다.
  - 예: R30 = 37,512 프레임 중 유효 시작점 36,492개.

### 3.2 C 보조 손실 — "손끝(TCP)이 얼마나 움직일지" 같이 예측
- 켜는 설정: `XVLA_EE_AUX=1`, `EE_AUX_SOURCE=state`, `EE_AUX_SCALE=10`, `EE_AUX_LAMBDA=2.0`.
- **정답 만들기**:
  1. 현재 follower 관절값 q_t와 미래 **실제 follower 관절값** q_{t+5+i}(명령값이 아님)를 각각 정기구학(FK, `rebot_fk_torch.py`)에 넣어 TCP 자세 행렬 T를 구한다.
  2. **현재 TCP 좌표계 기준 상대 이동** `inv(T_t) · T_{t+5+i}`의 위치(xyz)만 뽑는다.
  3. ×10을 곱한다(1.0 = 10 cm). 팔 Δq(라디안)와 크기 수준을 맞추기 위한 것이다.
  4. 왼팔 3개 + 오른팔 3개 = 6개 값을 **출력 14–19번 차원**의 정답으로 붙인다.
- 손실 = 14–19번 차원 MSE × 2.0. 팔 유효 마스크와 미래 상태 패딩 마스크를 적용한다.
- 의미: 모델이 관절 공간 행동과 함께 **작업 공간에서 손끝이 어디로 갈지**를 명시적으로 배우게 하는 보조 과제다.
- 참고값: 학습 시작 로그 기준(R30) 상대 이동 크기의 중앙값 36 mm, 95% 값 82 mm.

### 3.3 D FK 일관성 손실 — "예측한 관절값으로 가면 손끝이 정답 위치에 오는가"
- 켜는 설정: `EE_FK_LAMBDA=20`.
- 계산 방법:
  1. 예측한 절대 관절값 `q_t + Δq̂`를 FK에 넣어 TCP 위치(로봇 base 좌표계, m)를 구한다.
  2. 정답 절대 관절값 `q_t + Δq` = **명령값 q_cmd**의 FK 위치와의 **제곱거리(m²)**를 구한다.
  3. 팔 유효 마스크로 평균 낸 뒤 20을 곱한다. 예를 들어 1 cm 오차면 1e-4 × 20 = 2e-3이다.
- 미분 가능한 FK를 통해 **팔 Δq 출력(0–13번)에 직접 기울기**가 들어간다. 관절 오차 중에서 손끝 위치를 많이 틀리게 하는 오차에 더 큰 벌점을 주는 효과다.
- 로그의 `FK pos err (pred vs cmd) L p50 … R p50 … mm`가 이 D 오차다(청크 전체, mm 단위).
- 기하 계산(C·D)은 항상 fp32로 한다(autocast를 끔).

### 3.4 C·D 정답의 출처 — `EEF_TARGET_SOURCE=legacy`
- 이 레시피에서 C의 정답은 **실제 follower 상태의 FK**이고, D의 정답은 **명령값의 FK**다.
- `measured`는 사람 데이터 사전학습용(데이터의 `observation.ee.tcp_tgt`를 정답으로 씀)이다. **로봇 미세조정에서는 반드시 `legacy`.**
- 설정값을 바꾸면 정답의 정의가 바뀐다. 위 가중치와 출처는 실제 로봇에서 검증된 값이니 그대로 쓴다.

---

## 4. 최적화기와 학습률 (시작 로그에서 확인한 값)

| 항목 | 값 |
|---|---|
| 최적화기 | `xvla-adamw`: 최대 학습률 1e-4, betas (0.9, 0.95), eps 1e-8, weight decay 1e-4, gradient clip 10 |
| 파라미터 그룹 | VLM(Florence-2) 부분은 **학습률 × 0.1 = 1e-5**, weight decay × 0.1. soft prompt와 나머지(정책 transformer, 행동 입출력층)는 1e-4. 로그의 `lr:` 값은 VLM 그룹 기준이라 1e-5 근처로 보인다 |
| 스케줄 | `cosine_decay_with_warmup`: 1k 스텝 warmup → 최대값 → **400k에서 2.5e-6**까지 cosine 감소 |
| `XVLA_LR_GROUPS` | **설정하지 않음**. 설정하면 `lr_groups.py`가 그룹별 학습률을 덮어쓴다 |

> ⚠️ **LeRobot은 `--steps`가 감쇠 길이보다 작으면 감쇠 길이를 자동으로 줄인다**(`Auto-scaling LR scheduler` 로그).
> 300k만 돌릴 생각이어도 `--steps=600000 --policy.scheduler_decay_steps=400000`으로 두고 밖에서 멈춰야 학습률 곡선이 기준 실행과 같다.
> 데이터 크기와 상관없이 **같은 스텝 기준 스케줄**을 쓴다. 따라서 같은 스텝에서의 epoch 수는 데이터 크기마다 다르다(6장 표).

---

## 5. 코드

- 위치: `c8old/xvla/`. 4090은 `/home/bh-aiteam/c8old/xvla`, 5090은 `/srv/data/johann/c8old5090/xvla`.
- LeRobot은 `workspace/bh_rebot_LeRobot`의 가상환경을 쓴다. `train_bi.py`가 아래 패치들을 불러와 LeRobot의 X-VLA에 덧씌운다.
- **아래 8개 파일의 md5가 같아야 같은 레시피다.** 2026-09-26 이후 바뀌지 않았다.

```
70ea32ae91bc028d25d0a485d7802bb3  train_bi.py
6b63986b890ea815b3ea977e09205552  humanik_delta.py     ← 행동 변환, 마스크, C/D 손실. workspace/xvla/humanik_delta.py(db6d17c3…)와 다르다! 섞지 말 것
ad3de96c1fe89275a5accbad00205c82  eef_delta.py         (EEF_DELTA=1일 때만 쓰임, 이 레시피에서는 꺼져 있음)
2d49b7031a360bdfbab177002595d02d  contrastive_sampler.py (CONTRASTIVE=1일 때만, 꺼져 있음)
9a732c0bcd1c5e692e875fb63772758d  keep_indices.py
56c8f538e0adf25695e37ed1b90d5610  aug_defaults.py      (XVLA_AUG=1일 때만, 꺼져 있음)
0a50a52bcea34102122c0becf68d1a4a  lr_groups.py         (XVLA_LR_GROUPS 설정 시만, 꺼져 있음)
186c39c81138837f2c160b486ed3529f  rebot_fk_torch.py    ← C/D에 쓰는 미분 가능한 정기구학
```
확인 방법: `cd c8old/xvla && md5sum -c` (위 목록을 입력으로).

---

## 6. 학습 명령

```bash
export HF_HOME=/home/bh-aiteam/.cache/huggingface HF_HUB_OFFLINE=1
export HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd                         # 행동 = t+5부터 30개, 명령값 누적 Δq
export XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=200   # C·D 손실
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1           # 속도용
export EEF_TARGET_SOURCE=legacy CUDA_VISIBLE_DEVICES=<gpu 번호>

PY=/home/bh-aiteam/workspace/bh_rebot_LeRobot/.venv/bin/python
X=/home/bh-aiteam/c8old/xvla
R150=/home/bh-aiteam/holobrain-data/lerobot/rebot_3stack_R150_headview
EPS=$(cat r60_episodes.txt)        # 150개 전체를 쓰면 --dataset.episodes 인자를 빼면 된다

$PY $X/train_bi.py \
  --dataset.repo_id=rebot/rebot_3stack_R150_headview --dataset.root=$R150 "--dataset.episodes=$EPS" \
  --policy.path=lerobot/xvla-base --policy.dtype=float32 \
  --output_dir=runs/<이름> --job_name=<이름> \
  --steps=600000 --policy.scheduler_decay_steps=400000 --save_freq=10000 --log_freq=200 --seed=1000 \
  --batch_size=4 --num_workers=8 \
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --policy.normalization_mapping='{"STATE":"IDENTITY","ACTION":"IDENTITY","VISUAL":"IDENTITY"}' \
  --rename_map '{"observation.images.global":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}'
```

- `XVLA_TF32`(TF32 행렬곱)와 `XVLA_FUSED_ADAM`(fused AdamW)은 속도용이다. 기준 실행들이 모두 켜고 돌았으니 레시피의 일부로 보고 켜 둔다.
- 기존 실행 스크립트(위 명령 + 각종 검사): 4090 `/home/bh-aiteam/c8old/c8old_r{30,60,90,120}_scratch.sh <gpu 번호>`.
- **이어 학습하기**: `c8old_r*_scratch_resume.sh`를 쓴다.
  - 내용은 `c8old/xvla/resume_bi_c8old.py --config_path=<체크포인트>/pretrained_model/train_config.json --resume=true`다.
  - workspace 쪽 `resume_bi.py`는 다른 `humanik_delta.py`를 불러오므로 **쓰면 안 된다**.
- **GPU 작업 제출**: `ray job submit`으로 노드를 고정해서 제출한다.
  - 설정은 `entrypoint_num_gpus=0`, `entrypoint_resources={"node:100.64.0.2":0.001}`이고, GPU 번호는 스크립트 인자로 직접 준다.
  - 4090 GPU0은 다른 팀이 Ray 밖에서 쓰는 경우가 있다. `nvidia-smi`로 실제로 비어 있는지 먼저 확인한다.

---

## 7. 데이터와 부분집합 사다리

- 부분집합은 데이터를 복사하지 않고 `--dataset.episodes=[...]`로 고른다. 표본 추출기는 부분집합에서도 맞게 동작하도록 고쳐져 있다.
- 고정 파일: `~/c8/r150_nested_subset_v1.json` (sha256 `8996a20fc88a8e4bb32dad4b1260556552c911159a06c128eb5eeaa4b52b5744`). 목록 파일은 `~/c8/r{30,60,90,120}_episodes.txt`.
- **수집 세트 번호**(쌓는 순서마다 수집 시각 순으로 매긴 번호)로 자른다. jsonl의 `set_id`는 수집 세션마다 0부터 다시 시작하므로 그대로 쓰면 안 된다.

| | 세트 | 에피소드 | 순서당 | 유효 시작점 | 1 epoch (배치 4) | 250k 때 epoch |
|---|---|---|---|---|---|---|
| R30 | 0–4 | 30개 (ep 90–119, 세션 20260907_144206) | 5 | 36,492 | 9,123 스텝 | 27.4 |
| R60 | 0–9 | 60 | 10 | 72,016 | 18,004 | 13.9 |
| R90 | 0–14 | 90 | 15 | 108,058 | 27,015 | 9.3 |
| R120 | 0–19 | 120 | 20 | 139,020 | 34,755 | 7.2 |
| R150 | 0–24 | 150 | 25 | 168,718 | 42,180 | 5.9 |

- R30 ⊂ R60 ⊂ R90 ⊂ R120 ⊂ R150으로 서로 포함 관계다.
- 목록 순서까지 포함한 sha256(`json.dumps(list)` 기준): R30 `ffc042de…`, R60 `f1dcd2e6…`, R90 `b547244e…`, R120 `b8341112…`.
  같은 목록을 **같은 순서**로 줘야 표본 추출 순서가 같아진다.

---

## 8. 학습 시작 때 확인할 것

1. 코드 8개 md5, 부분집합 파일 sha256, xvla-base md5가 위와 같은지 본다.
2. 출력 폴더가 새 폴더인지 본다. 이미 있으면 LeRobot이 에러를 낸다.
3. 시작 로그의 `ot_train.py:212 {…}`(전체 설정 출력)를 기준 실행과 비교한다. 도구는 `~/c8/c8old/cfg_parity.py <기준 로그> <새 로그> <허용 항목>`이다.
   - 기준 로그는 `ref_B1old_R150_startup.log`다.
   - 달라도 되는 항목은 `output_dir`, `job_name`, `dataset.episodes`, `steps`, `save_freq`, `log_freq`다.
   - 다른 컴퓨터라면 `dataset.root`와 `eval.batch_size`도 달라도 된다. `eval.batch_size`는 CPU 개수로 정해지는 시뮬레이션 환경 수이고 학습에는 쓰이지 않는다.
   - `train_config.json`은 **첫 체크포인트 때 처음 생긴다.** 그래서 시작 직후에는 로그로 비교한다.
4. `humanik_delta` 시작 로그가 다음과 같이 나오는지 본다.
   - `installed: LEAD=5 ROBOT=True EE_AUX=True … EEF_TARGET_SOURCE=legacy`
   - `aux targets ON (source=state) … scale 10.0, lambda aux 2.0, lambda fk 20.0`
   - `first batch: action (4, 30, 20) … grip uniq [0.0, 1.0]`
5. 학습 로그를 확인한다.
   - 형식은 `step:22K loss:… grdn:… lr:…`이고 `\r`로 한 줄에 덮어쓴다. `tr '\r' '\n'`로 바꾼 뒤 grep해야 보인다.
   - C·D 값은 200 배치마다 `FK pos err … | aux_loss … fk_loss …`로 나온다.
   - 정상 범위(R30, 28k 스텝): 전체 손실 약 0.025, gradient norm 약 3, FK 오차 중앙값 약 9 mm.

---

## 9. 오프라인 평가

- 명령: `~/c8/c8old_mac_eval.py r150 <pretrained_model> <결과.json>`. Mac MPS에서 돌며, 실제 로봇 서빙과 같은 추론 경로다.
- 평가 세트는 probe val-10(`~/c8/probe_r150_split.json`)이다.
  - ⚠️ R150으로 학습하면 10개 모두 학습 데이터 안에 있다. 이 경우 "학습 데이터에 얼마나 맞췄나"만 보는 것이다.
  - 부분집합으로 학습하면 일부가 학습에 안 쓰인 에피소드가 된다: R120/R90은 8개 학습됨 / 2개 미학습, R60은 7 / 3, R30은 4 / 6.
- 지표:
  - 예측 구간 k ∈ {1, 4, 8, 16, 30}별 손끝 위치 오차(FK, 중앙값/90% 값, mm)
  - 관절 평균절대오차, 이동 방향 코사인, 행동 코사인
  - 붕괴 지표: 예측과 정답의 표준편차 비율. 1보다 많이 작으면 움직임이 줄어든 것이다.
- `geo_score` = k별 FK 오차 중앙값의 평균이다.
  - 표본의 60–75%가 거의 정지한 구간이라, 가만히 있는 예측도 점수가 좋게 나온다.
  - 그래서 **움직이는 구간만 따로 본 지표**(k30에서 손끝이 20 mm 이상 움직인 표본)를 반드시 같이 본다.
- 오프라인 지표는 체크포인트를 **고르는** 데만 쓰고, 성공 여부는 실제 로봇 실행으로 판단한다.

---

## 10. 실제 로봇에서 돌리기 (Mac)

```bash
# 1) 로봇 서비스 (왼팔/오른팔은 USB 위치 ID로 정한다)
cd ~/robot-cockpit; LEFT_LOC=51642368 RIGHT_LOC=1257472 ~/robot-env/bin/python robot_service.py   # 포트 8020

# 2) 추론 + 웹 UI
XVLA_CKPT=<실행>/<스텝>/pretrained_model E280_CAMS=middle,left,right E280_PROMPT_STYLE=robot \
E280_CLOSED_L=58 E280_CLOSED_R=68 HB_UI_PORT=8011 HB_MODEL_LABEL="<이름>" \
~/groot-infer-env/bin/python ~/holobrain-mac-model/mac_infer_ui_e280.py
```

- 시작 메시지에 `cams=['middle','left','right'] prompt_style=robot … grip cmd closed L/R 58.0/68.0 open 0.0`이 나와야 한다.
  기본값은 다른 데이터셋(R675) 기준이라, **환경변수를 빠뜨리면 에러 없이 틀린 설정으로 돌아간다.**
- UI 순서: "로봇 연결" → 시작 자세 → 실행.
  - "Rest 자세 (0°)" 버튼은 모든 관절을 천천히 0°로 보낸다. 실행 중에는 동작하지 않는다.
  - `/disconnect`는 모터 힘(torque)을 끈다. 팔이 떨어질 수 있으니 주의한다.
- 자주 생기는 문제
  - **카메라가 안 나옴**: 아이폰 연속성 카메라가 연결되면 OpenCV 카메라 번호가 밀린다. 아이폰 연결을 끊고 로봇 서비스를 재시작한다.
  - **그리퍼가 벌어진 채 멈추거나, 닫았는데 원시값이 0이 아님**: follower 그리퍼의 0점이 전원을 켤 때마다 틀어진다.
    `/gripper_zero`(풀기 → 손으로 닫기 → 설정)를 하거나, 그리퍼를 닫은 채로 전원을 다시 켠다. 추론 쪽은 값을 [−270, 0] 범위로 자른다.

---

## 11. 지금까지 관찰 (scratch 기준)

| 데이터 | 실제 로봇에서 쌓기가 처음 보인 체크포인트 | 그때 epoch |
|---|---|---|
| R150 (B1-old) | 200–250k | 4.7–5.9 |
| R120 | 약 190k | 약 5.5 |
| R90 | 약 100k | 약 3.7 |
| R60 | 100k | 약 5.6 |

- 대략 **4–6 epoch**쯤에서 쌓기 동작이 나오기 시작했다.
- 데이터가 적으면 같은 스텝에서 epoch이 더 많아 학습 손실이 낮게 보인다. 이것은 품질 차이가 아니라 epoch 차이다.
- 모두 seed 하나로 돌린 관찰이다.

---

## 12. 체크포인트 위치

- 외장 SSD `rebot_ckpts_archive/4090_c8old_20260925/<실행 이름>/<스텝>/`. 폴더 안 `ARCHIVED_OK`에 md5 목록이 있다.
  - scratch 실행 이름: `r30scratch600k`, `r60scratch600k`, `r90scratch600k`, `r120scratch600k`.
  - B1-old R150 원본 위치는 `R150_SPEC.md`를 참고한다.
- 평가 결과: `~/c8/c8old_runs/<실행 이름>/eval/<스텝>.json`.
