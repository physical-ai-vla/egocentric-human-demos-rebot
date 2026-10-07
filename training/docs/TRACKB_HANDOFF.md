# Track B — r150-umi-v4 인수인계 (2026-09-22)

X-VLA를 **body-frame incremental SE(3) delta**로 학습시키는 vanilla-UMI 베이스라인.
Track A(HandUMI/ORB-SLAM3)는 별도 문서. 이 문서만 읽고 Track B를 이어받을 수 있어야 한다.

## 0. 한 줄 정의

```
R150 UMI zarr (절대 TCP pose)
  → umi_to_lerobot_v4_3cam.py  (여기서 모든 기하 변환이 끝난다)
  → LeRobot v3 dataset (3-cam, 20D action, 20D state)
  → stock X-VLA full fine-tuning + language conditioning
```

X-VLA 학습 코드는 **한 줄도 고치지 않는다.** 변환기가 평범한 텐서를 내보내고 X-VLA는 그걸 그대로 먹는다.

## 1. 사용자가 못박은 제약 (변경 금지)

- v4를 joint-space X-VLA로 되돌리지 말 것. reBot IK를 학습 전제조건으로 쓰지 말 것.
- HandUMI를 예전 HaWoR ego 데이터셋과 혼동하지 말 것.
- vanilla UMI 원칙 보존: **zarr에는 절대 pose, loader에서 상대 SE(3), X-VLA는 Cartesian trajectory 예측.**
- **"shape만 맞추는 건 금지"** — state contract는 shape이 아니라 의미가 맞아야 한다.
- **missing proprio를 0-padding으로 때우지 말 것.**
- **38D 버전을 버리지 말고 보존할 것. v3를 덮어쓰지 말 것.**
- threshold 재튜닝 금지 (40 mm/12° soft, 70 mm/40° hard는 R150 분포에서 나온 값).
- 보고는 A–F 구조. 프로브 1개당 가설 1개. 패치 전에 측정.

## 2. 왜 upstream `pose_rep='delta'`를 안 쓰는가 (감사 완료, 결론 확정)

UMI upstream `convert_pose_mat_rep(pose_rep='delta')`는 **world-frame** 차분이다:

```
t_k − t_{k−1},    R_k @ inv(R_{k−1})
```

우리가 원하는 건 **body-frame** 증분 `inv(T_{k−1}) @ T_k`. 수치적으로 다르다는 걸 증명했다 (0.20 m / 0.34 rad 차이).
사용자 지시: "upstream `pose_rep='delta'`는 쓰지 말고, v4 전용 body-frame SE(3) delta를 직접 구현하는 게 맞아."

구현: `~/umi_bridge/umi_delta.py` — selftest 5 seed PASS, 왕복 오차 1.3e-07.

```python
def to_body_delta(future_mat, base_mat):
    prev = np.concatenate([base_mat[None], future_mat[:-1]], axis=0)
    return np.linalg.inv(prev) @ future_mat

def from_body_delta(delta_mat, base_mat):
    out = np.empty_like(delta_mat); cur = base_mat
    for k in range(len(delta_mat)):
        cur = cur @ delta_mat[k]; out[k] = cur
    return out
```

## 3. 데이터 계약 (변환기가 만드는 것)

프레임 t는 15 Hz (zarr 30 fps에서 `--stride 2`).

| key | dim | 정의 |
|---|---|---|
| `action` | **20** | 팔당 `[pos3, rot6d, gripper]`, `inv(T_t) @ T_{t+1}` 에서 |
| `observation.state` (slim20) | **20** | 팔당 `[prev_rel pos3+rot6d, gripper]`, `prev_rel = inv(T_t) @ T_{t-1}` |
| `observation.state` (full38) | 38 | 위 + 팔당 `wrt pos3+rot6d`, `wrt = inv(T_other_t) @ T_this_t` |
| `observation.images.global` | 224×224 | 소스 mp4에서 디코드 (no crop, INTER_AREA, half-frame tol) |
| `observation.images.left_wrist` | 224×224 | zarr 그대로 |
| `observation.images.right_wrist` | 224×224 | zarr 그대로 |

에피소드 경계는 **지어내지 않고 버린다**:
- 마지막 프레임은 t+1이 없으므로 **저장하지 않는다** (모든 저장 프레임이 진짜 action을 가진다).
- 첫 프레임은 t-1이 없으므로 prev_rel = identity, `state_prev_valid = 0`.
- identity action을 진짜 라벨인 척 쓰지 않는다.

16-step chunk를 `delta_timestamps`로 뽑으면 정확히 `[a_t … a_{t+15}]`, 연속 body-frame 증분이다.

## 4. 왜 38D가 아니라 20D인가 (결정적 발견)

`max_state_dim=38`로 pretrained `lerobot/xvla-base`를 로드하면:

```
RuntimeError: size mismatch for model.transformer.action_encoder.fc.weight:
  checkpoint [30, 73728] vs current [30, 92160]
```

즉 38D는 pretrained action encoder를 **깨뜨린다**. 그래서 cross-arm `wrt` 항을 뺀 slim20으로 간다.

사용자 주의: "pretrained layout을 과잉 주장하지 말 것" — 20D가 정답이라서가 아니라 **stock 체크포인트가 20을 요구**하기 때문이다.
**38D 버전(`r150_umi_v4_3cam`)은 보존한다.** 나중에 scratch 학습이나 encoder 확장 실험에 쓴다.

## 5. 3-cam인 이유

한때 내가 "2-cam으로 충분"이라고 했는데 **틀렸다**. 확인 결과:
- 소스 LeRobot 데이터셋: 카메라 **3개**
- 기존 UMI zarr: 2개만 (wrist만)
- `lerobot/xvla-base`: `num_image_views=3`, `empty_cameras: 0`

→ global view를 소스 mp4에서 다시 디코드해서 3-cam으로 재변환했다. wrist와 동일 정책(crop 없음, INTER_AREA 224, half-frame 허용오차).

## 6. 파일 위치

노드 `bh-aiteam@100.64.0.2` (28 core / 125 GB RAM), python = `~/miniforge3/envs/holobrain/bin/python`

| 경로 | 역할 |
|---|---|
| `~/umi_bridge/umi_to_lerobot_v4_3cam.py` | **정본 변환기.** `--state-mode {full38,slim20}` |
| `~/umi_bridge/umi_delta.py` | body-frame delta + selftest |
| `~/umi_bridge/umi_segment.py` | soft/hard gate 세그멘테이션 (Track A와 공유) |
| `~/umi_bridge/r150_v4_instructions.py` | `episode_table()` — 150 ep, 6 instruction, 정확히 25개씩 |
| `~/umi_bridge/r150_umi.zarr` | 소스 UMI replay buffer (4.2 GB) |
| `~/holobrain-data/lerobot/r150_umi_v4_3cam` | **38D 버전 (보존)** — 150 ep / 86,795 frame / 356 MB |
| `~/holobrain-data/lerobot/r150_umi_v4_3cam_s20` | **20D 버전 (변환 중)** |
| `~/holobrain-data/lerobot/r150_umi_v4_3cam_mini` | 4-ep 스모크용 |
| `~/umi_bridge/convert_s20.log` | 진행 로그 |

## 7. 이미 닫힌 게이트 (재검증 불필요)

| 항목 | 결과 |
|---|---|
| zarr ↔ LeRobot 에피소드 순서 | length 상관 1.0 |
| delta 의미 왕복 | 오차 1.3e-07 |
| 16-step 진화 | 통과 |
| 누적 재구성 | ~1e-13 mm |
| 6-instruction join | 150 ep = 6 × 25, 정확 |
| tail padding | pad = 15, 정확 |
| NaN / Inf | 0 |
| hard gate 위반 | 0 |
| Gate A (`__getitem__` 전체 override) | 통과 — `action_is_pad` 포함 |

주의: `_sample_to_data` override는 **죽은 코드였다.** upstream `UmiDataset`이 `__getitem__`에 전부 인라인해서
override가 호출되지 않는다. 반드시 `__getitem__`을 통째로 override할 것.

## 8. 지금 돌아가는 것

`umi_to_lerobot_v4_3cam.py --state-mode slim20` → `r150_umi_v4_3cam_s20`.
2026-09-22 기준 약 19분 경과, 307/356 MB (~86%). 완료까지 몇 분 남음.

```bash
ssh bh-aiteam@100.64.0.2 "ps -eo etime,args | grep '[u]mi_to_lerobot_v4_3cam'"
```

## 9. 다음 단계

### 9.1 20D 검증 (변환 완료 직후, **차이만** 본다)

사용자가 지정한 목록:
- frame / episode 수가 38D 버전과 **완전 동일** (150 ep / 86,795 frame)
- 3-camera byte/frame alignment 동일
- action tensor가 bitwise 또는 float-equivalent (state만 달라야 한다)
- `state20 == full38[:, selected_indices]` — 팔당 앞 10개(prev_rel 9 + gripper 1)
- instruction / task 동일
- hard gate 0
- stock `max_state_dim=20` pretrained load 성공
- processor 포함 one-step forward/backward 성공

### 9.2 런처 작성

`chunk_size=16`, `n_action_steps=16`, `max_state_dim=20`, `max_action_dim=20`, 3 views,
`--rename_map` (processor 단계에서 적용), fp32, **full fine-tuning**, loss는 `L_umi`만.

### 9.3 4-episode 고정 부분집합 overfit (500 step)

측정 항목:
- translation / rotation amplitude, pred vs GT
- left / right 각각의 activity
- 16-step 합성 endpoint 오차
- zero-chunk / constant-chunk collapse 여부

### 9.4 그 다음

150 ep 전체로 5k probe → 통과하면 long run.

순서는 사용자가 정했다: **v4-base → v4-C → v4-CD.** 지금은 v4-base
(full training + native UMI delta + language)만 간다.

## 10. 밟았던 지뢰 (다시 밟지 말 것)

- `zarr` 설치가 numpy를 2.4.6으로 올려 holobrain env의 `robo-orchard-lab`을 깨뜨린다. **numpy 1.26.4 고정.**
- 로컬 zsh에서 heredoc/quoting이 자주 깨진다. 스크립트를 파일로 쓰고 `scp`로 보낼 것.
- `pkill -f`가 자기 자신의 ssh를 죽인다(exit 143). `ps -eo pid,args | grep … | kill`을 쓸 것.
- rsync remote spec에서 `~`는 확장되지 않는다. 절대경로 `/home/bh-aiteam/…`를 쓸 것.
- 노드 `/`는 88% 사용 중(112 GB 여유). 큰 중간 산출물을 남기지 말 것.

## 11. 사용자에게 보고할 때

A–F 구조. 하나의 프로브는 하나의 가설만. 측정 없이 패치하지 말 것.
single-run 결과로 결론 내리지 말 것 (Track A에서 reset count만 run-robust였다).
