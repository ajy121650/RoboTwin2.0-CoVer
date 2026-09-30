# RoboTwin 2.0 × CoVer-style Verifier

RoboTwin 2.0 벤치마크([upstream](https://github.com/RoboTwin-Platform/RoboTwin), 기준 커밋 `45a853a`)를 포크해
Pi0.5 policy의 test-time scaling(instruction rephrase × action sampling)과 **CoVer 방식 contrastive
verifier** 학습 파이프라인을 얹은 저장소입니다. 이 문서는 **클라우드 GPU 서버에서 verifier 학습을
바로 돌리기 위한 세팅 절차**입니다. 벤치마크 자체(시뮬레이터, 평가)의 사용법은 upstream `README.md`를
보세요.

## 이 저장소에 추가된 것

| 경로 | 내용 |
|---|---|
| `verifier/` | verifier 데이터 빌더, 모델, 학습(DDP), 오프라인 평가. 상세: [`verifier/README.md`](verifier/README.md) |
| `scripts/eval_policy_xpolicylab.py` | `--rephrase_num`, `--policy_batch_inference_size`, `--noise_std`, `--merge_mode`, `--step_limit_scale` 등 TTS 평가 옵션 |
| `batch_eval/` | GPU 큐 기반 sweep 러너(`run_tts_sweep.sh`, `run_std_sweep.sh` 등)와 분석 스크립트 |
| `description/task_instruction/*.json` | unseen instruction 정비(task당 32개) |

verifier 학습에는 시뮬레이터, XPolicyLab 서브모듈, policy 체크포인트가 **필요 없습니다**. 데이터셋
(1.9 GB)과 `verifier/`만 있으면 됩니다.

## 설계 요약

- 학습 방식: CoVer Level-0 — frozen SigLIP2 (ViT-L-16-384) 이미지·텍스트 타워 + 학습 가능한 head,
  (head 카메라 이미지, instruction) ↔ expert의 다음 50 step joint action 윈도우 사이의 대칭 InfoNCE.
  negative는 in-batch만 사용(기본은 CoVer처럼 GPU별 자기 배치 63개).
- 데이터: `demo_clean` 50 task × 50 episode = 2,500 episode (Pi0.5 leaderboard 체크포인트의 학습셋과
  동일 — lerobot v30과 에피소드 단위로 일치 확인). stride 5, 프레임당 seen instruction 8개,
  task당 마지막 5 episode holdout, unseen instruction은 평가 전용.
- 액션: joint space 14-D `[L arm 6, L grip, R arm 6, R grip]`, chunk 시작 시점 qpos 대비 delta,
  `max(|q01|,|q99|)`로 대칭 정규화, 그리퍼 `2g−1`, 패딩 −5.
- 배치 샘플러: 한 배치에 같은 episode의 윈도우가 두 개 이상 들어가지 않음(false negative 방지),
  task당 epoch 샘플 수 cap.

## 클라우드 서버 세팅

### 1. 저장소 clone (서브모듈 제외)

```bash
git clone -b robotwin-verifier https://github.com/ajy121650/RoboTwin2.0-CoVer.git
cd RoboTwin2.0-CoVer
```

`--recursive`를 붙이지 마세요. `XPolicyLab` 서브모듈은 이 포크의 로컬 커밋을 가리키고 있어
업스트림에서 받을 수 없고, verifier 학습에는 필요하지 않습니다.

### 2. Python 환경

Python 3.10, CUDA용 PyTorch 2.4 이상. 이 저장소가 검증된 조합:

```
torch==2.4.1  torchvision==0.19.1  open_clip_torch==3.3.0  timm==1.0.30
transformers==4.57.6  numpy==1.26.4  pillow==11.3.0  pyyaml==6.0.3
```

```bash
conda create -n verifier python=3.10 -y && conda activate verifier
# torch는 서버의 CUDA 버전에 맞는 휠로 먼저 설치 (예: cu121)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r verifier/requirements.txt
```

주의: **torch < 2.5 이면 `transformers<5`** 를 유지해야 합니다(transformers 5.x는 torch 2.5+ 필요,
`requirements.txt`에 이미 핀 되어 있음). torch 2.5+ 를 쓴다면 이 핀은 풀어도 됩니다.

### 3. Hugging Face 캐시 위치

SigLIP2-L 가중치(약 3.3 GB)가 HF hub에서 자동 다운로드됩니다. 여유 있는 디스크로 캐시를 보내세요:

```bash
export HF_HOME=/data/hf_cache        # 매 셸에서 필요. .bashrc에 넣어두면 편함
```

hub 접근이 막힌 서버라면 다른 머신에서 받은 `hf_cache/`를 통째로 복사한 뒤 `HF_HUB_OFFLINE=1`.

### 4. 데이터셋

전처리가 끝난 학습셋 `demo_clean_s5_w50` (1.9 GB, JPEG 106,263장 + `actions.npy`).

**A. Hugging Face에서 받기**

```bash
export HF_DATASET_REPO=<HF_DATASET_REPO>     # 예: ajy1216/robotwin-verifier-data (추후 확정)
hf download "$HF_DATASET_REPO" demo_clean_s5_w50.tar --repo-type dataset --local-dir /data/verifier_data
cd /data/verifier_data && tar -xf demo_clean_s5_w50.tar
# 무결성: sha256 = 47e317971c1d21dd0bfe8b9a5266134dedb9fdc8278ffd6562da64d87891cc9e
```

**B. hdf5에서 직접 재생성** (호스팅 없이, 결정적 빌드라 A와 바이트 단위 동일)

```bash
pip install h5py
ROBOTWIN_DATA_ROOT=/data/robotwin bash scripts/download_xpolicylab_data.sh      # demo_clean 50 task, 약 33 GB
python verifier/build_dataset.py --data-root /data/robotwin/demo_clean --out /data/verifier_data/demo_clean_s5_w50
python verifier/inspect_dataset.py /data/verifier_data/demo_clean_s5_w50          # 통계 확인
```

기대 통계: frames 106,263 (train 95,543 / holdout 10,720), pairs 850,104, instructions 150,762.

**이미지 사전 리사이즈 캐시 (권장)** — 학습 중 CPU 이미지 작업을 없앱니다. 데이터셋 폴더 안에
`images_384.u8.npy` (47 GB, uint8 `(106263, 384, 384, 3)`)를 만들고, 학습은 이를 자동으로 사용합니다
(`image_cache: auto`). 온라인 전처리와 비트 단위로 동일합니다. 32코어에서 약 2분:

```bash
python verifier/cache_images.py /data/verifier_data/demo_clean_s5_w50 --workers 32
```

캐시가 있으면 `--set train.num_workers=2`로 충분합니다(없으면 기본 8 유지). 디스크가 부족하면
이 단계를 건너뛰어도 되고, 그 경우 JPEG 디코드·리사이즈를 DataLoader 워커가 수행합니다.

### 5. 학습

```bash
# 8 GPU (baseline: scratch, ViT-L)
torchrun --nproc_per_node 8 verifier/train.py \
    --config verifier/configs/default.yaml \
    --data /data/verifier_data/demo_clean_s5_w50 \
    --out  /data/verifier_ckpt/vitl_scratch

# CoVer Bridge 체크포인트로 text/vision head warm start (별도 실험)
torchrun --nproc_per_node 8 verifier/train.py \
    --config verifier/configs/default.yaml \
    --data /data/verifier_data/demo_clean_s5_w50 \
    --out  /data/verifier_ckpt/vitl_warm \
    --set warm_start=/data/cover_verifier_bridge.pt \
    --set model.token_scale=none --set model.text_mask=false
```

- 기본 설정(`verifier/configs/default.yaml`): GPU당 batch 64, negative는 CoVer와 같이 각 GPU의
  자기 배치 63개(`train.gather_negatives: true`면 전 GPU all-gather), lr 1e-5, 15 epoch,
  warmup 2,000 step, task당 epoch 40,000 쌍 cap. 8 GPU 기준 epoch당 약 1,490 step.
- 어떤 값이든 `--set train.lr=3e-5` 식으로 덮어쓸 수 있습니다. GPU 메모리가 작으면
  `--set train.batch_size=32`.
- 로그: `<out>/log.jsonl` (학습 loss/top-1/top-5, 500 step마다 validation). `--set train.wandb=true`로
  wandb 기록(`pip install wandb`).
- 체크포인트: `best.pt`(validation top-1 기준), `last.pt`, `epochNNN.pt`. head 가중치 + 설정 +
  정규화 통계가 들어 있어 추론에 다른 파일이 필요 없습니다.
- 중단 후 이어서: 같은 명령에 `--resume` (epoch 내 위치까지 복원).
- `cover_verifier_bridge.pt`는 `hf download cover-vla/cover-vla-bridge cover_verifier_bridge.pt`.

**CoVer 원본 조건 재현** — `configs/cover.yaml`은 이 포트가 CoVer와 다르게 둔 것(negative gather,
액션 인코더 위치 임베딩, `token_scale`, `text_mask`, weight decay 범위, lr)을 전부 CoVer 값으로
되돌린 설정입니다. "CoVer를 in-domain으로 재학습한 baseline"은 이걸로, 수정판은 `default.yaml`로:

```bash
torchrun --nproc_per_node 8 verifier/train.py --config verifier/configs/cover.yaml \
    --data /data/verifier_data/demo_clean_s5_w50 --out /data/verifier_ckpt/vitl_cover
```

빠른 동작 확인(작은 백본, 몇 step):

```bash
python verifier/train.py --data /data/verifier_data/demo_clean_s5_w50 --out /tmp/smoke \
    --set model.backbone=hf-hub:timm/ViT-B-16-SigLIP2-256 --set train.batch_size=8 \
    --set train.max_steps=3 --set train.val_pairs=32 --set train.val_batch_size=16 \
    --set train.val_every=3 --set train.log_every=1 --set train.warmup_steps=2
```

### 6. 오프라인 평가

```bash
python verifier/eval_offline.py /data/verifier_ckpt/vitl_scratch/best.pt \
    --out /data/verifier_ckpt/vitl_scratch/eval_holdout.json
```

holdout 프레임마다 정답 윈도우가 (a) 전체 holdout, (b) 같은 task, (c) 같은 episode 풀에서 몇 등인지를
seen / unseen instruction 각각으로 보고합니다. (c)가 policy 후보 순위 매기기에 가장 가까운 지표이고,
각 지표 옆에 chance 수준이 함께 찍힙니다.

## 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| `transformers ... PyTorch >= 2.5 is required` | torch 2.4 + transformers 5.x 조합. `pip install "transformers<5"` |
| `--data and --out are required` | 설정 파일에 경로가 없는 것이 정상. 두 인자를 명령줄에 지정 |
| 첫 step까지 오래 걸림 | SigLIP2-L 3.3 GB 다운로드(rank 0가 먼저 받고 나머지가 대기). `HF_HOME` 디스크 확인 |
| DataLoader가 병목 | `cache_images.py`로 이미지 캐시를 만들면 CPU 이미지 작업이 없어짐. 캐시 없이 돌릴 땐 `train.num_workers`(기본 8/GPU) 조정 |
| GPU 메모리 부족 | `--set train.batch_size=32` (negative 수도 절반이 됨) |

## 로컬(원본 서버) 참고

- hdf5 원본: `/work2/junyoung/RoboTwin/data/demo_clean/` (33 GB)
- 전처리 데이터: `/work2/junyoung/RoboTwin/verifier_data/demo_clean_s5_w50/` (`data/verifier/` 심볼릭 링크)
- 평가 결과·후보 덤프: `eval_result/`, `batch_eval/{runs,candidates}` → `/work2` 심볼릭 링크
