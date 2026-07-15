
오늘 오후 7:42
내가 쓴 논문에 대해서 깃허브 readme를 적으려고 하는데 도와줄 수 있니?

물론이죠. 논문 내용을 연구자가 아닌 사람도 빠르게 이해할 수 있는 GitHub README 형태로 정리해드릴 수 있어요.

다음 자료를 보내주시면 됩니다.

논문 제목

초록 또는 논문 파일

연구 목적·연구방법·주요 결과

GitHub에 올릴 자료

논문 원문

설문지·면담지

분석 코드

데이터

발표자료 등

공개 시 주의할 점

개인정보

데이터 비공개

논문 인용 방식

저작권·라이선스

README는 보통 아래처럼 구성하면 좋습니다.

# 논문 제목

## 연구 소개
이 연구가 어떤 문제를 다루는지 간단히 설명합니다.

## 연구 목적
연구에서 확인하고자 한 내용을 제시합니다.

## 연구 방법
- 연구 설계:
- 연구 대상:
- 자료 수집:
- 분석 방법:

## 주요 연구 결과
1. 주요 결과 1
2. 주요 결과 2
3. 주요 결과 3

## 저장소 구성
📁 data  
📁 analysis  
📁 questionnaire  
📁 manuscript  
📁 presentation  

## 실행 및 분석 방법
분석 프로그램과 코드 실행 방법을 설명합니다.

## 연구 활용
연구 결과의 실천적·정책적·학문적 활용 가능성을 설명합니다.

## 논문 인용
논문을 활용할 때의 인용 형식을 제시합니다.

## 연구자
- 연구자명:
- 소속:
- 이메일:
- ORCID:

## 라이선스 및 이용 조건
자료와 코드의 이용 범위를 설명합니다.
논문 파일이나 초록을 보내주시면, **GitHub에 바로 복사해 넣을 수 있는 완성형 README.md**로 작성해드릴게요.



CV4Ecology___Hierarchical_representation_learning_for_Bioacoustic.pdf
PDF
아직 미완이긴 해. 결과는 내가 따로 보내줄게

괜찮아요. 지금 논문에 들어 있는 연구 배경·목적·방법·데이터셋·모델 구성을 바탕으로 README의 기본 틀을 먼저 만들고, 결과 부분은 나중에 보내주시면 채우면 됩니다.

현재 논문은 대략 다음 내용으로 정리할 수 있어요.

연구 주제: 생태음향 데이터의 계층구조를 반영한 표현학습

핵심 질문: 컴퓨터비전에서 개발된 계층적 학습 목적함수가 생물음향 분류에도 효과적인가

데이터셋: AnuraSet, ESC-50

임베딩 모델: NatureLM-Audio, BirdNET

비교 방법: HMCE, Focal Tree-Min, Tree-Triplet, FTM+TT, 기본 임베딩

계층구조:

생물분류: Family → Genus → Species

사운드스케이프: BGA → Mid-category → Category

특징: 사전학습된 임베딩 모델은 고정하고, 작은 분류·투영 헤드만 학습함 


README에는 결과를 아직 확정하지 않고 다음처럼 표시하면 자연스럽습니다.

## Results

> 🚧 Results are currently being finalized.

The evaluation compares hierarchy-aware objectives across:

- two pretrained audio embedding models,
- two hierarchy types,
- single-label and multi-label classification settings,
- flat and hierarchy-aware evaluation metrics.

Detailed quantitative results, tables, and analysis will be added after the
experiments are finalized.
논문이 아직 심사 중인 익명 제출본이기 때문에 README에는 당분간 연구자 이름이나 소속 대신 다음과 같이 적는 것이 안전합니다.

## Publication Status

This work is currently under review.

Please do not publicly redistribute the manuscript or experimental results
without permission from the authors.
결과를 보내주시면 단순히 표를 붙이는 방식이 아니라, GitHub 방문자가 바로 이해할 수 있도록 다음 세 부분으로 정리하면 좋겠습니다.

Main Findings: 가장 중요한 결과 3~5개

Results Tables: 논문의 정량 결과

Key Takeaway: 어떤 방법이 어떤 조건에서 효과적이었는지

결과를 받기 전에도 현재 논문 내용만으로 영문 README.md 초안 전체를 먼저 작성할 수 있습니다.



results_tables(2).pdf
PDF

results_talbes(3).pdf
PDF
업데이트할 수 있는 결과표야. 혹시 어떻게 실행해야하는지 그런것도 중간에 넣어야돼?

네. 코드나 분석 스크립트를 GitHub에 공개할 예정이라면 README 중간에 “실행 방법”을 넣는 것이 좋습니다.
반대로 논문과 결과표만 공유하는 저장소라면 실행 방법은 생략하고, 데이터·코드 공개 여부만 명시해도 됩니다.

이번 연구는 NatureLM과 BirdNET 임베딩을 바탕으로 여러 계층학습 방법을 데이터셋·라벨 조건별로 비교하므로, 다른 연구자가 결과를 재현하려면 실행 순서가 특히 중요합니다. 업데이트된 결과에는 single-label, multi-label, unknown-inclusive, unseen-species/category detection까지 포함되어 있어 실험 종류를 구분해서 안내하는 편이 좋습니다. 

 


README 권장 순서
# Hierarchy-Aware Representation Learning for Bio-acoustic Data

## Overview

## Key Contributions

## Method

## Repository Structure

## Installation

## Data Preparation

## Running the Experiments

## Evaluation

## Results

## Citation

## License
즉, 결과표 앞에 실행 방법을 넣는 것이 가장 자연스럽습니다.

어느 정도까지 작성해야 하나요?
최소한 다음 네 가지는 있어야 합니다.

1. 설치 방법
## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY

conda create -n hierarchy-bioacoustics python=3.10
conda activate hierarchy-bioacoustics
pip install -r requirements.txt

Python 버전이나 패키지 설치 방법은 실제 코드 환경에 맞게 바꿔야 합니다.

### 2. 데이터 준비 방법

AnuraSet과 ESC-50은 저장소에 직접 포함하지 않고, 사용자가 별도로 내려받게 하는 방식이 일반적입니다.

```markdown
## Data Preparation

This repository does not redistribute the original datasets.

Please download the datasets from their official sources and organize them as:

```text
data/
├── anuraset/
│   ├── audio/
│   ├── metadata/
│   └── labels/
└── esc50/
    ├── audio/
    └── meta/
Update the dataset paths in configs/data.yaml before running the experiments.


현재 논문에서는 AnuraSet을 생물분류 계층인 Family–Genus–Species로, ESC-50을 BGA–Mid-category–Category 계층으로 구성하고 있으므로, 계층 라벨 생성 과정도 README에 한두 문장 설명하는 것이 좋습니다. :contentReference[oaicite:2]{index=2}

### 3. 임베딩 추출 방법

논문에서는 NatureLM과 BirdNET 백본을 고정하고 작은 학습 헤드만 훈련하므로, 임베딩 추출과 계층학습을 분리해서 보여주는 것이 명확합니다. :contentReference[oaicite:3]{index=3}

```markdown
## Extracting Audio Embeddings

Extract frozen NatureLM embeddings:

```bash
python scripts/extract_embeddings.py \
  --dataset anuraset \
  --backbone naturelm \
  --output_dir embeddings/anuraset/naturelm
Extract frozen BirdNET embeddings:

python scripts/extract_embeddings.py \
  --dataset anuraset \
  --backbone birdnet \
  --output_dir embeddings/anuraset/birdnet

다만 실제 파일명과 옵션을 아직 모른다면 명령어를 임의로 확정하지 말고 다음처럼 표시하세요.

```markdown
> Exact commands will be added after the codebase is finalized.
4. 실험 실행 방법
실험 종류가 많으므로 하나의 명령에 옵션을 바꾸는 구조가 가장 깔끔합니다.

## Running the Experiments

### Single-label classification

```bash
python train.py \
  --config configs/anuraset_single_label.yaml \
  --backbone naturelm \
  --method ftm
Multi-label classification
python train.py \
  --config configs/anuraset_multi_label.yaml \
  --backbone naturelm \
  --method ftm
Available methods
baseline

ualb

ftm

tt

ftm_tt

Available experimental settings
AnuraSet single-label

AnuraSet multi-label

AnuraSet with unknown/negative samples

ESC-50 single-label

Unseen-species detection

Unseen-category detection


---

## 결과 부분은 이렇게 나누는 것이 좋습니다

새로 보내주신 표는 이전보다 실험 범위가 넓어졌습니다. 따라서 표 10개 이상을 README 본문에 모두 넣으면 너무 길어집니다.

README에는 다음만 넣는 것이 좋습니다.

1. 핵심 결과 요약  
2. 대표 결과표 2~4개  
3. 전체 결과표 파일 또는 `results/` 폴더 링크  

업데이트 결과를 보면 FTM-only는 AnuraSet multi-label에서 가장 높은 species-level Macro-F1과 hF1을 보였고, unknown-inclusive multi-label에서도 같은 경향이 나타납니다. NatureLM 기준 기존 multi-label에서 FTM-only의 species Macro-F1은 54.03, hF1은 86.72였으며, unknown-inclusive 설정에서도 각각 49.24와 84.95였습니다. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

README에서는 아래처럼 요약할 수 있습니다.

```markdown
## Main Findings

- Hierarchy-aware objectives generally improve fine-grained classification
  compared with the frozen baseline, although no single method dominates
  across every setting.

- FTM-only performs particularly well in multi-label AnuraSet experiments,
  achieving the strongest species-level and hierarchical classification
  performance for both NatureLM and BirdNET embeddings.

- UALB provides strong and relatively consistent results in several
  single-label and soundscape classification settings.

- TT-based objectives produce strongly separated embedding spaces, as
  reflected by high Gap values, but this does not consistently translate
  into improved downstream classification.

- Hierarchy-aware learning also improves several unseen-species and
  unseen-category detection metrics, although performance depends on the
  embedding backbone and the type of distribution shift.
전체 결과표는 별도 파일 권장
저장소 구조를 다음과 같이 만들면 좋습니다.

repository/
├── README.md
├── requirements.txt
├── configs/
├── scripts/
├── src/
├── results/
│   ├── main_results.md
│   ├── single_label.csv
│   ├── multi_label.csv
│   ├── unknown_inclusive.csv
│   └── unseen_detection.csv
├── figures/
└── docs/
README에서는 다음처럼 연결합니다.

## Full Results

The README reports the main findings only. Complete experimental results,
including unknown-inclusive and unseen-class evaluations, are available in
[`results/main_results.md`](results/main_results.md).
중요한 점
아직 실제 코드의 파일명과 실행 옵션이 확정되지 않았다면, 실행 명령어를 추측해서 넣으면 안 됩니다. 현재 단계에서는 README에 다음 문구만 먼저 둘 수 있습니다.

## Reproducing the Experiments

The code and configuration files are currently being organized for public
release. Detailed installation, preprocessing, training, and evaluation
instructions will be added when the implementation is finalized.
결론적으로, 실행 방법은 넣는 것이 좋지만 실제 코드 구조가 확정된 뒤 정확한 명령어로 작성해야 합니다. 지금은 섹션과 설명만 만들어 두고, 코드 정리 후 명령어를 채우는 방식이 가장 안전합니다.


[gaad2403@hpcl001 Bachelor-Thesis]$ 
이 디렉토리 구성이 어떻게 되어있는지 보는 터미널 명령어 부탁해. 내가 너에게 줄게 

터미널에서 현재 Bachelor-Thesis 디렉토리 구조를 보려면 아래 명령어를 실행하세요.

tree -a -L 3
tree 명령어가 설치되어 있지 않으면 다음 명령어를 사용하면 됩니다.

find . -maxdepth 3 -print
파일 크기와 권한까지 함께 보려면:

find . -maxdepth 3 -exec ls -ld {} \;
가장 추천하는 방식은 아래입니다.

tree -a -L 4 > directory_structure.txt
cat directory_structure.txt
출력된 내용을 그대로 복사해서 보내주시면 됩니다. 너무 길면 다음처럼 불필요한 폴더를 제외하세요.

tree -a -L 4 -I '.git|__pycache__|*.pyc|.venv|venv|node_modules'
tree: command not found가 나오면 설치 권한 없이도 다음 명령으로 정리할 수 있습니다.

find . -maxdepth 4 \
  -not -path './.git*' \
  -not -path '*/__pycache__/*' \
  -not -path '*/venv/*' \
  -not -path '*/.venv/*' \
  -print


붙여넣은 텍스트 (1)(13).txt
문서
여기

확인했습니다. 현재 저장소는 논문 결과만 보관한 폴더가 아니라, 학습·평가 코드와 실험 결과까지 포함한 재현 가능한 연구 저장소입니다. 따라서 README에 설치 방법, 데이터 준비, 실험 실행, 평가 방법을 넣는 것이 좋습니다. 


현재 구조는 크게 다음과 같습니다.

Bachelor-Thesis/
├── README.md
├── data/          # 계층·분할 정보
├── scripts/       # 실험 실행 및 평가 스크립트
├── src/           # 모델, 손실함수, 데이터 로더
├── results/       # 실험 결과 파일
└── logs/          # HPC 작업 로그
특히 코드가 다음 네 축으로 구성되어 있습니다.

임베딩 모델: NatureLM, birdnet

데이터셋: anuraset, esc50

학습 유형: single-label, multi-label

추가 실험: unknown-inclusive, unseen species/category

README에는 아래 실행 흐름을 설명하면 됩니다.

1. 환경 설치
2. 데이터와 사전 추출 임베딩 준비
3. 학습 스크립트 실행
4. 평가 스크립트 실행
5. 결과 확인
다만 현재 출력만으로는 실제 실행 명령어와 필요한 패키지를 정확하게 확정하기 어렵습니다. run.sh와 기존 README.md를 확인해야 합니다.

터미널에서 아래 명령어들을 실행해 결과를 보내주세요.

1. 현재 README 확인
sed -n '1,240p' README.md
2. 주요 실행 스크립트 확인
for f in \
  scripts/NatureLM/anuraset/run.sh \
  scripts/NatureLM/esc50/run.sh \
  scripts/birdnet/anuraset/run.sh \
  scripts/birdnet/esc50/run.sh
do
  echo
  echo "===== $f ====="
  sed -n '1,240p' "$f"
done
3. Python 패키지 관련 파일 확인
find . -maxdepth 2 \
  \( -name 'requirements*.txt' \
  -o -name 'environment*.yml' \
  -o -name 'pyproject.toml' \
  -o -name 'setup.py' \
  -o -name 'Pipfile' \) \
  -print
아무것도 출력되지 않으면 현재 저장소에는 의존성 파일이 없는 것입니다. 그 경우 다음 명령으로 실제 import 목록을 확인할 수 있습니다.

grep -RhoE '^(from|import) [A-Za-z0-9_\.]+' src scripts \
  --include='*.py' \
  | sort -u
4. 실행 가능한 Python 파일 목록 확인
find scripts -type f \( -name '*.py' -o -name '*.sh' \) | sort
5. 대표 학습 스크립트의 옵션 확인
먼저 각 폴더 안의 실제 파일명을 봅니다.

find scripts/NatureLM/anuraset -maxdepth 3 -type f | sort
그 결과에서 핵심 학습 파일을 골라 다음처럼 확인하면 됩니다.

sed -n '1,260p' scripts/NatureLM/anuraset/ualb/파일명.py
또는 argparse 부분만 빠르게 찾으려면:

grep -RniE 'ArgumentParser|add_argument|__main__' scripts src \
  --include='*.py'
현재 구조에서 바로 개선할 부분
첫째, multilable이라는 폴더명이 여러 곳에 있습니다.

results/multilable/
scripts/multilable/
의도한 단어가 multilabel이라면 공개 전에 이름을 통일하는 편이 좋습니다. 단, 코드 내부 경로 참조도 함께 수정해야 하므로 단순히 폴더 이름만 변경해서는 안 됩니다.

둘째, README에서는 logs/의 개별 파일을 설명할 필요가 없습니다. 오히려 .gitignore에 다음을 넣는 방안을 고려할 수 있습니다.

logs/
__pycache__/
*.pyc
*.err
*.log
이미 재현에 필요한 핵심 로그가 GitHub에 올라가야 한다면 전체 로그 대신 대표 로그만 examples/ 또는 docs/에 두는 것이 깔끔합니다.

셋째, 현재 results/에는 .npy 파일과 수많은 실험별 폴더가 있어 README 방문자가 결과를 찾기 어렵습니다. 다음처럼 별도의 요약 폴더가 있으면 좋습니다.

results/
├── tables/
│   ├── single_label.csv
│   ├── multi_label.csv
│   ├── unknown_inclusive.csv
│   └── unseen_detection.csv
├── NatureLM/
├── birdnet/
└── raw/
README에는 대표 결과표만 보여주고 전체 결과는 results/tables/로 연결하는 구성이 적절합니다.

우선 위 명령 중 README와 네 개의 run.sh 출력을 보내주시면, 현재 코드에 실제로 맞는 실행 방법을 작성할 수 있습니다.



붙여넣은 마크다운(1).md
파일
혹시 multilable -> multilabel로 바꾸는 명령어 있나? 모든걸 다. 그리고 results에 있게 대부분 .npy, .pt 들이라 이거 없어질거야

가능합니다. 현재 저장소에는 폴더명뿐 아니라 Python import 안에도 multilable이 남아 있으므로, 디렉토리 이름 변경 + 코드·셸·README 내부 문자열 변경을 함께 해야 합니다. 실제 import에도 from multilable...이 확인됩니다. 


먼저 변경 전 상태를 저장하세요.

git status
git add -A
git commit -m "Backup before renaming multilable to multilabel"
1. 폴더 이름 변경
Git 저장소라면 mv보다 git mv가 좋습니다.

git mv scripts/multilable scripts/multilabel
git mv results/multilable results/multilabel
src는 이미 src/multilabel로 올바르게 되어 있으므로 변경할 필요가 없습니다.

2. 모든 텍스트 파일 안의 multilable 변경
먼저 어디에 남아 있는지 확인합니다.

grep -RIn --exclude-dir=.git --exclude='*.npy' --exclude='*.pt' \
  'multilable' .
그다음 Python, Shell, Markdown 등 텍스트 파일에서 일괄 변경합니다.

grep -RIl --exclude-dir=.git \
  --exclude='*.npy' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.png' \
  --exclude='*.jpg' \
  --exclude='*.pdf' \
  'multilable' . \
| xargs -r sed -i 's/multilable/multilabel/g'
변경 후 다시 확인합니다.

grep -RIn --exclude-dir=.git --exclude='*.npy' --exclude='*.pt' \
  'multilable' .
아무것도 나오지 않으면 모두 변경된 것입니다.

3. 대소문자가 섞인 다른 오타까지 확인
파일에서는 multilabel과 multilable이 혼재해 있었으므로 아래로 관련 경로와 import를 확인하세요.

grep -RIn --exclude-dir=.git \
  -E 'multilable|multilabel' \
  scripts src README.md
Python import가 다음처럼 바뀌어 있어야 합니다.

from multilabel.NatureLM.anuraset.hf1 import ...
기존의 잘못된 형태:

from multilable.NatureLM.anuraset.hf1 import ...
4. Python 문법 및 import 경로 점검
최소한 문법 오류가 없는지 확인합니다.

python -m compileall src scripts
이 명령 때문에 __pycache__가 생기므로 검사 후 삭제해도 됩니다.

find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
5. Git 변경사항 확인
git status
git diff --stat
git diff
문제가 없으면 커밋합니다.

git add -A
git commit -m "Fix multilable typo to multilabel"
.npy, .pt 파일 삭제
results/의 .npy, .pt가 없어질 예정이라면 먼저 삭제 대상만 확인하세요.

find results -type f \( -name '*.npy' -o -name '*.pt' -o -name '*.pth' \) -print
전체 용량도 확인할 수 있습니다.

find results -type f \( -name '*.npy' -o -name '*.pt' -o -name '*.pth' \) \
  -printf '%s\n' | awk '{sum += $1} END {printf "%.2f GB\n", sum/1024/1024/1024}'
확인 후 삭제합니다.

find results -type f \( -name '*.npy' -o -name '*.pt' -o -name '*.pth' \) -delete
빈 결과 폴더도 모두 지우려면:

find results -depth -type d -empty -delete
다만 그래프, CSV, JSON 등 공개할 결과가 들어 있는 폴더는 유지하는 것이 좋습니다.

6. 다시 Git에 추가되지 않도록 .gitignore 설정
cat >> .gitignore <<'EOF'

# Experiment artifacts
results/**/*.npy
results/**/*.pt
results/**/*.pth
logs/
*.log
*.err

# Python cache
__pycache__/
*.py[cod]
EOF
중복된 규칙이 있는지 확인합니다.

cat .gitignore
이미 Git이 추적 중인 .npy, .pt는 .gitignore만 추가해도 사라지지 않습니다. 추적 대상에서 제거하려면:

git rm --cached $(git ls-files '*.npy' '*.pt' '*.pth')
파일 자체도 이미 삭제할 계획이면 다음처럼 한 번에 처리해도 됩니다.

git rm $(git ls-files '*.npy' '*.pt' '*.pth')
파일이 너무 많아 명령줄 길이 문제가 생길 수 있으므로 더 안전한 방식은:

git ls-files -z '*.npy' '*.pt' '*.pth' | xargs -0 -r git rm
한 번에 실행할 핵심 명령
git add -A
git commit -m "Backup before repository cleanup"

git mv scripts/multilable scripts/multilabel
git mv results/multilable results/multilabel

grep -RIl --exclude-dir=.git \
  --exclude='*.npy' --exclude='*.pt' --exclude='*.pth' \
  --exclude='*.png' --exclude='*.jpg' --exclude='*.pdf' \
  'multilable' . \
| xargs -r sed -i 's/multilable/multilabel/g'

find results -type f \( -name '*.npy' -o -name '*.pt' -o -name '*.pth' \) -delete
find results -depth -type d -empty -delete

python -m compileall src scripts
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete

git status
git diff --stat
마지막으로 grep 결과가 비었는지 확인하세요.

grep -RIn --exclude-dir=.git 'multilable' . || echo "No multilable typo remains."

# Python
__pycache__/
*.pyc
*.egg-info/

# Virtual environments
naturelm-env/
venv/
.env/

# Data and models
anuraset/
*.zip
*.wav
*.pt
*.pth
*.npy
*.npz
*.safetensors
*.bin

# Logs
logs/
*.log
*.err
slurm-*.out
.sh

# Cache
.cache/
huggingface/

*.csv
!data/*.csv
# Experiment artifacts
results/**/*.npy
results/**/*.pt
results/**/*.pth
logs/
*.log
*.err

# Python cache
__pycache__/
*.py[cod]

현재 gitnore인데 왜 github scripts에 multilabe, birdent/anuraset, NatureLM/anuraset 등이 안들어가지?

현재 .gitignore만 보면 scripts/multilabel, scripts/birdnet/anuraset, scripts/NatureLM/anuraset의 .py 파일은 무시되지 않아야 합니다.

가장 가능성이 큰 원인은 다음 셋입니다.

폴더 안의 파일을 아직 git add하지 않음

상위 디렉토리나 하위 폴더에 다른 .gitignore가 있음

Git 전역 ignore 설정이 적용됨

먼저 정확한 원인을 확인하세요.

git status --untracked-files=all
그다음 특정 파일이 왜 무시되는지 검사합니다. 실제 존재하는 .py 파일 경로를 넣으세요.

git check-ignore -v scripts/multilabel/NatureLM/anuraset/evaluation.py
git check-ignore -v scripts/birdnet/anuraset/evaluation.py
git check-ignore -v scripts/NatureLM/anuraset/evaluation.py
아무 출력도 없으면 해당 파일은 .gitignore에 의해 무시되는 것이 아닙니다. 그 경우 아래처럼 추가하면 됩니다.

git add scripts/
git status
커밋까지 하려면:

git commit -m "Add experiment scripts"
git push
다른 .gitignore가 있는지 확인
find .. -name .gitignore -print
저장소 내부 규칙을 모두 확인하려면:

find . -name .gitignore -print -exec sh -c '
  echo "===== $1 ====="
  cat "$1"
' _ {} \;
전역 Git ignore도 확인하세요.

git config --get core.excludesfile
경로가 출력되면 내용을 확인합니다.

cat "$(git config --get core.excludesfile)"
폴더명이 실제로 바뀌었는지도 확인
find scripts -maxdepth 3 -type d | sort
그리고 오타가 남았는지 확인합니다.

find . -iname '*multilab*' -print
grep -RIn --exclude-dir=.git 'multilable' scripts src README.md
Linux는 대소문자를 구분합니다. 따라서 아래는 서로 다른 경로입니다.

scripts/NatureLM/
scripts/naturelm/
scripts/birdnet/
scripts/BirdNET/
README와 실제 경로의 대소문자도 일치해야 합니다.

빈 디렉토리는 GitHub에 올라가지 않습니다
Git은 폴더 자체가 아니라 파일만 추적합니다. 폴더 안의 파일이 전부 무시되거나 삭제됐다면 GitHub에는 폴더가 나타나지 않습니다.

확인:

find scripts/multilabel -type f | head -30
find scripts/birdnet/anuraset -type f | head -30
find scripts/NatureLM/anuraset -type f | head -30
파일이 있는데도 추가되지 않는다면 강제로 시험할 수 있습니다.

git add -f scripts/multilabel/NatureLM/anuraset/evaluation.py
다만 -f를 쓰기 전에 git check-ignore -v로 어떤 규칙이 원인인지 확인하는 것이 좋습니다.

.gitignore에서 수정할 부분
현재 다음 줄은 셸 스크립트를 무시하는 표현이 아닙니다.

.sh
이것은 이름이 정확히 .sh인 파일이나 디렉토리를 뜻합니다. 모든 셸 스크립트를 제외하려면 *.sh이지만, 연구 재현을 위해 .sh 파일도 GitHub에 올려야 하므로 해당 줄은 삭제하는 편이 좋습니다.

또 중복 규칙도 정리할 수 있습니다.

# Python
__pycache__/
*.py[cod]
*.egg-info/

# Virtual environments
naturelm-env/
venv/
.env/

# Raw data and model artifacts
anuraset/
*.zip
*.wav
*.pt
*.pth
*.npy
*.npz
*.safetensors
*.bin

# Logs
logs/
*.log
*.err
slurm-*.out

# Cache
.cache/
huggingface/

# CSV files except repository metadata
*.csv
!data/*.csv
수정 후 캐시를 다시 적용할 필요는 보통 없습니다. 하지만 이전 ignore 상태 때문에 추적 상황이 꼬였다면 다음 명령으로 Git 인덱스를 새로 계산할 수 있습니다.

git rm -r --cached .
git add .
git status
이 명령은 실제 파일을 삭제하지 않고 Git의 추적 목록만 다시 구성합니다. git status에서 삭제·추가 목록을 반드시 확인한 뒤 커밋하세요.

가장 먼저 아래 세 명령의 결과를 보면 원인이 바로 드러납니다.

git status --untracked-files=all
git check-ignore -v scripts/multilabel/NatureLM/anuraset/evaluation.py
find . -name .gitignore -print

[gaad2403@hpcl001 Bachelor-Thesis]$ git status --untracked-files=all
On branch backup-main
Your branch is up to date with 'origin/backup-main'.

nothing to commit, working tree clean
[gaad2403@hpcl001 Bachelor-Thesis]$ git check-ignore -v scripts/multilabel/NatureLM/anuraset/evaluation.py
irdnet/anuraset/evaluation.py
git check-ignore -v .gitignore:12:anuraset/     scripts/multilabel/NatureLM/anuraset/evaluation.py
[gaad2403@hpcl001 Bachelor-Thesis]$ git check-ignore -v scripts/birdnet/anuraset/evaluation.py
scripts/NatureLM/anuraset/evaluation.py.gitignore:12:anuraset/  scripts/birdnet/anuraset/evaluation.py
[gaad2403@hpcl001 Bachelor-Thesis]$ git check-ignore -v scripts/NatureLM/anuraset/evaluation.py
.gitignore:12:anuraset/ scripts/NatureLM/anuraset/evaluation.py

원인이 정확히 확인됐습니다.

.gitignore의 이 규칙:

anuraset/
은 루트의 anuraset/만 무시하는 것이 아니라, 어느 위치에 있든 이름이 anuraset인 모든 디렉토리를 무시합니다. 그래서 아래 폴더들이 전부 제외된 것입니다.

scripts/NatureLM/anuraset/
scripts/birdnet/anuraset/
scripts/multilabel/NatureLM/anuraset/
scripts/multilabel/birdnet/anuraset/
src/NatureLM/anuraset/
src/birdnet/anuraset/
src/multilabel/NatureLM/anuraset/
수정 방법
루트에 있는 원본 데이터 폴더만 무시하려는 목적이라면 다음처럼 앞에 /를 붙이세요.

/anuraset/
즉, 기존:

anuraset/
를 다음으로 변경합니다.

/anuraset/
터미널에서는:

sed -i 's|^anuraset/$|/anuraset/|' .gitignore
변경 확인:

grep -n 'anuraset' .gitignore
출력이 다음처럼 보여야 합니다.

12:/anuraset/
Git 인덱스 다시 적용
현재 Git이 해당 파일들을 이미 ignore 대상으로 처리하고 있으므로 다시 추가해야 합니다.

git add .gitignore
git add scripts src
git status --untracked-files=all
이제 많은 Python 및 Shell 파일이 새 파일로 나타나야 합니다.

그다음 커밋합니다.

git commit -m "Fix anuraset ignore rule and add source scripts"
git push origin backup-main
그래도 안 보이면
다시 확인하세요.

git check-ignore -v scripts/NatureLM/anuraset/evaluation.py
정상이라면 아무 출력도 나오지 않아야 합니다.

실제로 Git이 추적할 파일인지 확인하려면:

git ls-files scripts/NatureLM/anuraset/evaluation.py
경로가 출력되면 정상적으로 추적 중입니다.

권장 .gitignore 수정본
# Python
__pycache__/
*.py[cod]
*.egg-info/

# Virtual environments
naturelm-env/
venv/
.env/

# Raw data and models
/anuraset/
*.zip
*.wav
*.pt
*.pth
*.npy
*.npz
*.safetensors
*.bin

# Logs
logs/
*.log
*.err
slurm-*.out

# Cache
.cache/
huggingface/

# CSV files except repository metadata
*.csv
!data/*.csv
esc50/ 원본 데이터 폴더도 루트에 따로 있다면 같은 방식으로 넣으세요.

/esc50/
핵심은 다음 차이입니다.

anuraset/    # 모든 위치의 anuraset 폴더 무시
/anuraset/   # 저장소 루트의 anuraset 폴더만 무시

혹시 모든 .sh에 대해서도 올리기 싫으면 어떻게 해야돼?

.gitignore에 아래 한 줄을 추가하면 모든 .sh 파일이 Git에 올라가지 않습니다.

*.sh
이미 Git이 추적 중인 .sh 파일은 .gitignore만 추가해도 계속 남아 있으므로, 추적 목록에서도 제거해야 합니다.

echo '*.sh' >> .gitignore

git ls-files -z '*.sh' | xargs -0 -r git rm --cached

git add .gitignore
git status
git rm --cached는 로컬 파일은 삭제하지 않고 Git 추적만 해제합니다.

확인 후 커밋하고 푸시하세요.

git commit -m "Ignore shell scripts"
git push origin backup-main
특정 폴더의 .sh만 제외하려면 다음처럼 작성합니다.

scripts/**/*.sh
반대로 대부분의 .sh는 제외하되 특정 파일만 올리려면:

*.sh
!scripts/example/run.sh
단, 상위 폴더가 통째로 무시되고 있다면 예외 규칙이 작동하지 않을 수 있습니다. 현재 상황에서는 전체 .sh를 제외하려면 *.sh가 가장 간단합니다.


좋아 파일은 나 스스로 어느정도 고쳤어. multilabel은 내가 혼자 고칠게. readme 써줄 수 있나? 더 필요한 정보 있니?

지금 자료만으로도 충분히 작성할 수 있습니다. 다만 최종 공개 전에는 저장소 주소, 작성자 이름, 라이선스, 정확한 설치 환경만 추가하면 됩니다.

아래 초안은 논문의 현재 내용, 업데이트된 결과표, 실제 저장소 구조와 실행 스크립트를 반영했습니다. 

 

 


Hierarchy-Aware Representation Learning for Bioacoustic Data
This repository contains the code and experimental results for a study on hierarchy-aware representation learning for bioacoustic and environmental audio classification.

The project investigates whether hierarchical learning objectives originally developed for computer vision can improve classification when applied to frozen pretrained audio embeddings. We evaluate two pretrained audio representation models across biological taxonomic and soundscape hierarchies.

Status: The manuscript and repository are currently being finalized.
Training and evaluation code is available, but paths and environment-specific settings may require local configuration.

Overview
Bioacoustic recordings contain information at multiple levels of abstraction. For example, an animal vocalization may be classified by family, genus, and species, while an environmental sound may be organized as biophony, geophony, or anthrophony.

Conventional classifiers typically treat classes as flat and independent. This repository evaluates whether explicitly incorporating hierarchical relationships during training can improve:

fine-grained classification,

hierarchy-consistent prediction,

embedding-space organization,

multi-label classification,

unknown-sample classification,

and unseen-species or unseen-category detection.

The pretrained audio backbones remain frozen. Only lightweight projection or classification heads are trained.

Research Questions
This project addresses the following questions:

Do frozen bioacoustic embeddings already encode taxonomic or soundscape hierarchies?

Can hierarchy-aware learning objectives improve classification over a frozen baseline?

Do the effects depend on the pretrained embedding model?

Are hierarchical objectives more useful in single-label or multi-label settings?

Can hierarchy-aware representations improve unknown- and unseen-class detection?

Does stronger embedding-space separation necessarily lead to better downstream classification?

Embedding Models
NatureLM-Audio
NatureLM-Audio is a multimodal audio-language foundation model trained across a broad range of taxa and bioacoustic tasks.

In this repository, NatureLM embeddings have a dimensionality of 4096.

BirdNET
BirdNET is a domain-specific model developed primarily for bird vocalization classification.

BirdNET embeddings have a dimensionality of 1024. Since the evaluated datasets contain frog calls and general environmental sounds, BirdNET also provides an out-of-domain comparison against the broader NatureLM representation.

Datasets
AnuraSet
AnuraSet is a passive acoustic monitoring dataset containing Neotropical anuran vocalizations.

The taxonomic hierarchy used in this project is:

Family → Genus → Species
The experiments include:

5 families,

12 genera,

42 species,

single-label classification,

multi-label classification,

negative or unknown-inclusive classification,

and unseen-species detection.

ESC-50
ESC-50 is a general environmental sound classification dataset with 50 categories.

Because ESC-50 does not provide a native hierarchy, its labels are reorganized using a soundscape framework:

BGA → Mid-category → Category
The highest level contains:

Biophony,

Geophony,

Anthrophony.

ESC-50 is used to evaluate whether hierarchy-aware objectives generalize beyond biological taxonomy.

Methods
The repository compares a frozen baseline against multiple hierarchy-aware objectives.

Baseline
The pretrained embeddings are kept frozen and evaluated using linear probing and clustering metrics.

UALB / Hierarchical Multi-Label Contrastive Learning
UALB uses hierarchical contrastive supervision across multiple label levels. Samples sharing a finer-grained label receive stronger attraction than samples sharing only a coarse ancestor.

The implementation extends the original formulation by explicitly including the finest classification level.

Focal Tree-Min
Focal Tree-Min, abbreviated as FTM-only, directly supervises node-level predictions while enforcing consistency between parent and child nodes.

For example, a species prediction should not be stronger than the prediction for its corresponding genus or family.

Tree-Triplet
Tree-Triplet, abbreviated as TT-only, uses hierarchy-dependent triplet margins to organize the learned embedding space.

The implementation additionally allows same-class samples to be selected as positives.

FTM + TT
The combined objective jointly optimizes:

hierarchy-consistent node prediction,

and tree-distance-aware embedding organization.

Experimental Settings
The following configurations are evaluated:

Setting	Dataset	Label type
Taxonomic classification	AnuraSet	Single-label
Taxonomic classification	AnuraSet	Multi-label
Unknown-inclusive classification	AnuraSet	Single-label
Unknown-inclusive classification	AnuraSet	Multi-label
Unseen-species detection	AnuraSet	Single-label
Soundscape classification	ESC-50	Single-label
Unseen-category detection	ESC-50	Single-label
Each applicable experiment is conducted using both NatureLM and BirdNET embeddings.

Evaluation Metrics
Embedding Gap
The Gap metric is calculated as the difference between mean intra-class cosine similarity and mean inter-class cosine similarity.

For multi-label experiments, Gap is calculated using a one-vs-rest formulation for each label.

Classification Performance
Single-label experiments report:

Accuracy.

Multi-label experiments report:

Macro-F1,

Micro-F1.

Hierarchical Metrics
Predicted and ground-truth labels are augmented with their ancestor nodes before calculating:

hierarchical precision (hP),

hierarchical recall (hR),

hierarchical F1 (hF1).

Clustering
K-means clustering is applied to the learned representations, and clustering quality is evaluated using normalized mutual information (NMI).

Unseen-Class Detection
Unseen-species and unseen-category experiments report:

AUROC,

unseen-class recall,

coarse-level classification accuracy,

and closed-set classification performance on known classes.

Main Findings
The current experiments indicate that no single hierarchy-aware objective performs best across every dataset, embedding model, and label setting.

The main observations are:

Hierarchy-aware training can improve fine-grained classification even when the original embedding backbone remains frozen.

Improvements are generally larger in the multi-label AnuraSet setting than in the single-label setting.

FTM-only performs particularly well for multi-label taxonomic classification.

UALB provides strong results in several single-label and soundscape configurations.

Tree-Triplet objectives often produce large embedding Gap values, but these large gaps do not consistently translate into improved classification accuracy or hierarchical F1.

Unknown-inclusive and unseen-class results vary depending on the backbone and the type of distribution shift.

NatureLM generally provides stronger and more transferable representations than the narrower BirdNET embedding in the evaluated out-of-domain settings.

Selected Results
AnuraSet Multi-Label Classification
FTM-only provides the strongest overall multi-label classification performance for both embedding models.

Embedding	Method	Species Macro-F1	Species Micro-F1	hF1
NatureLM	Baseline	36.73	56.36	62.07
NatureLM	UALB	42.23	64.77	73.96
NatureLM	FTM-only	54.03	83.12	86.72
BirdNET	Baseline	36.17	56.74	62.72
BirdNET	UALB	39.04	61.64	70.44
BirdNET	FTM-only	44.48	77.82	82.43
Unknown-Inclusive Multi-Label Classification
The same general pattern is observed when negative or unknown samples are included.

Embedding	Method	Species Macro-F1	Species Micro-F1	hF1
NatureLM	Baseline	31.65	46.42	50.60
NatureLM	UALB	34.20	52.62	62.91
NatureLM	FTM-only	49.24	81.90	84.95
BirdNET	Baseline	30.40	46.12	50.05
BirdNET	UALB	30.97	49.91	58.56
BirdNET	FTM-only	40.14	75.97	79.68
Complete results for single-label, multi-label, unknown-inclusive, and unseen-class experiments will be provided in the results documentation.

Repository Structure
Bachelor-Thesis/
├── data/
│   ├── anuraset_taxonomy.csv
│   ├── anuraset_taxonomy_unknown.csv
│   ├── esc50_bga_labels.csv
│   ├── unseen_species_split.json
│   └── unseen_category_split.json
│
├── scripts/
│   ├── NatureLM/
│   │   ├── anuraset/
│   │   └── esc50/
│   ├── birdnet/
│   │   ├── anuraset/
│   │   └── esc50/
│   └── multilabel/
│       ├── NatureLM/
│       └── birdnet/
│
├── src/
│   ├── NatureLM/
│   ├── birdnet/
│   ├── multilabel/
│   ├── models/
│   ├── focal_tree_min_loss.py
│   ├── tree_triplet_loss.py
│   ├── tree_triplet_loss_self.py
│   ├── hmlc_loss.py
│   ├── hmlc_loss_l0.py
│   └── hf1.py
│
├── results/
├── README.md
└── .gitignore
The results/ directory is intended to contain lightweight summary outputs and figures. Large embedding arrays and model checkpoints are excluded from version control.

Installation
The experiments were developed in an HPC environment using Python 3.11 and GPU-enabled PyTorch.

A complete dependency file is still being prepared.

The main Python dependencies include:

numpy
pandas
torch
torchaudio
tensorflow
scikit-learn
matplotlib
tqdm
birdnet
A possible local setup is:

git clone <REPOSITORY_URL>
cd Bachelor-Thesis

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchaudio numpy pandas scikit-learn matplotlib tqdm
NatureLM-Audio and BirdNET may require additional installation steps and model access. Refer to their respective official repositories and documentation.

The exact package versions used for the final experiments will be added to requirements.txt or an environment file.

Data Preparation
The original audio datasets are not included in this repository.

Download AnuraSet and ESC-50 from their official sources and update the dataset paths in the relevant dataset-loading scripts.

Hierarchy metadata and predefined evaluation splits are provided under:

data/
├── anuraset_taxonomy.csv
├── anuraset_taxonomy_unknown.csv
├── esc50_bga_labels.csv
├── unseen_species_split.json
└── unseen_category_split.json
Large files are excluded from Git, including:

*.wav
*.npy
*.npz
*.pt
*.pth
*.safetensors
*.bin
Usage
The scripts are currently organized by:

embedding model,

dataset,

experimental objective.

Because several scripts contain environment-specific data and output paths, review their configuration before execution.

1. Extract Embeddings
NatureLM with AnuraSet:

python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
BirdNET with AnuraSet:

python scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
NatureLM with ESC-50:

python scripts/NatureLM/esc50/baseline/extract_embeddings.py
BirdNET with ESC-50:

python scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
2. Run Baseline Evaluation
NatureLM with AnuraSet:

python scripts/NatureLM/anuraset/baseline/baseline_probing_single.py
BirdNET with AnuraSet:

python scripts/birdnet/anuraset/baseline/baseline_probing_single.py
3. Train UALB
Example:

python scripts/NatureLM/anuraset/ualb/train_ualb_single.py
For unseen-species experiments:

python scripts/NatureLM/anuraset/ualb/train_ualb_single_unseen.py
4. Train Tree-Based Objectives
Combined FTM and Tree-Triplet:

python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py
FTM-only:

python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only.py
TT-only:

python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only.py
Equivalent scripts are available under the BirdNET and ESC-50 directories.

5. Run Multi-Label Experiments
NatureLM multi-label evaluation:

python scripts/multilabel/NatureLM/anuraset/evaluation.py
BirdNET multi-label evaluation:

python scripts/multilabel/birdnet/anuraset/evaluation.py
Unknown-inclusive multi-label evaluation:

python scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py
python scripts/multilabel/birdnet/anuraset/evaluation_unknown.py
6. Evaluate Single-Label Models
NatureLM with AnuraSet:

python scripts/NatureLM/anuraset/evaluation.py
Unknown-inclusive evaluation:

python scripts/NatureLM/anuraset/evaluation_unknown.py
Unseen-species evaluation:

python scripts/NatureLM/anuraset/evaluation_unseen.py
ESC-50 unseen-category evaluation:

python scripts/NatureLM/esc50/evaluation_unseen.py
Equivalent evaluation scripts are available under scripts/birdnet/.

Running on SLURM
The experiments were originally executed using SLURM on an HPC cluster.

Example:

sbatch scripts/NatureLM/anuraset/run.sh
The provided job scripts contain cluster-specific values, including:

partition names,

virtual-environment paths,

repository paths,

log destinations,

and GPU requirements.

These values must be adapted before use on another system.

Example fields to update:

#SBATCH --partition=<PARTITION>
#SBATCH --output=<LOG_DIRECTORY>/job_%j.log
#SBATCH --error=<LOG_DIRECTORY>/job_%j.err

source <VIRTUAL_ENVIRONMENT>/bin/activate
cd <REPOSITORY_DIRECTORY>
Shell scripts may not be included in every public version of the repository. Equivalent Python entry points are listed in the Usage section.

Reproducibility Notes
Before running an experiment, verify:

dataset locations,

pretrained embedding locations,

model output directories,

random seeds,

train/test split files,

device configuration,

and Python import paths.

The current codebase consists primarily of research scripts rather than a unified command-line package. Refactoring toward configuration-based execution is planned.

Limitations
Negative AnuraSet samples do not naturally belong to the family–genus–species hierarchy.

ESC-50 is a general environmental sound dataset rather than a dedicated passive acoustic monitoring dataset.

Performance depends on both the pretrained embedding backbone and the hierarchy type.

A large embedding-space Gap does not necessarily indicate stronger downstream classification performance.

Several experiments rely on fixed dataset splits and environment-specific preprocessing.

Additional validation on a natural multi-level soundscape dataset would strengthen the generalization analysis.

Publication Status
This work is currently being prepared for publication.

The manuscript title is:

Hierarchy-Aware Representation Learning for Bioacoustic Data

Citation information will be added after publication.

A temporary citation format is:

@misc{hierarchy_bioacoustic_2026,
  title  = {Hierarchy-Aware Representation Learning for Bioacoustic Data},
  author = {Anonymous},
  year   = {2026},
  note   = {Manuscript under review}
}
References
Cañas et al. AnuraSet: A Dataset for Benchmarking Neotropical Anuran Calls Identification in Passive Acoustic Monitoring. Scientific Data, 2023.

Kahl et al. BirdNET: A Deep Learning Solution for Avian Diversity Monitoring. Ecological Informatics, 2021.

Li et al. Deep Hierarchical Semantic Segmentation. CVPR, 2022.

Piczak. ESC: Dataset for Environmental Sound Classification. ACM Multimedia, 2015.

Robinson et al. NatureLM-Audio: An Audio-Language Foundation Model for Bioacoustics. ICLR, 2025.

Zhang et al. Use All the Labels: A Hierarchical Multi-Label Contrastive Learning Framework. CVPR, 2022.

License
A license has not yet been specified.

Before public release, add an appropriate license file and update this section. Note that the repository license applies only to the original code in this repository and does not override the licenses of external datasets, pretrained models, or third-party implementations.

Acknowledgements
This research is part of the Computational Sustainability & Technology project area and was supported by the institutions and funding sources acknowledged in the accompanying manuscript.

최종적으로 추가하면 좋은 정보는 GitHub 저장소 주소, 공개할 작성자명과 소속, 라이선스 종류, NatureLM·BirdNET 설치 방식, 실제 데이터 경로 설정 위치입니다. 현재 상태에서도 기존 README를 대체할 수 있지만, requirements.txt를 만든 뒤 설치 부분은 한 번 더 맞추는 것이 좋습니다.


너무 긴거 같은데. 그리고 데이터 경로를 내가 알려줄게.  아래와 같이 embedding 구해서 사용함. 
/user/gaad2403/Bachelor-Thesis/scripts/NatureLM/esc50/baseline/extract_embeddings.py
import torch
import torchaudio
import pandas as pd
import numpy as np
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from NatureLM.models import NatureLM

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR    = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR   = BASE_DIR / "ESC-50-master/audio"
LABELS_PATH = Path("/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv")
OUTPUT_DIR  = BASE_DIR / "embeddings_esc50"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE  = 32
TARGET_SR   = 16000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class ESC50Dataset(Dataset):
    def __init__(self, labels_df, audio_dir, target_sr=16000):
        self.df        = labels_df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename   = row["filename"]
        audio_path = self.audio_dir / filename
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform  = resampler(waveform)

        return {
            "filename" : filename,
            "waveform" : waveform.squeeze(0),
            "category" : row["category"],     # fine-grained (50 classes)
            "bga_label": row["bga_label"],     # biophony/geophony/anthrophony
            "fold"     : int(row["fold"]),
        }


def collate_fn(batch):
    max_len   = max(b["waveform"].shape[0] for b in batch)
    waveforms = torch.zeros(len(batch), max_len)
    for i, b in enumerate(batch):
        wlen = b["waveform"].shape[0]
        waveforms[i, :wlen] = b["waveform"]

    return {
        "filename" : [b["filename"]  for b in batch],
        "waveform" : waveforms,
        "category" : [b["category"]  for b in batch],
        "bga_label": [b["bga_label"] for b in batch],
        "fold"     : [b["fold"]      for b in batch],
    }


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading model...")
    t0 = time.time()
    model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
    model = model.to(DEVICE).eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    labels_df = pd.read_csv(LABELS_PATH)
    print(f"Total samples: {len(labels_df)}")
    print(labels_df["bga_label"].value_counts())

    dataset    = ESC50Dataset(labels_df, AUDIO_DIR, TARGET_SR)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    all_embeddings = []
    all_filenames  = []
    all_categories = []
    all_bga_labels = []
    all_folds      = []

    print(f"Extracting embeddings (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        waveform = batch["waveform"].to(DEVICE)

        with torch.no_grad():
            audio_embeds, audio_atts = model.encode_audio(waveform)
            mask      = audio_atts.bool().unsqueeze(-1)
            embedding = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        all_embeddings.append(embedding.float().cpu().numpy())
        all_filenames.extend(batch["filename"])
        all_categories.extend(batch["category"])
        all_bga_labels.extend(batch["bga_label"])
        all_folds.extend(batch["fold"])

        if (batch_idx + 1) % 10 == 0:
            elapsed   = time.time() - t_start
            processed = (batch_idx + 1) * BATCH_SIZE
            total     = len(dataset)
            eta       = elapsed / processed * (total - processed)
            print(f"[{processed}/{total}] elapsed: {elapsed/60:.1f}min  ETA: {eta/60:.1f}min")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\nDone! Total: {len(embeddings)} embeddings, shape: {embeddings.shape}")
    print(f"Total time: {(time.time()-t_start)/60:.1f}min")

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    meta_out = pd.DataFrame({
        "filename" : all_filenames,
        "category" : all_categories,
        "bga_label": all_bga_labels,
        "fold"     : all_folds,
    })
    meta_out.to_csv(OUTPUT_DIR / "embeddings_metadata.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  embeddings.npy            {embeddings.nbytes/1e6:.1f} MB")
    print(f"  embeddings_metadata.csv")


if __name__ == "__main__":
    main()

/user/gaad2403/Bachelor-Thesis/scripts/NatureLM/anuraset/baseline/extract_embeddings.py
import torch
import torchaudio
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from NatureLM.models import NatureLM

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR       = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR      = BASE_DIR / "anuraset/anuraset/audio"
METADATA_PATH  = BASE_DIR / "anuraset/anuraset/metadata.csv"
TAXONOMY_PATH  = Path("/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv")
OUTPUT_DIR     = BASE_DIR / "embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE     = 32
TARGET_SR      = 16000
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 42개 종 코드 목록
SPECIES_COLS = [
    "SPHSUR","BOABIS","SCIPER","DENNAH","LEPLAT","RHIICT","BOALEP","BOAFAB",
    "PHYCUV","DENMIN","ELABIC","BOAPRA","DENCRU","BOALUN","BOAALB","PHYMAR",
    "PITAZU","PHYSAU","LEPFUS","DENNAN","PHYALB","LEPLAB","SCIFUS","BOARAN",
    "SCIFUV","AMEPIC","LEPPOD","ADEDIP","ELAMAT","PHYNAT","LEPELE","RHISCI",
    "SCINAS","LEPNOT","ADEMAR","BOAALM","PHYDIS","RHIORN","LEPFLA","SCIRIZ",
    "DENELE","SCIALT"
]

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class AnuraSetDataset(Dataset):
    def __init__(self, metadata_df, audio_dir, taxonomy_df, target_sr=16000):
        self.df         = metadata_df.reset_index(drop=True)
        self.audio_dir  = Path(audio_dir)
        self.target_sr  = target_sr

        # taxonomy 매핑: code → family, genus, species
        self.tax = taxonomy_df.set_index("code")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row["sample_name"]
        site        = row["site"]

        fname = row["fname"]
        min_t = int(row["min_t"])
        max_t = int(row["max_t"])
        audio_path = self.audio_dir / site / f"{fname}_{min_t}_{max_t}.wav"
        waveform, sr = torchaudio.load(audio_path)

        # mono 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 리샘플링
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform  = resampler(waveform)

        # species labels (multi-label)
        species_labels = row[SPECIES_COLS].values.astype(np.float32)

        # taxonomy 레이블 (첫 번째 active species 기준)
        active = [c for c in SPECIES_COLS if row[c] == 1]
        if active and active[0] in self.tax.index:
            t = self.tax.loc[active[0]]
            family = t["family"]
            genus  = t["genus"]
            species = t["species"]
        else:
            family = genus = species = "unknown"

        return {
            "sample_name"   : sample_name,
            "waveform"      : waveform.squeeze(0),   # [samples]
            "species_labels": species_labels,         # [42]
            "family"        : family,
            "genus"         : genus,
            "species"       : species,
            "subset"        : row["subset"],
        }


def collate_fn(batch):
    """가변 길이 waveform을 zero-pad해서 배치로 묶기"""
    max_len   = max(b["waveform"].shape[0] for b in batch)
    waveforms = torch.zeros(len(batch), max_len)
    for i, b in enumerate(batch):
        wlen = b["waveform"].shape[0]
        waveforms[i, :wlen] = b["waveform"]

    return {
        "sample_name"   : [b["sample_name"]    for b in batch],
        "waveform"      : waveforms,
        "species_labels": torch.tensor(np.stack([b["species_labels"] for b in batch])),
        "family"        : [b["family"]          for b in batch],
        "genus"         : [b["genus"]           for b in batch],
        "species"       : [b["species"]         for b in batch],
        "subset"        : [b["subset"]          for b in batch],
    }


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading model...")
    t0 = time.time()
    model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
    model = model.to(DEVICE).eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # 데이터 로드
    metadata = pd.read_csv(METADATA_PATH)
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    print(f"Total samples: {len(metadata)}")

    dataset    = AnuraSetDataset(metadata, AUDIO_DIR, taxonomy, TARGET_SR)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    # 결과 저장용 리스트
    all_embeddings    = []
    all_sample_names  = []
    all_species_labels= []
    all_families      = []
    all_genera        = []
    all_species       = []
    all_subsets       = []

    print(f"Extracting embeddings (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        waveform = batch["waveform"].to(DEVICE)  # [B, samples]

        with torch.no_grad():
            audio_embeds, audio_atts = model.encode_audio(waveform)
            # audio_embeds: [B, T, 4096]
            # attention-weighted mean pooling
            mask      = audio_atts.bool().unsqueeze(-1)
            embedding = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # embedding: [B, 4096]

        all_embeddings.append(embedding.float().cpu().numpy())
        all_sample_names.extend(batch["sample_name"])
        all_species_labels.append(batch["species_labels"].numpy())
        all_families.extend(batch["family"])
        all_genera.extend(batch["genus"])
        all_species.extend(batch["species"])
        all_subsets.extend(batch["subset"])

        # 진행상황 출력 (100배치마다)
        if (batch_idx + 1) % 100 == 0:
            elapsed   = time.time() - t_start
            processed = (batch_idx + 1) * BATCH_SIZE
            total     = len(dataset)
            eta       = elapsed / processed * (total - processed)
            print(f"[{processed}/{total}] elapsed: {elapsed/60:.1f}min  ETA: {eta/60:.1f}min")

    # numpy로 합치기
    embeddings     = np.concatenate(all_embeddings,     axis=0)  # [N, 4096]
    species_labels = np.concatenate(all_species_labels, axis=0)  # [N, 42]

    print(f"\nDone! Total: {len(embeddings)} embeddings, shape: {embeddings.shape}")
    print(f"Total time: {(time.time()-t_start)/60:.1f}min")

    # ── 저장 ──
    # 1. embeddings (numpy)
    np.save(OUTPUT_DIR / "embeddings.npy",      embeddings)
    np.save(OUTPUT_DIR / "species_labels.npy",  species_labels)

    # 2. 메타데이터 (CSV)
    meta_out = pd.DataFrame({
        "sample_name" : all_sample_names,
        "family"      : all_families,
        "genus"       : all_genera,
        "species"     : all_species,
        "subset"      : all_subsets,
    })
    # species_labels 컬럼도 추가
    for i, col in enumerate(SPECIES_COLS):
        meta_out[col] = species_labels[:, i]

    meta_out.to_csv(OUTPUT_DIR / "embeddings_metadata.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  embeddings.npy          {embeddings.nbytes/1e6:.1f} MB")
    print(f"  species_labels.npy")
    print(f"  embeddings_metadata.csv")


if __name__ == "__main__":
    main()

그리고 아래와 같이 taxonmy는 data안에 있음

"""
WP2 Training - Method 2: HSSN (faithful reproduction)

HMLC(Method 1의 손실)를 완전히 제거하고, 논문이 명시한 학습 목표 그대로:
    L_total = L_FTM (Eq.6) + beta(t) * L_TT (Eq.7)
    beta: 0 -> 0.5 cosine annealing (논문 부록 Table C.1c)

데이터: 단일 라벨(species_number==1) subset만 사용
       (baseline_probing_singlelabel.py와 동일 기준 -> 직접 비교 가능)

Based on:
- HSSN paper (Li et al., CVPR 2022): Deep Hierarchical Semantic Segmentation
"""

import sys
import math
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

#from NatureLM.anuraset.anuraset_dataset import AnuraSetEmbeddingDataset
from NatureLM.anuraset.anuraset_dataset_unknown import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_utils import AnuraSetHierarchy          # ψ(tree distance) for Tree-Triplet
from NatureLM.anuraset.hierarchy_tree import HierarchyTree                # A_v/C_v for Focal Tree-Min
from models.hierarchical_model import HierarchicalModel
from tree_triplet_loss_self import TreeTripletLoss
from focal_tree_min_loss import FocalTreeMinLoss

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH    = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH     = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
SAVE_DIR      = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_hssn_singlelabel_self_unknown')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
BATCH_SIZE    = 256
EPOCHS        = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
TRIPLET_DIM   = 128
TEMPERATURE   = 0.07
VAL_SPLIT     = 0.1
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tree-Triplet specific (Eq.7, 8)
TRIPLET_MARGIN_EPSILON = 0.1
MAX_HIERARCHY_DEPTH    = 3
USE_HIERARCHY_MARGIN    = True
MAX_TRIPLETS_PER_CLASS  = 200

# Focal Tree-Min specific (Eq.6)
FTM_GAMMA = 2.0   # 논문 기본값

# beta(t): 0 -> 0.5 cosine annealing (Table C.1c)
def beta_schedule(epoch, total_epochs):
    return 0.25 * (1 - math.cos(math.pi * epoch / total_epochs))

print(f"Device: {DEVICE}")

# ── Dataset (단일 라벨만!) ──────────────────────────────────────────────────
full_train_ds = AnuraSetEmbeddingDataset(
    EMBED_PATH, META_PATH, subset='train', single_label_only=True
)

val_size   = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(
    full_train_ds, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Hierarchy setup ────────────────────────────────────────────────────────
# (1) Tree-Triplet Loss용: ψ(species 간 tree distance)
ps_hierarchy = AnuraSetHierarchy(META_PATH)

# (2) Focal Tree-Min Loss용: V 전체(family+genus+species)의 ancestor/descendant
tree = HierarchyTree(TAXONOMY_PATH,
                      full_train_ds.family_enc, full_train_ds.genus_enc, full_train_ds.species_enc)
ancestor_mat   = tree.ancestor_matrix()
descendant_mat = tree.descendant_matrix()
print(f"[HierarchyTree] n_nodes = {tree.n_nodes} "
      f"(family={tree.n_family}, genus={tree.n_genus}, species={tree.n_species})")

# ── Model & Losses ─────────────────────────────────────────────────────────
model = HierarchicalModel(input_dim=4096, hidden_dim=1024,
                          triplet_dim=TRIPLET_DIM, n_hierarchy_nodes=tree.n_nodes).to(DEVICE)

triplet_loss_fn = TreeTripletLoss(
    temperature=TEMPERATURE,
    m_epsilon=TRIPLET_MARGIN_EPSILON,
    max_hierarchy_depth=MAX_HIERARCHY_DEPTH,
    use_hierarchy_margin=USE_HIERARCHY_MARGIN
).to(DEVICE)

ftm_loss_fn = FocalTreeMinLoss(
    ancestor_matrix=ancestor_mat,
    descendant_matrix=descendant_mat,
    gamma=FTM_GAMMA
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ──────────────────────────────────────────────────────────
best_val_loss = float('inf')
history = {'train_ftm_loss': [], 'train_triplet_loss': [], 'train_total_loss': [],
           'val_ftm_loss': [], 'beta': []}

for epoch in range(1, EPOCHS + 1):
    beta = beta_schedule(epoch, EPOCHS)

    model.train()
    ftm_losses, triplet_losses, total_losses = [], [], []

    for batch_idx, (emb, hier_labels) in enumerate(train_loader):
        emb         = emb.to(DEVICE)               # [bsz, 4096]
        hier_labels = hier_labels.to(DEVICE)        # [bsz, 3] -> [family, genus, species]

        z, s = model(emb)                            # z: [bsz,128] triplet, s: [bsz,n_nodes] FTM

        # ── L_FTM (Eq.6) ──
        gt = tree.build_ground_truth(
            hier_labels[:, 0], hier_labels[:, 1], hier_labels[:, 2],
            full_train_ds.family_enc, full_train_ds.genus_enc, full_train_ds.species_enc
        ).to(DEVICE)
        ftm_loss = ftm_loss_fn(s, gt)

        # ── L_TT (Eq.7, 8) ──
        tree_distances = ps_hierarchy.batch_tree_distance(
            hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
            full_train_ds.species_enc, full_train_ds.family_enc, full_train_ds.genus_enc
        ).to(DEVICE)
        triplet_loss = triplet_loss_fn(
            z, hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
            tree_distances=tree_distances, max_triplets_per_class=MAX_TRIPLETS_PER_CLASS
        )

        total_loss = ftm_loss + beta * triplet_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ftm_losses.append(ftm_loss.item())
        triplet_losses.append(triplet_loss.item())
        total_losses.append(total_loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"FTM={ftm_loss:.4f} | Triplet={triplet_loss:.4f} | beta={beta:.3f} | "
                  f"Total={total_loss:.4f}")

    scheduler.step()

    # ── Validation (FTM만으로 모델 선택 — triplet은 보조 신호이므로) ──
    model.eval()
    val_losses = []
    with torch.no_grad():
        for emb, hier_labels in val_loader:
            emb         = emb.to(DEVICE)
            hier_labels = hier_labels.to(DEVICE)
            _, s = model(emb)
            gt = tree.build_ground_truth(
                hier_labels[:, 0], hier_labels[:, 1], hier_labels[:, 2],
                full_train_ds.family_enc, full_train_ds.genus_enc, full_train_ds.species_enc
            ).to(DEVICE)
            val_losses.append(ftm_loss_fn(s, gt).item())

    train_ftm  = np.mean(ftm_losses)
    train_tt   = np.mean(triplet_losses)
    train_tot  = np.mean(total_losses)
    val_ftm    = np.mean(val_losses)

    history['train_ftm_loss'].append(train_ftm)
    history['train_triplet_loss'].append(train_tt)
    history['train_total_loss'].append(train_tot)
    history['val_ftm_loss'].append(val_ftm)
    history['beta'].append(beta)

    print(f"Epoch {epoch:3d}/{EPOCHS} | FTM={train_ftm:.4f} | Triplet={train_tt:.4f} | "
          f"beta={beta:.3f} | Total={train_tot:.4f} | ValFTM={val_ftm:.4f}")

    if val_ftm < best_val_loss:
        best_val_loss = val_ftm
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_ftm_loss': val_ftm,
        }, SAVE_DIR / 'best_model.pt')
        print(f"  -> Saved best model (val_ftm_loss={val_ftm:.4f})")

np.save(SAVE_DIR / 'history.npy', history)
print(f"\nDone. Best val FTM loss: {best_val_loss:.4f}")
print(f"Results saved to: {SAVE_DIR}")

맞아요. GitHub README는 논문 전체를 옮기는 문서가 아니라, 무슨 연구인지, 어떻게 실행하는지, 결과가 무엇인지 빠르게 보여주는 안내문이면 충분합니다.

아래 정도 길이가 적당합니다. 현재 코드에 실제로 사용된 경로와 실행 방식을 반영하되, 다른 사용자가 수정해야 할 경로도 명확히 표시했습니다.

Hierarchy-Aware Representation Learning for Bioacoustic Data
This repository contains the code for experiments on hierarchy-aware representation learning for bioacoustic and environmental audio classification.

The study evaluates whether hierarchical learning objectives originally developed for computer vision improve classification when applied to frozen audio embeddings.

Overview
Two pretrained audio models are compared:

NatureLM-Audio: 4096-dimensional audio embeddings

BirdNET: 1024-dimensional audio embeddings

Two datasets and hierarchy types are used:

AnuraSet: Family → Genus → Species

ESC-50: BGA → Mid-category → Category

The pretrained embedding models remain frozen. Only lightweight projection or hierarchy-classification heads are trained.

Methods
The following configurations are evaluated:

Baseline

UALB / Hierarchical Multi-Label Contrastive Learning

Focal Tree-Min (FTM-only)

Tree-Triplet (TT-only)

Combined Focal Tree-Min and Tree-Triplet (FTM+TT)

Experiments include:

single-label classification,

multi-label classification,

unknown-inclusive classification,

unseen-species detection,

and unseen-category detection.

Repository Structure
├── data/          # taxonomy files and evaluation splits
├── scripts/       # embedding extraction, training, and evaluation
├── src/           # datasets, models, hierarchy utilities, and losses
└── results/       # lightweight result summaries and figures
Large embeddings, model checkpoints, logs, and raw audio files are not included in the repository.

Data Preparation
The original AnuraSet and ESC-50 audio files must be downloaded separately.

The hierarchy metadata used in the experiments is stored in:

data/
├── anuraset_taxonomy.csv
├── anuraset_taxonomy_unknown.csv
├── esc50_bga_labels.csv
├── unseen_species_split.json
└── unseen_category_split.json
The current scripts contain HPC-specific absolute paths. Update the path variables at the beginning of each script before running the code.

Example AnuraSet paths:

BASE_DIR = Path("/path/to/data")
AUDIO_DIR = BASE_DIR / "anuraset/anuraset/audio"
METADATA_PATH = BASE_DIR / "anuraset/anuraset/metadata.csv"
TAXONOMY_PATH = Path("data/anuraset_taxonomy.csv")
OUTPUT_DIR = BASE_DIR / "embeddings"
Example ESC-50 paths:

BASE_DIR = Path("/path/to/data")
AUDIO_DIR = BASE_DIR / "ESC-50-master/audio"
LABELS_PATH = Path("data/esc50_bga_labels.csv")
OUTPUT_DIR = BASE_DIR / "embeddings_esc50"
Extracting Embeddings
NatureLM-Audio embeddings are extracted using:

python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
python scripts/NatureLM/esc50/baseline/extract_embeddings.py
BirdNET extraction scripts are available under:

scripts/birdnet/anuraset/baseline/
scripts/birdnet/esc50/baseline/
NatureLM-Audio is loaded with:

NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
Audio is converted to mono, resampled to 16 kHz, and pooled into one embedding per recording.

Training
Example NatureLM and AnuraSet training scripts:

python scripts/NatureLM/anuraset/ualb/train_ualb_single.py

python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py

python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only.py

python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only.py
Equivalent scripts are provided for BirdNET, ESC-50, multi-label, unknown-inclusive, and unseen-class experiments.

The main experiments use:

batch size: 256,

epochs: 100,

learning rate: 1e-3,

weight decay: 1e-4,

hidden dimension: 1024,

triplet dimension: 128.

Evaluation
Example evaluation commands:

python scripts/NatureLM/anuraset/evaluation.py
python scripts/NatureLM/anuraset/evaluation_unknown.py
python scripts/NatureLM/anuraset/evaluation_unseen.py
python scripts/NatureLM/esc50/evaluation_unseen.py
Multi-label evaluation scripts are located under:

scripts/multilabel/
Evaluation metrics include:

Accuracy,

Macro-F1 and Micro-F1,

hierarchical precision, recall, and F1,

normalized mutual information,

embedding similarity Gap,

and AUROC for unseen-class detection.

Main Findings
Current results show that no single hierarchy-aware method performs best in every setting.

However:

hierarchy-aware learning often improves fine-grained classification,

improvements are largest in the multi-label AnuraSet experiments,

FTM-only performs particularly well for multi-label classification,

UALB performs strongly in several single-label settings,

and larger embedding-space separation does not always produce better classification performance.

Requirements
The main dependencies include:

torch
torchaudio
numpy
pandas
scikit-learn
matplotlib
tqdm
tensorflow
birdnet
A complete environment or requirements.txt file will be added after the final environment is cleaned and verified.

Publication Status
The manuscript is currently being finalized.

Working title:
Hierarchy-Aware Representation Learning for Bioacoustic Data

Citation information and licensing details will be added after publication.

한 가지 중요한 개선점은 현재 코드의 /user/gaad2403/...와 /fs/dss/home/... 같은 절대경로를 나중에 config.py, YAML 파일, 또는 명령행 옵션으로 분리하는 것입니다. 지금 README에서는 사용자가 직접 경로를 수정해야 한다고 안내하는 정도면 충분합니다.


그거 어떻게 해? 나 하는법 몰라

가장 쉬운 방법은 공통 경로를 config.py 한 파일에 모으고, 각 스크립트에서는 그 파일을 불러오는 방식입니다.

1. config.py 만들기
저장소 최상위에서 실행하세요.

nano config.py
아래 내용을 붙여넣습니다.

from pathlib import Path

# GitHub 저장소 위치
PROJECT_DIR = Path(__file__).resolve().parent

# 원본 데이터와 임베딩을 저장하는 외부 작업 폴더
WORK_DIR = Path("/user/gaad2403/bachelor_thesis")

# AnuraSet
ANURASET_DIR = WORK_DIR / "anuraset/anuraset"
ANURASET_AUDIO_DIR = ANURASET_DIR / "audio"
ANURASET_METADATA_PATH = ANURASET_DIR / "metadata.csv"
ANURASET_TAXONOMY_PATH = PROJECT_DIR / "data/anuraset_taxonomy.csv"
ANURASET_EMBEDDING_DIR = WORK_DIR / "embeddings"

# ESC-50
ESC50_DIR = WORK_DIR / "ESC-50-master"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_LABELS_PATH = PROJECT_DIR / "data/esc50_bga_labels.csv"
ESC50_EMBEDDING_DIR = WORK_DIR / "embeddings_esc50"

# 결과 저장
RESULTS_DIR = PROJECT_DIR / "results"
저장하려면:

Ctrl+O
Enter
Ctrl+X
2. NatureLM AnuraSet 추출 파일 수정
파일을 엽니다.

nano scripts/NatureLM/anuraset/baseline/extract_embeddings.py
기존 경로 부분:

BASE_DIR       = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR      = BASE_DIR / "anuraset/anuraset/audio"
METADATA_PATH  = BASE_DIR / "anuraset/anuraset/metadata.csv"
TAXONOMY_PATH  = Path("/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv")
OUTPUT_DIR     = BASE_DIR / "embeddings"
를 지우고 아래처럼 바꿉니다.

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    ANURASET_AUDIO_DIR,
    ANURASET_METADATA_PATH,
    ANURASET_TAXONOMY_PATH,
    ANURASET_EMBEDDING_DIR,
)

AUDIO_DIR = ANURASET_AUDIO_DIR
METADATA_PATH = ANURASET_METADATA_PATH
TAXONOMY_PATH = ANURASET_TAXONOMY_PATH
OUTPUT_DIR = ANURASET_EMBEDDING_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
이미 위쪽에 다음 줄이 있다면 중복되지 않도록 하나만 남기세요.

from pathlib import Path
3. ESC-50 추출 파일 수정
nano scripts/NatureLM/esc50/baseline/extract_embeddings.py
기존 경로 부분을 아래처럼 바꿉니다.

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    ESC50_AUDIO_DIR,
    ESC50_LABELS_PATH,
    ESC50_EMBEDDING_DIR,
)

AUDIO_DIR = ESC50_AUDIO_DIR
LABELS_PATH = ESC50_LABELS_PATH
OUTPUT_DIR = ESC50_EMBEDDING_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
4. 학습 파일도 같은 방식으로 수정
예를 들어 기존 코드:

sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

EMBED_PATH = "/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy"
META_PATH = "/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv"
TAXONOMY_PATH = "/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv"
SAVE_DIR = Path("/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_hssn_singlelabel_self_unknown")
를 다음처럼 바꿉니다.

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    ANURASET_EMBEDDING_DIR,
    ANURASET_TAXONOMY_PATH,
    RESULTS_DIR,
)

EMBED_PATH = ANURASET_EMBEDDING_DIR / "embeddings.npy"
META_PATH = ANURASET_EMBEDDING_DIR / "embeddings_metadata.csv"
TAXONOMY_PATH = ANURASET_TAXONOMY_PATH

SAVE_DIR = (
    RESULTS_DIR
    / "NatureLM"
    / "anuraset"
    / "wp2_hssn_singlelabel_self_unknown"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
Path 객체는 numpy, pandas, torch에서 대부분 그대로 사용할 수 있습니다.

5. 제대로 연결됐는지 확인
저장소 최상위에서:

python - <<'PY'
from config import *

print("PROJECT_DIR:", PROJECT_DIR)
print("AnuraSet audio:", ANURASET_AUDIO_DIR)
print("AnuraSet metadata:", ANURASET_METADATA_PATH)
print("AnuraSet taxonomy:", ANURASET_TAXONOMY_PATH)
print("ESC-50 audio:", ESC50_AUDIO_DIR)
print("Results:", RESULTS_DIR)
PY
파일 존재 여부까지 확인하려면:

python - <<'PY'
from config import *

paths = {
    "AnuraSet audio": ANURASET_AUDIO_DIR,
    "AnuraSet metadata": ANURASET_METADATA_PATH,
    "AnuraSet taxonomy": ANURASET_TAXONOMY_PATH,
    "ESC-50 audio": ESC50_AUDIO_DIR,
    "ESC-50 labels": ESC50_LABELS_PATH,
}

for name, path in paths.items():
    print(f"{name}: {path} | exists={path.exists()}")
PY
더 간단한 대안
모든 파일을 한꺼번에 고치는 것이 부담되면 우선 config.py를 만든 뒤, embedding 추출 파일 2개와 대표 학습 파일 1개만 먼저 수정하면 됩니다. 정상 실행되는 것을 확인한 후 나머지 파일에 같은 형태를 적용하세요.

GitHub에 공개할 때는 config.py 안에 개인 경로가 보이지 않게 다음처럼 바꾸는 것이 좋습니다.

WORK_DIR = Path.home() / "bioacoustic_data"
그러면 사용자마다 홈 디렉토리 아래에 다음 구조를 만들면 됩니다.

~/bioacoustic_data/
├── anuraset/
├── ESC-50-master/
├── embeddings/
└── embeddings_esc50/

아 지금 해야겠다. 근데 지금 파일이 엄청 많잖아. 어떻게 해? 상관없어?

파일이 많아도 괜찮습니다. 한꺼번에 전부 고치지 말고, 공통 경로 파일을 만든 다음 자동 치환하되 단계별로 검증하면 됩니다.

가장 안전한 순서는 이렇습니다.

1. 먼저 백업 커밋
git add -A
git commit -m "Backup before path refactor"
2. 절대경로가 들어간 파일부터 찾기
grep -RIn \
  --include='*.py' \
  --include='*.sh' \
  -E '/user/gaad2403|/fs/dss/home/gaad2403' \
  scripts src
파일 개수만 보려면:

grep -RIl \
  --include='*.py' \
  --include='*.sh' \
  -E '/user/gaad2403|/fs/dss/home/gaad2403' \
  scripts src | wc -l
3. 지금은 전부 고칠 필요 없음
우선 아래 세 종류만 고치면 됩니다.

embedding extraction 파일

실제로 논문 결과에 사용한 training 파일

evaluation 파일

오래된 실험 파일이나 더 이상 사용하지 않는 파일은 그대로 둬도 됩니다. 대신 README에 “research scripts may contain environment-specific paths”라고 적으면 됩니다.

4. 가장 쉬운 현실적인 방법
config.py 하나를 만들고, 활성 스크립트만 수정하세요.

예를 들어:

scripts/NatureLM/anuraset/baseline/extract_embeddings.py
scripts/NatureLM/esc50/baseline/extract_embeddings.py
scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
그리고 각 최종 실험별로 실제 사용한 학습·평가 파일만 수정합니다.

5. 어떤 파일을 고쳐야 할지 목록 만들기
절대경로가 들어간 파일을 텍스트로 저장하세요.

grep -RIl \
  --include='*.py' \
  -E '/user/gaad2403|/fs/dss/home/gaad2403' \
  scripts src \
  | sort > files_with_absolute_paths.txt
확인:

cat files_with_absolute_paths.txt
이 목록을 보고 사용하는 파일만 하나씩 체크하면 됩니다.

6. 경로만 단순 치환하는 것도 가능
예를 들어 프로젝트 경로:

/user/gaad2403/Bachelor-Thesis
를 Python에서 자동 계산하도록 바꾸는 작업은 파일마다 import 위치가 달라서 완전 자동화는 위험합니다.

반면 데이터 루트 경로는 환경변수 방식으로 비교적 쉽게 통일할 수 있습니다.

config.py:

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

WORK_DIR = Path(
    os.environ.get(
        "BIOACOUSTIC_DATA_DIR",
        "/user/gaad2403/bachelor_thesis",
    )
)

ANURASET_DIR = WORK_DIR / "anuraset/anuraset"
ANURASET_AUDIO_DIR = ANURASET_DIR / "audio"
ANURASET_METADATA_PATH = ANURASET_DIR / "metadata.csv"
ANURASET_TAXONOMY_PATH = PROJECT_DIR / "data/anuraset_taxonomy.csv"
ANURASET_EMBEDDING_DIR = WORK_DIR / "embeddings"

ESC50_DIR = WORK_DIR / "ESC-50-master"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_LABELS_PATH = PROJECT_DIR / "data/esc50_bga_labels.csv"
ESC50_EMBEDDING_DIR = WORK_DIR / "embeddings_esc50"

RESULTS_DIR = PROJECT_DIR / "results"
그러면 현재 서버에서는 별도 설정 없이 기존 경로가 사용됩니다.

다른 사용자는 이렇게 실행할 수 있습니다.

export BIOACOUSTIC_DATA_DIR=/path/to/bioacoustic_data
python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
7. 전부 수정하지 않아도 GitHub 공개 가능
논문 코드 저장소라면 다음 조건만 충족하면 충분합니다.

대표 실행 경로가 작동함

README에 경로 수정법이 있음

최종 실험 스크립트가 구분됨

오래된 스크립트는 legacy/ 또는 archive/로 이동하거나 설명함

가장 권장하는 구조는:

scripts/
├── main/
│   ├── extract/
│   ├── train/
│   └── evaluate/
└── legacy/
하지만 지금 당장 구조까지 바꿀 필요는 없습니다.

결론적으로 파일이 많아도 상관없고, 실제 논문 결과에 사용한 핵심 파일만 우선 수정하면 됩니다. 나머지는 나중에 정리해도 됩니다.



scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py

글쓰기 시작

scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py



붙여넣은 텍스트 (1)(14).txt
문서
이렇게 많은데 이거 하는법 알려줘

파일이 너무 많아서 하나씩 수정하면 안 됩니다. 현재 절대경로가 NatureLM, BirdNET, single-label, multi-label 파일 전반에 반복되어 있으므로 자동 변환하는 편이 낫습니다. 


아래 방법은 .py 파일만 수정합니다. .sh 파일을 GitHub에 올리지 않을 예정이라면 .sh는 그대로 두어도 됩니다.

1. 먼저 백업
git add -A
git commit -m "Backup before replacing absolute paths"
2. 자동 변환 프로그램 만들기
저장소 최상위 Bachelor-Thesis에서 실행하세요.

nano refactor_paths.py
아래 내용을 전부 붙여넣으세요.

#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


REPO_ROOT = Path.cwd().resolve()

# GitHub 저장소 내부 경로
PROJECT_PREFIXES = [
    "/user/gaad2403/Bachelor-Thesis",
    "/fs/dss/home/gaad2403/Bachelor-Thesis",
]

# 데이터, embeddings, 환경 등이 있는 외부 작업 경로
WORK_PREFIXES = [
    "/user/gaad2403/bachelor_thesis",
    "/fs/dss/home/gaad2403/bachelor_thesis",
]


def replacement_expression(variable: str, relative_path: str) -> str:
    """Path expression을 문자열로 반환합니다."""
    relative_path = relative_path.lstrip("/")

    if not relative_path:
        return f"str({variable})"

    return f'str({variable} / "{relative_path}")'


def replace_quoted_paths(text: str) -> tuple[str, int]:
    """
    작은따옴표 또는 큰따옴표 안의 절대경로만 변경합니다.

    예:
    "/user/gaad2403/Bachelor-Thesis/data/file.csv"

    변경 후:
    str(PROJECT_DIR / "data/file.csv")
    """
    count = 0

    all_prefixes = [
        *(("PROJECT_DIR", prefix) for prefix in PROJECT_PREFIXES),
        *(("WORK_DIR", prefix) for prefix in WORK_PREFIXES),
    ]

    for variable, prefix in all_prefixes:
        pattern = re.compile(
            rf"""(?P<quote>["']){re.escape(prefix)}(?P<relative>[^"']*)(?P=quote)"""
        )

        def repl(match: re.Match) -> str:
            nonlocal count
            count += 1
            relative = match.group("relative")
            return replacement_expression(variable, relative)

        text = pattern.sub(repl, text)

    return text, count


def has_definition(text: str, variable: str) -> bool:
    return bool(re.search(rf"^\s*{variable}\s*=", text, flags=re.MULTILINE))


def find_insertion_index(lines: list[str]) -> int:
    """
    shebang, 모듈 docstring, from __future__ import 뒤에
    공통 경로 설정을 삽입합니다.
    """
    index = 0

    if lines and lines[0].startswith("#!"):
        index = 1

    # 앞부분의 빈 줄과 주석 통과
    while index < len(lines) and (
        not lines[index].strip() or lines[index].lstrip().startswith("#")
    ):
        index += 1

    # 단순한 모듈 docstring 처리
    if index < len(lines):
        stripped = lines[index].lstrip()

        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]

            # 한 줄짜리 docstring
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                index += 1
            else:
                index += 1
                while index < len(lines):
                    if quote in lines[index]:
                        index += 1
                        break
                    index += 1

    # from __future__ import는 항상 다른 import보다 먼저 있어야 함
    while index < len(lines):
        stripped = lines[index].strip()

        if stripped.startswith("from __future__ import"):
            index += 1
        elif not stripped:
            index += 1
        else:
            break

    return index


def add_path_configuration(text: str) -> str:
    additions = []

    if not re.search(r"^\s*import os\s*$", text, flags=re.MULTILINE):
        additions.append("import os")

    if not re.search(
        r"^\s*from pathlib import .*Path", text, flags=re.MULTILINE
    ):
        additions.append("from pathlib import Path")

    if not has_definition(text, "PROJECT_DIR"):
        additions.extend(
            [
                "",
                "# Repository root: directory containing src/, scripts/, and data/",
                "PROJECT_DIR = next(",
                "    parent",
                "    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]",
                '    if (parent / "src").exists() and (parent / "scripts").exists()',
                ")",
            ]
        )

    if not has_definition(text, "WORK_DIR"):
        additions.extend(
            [
                "",
                "# External datasets and generated embeddings",
                "WORK_DIR = Path(",
                "    os.environ.get(",
                '        "BIOACOUSTIC_WORK_DIR",',
                '        "/user/gaad2403/bachelor_thesis",',
                "    )",
                ")",
            ]
        )

    if not additions:
        return text

    lines = text.splitlines(keepends=True)
    insertion_index = find_insertion_index(lines)

    block = "\n".join(additions).strip() + "\n\n"
    lines.insert(insertion_index, block)

    return "".join(lines)


def process_file(path: Path, apply_changes: bool) -> int:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[SKIP: encoding] {path}")
        return 0

    updated, replacements = replace_quoted_paths(original)

    if replacements == 0:
        return 0

    updated = add_path_configuration(updated)

    relative = path.relative_to(REPO_ROOT)

    if apply_changes:
        path.write_text(updated, encoding="utf-8")
        print(f"[UPDATED] {relative}: {replacements} path(s)")
    else:
        print(f"[DRY RUN] {relative}: {replacements} path(s)")

    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace personal absolute paths in Python scripts."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files. Without this option, only previews changes.",
    )
    args = parser.parse_args()

    targets = [
        *REPO_ROOT.joinpath("scripts").rglob("*.py"),
        *REPO_ROOT.joinpath("src").rglob("*.py"),
    ]

    total_files = 0
    total_paths = 0

    for path in sorted(set(targets)):
        replacements = process_file(path, args.apply)

        if replacements:
            total_files += 1
            total_paths += replacements

    mode = "UPDATED" if args.apply else "DRY RUN"

    print()
    print(f"{mode}: {total_files} files, {total_paths} absolute paths")

    if not args.apply and total_files:
        print("Run again with --apply to modify the files.")


if __name__ == "__main__":
    main()
저장합니다.

Ctrl+O
Enter
Ctrl+X
3. 먼저 미리보기만 실행
이 명령은 파일을 수정하지 않습니다.

python refactor_paths.py
출력 예시는 다음과 같습니다.

[DRY RUN] scripts/NatureLM/anuraset/evaluation.py: 9 path(s)
[DRY RUN] scripts/birdnet/anuraset/evaluation.py: 9 path(s)

DRY RUN: 48 files, 210 absolute paths
4. 실제로 변환
미리보기 결과가 정상이라면:

python refactor_paths.py --apply
예를 들어 기존 코드:

sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

EMBED_PATH = "/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy"
TAXONOMY_PATH = "/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv"
는 다음처럼 바뀝니다.

sys.path.append(str(PROJECT_DIR / "src"))

EMBED_PATH = str(WORK_DIR / "embeddings/embeddings.npy")
TAXONOMY_PATH = str(PROJECT_DIR / "data/anuraset_taxonomy.csv")
각 수정 파일에는 다음 설정도 자동으로 들어갑니다.

import os
from pathlib import Path

PROJECT_DIR = next(
    parent
    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (parent / "src").exists() and (parent / "scripts").exists()
)

WORK_DIR = Path(
    os.environ.get(
        "BIOACOUSTIC_WORK_DIR",
        "/user/gaad2403/bachelor_thesis",
    )
)
5. 개인 절대경로가 남았는지 확인
grep -RIn \
  --include='*.py' \
  -E '/user/gaad2403|/fs/dss/home/gaad2403' \
  scripts src
기본값으로 남겨 둔 다음 한 줄은 출력될 수 있습니다.

"/user/gaad2403/bachelor_thesis"
이것은 현재 서버에서 계속 실행할 수 있도록 둔 기본값입니다. GitHub에 개인 경로를 전혀 남기고 싶다면 자동 변환 후 다음 명령을 실행하세요.

sed -i \
  's|"/user/gaad2403/bachelor_thesis"|str(Path.home() / "bioacoustic_data")|g' \
  $(grep -RIl --include='*.py' '/user/gaad2403/bachelor_thesis' scripts src)
그러면 기본 경로가 다음으로 바뀝니다.

WORK_DIR = Path(
    os.environ.get(
        "BIOACOUSTIC_WORK_DIR",
        str(Path.home() / "bioacoustic_data"),
    )
)
6. 문법 오류 확인
python -m compileall -q scripts src
오류가 없으면 아무 출력이 없거나 정상 종료됩니다.

생성된 캐시를 삭제합니다.

find scripts src -type d -name '__pycache__' -prune -exec rm -rf {} +
find scripts src -type f -name '*.pyc' -delete
7. 수정 결과 확인
git diff --stat
git diff -- scripts/NatureLM/anuraset/evaluation.py
문제가 있다면 백업 상태로 전부 되돌릴 수 있습니다.

git reset --hard HEAD
정상이면 커밋합니다.

git add scripts src refactor_paths.py
git commit -m "Replace personal absolute paths with portable paths"
다른 컴퓨터에서 실행할 때
사용자는 데이터와 embeddings가 있는 폴더만 환경변수로 지정하면 됩니다.

export BIOACOUSTIC_WORK_DIR=/path/to/bioacoustic_data
python scripts/NatureLM/anuraset/evaluation.py
현재 서버에서는 기존 위치를 사용할 경우:

export BIOACOUSTIC_WORK_DIR=/user/gaad2403/bachelor_thesis
중요한 점은 자동 변환 후 바로 모든 실험을 다시 돌릴 필요는 없다는 것입니다. 우선 compileall 검사 후 대표 파일 하나만 실행해 확인하면 됩니다.


(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ python refactor_paths.py
[DRY RUN] scripts/NatureLM/anuraset/baseline/baseline_probing_single.py: 2 path(s)
[DRY RUN] scripts/NatureLM/anuraset/baseline/extract_embeddings.py: 2 path(s)
[DRY RUN] scripts/NatureLM/anuraset/baseline/test_embedding.py: 1 path(s)
[DRY RUN] scripts/NatureLM/anuraset/evaluation.py: 9 path(s)
[DRY RUN] scripts/NatureLM/anuraset/evaluation_unknown.py: 9 path(s)
[DRY RUN] scripts/NatureLM/anuraset/evaluation_unseen.py: 10 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/eval_tree_triplet.py: 6 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/eval_tree_triplet_abalation.py: 8 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py: 5 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only.py: 5 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only_unseen.py: 6 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only.py: 5 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only_unseen.py: 6 path(s)
[DRY RUN] scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_unseen.py: 6 path(s)
[DRY RUN] scripts/NatureLM/anuraset/ualb/eval_ualb_single.py: 5 path(s)
[DRY RUN] scripts/NatureLM/anuraset/ualb/train_ualb_single.py: 4 path(s)
[DRY RUN] scripts/NatureLM/anuraset/ualb/train_ualb_single_unseen.py: 5 path(s)
[DRY RUN] scripts/NatureLM/esc50/baseline/baseline_probing.py: 3 path(s)
[DRY RUN] scripts/NatureLM/esc50/baseline/build_bga_labels.py: 2 path(s)
[DRY RUN] scripts/NatureLM/esc50/baseline/extract_embeddings.py: 2 path(s)
[DRY RUN] scripts/NatureLM/esc50/evaluation.py: 8 path(s)
[DRY RUN] scripts/NatureLM/esc50/evaluation_unseen.py: 9 path(s)
[DRY RUN] scripts/NatureLM/esc50/tree_triplet/train_triplet_single.py: 4 path(s)
[DRY RUN] scripts/NatureLM/esc50/tree_triplet/train_triplet_single_ftm_only.py: 4 path(s)
[DRY RUN] scripts/NatureLM/esc50/tree_triplet/train_triplet_single_ftm_only_unseen.py: 5 path(s)
[DRY RUN] scripts/NatureLM/esc50/tree_triplet/train_triplet_single_tt_only.py: 4 path(s)
[DRY RUN] scripts/NatureLM/esc50/tree_triplet/train_triplet_single_tt_only_unseen.py: 5 path(s)
[DRY RUN] scripts/NatureLM/esc50/tree_triplet/train_triplet_single_unseen.py: 5 path(s)
[DRY RUN] scripts/NatureLM/esc50/ualb/eval_ualb.py: 5 path(s)
[DRY RUN] scripts/NatureLM/esc50/ualb/train_ualb.py: 4 path(s)
[DRY RUN] scripts/NatureLM/esc50/ualb/train_ualb_unseen.py: 5 path(s)
[DRY RUN] scripts/birdnet/anuraset/baseline/baseline_probing_single.py: 2 path(s)
[DRY RUN] scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py: 2 path(s)
[DRY RUN] scripts/birdnet/anuraset/evaluation.py: 9 path(s)
[DRY RUN] scripts/birdnet/anuraset/evaluation_unknown.py: 9 path(s)
[DRY RUN] scripts/birdnet/anuraset/evaluation_unseen.py: 10 path(s)
[DRY RUN] scripts/birdnet/anuraset/tree_triplet/eval_tree_triplet.py: 6 path(s)
[DRY RUN] scripts/birdnet/anuraset/tree_triplet/train_triplet_single.py: 5 path(s)
[DRY RUN] scripts/birdnet/anuraset/tree_triplet/train_triplet_single_ftm_only.py: 5 path(s)
[DRY RUN] scripts/birdnet/anuraset/tree_triplet/train_triplet_single_ftm_only_unseen.py: 6 path(s)
[DRY RUN] scripts/birdnet/anuraset/tree_triplet/train_triplet_single_tt_only.py: 5 path(s)
[DRY RUN] scripts/birdnet/anuraset/tree_triplet/train_triplet_single_tt_only_unseen.py: 6 path(s)
[DRY RUN] scripts/birdnet/anuraset/tree_triplet/train_triplet_single_unseen.py: 6 path(s)
[DRY RUN] scripts/birdnet/anuraset/ualb/eval_ualb_single.py: 5 path(s)
[DRY RUN] scripts/birdnet/anuraset/ualb/train_ualb_single.py: 4 path(s)
[DRY RUN] scripts/birdnet/anuraset/ualb/train_ualb_single_unseen.py: 5 path(s)
[DRY RUN] scripts/birdnet/esc50/baseline/baseline_probing_birdnet.py: 3 path(s)
[DRY RUN] scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py: 2 path(s)
[DRY RUN] scripts/birdnet/esc50/evaluation.py: 8 path(s)
[DRY RUN] scripts/birdnet/esc50/evaluation_unseen.py: 9 path(s)
[DRY RUN] scripts/birdnet/esc50/tree_triplet/train_triplet_single.py: 4 path(s)
[DRY RUN] scripts/birdnet/esc50/tree_triplet/train_triplet_single_ftm_only.py: 4 path(s)
[DRY RUN] scripts/birdnet/esc50/tree_triplet/train_triplet_single_ftm_only_unseen.py: 5 path(s)
[DRY RUN] scripts/birdnet/esc50/tree_triplet/train_triplet_single_tt_only.py: 4 path(s)
[DRY RUN] scripts/birdnet/esc50/tree_triplet/train_triplet_single_tt_only_unseen.py: 5 path(s)
[DRY RUN] scripts/birdnet/esc50/tree_triplet/train_triplet_single_unseen.py: 5 path(s)
[DRY RUN] scripts/birdnet/esc50/ualb/eval_ualb.py: 5 path(s)
[DRY RUN] scripts/birdnet/esc50/ualb/train_ualb.py: 4 path(s)
[DRY RUN] scripts/birdnet/esc50/ualb/train_ualb_unseen.py: 5 path(s)
[DRY RUN] scripts/birdnet/test_birdnet_api.py: 1 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/baseline/baseline_probing_multi.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/baseline/test.py: 4 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/evaluation.py: 10 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py: 10 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_ftm.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_ftm_unknown.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_ftm_unseen.py: 6 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_tt.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_tt_unknown.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_tt_unseen.py: 6 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_unknown.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/tree_triplet/train_triplet_multi_unseen.py: 6 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/ualb/eval_ualb_multi.py: 6 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/ualb/train_ualb_multi.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/ualb/train_ualb_multi_unknown.py: 5 path(s)
[DRY RUN] scripts/multilabel/NatureLM/anuraset/ualb/train_ualb_multi_unseen.py: 6 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/baseline/baseline_probing_multi.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/evaluation.py: 10 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/evaluation_unknown.py: 10 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/evaluation_unknown_neg.py: 8 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/tree_triplet/train_triplet_multi.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/tree_triplet/train_triplet_multi_ftm.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/tree_triplet/train_triplet_multi_ftm_unknown.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/tree_triplet/train_triplet_multi_tt.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/tree_triplet/train_triplet_multi_tt_unknown.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/tree_triplet/train_triplet_multi_unknown.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/ualb/eval_ualb_multi.py: 6 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/ualb/train_ualb_multi.py: 5 path(s)
[DRY RUN] scripts/multilabel/birdnet/anuraset/ualb/train_ualb_multi_unknown.py: 5 path(s)
[DRY RUN] src/NatureLM/anuraset/unseen_eval.py: 6 path(s)
[DRY RUN] src/NatureLM/anuraset/unseen_test.py: 3 path(s)
[DRY RUN] src/NatureLM/esc50/unseen_test.py: 2 path(s)
[DRY RUN] src/multilabel/NatureLM/anuraset/anuraset_dataset.py: 7 path(s)
[DRY RUN] src/multilabel/multilabel_hierarchy_tree.py: 5 path(s)
[DRY RUN] src/multilabel/species_distance_matrix.py: 5 path(s)

DRY RUN: 96 files, 514 absolute paths
Run again with --apply to modify the files.

이렇게 나왔어

정상적으로 잘 된 것입니다. 아직 DRY RUN이라 실제 파일은 하나도 수정되지 않았고, 수정 대상이 96개 파일, 절대경로 514개라는 뜻입니다.

다음 순서대로 진행하세요.

1. 백업 커밋 확인
먼저 현재 상태를 저장합니다.

git status
git add -A
git commit -m "Backup before path refactor"
nothing to commit이 나오면 이미 백업된 상태입니다.

2. 실제 변환 실행
python refactor_paths.py --apply
3. 대표 파일 하나 확인
git diff -- scripts/NatureLM/anuraset/evaluation.py
아래와 비슷하게 바뀌었는지 보세요.

sys.path.append(str(PROJECT_DIR / "src"))

EMBED_PATH = str(WORK_DIR / "embeddings/embeddings.npy")
META_PATH = str(WORK_DIR / "embeddings/embeddings_metadata.csv")
TAXONOMY_PATH = str(PROJECT_DIR / "data/anuraset_taxonomy.csv")
결과 경로는 다음처럼 되어야 합니다.

OUTPUT_DIR = Path(
    str(PROJECT_DIR / "results/NatureLM/anuraset/wp3_eval_total_hf1")
)
Path(str(...)) 형태는 조금 중복되지만 정상적으로 작동합니다.

4. Python 문법 검사
python -m compileall -q scripts src
echo $?
마지막 출력이:

0
이면 문법 검사를 통과한 것입니다.

오류가 생기면 오류 메시지에 표시된 파일을 보내주면 됩니다. 아직 커밋 전이라 쉽게 되돌릴 수 있습니다.

5. 개인 경로가 얼마나 남았는지 확인
grep -RIn \
  --include='*.py' \
  -E '/user/gaad2403|/fs/dss/home/gaad2403' \
  scripts src
대부분 없어져야 합니다. 다만 각 파일에 자동으로 들어간 기본값:

"/user/gaad2403/bachelor_thesis"
은 남아 있을 수 있습니다. 이것은 현재 HPC에서 기존처럼 실행되도록 일부러 남긴 것입니다.

개인 경로를 GitHub에서 완전히 없애고 싶다면 나중에 기본값을 다음처럼 바꾸면 됩니다.

WORK_DIR = Path(
    os.environ.get(
        "BIOACOUSTIC_WORK_DIR",
        str(Path.home() / "bioacoustic_data"),
    )
)
하지만 지금은 기존 실험 실행을 유지하려면 그대로 두는 편이 안전합니다.

6. 변경량 확인
git diff --stat
git status
96개 파일이 수정되었다고 나오는 것이 정상입니다.

7. 대표 스크립트 경로만 시험
실제 학습을 돌릴 필요 없이 경로 설정 부분만 시험할 수 있습니다.

python - <<'PY'
import os
from pathlib import Path

os.environ["BIOACOUSTIC_WORK_DIR"] = "/user/gaad2403/bachelor_thesis"

WORK_DIR = Path(os.environ["BIOACOUSTIC_WORK_DIR"])
PROJECT_DIR = Path.cwd()

print("PROJECT_DIR:", PROJECT_DIR)
print("WORK_DIR:", WORK_DIR)
print("Embedding exists:", (WORK_DIR / "embeddings/embeddings.npy").exists())
print("Taxonomy exists:", (PROJECT_DIR / "data/anuraset_taxonomy.csv").exists())
PY
두 파일이 실제로 있다면 다음처럼 나와야 합니다.

Embedding exists: True
Taxonomy exists: True
8. 이상 없으면 커밋
find scripts src -type d -name '__pycache__' -prune -exec rm -rf {} +
find scripts src -type f -name '*.pyc' -delete

git add scripts src refactor_paths.py
git commit -m "Replace absolute paths with portable project paths"
문제가 생기면 커밋 전에 다음 명령으로 전부 원래 상태로 되돌릴 수 있습니다.

git reset --hard HEAD
현재 출력 자체는 문제 없습니다. 다음 명령은 python refactor_paths.py --apply입니다.


(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ git diff -- scripts/NatureLM/anuraset/evaluation.py
diff --git a/scripts/NatureLM/anuraset/evaluation.py b/scripts/NatureLM/anuraset/evaluation.py
index a8d1b07..a8694ee 100644
--- a/scripts/NatureLM/anuraset/evaluation.py
+++ b/scripts/NatureLM/anuraset/evaluation.py
@@ -2,8 +2,25 @@
 WP3 full evaluation on AnuraSet (NatureLM embeddings, single-label subset, NORMAL/42-species) + hF1.
 """
 
+import os
+
+# Repository root: directory containing src/, scripts/, and data/
+PROJECT_DIR = next(
+    parent
+    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
+    if (parent / "src").exists() and (parent / "scripts").exists()
+)
+
+# External datasets and generated embeddings

이렇게 나와

지금 상태는 수정이 필요합니다. PROJECT_DIR에서 Path를 사용하는데, from pathlib import Path가 그 아래에 있다면 실행할 때 NameError가 납니다.

먼저 확인해보세요.

sed -n '1,45p' scripts/NatureLM/anuraset/evaluation.py
현재 대략 이렇게 되어 있을 가능성이 큽니다.

import os

PROJECT_DIR = next(
    parent
    for parent in [Path(__file__).resolve().parent, ...]
)

# 아래쪽에 있음
from pathlib import Path
Path를 사용하기 전에 import해야 합니다.

가장 안전한 해결 방법
아직 커밋하지 않았다면 전체 자동 변경을 되돌리세요.

git reset --hard HEAD
refactor_paths.py가 커밋되지 않은 새 파일이었다면 이 파일은 남을 수 있습니다.

그다음 refactor_paths.py의 add_path_configuration() 함수에서 다음 부분을 찾으세요.

if not re.search(
    r"^\s*from pathlib import .*Path", text, flags=re.MULTILINE
):
    additions.append("from pathlib import Path")
이를 아래처럼 바꿉니다.

# 경로 설정 블록 안에서 Path를 즉시 사용할 수 있도록 항상 import
additions.append("from pathlib import Path")
그리고 중복 import가 생겨도 Python 실행에는 문제가 없지만, 깔끔하게 하려면 기존 from pathlib import Path를 제거하도록 아래 줄도 추가해야 합니다. 더 쉬운 방법은 add_path_configuration() 함수 전체를 다음처럼 수정하는 것입니다.

def add_path_configuration(text: str) -> str:
    # 기존 pathlib import를 먼저 제거합니다.
    text = re.sub(
        r"^\s*from pathlib import\s+Path\s*\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    additions = []

    if not re.search(r"^\s*import os\s*$", text, flags=re.MULTILINE):
        additions.append("import os")

    # PROJECT_DIR에서 Path를 사용하기 전에 반드시 import
    additions.append("from pathlib import Path")

    if not has_definition(text, "PROJECT_DIR"):
        additions.extend(
            [
                "",
                "# Repository root: directory containing src/, scripts/, and data/",
                "PROJECT_DIR = next(",
                "    parent",
                "    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]",
                '    if (parent / "src").exists() and (parent / "scripts").exists()',
                ")",
            ]
        )

    if not has_definition(text, "WORK_DIR"):
        additions.extend(
            [
                "",
                "# External datasets and generated embeddings",
                "WORK_DIR = Path(",
                "    os.environ.get(",
                '        "BIOACOUSTIC_WORK_DIR",',
                '        "/user/gaad2403/bachelor_thesis",',
                "    )",
                ")",
            ]
        )

    lines = text.splitlines(keepends=True)
    insertion_index = find_insertion_index(lines)

    block = "\n".join(additions).strip() + "\n\n"
    lines.insert(insertion_index, block)

    return "".join(lines)
다시 미리보기 후 적용하세요.

python refactor_paths.py
python refactor_paths.py --apply
대표 파일을 다시 확인합니다.

sed -n '1,40p' scripts/NatureLM/anuraset/evaluation.py
정상적인 순서는 아래와 같아야 합니다.

import os
from pathlib import Path

PROJECT_DIR = next(
    parent
    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (parent / "src").exists() and (parent / "scripts").exists()
)

WORK_DIR = Path(
    os.environ.get(
        "BIOACOUSTIC_WORK_DIR",
        "/user/gaad2403/bachelor_thesis",
    )
)
마지막으로 전체 문법 검사를 하세요.

python -m compileall -q scripts src
echo $?
0이 나오면 문법상 정상입니다. 현재 보인 diff 상태로는 바로 커밋하지 않는 것이 좋습니다.


(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ sed -n '1,40p' scripts/NatureLM/anuraset/evaluation.py
"""
WP3 full evaluation on AnuraSet (NatureLM embeddings, single-label subset, NORMAL/42-species) + hF1.
"""

import os
from pathlib import Path

# Repository root: directory containing src/, scripts/, and data/
PROJECT_DIR = next(
    parent
    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (parent / "src").exists() and (parent / "scripts").exists()
)

# External datasets and generated embeddings
WORK_DIR = Path(
    os.environ.get(
        "BIOACOUSTIC_WORK_DIR",
        "/user/gaad2403/bachelor_thesis",
    )
)

import sys
sys.path.append(str(PROJECT_DIR / "src"))

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from NatureLM.anuraset.anuraset_dataset import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel
from hf1 import (joint_probe_singlelabel, ftm_only_preds_singlelabel,
                                    ground_truth_singlelabel, hierarchical_f1)

EMBED_PATH    = str(WORK_DIR / "embeddings/embeddings.npy")
(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ python -m compileall -q scripts src
(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ echo $?
0

정상입니다. Path가 사용되기 전에 import되었고, 전체 compileall 검사도 종료코드 0으로 통과했습니다.

이제 아래만 진행하면 됩니다.

grep -RIn \
  --include='*.py' \
  -E '/user/gaad2403|/fs/dss/home/gaad2403' \
  scripts src
여기서 각 파일의 BIOACOUSTIC_WORK_DIR 기본값만 남는 것은 정상입니다.

그다음 변경량을 확인하세요.

git diff --stat
git status
문제가 없으면 커밋합니다.

find scripts src -type d -name '__pycache__' -prune -exec rm -rf {} +
find scripts src -type f -name '*.pyc' -delete

git add scripts src refactor_paths.py
git commit -m "Replace absolute paths with portable project paths"
현재 서버에서 실행할 때는 기존 기본 경로가 자동으로 사용됩니다. 다른 환경에서는 다음처럼 지정하면 됩니다.

export BIOACOUSTIC_WORK_DIR=/path/to/bioacoustic_data
그리고 실행:

python scripts/NatureLM/anuraset/evaluation.py
지금 상태면 다음 단계로 README에 환경변수 사용법을 반영해도 됩니다.


지금은 yaml이나 config로 한게 아니야? 나 그렇게 하고 싶은데? 아직 커밋 안했어

맞아요. 지금 적용한 방식은 YAML이나 공통 config 파일이 아니라, 각 Python 파일에 PROJECT_DIR, WORK_DIR 설정을 반복 삽입한 방식입니다.

아직 커밋하지 않았으니 지금 되돌리고, config.yaml 하나 + 공통 로더 하나로 바꾸는 것이 훨씬 깔끔합니다.

1. 방금 자동 수정한 내용 되돌리기
git reset --hard HEAD
refactor_paths.py가 새로 만든 추적되지 않은 파일이면 그대로 남습니다. 확인하세요.

git status --short
필요 없으면 삭제합니다.

rm -f refactor_paths.py
2. config.yaml 만들기
저장소 최상위에서:

nano config.yaml
다음을 붙여넣으세요.

project:
  root: /user/gaad2403/Bachelor-Thesis

data:
  root: /user/gaad2403/bachelor_thesis

  anuraset:
    audio_dir: anuraset/anuraset/audio
    metadata_path: anuraset/anuraset/metadata.csv
    taxonomy_path: data/anuraset_taxonomy.csv
    taxonomy_unknown_path: data/anuraset_taxonomy_unknown.csv
    unseen_split_path: data/unseen_species_split.json

  esc50:
    audio_dir: ESC-50-master/audio
    labels_path: data/esc50_bga_labels.csv
    unseen_split_path: data/unseen_category_split.json

embeddings:
  naturelm:
    anuraset: embeddings
    esc50: embeddings_esc50

  birdnet:
    anuraset: embeddings_anuraset_birdnet
    esc50: embeddings_esc50_birdnet

results:
  root: results
여기서:

project.root는 Git 저장소 위치

data.root는 원본 데이터와 임베딩이 있는 작업 디렉토리입니다.

3. YAML 패키지 설치
pip install pyyaml
4. 공통 설정 로더 만들기
nano src/config.py
아래 내용을 넣으세요.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load repository configuration from YAML."""

    selected_path = Path(
        config_path
        or os.environ.get("BIOACOUSTIC_CONFIG", DEFAULT_CONFIG_PATH)
    ).expanduser().resolve()

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {selected_path}"
        )

    with selected_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    return config


CONFIG = load_config()

# Repository root
PROJECT_ROOT = Path(
    os.environ.get(
        "BIOACOUSTIC_PROJECT_ROOT",
        CONFIG.get("project", {}).get("root", PROJECT_DIR),
    )
).expanduser()

# Raw datasets and generated embeddings
DATA_ROOT = Path(
    os.environ.get(
        "BIOACOUSTIC_DATA_ROOT",
        CONFIG["data"]["root"],
    )
).expanduser()


def project_path(*parts: str) -> Path:
    """Return a path relative to the Git repository."""

    return PROJECT_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Return a path relative to the external data directory."""

    return DATA_ROOT.joinpath(*parts)


# AnuraSet
ANURASET_AUDIO_DIR = data_path(
    CONFIG["data"]["anuraset"]["audio_dir"]
)
ANURASET_METADATA_PATH = data_path(
    CONFIG["data"]["anuraset"]["metadata_path"]
)
ANURASET_TAXONOMY_PATH = project_path(
    CONFIG["data"]["anuraset"]["taxonomy_path"]
)
ANURASET_TAXONOMY_UNKNOWN_PATH = project_path(
    CONFIG["data"]["anuraset"]["taxonomy_unknown_path"]
)
ANURASET_UNSEEN_SPLIT_PATH = project_path(
    CONFIG["data"]["anuraset"]["unseen_split_path"]
)

# ESC-50
ESC50_AUDIO_DIR = data_path(
    CONFIG["data"]["esc50"]["audio_dir"]
)
ESC50_LABELS_PATH = project_path(
    CONFIG["data"]["esc50"]["labels_path"]
)
ESC50_UNSEEN_SPLIT_PATH = project_path(
    CONFIG["data"]["esc50"]["unseen_split_path"]
)

# Embeddings
NATURELM_ANURASET_EMBED_DIR = data_path(
    CONFIG["embeddings"]["naturelm"]["anuraset"]
)
NATURELM_ESC50_EMBED_DIR = data_path(
    CONFIG["embeddings"]["naturelm"]["esc50"]
)

BIRDNET_ANURASET_EMBED_DIR = data_path(
    CONFIG["embeddings"]["birdnet"]["anuraset"]
)
BIRDNET_ESC50_EMBED_DIR = data_path(
    CONFIG["embeddings"]["birdnet"]["esc50"]
)

# Results
RESULTS_DIR = project_path(CONFIG["results"]["root"])
5. 설정이 제대로 읽히는지 확인
저장소 최상위에서:

PYTHONPATH=src python - <<'PY'
from config import (
    PROJECT_ROOT,
    DATA_ROOT,
    ANURASET_AUDIO_DIR,
    ANURASET_METADATA_PATH,
    ANURASET_TAXONOMY_PATH,
    NATURELM_ANURASET_EMBED_DIR,
    BIRDNET_ANURASET_EMBED_DIR,
    ESC50_AUDIO_DIR,
    ESC50_LABELS_PATH,
    RESULTS_DIR,
)

paths = {
    "PROJECT_ROOT": PROJECT_ROOT,
    "DATA_ROOT": DATA_ROOT,
    "ANURASET_AUDIO_DIR": ANURASET_AUDIO_DIR,
    "ANURASET_METADATA_PATH": ANURASET_METADATA_PATH,
    "ANURASET_TAXONOMY_PATH": ANURASET_TAXONOMY_PATH,
    "NATURELM_ANURASET_EMBED_DIR": NATURELM_ANURASET_EMBED_DIR,
    "BIRDNET_ANURASET_EMBED_DIR": BIRDNET_ANURASET_EMBED_DIR,
    "ESC50_AUDIO_DIR": ESC50_AUDIO_DIR,
    "ESC50_LABELS_PATH": ESC50_LABELS_PATH,
    "RESULTS_DIR": RESULTS_DIR,
}

for name, path in paths.items():
    print(f"{name}: {path}")
    print(f"  exists: {path.exists()}")
PY
6. 개별 스크립트에서 사용하는 방법
예를 들어 NatureLM AnuraSet embedding 추출 파일의 기존 경로 설정:

BASE_DIR = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR = BASE_DIR / "anuraset/anuraset/audio"
METADATA_PATH = BASE_DIR / "anuraset/anuraset/metadata.csv"
TAXONOMY_PATH = Path(
    "/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv"
)
OUTPUT_DIR = BASE_DIR / "embeddings"
을 다음처럼 바꿉니다.

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import (
    ANURASET_AUDIO_DIR,
    ANURASET_METADATA_PATH,
    ANURASET_TAXONOMY_PATH,
    NATURELM_ANURASET_EMBED_DIR,
)

AUDIO_DIR = ANURASET_AUDIO_DIR
METADATA_PATH = ANURASET_METADATA_PATH
TAXONOMY_PATH = ANURASET_TAXONOMY_PATH
OUTPUT_DIR = NATURELM_ANURASET_EMBED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ESC-50 NatureLM 추출 파일은:

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import (
    ESC50_AUDIO_DIR,
    ESC50_LABELS_PATH,
    NATURELM_ESC50_EMBED_DIR,
)

AUDIO_DIR = ESC50_AUDIO_DIR
LABELS_PATH = ESC50_LABELS_PATH
OUTPUT_DIR = NATURELM_ESC50_EMBED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
7. 학습 파일에서는 이렇게 사용
기존:

sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

EMBED_PATH = "/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy"
META_PATH = "/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv"
TAXONOMY_PATH = "/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv"
SAVE_DIR = Path(
    "/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/..."
)
변경:

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import (
    ANURASET_TAXONOMY_PATH,
    NATURELM_ANURASET_EMBED_DIR,
    RESULTS_DIR,
)

EMBED_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
META_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"
TAXONOMY_PATH = ANURASET_TAXONOMY_PATH

SAVE_DIR = (
    RESULTS_DIR
    / "NatureLM"
    / "anuraset"
    / "wp2_hssn_singlelabel_self_unknown"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
8. 다른 사람이 자신의 설정을 사용하려면
GitHub에는 개인용 config.yaml을 직접 올리지 않고 예시 파일을 올리는 방식이 좋습니다.

cp config.yaml config.example.yaml
.gitignore에는:

config.yaml
Git에는 예시만 올립니다.

git add config.example.yaml src/config.py .gitignore
다른 사용자는:

cp config.example.yaml config.yaml
nano config.yaml
로 자기 경로를 수정하면 됩니다.

현재는 96개 파일을 한 번에 다시 건드리지 말고, 먼저 config.yaml과 src/config.py를 만든 뒤 대표 파일 1개만 변경해서 실행 확인하는 것이 안전합니다. 정상 작동하면 같은 패턴으로 자동 변환하면 됩니다.


아 잠시만 근데 그 전에 너가 내 data에 대해서 완전히 알고 있는게 필요할것 같아. 내가 몇개 .py줄테니까 분석해서 ymal 만들자 어때? 

좋아요. 그게 가장 안전합니다.

.py 파일들을 보내주시면 제가 먼저 다음을 정리하겠습니다.

실제 원본 데이터 위치

메타데이터 파일 위치

taxonomy와 split 파일

NatureLM·BirdNET별 embedding 위치

single-label·multi-label·unknown·unseen 설정

결과 저장 경로

파일마다 서로 다르거나 잘못 연결된 경로

그다음 이를 바탕으로:

config.example.yaml

src/config.py

각 스크립트에서 불러오는 통일된 방식

자동 변환할 수 있는 수정 규칙

까지 만들면 됩니다.

특히 현재 코드에는 일부 BirdNET 평가 파일이 NatureLM embedding이나 NatureLM 결과 경로를 참조하는 것으로 보이는 부분도 있어서, 단순 치환보다 먼저 확인하는 게 맞습니다.

보내기 좋은 파일은 우선 아래 종류입니다.

embedding extraction 4개
- NatureLM / AnuraSet
- NatureLM / ESC-50
- BirdNET / AnuraSet
- BirdNET / ESC-50

dataset loader
- single-label AnuraSet
- multi-label AnuraSet
- unknown / unseen용 loader

대표 학습 파일
- UALB
- FTM-only
- TT-only
- FTM+TT

대표 evaluation 파일
- single-label
- multi-label
- unknown
- unseen
여러 파일을 한꺼번에 올려도 됩니다. 제가 경로와 데이터 흐름을 표로 정리한 뒤 YAML을 만들겠습니다.



extract_embeddings.py
Python

extract_embeddings_birdnet.py
Python

extract_embeddings_birdnet_esc50.py
Python

extract_embeddings_esc50.py
Python
/user/gaad2403/Bachelor-Thesis/scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
/user/gaad2403/Bachelor-Thesis/scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
/user/gaad2403/Bachelor-Thesis/scripts/NatureLM/anuraset/baseline/extract_embeddings.py
/user/gaad2403/Bachelor-Thesis/scripts/NatureLM/esc50/baseline/extract_embeddings.py

좋아요. 이 4개 파일만으로도 원본 데이터와 임베딩 생성 경로 구조는 거의 확정할 수 있습니다.

현재 데이터 흐름은 아래처럼 정리됩니다.

확인된 경로 구조
/user/gaad2403/bachelor_thesis/
├── anuraset/
│   └── anuraset/
│       ├── audio/
│       └── metadata.csv
│
├── ESC-50-master/
│   └── audio/
│
├── embeddings/                    # NatureLM × AnuraSet
├── embeddings_anuraset_birdnet/   # BirdNET × AnuraSet
├── embeddings_esc50/              # NatureLM × ESC-50
└── embeddings_esc50_birdnet/      # BirdNET × ESC-50
Git 저장소 내부에는 다음 파일들이 있습니다.

/user/gaad2403/Bachelor-Thesis/
└── data/
    ├── anuraset_taxonomy.csv
    └── esc50_bga_labels.csv
NatureLM AnuraSet은 원본 오디오와 metadata.csv를 읽고, taxonomy는 저장소 내부 CSV를 사용합니다. 출력은 embeddings/에 저장됩니다. 


BirdNET AnuraSet도 같은 원본 데이터와 taxonomy를 사용하지만 출력은 embeddings_anuraset_birdnet/입니다. 


NatureLM ESC-50은 ESC-50-master/audio와 저장소 내부 esc50_bga_labels.csv를 사용하고, 출력은 embeddings_esc50/입니다. 


BirdNET ESC-50 출력은 embeddings_esc50_birdnet/입니다. 


생성되는 파일
AnuraSet
NatureLM과 BirdNET 모두 다음 구조를 만듭니다.

embeddings.npy
species_labels.npy
embeddings_metadata.csv
BirdNET은 추가로:

summary.txt
를 생성합니다.

embeddings_metadata.csv에는 다음 정보가 들어갑니다.

sample_name
family
genus
species
subset
42개 species indicator columns
ESC-50
두 모델 모두:

embeddings.npy
embeddings_metadata.csv
를 생성하며, 메타데이터에는 다음이 들어갑니다.

filename
category
bga_label
fold
BirdNET은 추가로 summary.txt를 생성합니다.

YAML 초안 구조
현재 정보로는 다음 구조가 가장 적절합니다.

project:
  root: /user/gaad2403/Bachelor-Thesis

storage:
  root: /user/gaad2403/bachelor_thesis

datasets:
  anuraset:
    audio_dir: anuraset/anuraset/audio
    metadata_path: anuraset/anuraset/metadata.csv
    taxonomy_path: data/anuraset_taxonomy.csv

  esc50:
    audio_dir: ESC-50-master/audio
    labels_path: data/esc50_bga_labels.csv

embeddings:
  naturelm:
    anuraset: embeddings
    esc50: embeddings_esc50

  birdnet:
    anuraset: embeddings_anuraset_birdnet
    esc50: embeddings_esc50_birdnet

models:
  naturelm:
    pretrained_id: EarthSpeciesProject/NatureLM-audio
    embedding_dim: 4096
    target_sample_rate: 16000

  birdnet:
    model_type: acoustic
    version: "2.4"
    backend: tf
    embedding_dim: 1024

extraction:
  naturelm:
    batch_size: 32
    num_workers: 4

  birdnet:
    anuraset_batch_size: 128
    esc50_batch_size: 32
다만 최종 YAML을 만들기 전에 학습 파일과 평가 파일의 경로 구조도 확인하는 것이 좋습니다. 특히 results 하위 폴더명과 unknown/unseen split 파일까지 포함해야 하기 때문입니다.

발견한 수정 후보
BirdNET AnuraSet 코드에는 아래 함수가 있습니다.

def get_audio_path(row):
    return str(
        AUDIO_DIR
        / row["site"]
        / f"{row['fname']}_{int(row['min_t'])}={int(row['max_t'])}.wav"
    )
여기에는 중간 구분자가 =로 되어 있습니다. 하지만 이 함수는 실제로 사용되지 않고, 바로 아래에서 _를 사용한 audio_paths를 다시 만들고 있습니다. 따라서 현재 실행에는 영향이 없을 가능성이 크지만, 혼동을 막기 위해 이 미사용 함수를 삭제하거나 _로 수정하는 편이 좋습니다. 


또한 네 파일 모두 출력 디렉토리 생성이:

OUTPUT_DIR.mkdir(exist_ok=True)
로 되어 있습니다. 상위 폴더가 없는 환경에서도 안전하게 만들려면 다음이 낫습니다.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
다음으로 필요한 파일
이제 다음 종류에서 대표 파일만 주시면 됩니다.

1. NatureLM AnuraSet single-label 학습 파일 1개
2. BirdNET AnuraSet single-label 학습 파일 1개
3. NatureLM ESC-50 학습 파일 1개
4. multi-label 학습 파일 1개
5. evaluation.py 1~2개
6. unknown 또는 unseen 평가 파일 1개
이 파일들을 보면 results, split, checkpoint, unknown taxonomy까지 포함한 최종 config.example.yaml과 src/config.py를 정확하게 설계할 수 있습니다.



evaluation.py
Python

train_ualb_single.py
Python

train_triplet_single.py
Python

train_triplet_single_ftm_only.py
Python

train_ualb_multi.py
Python
각각의 위치도 중요한가?

네, 각 파일의 원래 위치도 중요합니다. 특히 YAML로 통합하려면 파일 경로에서 다음을 알아야 합니다.

해당 스크립트가 NatureLM인지 BirdNET인지

AnuraSet인지 ESC-50인지

single-label인지 multi-label인지

train인지 evaluation인지

unknown/unseen 실험인지

결과 저장 폴더가 어느 계층인지

예를 들어 같은 train_triplet_single.py라는 이름이어도 아래 두 파일은 설정이 다릅니다.

scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py
scripts/NatureLM/esc50/tree_triplet/train_triplet_single.py
첫 번째는 AnuraSet taxonomy와 embeddings/를 사용하고, 두 번째는 ESC-50 BGA 라벨과 embeddings_esc50/를 사용합니다. 


또 지금 보내주신 train_triplet_single_ftm_only.py는 BirdNET AnuraSet용으로 보입니다. 입력 차원이 1024이고 BirdNET 임베딩 폴더를 사용합니다. 


따라서 파일을 올릴 때 아래처럼 원래 위치를 함께 적어주시면 가장 정확합니다.

scripts/birdnet/anuraset/tree_triplet/train_triplet_single_ftm_only.py
scripts/NatureLM/esc50/tree_triplet/train_triplet_single.py
scripts/multilabel/NatureLM/anuraset/ualb/train_ualb_multi.py
scripts/multilabel/NatureLM/anuraset/evaluation.py
현재 보내주신 파일로 확인되는 내용은 다음과 같습니다.

train_ualb_single.py: NatureLM × AnuraSet × single-label × unknown-inclusive 결과 폴더 


train_triplet_single.py: NatureLM × ESC-50 × FTM+TT 


train_triplet_single_ftm_only.py: BirdNET × AnuraSet × FTM-only × unknown-inclusive 


train_ualb_multi.py: NatureLM × AnuraSet × multi-label × UALB 


evaluation.py: NatureLM × AnuraSet × multi-label 전체 평가 


그리고 지금 코드에서 설정 파일로 분리할 항목은 두 종류입니다.

공통 경로

project_root:
storage_root:
taxonomy:
splits:
embeddings:
results:
실험별 설정

backbone:
dataset:
label_mode:
input_dim:
batch_size:
epochs:
learning_rate:
output_subdir:
checkpoint_paths:
즉, 파일 위치는 중요하지만 모든 파일을 다시 올릴 필요는 없습니다. 앞으로는 파일을 보낼 때 원래 상대경로 한 줄 + 파일 업로드만 같이 주시면 됩니다. 그러면 그 위치까지 반영해서 최종 config.example.yaml을 만들 수 있습니다.



evaluation_unknown.py
Python
/user/gaad2403/Bachelor-Thesis/scripts/NatureLM/anuraset/ualb/train_ualb_single.py
/user/gaad2403/Bachelor-Thesis/scripts/NatureLM/esc50/tree_triplet/train_triplet_single.py
/user/gaad2403/Bachelor-Thesis/scripts/birdnet/anuraset/tree_triplet/train_triplet_single_ftm_only.py
/user/gaad2403/Bachelor-Thesis/scripts/multilabel/NatureLM/anuraset/ualb/train_ualb_multi.py
/user/gaad2403/Bachelor-Thesis/scripts/multilabel/NatureLM/anuraset/evaluation.py
/user/gaad2403/Bachelor-Thesis/scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py

좋아요. 이제 각 파일의 정확한 위치와 역할이 연결됐습니다.

현재 확인된 매핑은 다음과 같습니다.

scripts/NatureLM/anuraset/ualb/train_ualb_single.py
→ NatureLM × AnuraSet × single-label × UALB × unknown 설정

scripts/NatureLM/esc50/tree_triplet/train_triplet_single.py
→ NatureLM × ESC-50 × single-label × FTM+TT

scripts/birdnet/anuraset/tree_triplet/train_triplet_single_ftm_only.py
→ BirdNET × AnuraSet × single-label × FTM-only × unknown 설정

scripts/multilabel/NatureLM/anuraset/ualb/train_ualb_multi.py
→ NatureLM × AnuraSet × multi-label × UALB

scripts/multilabel/NatureLM/anuraset/evaluation.py
→ NatureLM × AnuraSet × multi-label 일반 평가

scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py
→ NatureLM × AnuraSet × multi-label unknown-inclusive 평가
특히 evaluation_unknown.py에서는 일반 multi-label loader가 아니라 anuraset_dataset_un을 사용하고, train/test 모두 include_negative=True로 평가합니다. 결과 폴더와 네 개 checkpoint 경로도 일반 평가와 별도로 나뉘어 있습니다. 


이제 YAML에서는 다음 수준까지 분리해야 정확합니다.

datasets:
  anuraset:
    normal:
    unknown:
    unseen:

experiments:
  single_label:
  multi_label:

backbones:
  naturelm:
  birdnet:

methods:
  baseline:
  ualb:
  ftm_only:
  tt_only:
  ftm_tt:

evaluation:
  normal:
  unknown:
  unseen:
그리고 경로도 단순히 results.root 하나만 두는 것보다 이렇게 구성하는 게 좋습니다.

paths:
  project_root:
  storage_root:

  embeddings:
    naturelm:
      anuraset:
      esc50:
    birdnet:
      anuraset:
      esc50:

  metadata:
    anuraset:
    esc50:

  taxonomy:
    anuraset:
    anuraset_unknown:

  splits:
    unseen_species:
    unseen_category:

  results:
    naturelm:
      anuraset:
      esc50:
    birdnet:
      anuraset:
      esc50:
    multilabel:
      naturelm:
      birdnet:
지금 단계에서 중요한 건 파일명을 더 받는 것보다, 실제 사용하는 최종 학습·평가 파일이 무엇인지 확인하는 것입니다. 오래된 파일과 현재 논문 결과에 사용한 파일이 섞여 있기 때문입니다.

다음으로 있으면 좋은 파일은 딱 세 종류입니다.

scripts/NatureLM/anuraset/evaluation.py
scripts/NatureLM/anuraset/evaluation_unseen.py
scripts/birdnet/anuraset/evaluation.py
이 세 개까지 보면 single-label normal/unknown/unseen과 NatureLM/BirdNET 결과 구조가 완성됩니다. 그다음 최종 config.example.yaml과 공통 config loader를 만들면 됩니다.



evaluation_unseen.py
Python

evaluation_birdent.py
Python

evaluation(1).py
Python

좋아요. 이제 single-label normal / unseen, NatureLM / BirdNET, multi-label normal / unknown 경로까지 충분히 파악됐습니다.

확인된 핵심 구조는 다음과 같습니다.

NatureLM × AnuraSet
- embedding: /user/gaad2403/bachelor_thesis/embeddings
- input_dim: 4096
- results: results/NatureLM/anuraset/

BirdNET × AnuraSet
- embedding: /user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet
- input_dim: 1024
- results: results/birdnet/anuraset/

NatureLM × ESC-50
- embedding: /user/gaad2403/bachelor_thesis/embeddings_esc50
- input_dim: 4096

BirdNET × ESC-50
- embedding: /user/gaad2403/bachelor_thesis/embeddings_esc50_birdnet
- input_dim: 1024
또한 unseen 평가는 data/unseen_species_split.json을 사용하고, NatureLM의 unseen 결과와 네 가지 checkpoint가 별도 폴더에 저장됩니다. 


BirdNET normal 평가는 BirdNET 전용 embedding과 1024차원 모델을 사용합니다. 

 NatureLM normal 평가는 같은 실험 구조이지만 4096차원 embedding을 사용합니다. 


이제 YAML은 다음처럼 설계하는 것이 적절합니다.

project:
  root: /user/gaad2403/Bachelor-Thesis

storage:
  root: /user/gaad2403/bachelor_thesis

datasets:
  anuraset:
    audio_dir: anuraset/anuraset/audio
    metadata_path: anuraset/anuraset/metadata.csv
    taxonomy_path: data/anuraset_taxonomy.csv
    taxonomy_unknown_path: data/anuraset_taxonomy_unknown.csv
    unseen_split_path: data/unseen_species_split.json

  esc50:
    audio_dir: ESC-50-master/audio
    labels_path: data/esc50_bga_labels.csv
    unseen_split_path: data/unseen_category_split.json

embeddings:
  naturelm:
    anuraset: embeddings
    esc50: embeddings_esc50
    input_dim: 4096

  birdnet:
    anuraset: embeddings_anuraset_birdnet
    esc50: embeddings_esc50_birdnet
    input_dim: 1024

results:
  root: results

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  projection_dim: 128
  hidden_dim: 1024
  validation_split: 0.1
  seed: 42

evaluation:
  probe_epochs: 300
  probe_learning_rate: 0.01
  threshold: 0.5
  seed: 42
다만 SAVE_DIR와 MODEL_PATHS까지 YAML에 모두 넣으면 파일이 너무 복잡해집니다. 더 좋은 방식은 결과 루트만 YAML에 두고, Python에서 실험명으로 경로를 조합하는 것입니다.

예를 들어:

SAVE_DIR = (
    RESULTS_ROOT
    / backbone
    / dataset
    / experiment_name
)
checkpoint도:

MODEL_PATHS = {
    "UALB": RESULTS_ROOT / backbone / dataset / "wp2_ualb_singlelabel_l0" / "best_model.pt",
    "FTM-only": RESULTS_ROOT / backbone / dataset / "wp2_ftm_only_singlelabel" / "best_model.pt",
}
처럼 만들면 YAML이 훨씬 짧아집니다.

이제 다음 단계는 충분한 정보를 모았으므로 config.example.yaml과 src/config.py를 실제로 완성하고, 대표 스크립트 하나를 변환하는 것입니다.


그래 어떻게 하면 되니?

좋습니다. 아직 커밋하지 않았다면, 지금부터 config.example.yaml + 개인용 config.yaml + src/config.py 구조로 바꾸면 됩니다.

1. 이전 자동 수정 되돌리기
아직 커밋하지 않았으므로 먼저 이전 경로 자동수정 내용을 되돌립니다.

git reset --hard HEAD
refactor_paths.py가 추적되지 않은 파일로 남아 있다면 삭제합니다.

rm -f refactor_paths.py
git status
working tree clean이 나오면 준비가 끝난 것입니다.

2. YAML 사용 패키지 설치
pip install pyyaml
설치 확인:

python -c "import yaml; print(yaml.__version__)"
3. config.example.yaml 만들기
저장소 최상위에서 실행합니다.

nano config.example.yaml
아래 내용을 붙여넣으세요.

project:
  root: /path/to/Bachelor-Thesis

storage:
  root: /path/to/bioacoustic_storage

datasets:
  anuraset:
    audio_dir: anuraset/anuraset/audio
    metadata_path: anuraset/anuraset/metadata.csv
    taxonomy_path: data/anuraset_taxonomy.csv
    taxonomy_unknown_path: data/anuraset_taxonomy_unknown.csv
    unseen_split_path: data/unseen_species_split.json

  esc50:
    audio_dir: ESC-50-master/audio
    labels_path: data/esc50_bga_labels.csv
    unseen_split_path: data/unseen_category_split.json

embeddings:
  naturelm:
    input_dim: 4096
    anuraset_dir: embeddings
    esc50_dir: embeddings_esc50

  birdnet:
    input_dim: 1024
    anuraset_dir: embeddings_anuraset_birdnet
    esc50_dir: embeddings_esc50_birdnet

models:
  naturelm:
    pretrained_id: EarthSpeciesProject/NatureLM-audio
    target_sample_rate: 16000

  birdnet:
    model_type: acoustic
    version: "2.4"
    backend: tf

results:
  root: results

training:
  batch_size: 256
  epochs:
    anuraset: 100
    esc50: 450
  learning_rate: 0.001
  weight_decay: 0.0001
  hidden_dim: 1024
  projection_dim: 128
  triplet_dim: 128
  temperature: 0.07
  validation_split: 0.1
  seed: 42

evaluation:
  probe_epochs: 300
  probe_learning_rate: 0.01
  threshold: 0.5
  inter_class_multiplier: 5
  seed: 42
저장:

Ctrl+O
Enter
Ctrl+X
4. 실제 서버용 config.yaml 만들기
예시 파일을 복사합니다.

cp config.example.yaml config.yaml
nano config.yaml
맨 위 두 경로만 현재 서버 경로로 바꿉니다.

project:
  root: /user/gaad2403/Bachelor-Thesis

storage:
  root: /user/gaad2403/bachelor_thesis
나머지는 그대로 두면 됩니다.

5. 개인용 YAML은 GitHub에서 제외
.gitignore 마지막에 추가합니다.

echo 'config.yaml' >> .gitignore
확인:

tail -5 .gitignore
GitHub에는 config.example.yaml만 올라가고, 실제 개인 경로가 있는 config.yaml은 올라가지 않습니다.

6. 공통 설정 로더 만들기
nano src/config.py
아래 내용을 붙여넣으세요.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = Path(
    os.environ.get(
        "BIOACOUSTIC_CONFIG",
        REPOSITORY_ROOT / "config.yaml",
    )
).expanduser().resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Copy config.example.yaml to config.yaml and update the paths."
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config


CONFIG = load_yaml(CONFIG_PATH)

PROJECT_ROOT = Path(CONFIG["project"]["root"]).expanduser().resolve()
STORAGE_ROOT = Path(CONFIG["storage"]["root"]).expanduser().resolve()


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def storage_path(relative_path: str) -> Path:
    return STORAGE_ROOT / relative_path


# AnuraSet
ANURASET_AUDIO_DIR = storage_path(
    CONFIG["datasets"]["anuraset"]["audio_dir"]
)
ANURASET_METADATA_PATH = storage_path(
    CONFIG["datasets"]["anuraset"]["metadata_path"]
)
ANURASET_TAXONOMY_PATH = project_path(
    CONFIG["datasets"]["anuraset"]["taxonomy_path"]
)
ANURASET_TAXONOMY_UNKNOWN_PATH = project_path(
    CONFIG["datasets"]["anuraset"]["taxonomy_unknown_path"]
)
ANURASET_UNSEEN_SPLIT_PATH = project_path(
    CONFIG["datasets"]["anuraset"]["unseen_split_path"]
)

# ESC-50
ESC50_AUDIO_DIR = storage_path(
    CONFIG["datasets"]["esc50"]["audio_dir"]
)
ESC50_LABELS_PATH = project_path(
    CONFIG["datasets"]["esc50"]["labels_path"]
)
ESC50_UNSEEN_SPLIT_PATH = project_path(
    CONFIG["datasets"]["esc50"]["unseen_split_path"]
)

# Embeddings
NATURELM_ANURASET_EMBED_DIR = storage_path(
    CONFIG["embeddings"]["naturelm"]["anuraset_dir"]
)
NATURELM_ESC50_EMBED_DIR = storage_path(
    CONFIG["embeddings"]["naturelm"]["esc50_dir"]
)

BIRDNET_ANURASET_EMBED_DIR = storage_path(
    CONFIG["embeddings"]["birdnet"]["anuraset_dir"]
)
BIRDNET_ESC50_EMBED_DIR = storage_path(
    CONFIG["embeddings"]["birdnet"]["esc50_dir"]
)

NATURELM_INPUT_DIM = int(
    CONFIG["embeddings"]["naturelm"]["input_dim"]
)
BIRDNET_INPUT_DIM = int(
    CONFIG["embeddings"]["birdnet"]["input_dim"]
)

RESULTS_ROOT = project_path(CONFIG["results"]["root"])

TRAINING = CONFIG["training"]
EVALUATION = CONFIG["evaluation"]
MODELS = CONFIG["models"]
7. 설정 파일 정상 작동 확인
저장소 최상위에서 실행하세요.

PYTHONPATH=src python - <<'PY'
from config import (
    PROJECT_ROOT,
    STORAGE_ROOT,
    ANURASET_AUDIO_DIR,
    ANURASET_METADATA_PATH,
    ANURASET_TAXONOMY_PATH,
    ESC50_AUDIO_DIR,
    ESC50_LABELS_PATH,
    NATURELM_ANURASET_EMBED_DIR,
    BIRDNET_ANURASET_EMBED_DIR,
    NATURELM_ESC50_EMBED_DIR,
    BIRDNET_ESC50_EMBED_DIR,
    RESULTS_ROOT,
)

paths = {
    "PROJECT_ROOT": PROJECT_ROOT,
    "STORAGE_ROOT": STORAGE_ROOT,
    "ANURASET_AUDIO_DIR": ANURASET_AUDIO_DIR,
    "ANURASET_METADATA_PATH": ANURASET_METADATA_PATH,
    "ANURASET_TAXONOMY_PATH": ANURASET_TAXONOMY_PATH,
    "ESC50_AUDIO_DIR": ESC50_AUDIO_DIR,
    "ESC50_LABELS_PATH": ESC50_LABELS_PATH,
    "NATURELM_ANURASET_EMBED_DIR": NATURELM_ANURASET_EMBED_DIR,
    "BIRDNET_ANURASET_EMBED_DIR": BIRDNET_ANURASET_EMBED_DIR,
    "NATURELM_ESC50_EMBED_DIR": NATURELM_ESC50_EMBED_DIR,
    "BIRDNET_ESC50_EMBED_DIR": BIRDNET_ESC50_EMBED_DIR,
    "RESULTS_ROOT": RESULTS_ROOT,
}

for name, path in paths.items():
    print(f"{name}: {path}")
    print(f"  exists: {path.exists()}")
PY
필요한 실제 파일과 폴더가 있다면 대부분 exists: True가 나와야 합니다.

8. 대표 파일 하나만 먼저 변경
먼저 이 파일로 시험하는 것이 좋습니다.

scripts/NatureLM/anuraset/ualb/train_ualb_single.py
엽니다.

nano scripts/NatureLM/anuraset/ualb/train_ualb_single.py
기존 상단 부분:

import sys
import os
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")
를 아래로 바꿉니다.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
기존 import 아래에 다음을 추가합니다.

from config import (
    NATURELM_ANURASET_EMBED_DIR,
    RESULTS_ROOT,
    TRAINING,
)
기존 경로 부분:

EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH  = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
SAVE_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_ualb_singlelabel_l0_unknown')
를 다음으로 변경합니다.

EMBED_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
META_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"

SAVE_DIR = (
    RESULTS_ROOT
    / "NatureLM"
    / "anuraset"
    / "wp2_ualb_singlelabel_l0_unknown"
)
하이퍼파라미터도 YAML에서 읽게 하려면 기존:

BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PROJ_DIM     = 128
TEMPERATURE  = 0.07
VAL_SPLIT    = 0.1
를 다음으로 바꿉니다.

BATCH_SIZE = int(TRAINING["batch_size"])
EPOCHS = int(TRAINING["epochs"]["anuraset"])
LR = float(TRAINING["learning_rate"])
WEIGHT_DECAY = float(TRAINING["weight_decay"])
PROJ_DIM = int(TRAINING["projection_dim"])
TEMPERATURE = float(TRAINING["temperature"])
VAL_SPLIT = float(TRAINING["validation_split"])
9. 실행 전에 경로만 검사
학습을 바로 돌리지 말고 우선 Python import와 경로를 확인하세요.

PYTHONPATH=src python - <<'PY'
from config import (
    NATURELM_ANURASET_EMBED_DIR,
    RESULTS_ROOT,
)

embed = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
meta = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"
result = RESULTS_ROOT / "NatureLM" / "anuraset"

print("Embedding:", embed, embed.exists())
print("Metadata:", meta, meta.exists())
print("Result root:", result)
PY
10. 문법 검사
python -m py_compile scripts/NatureLM/anuraset/ualb/train_ualb_single.py
아무것도 출력되지 않으면 문법상 정상입니다.

그다음 전체 설정 로더도 검사합니다.

python -m py_compile src/config.py
11. Git 상태 확인
git status
git diff --stat
정상이라면 우선 이 단계만 커밋해도 됩니다.

git add config.example.yaml src/config.py .gitignore
git add scripts/NatureLM/anuraset/ualb/train_ualb_single.py
git commit -m "Add YAML configuration and migrate first training script"
이 대표 파일이 정상 작동하는 것을 확인한 뒤 나머지 95개 파일도 같은 방식으로 자동 변환하는 것이 안전합니다.


(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ PYTHONPATH=src python - <<'PY'
> from config import (
>     PROJECT_ROOT,
>     STORAGE_ROOT,
>     ANURASET_AUDIO_DIR,
>     ANURASET_METADATA_PATH,
>     ANURASET_TAXONOMY_PATH,
>     ESC50_AUDIO_DIR,
>     ESC50_LABELS_PATH,
>     NATURELM_ANURASET_EMBED_DIR,
>     BIRDNET_ANURASET_EMBED_DIR,
>     NATURELM_ESC50_EMBED_DIR,
>     BIRDNET_ESC50_EMBED_DIR,
>     RESULTS_ROOT,
> )
> 
> paths = {
>     "PROJECT_ROOT": PROJECT_ROOT,
>     "STORAGE_ROOT": STORAGE_ROOT,
>     "ANURASET_AUDIO_DIR": ANURASET_AUDIO_DIR,
>     "ANURASET_METADATA_PATH": ANURASET_METADATA_PATH,
>     "ANURASET_TAXONOMY_PATH": ANURASET_TAXONOMY_PATH,
>     "ESC50_AUDIO_DIR": ESC50_AUDIO_DIR,
>     "ESC50_LABELS_PATH": ESC50_LABELS_PATH,
>     "NATURELM_ANURASET_EMBED_DIR": NATURELM_ANURASET_EMBED_DIR,
>     "BIRDNET_ANURASET_EMBED_DIR": BIRDNET_ANURASET_EMBED_DIR,
>     "NATURELM_ESC50_EMBED_DIR": NATURELM_ESC50_EMBED_DIR,
>     "BIRDNET_ESC50_EMBED_DIR": BIRDNET_ESC50_EMBED_DIR,
>     "RESULTS_ROOT": RESULTS_ROOT,
> }
> 
> for name, path in paths.items():
>     print(f"{name}: {path}")
>     print(f"  exists: {path.exists()}")
> PY
PROJECT_ROOT: /path/to/Bachelor-Thesis
  exists: False
STORAGE_ROOT: /path/to/bioacoustic_storage
  exists: False
ANURASET_AUDIO_DIR: /path/to/bioacoustic_storage/anuraset/anuraset/audio
  exists: False
ANURASET_METADATA_PATH: /path/to/bioacoustic_storage/anuraset/anuraset/metadata.csv
  exists: False
ANURASET_TAXONOMY_PATH: /path/to/Bachelor-Thesis/data/anuraset_taxonomy.csv
  exists: False
ESC50_AUDIO_DIR: /path/to/bioacoustic_storage/ESC-50-master/audio
  exists: False
ESC50_LABELS_PATH: /path/to/Bachelor-Thesis/data/esc50_bga_labels.csv
  exists: False
NATURELM_ANURASET_EMBED_DIR: /path/to/bioacoustic_storage/embeddings
  exists: False
BIRDNET_ANURASET_EMBED_DIR: /path/to/bioacoustic_storage/embeddings_anuraset_birdnet
  exists: False
NATURELM_ESC50_EMBED_DIR: /path/to/bioacoustic_storage/embeddings_esc50
  exists: False
BIRDNET_ESC50_EMBED_DIR: /path/to/bioacoustic_storage/embeddings_esc50_birdnet
  exists: False
RESULTS_ROOT: /path/to/Bachelor-Thesis/results
  exists: False

이렇게 나왔어

config.yaml이 아니라 config.example.yaml의 예시 경로를 읽고 있는 상태입니다.

즉, config.yaml 파일이 없거나, 내용이 아직 예시 그대로입니다.

먼저 확인하세요.

ls -l config.yaml config.example.yaml
1. config.yaml이 없다면
cp config.example.yaml config.yaml
nano config.yaml
맨 위를 이렇게 바꾸세요.

project:
  root: /user/gaad2403/Bachelor-Thesis

storage:
  root: /user/gaad2403/bachelor_thesis
저장 후 다시 실행합니다.

PYTHONPATH=src python - <<'PY'
from config import PROJECT_ROOT, STORAGE_ROOT
print(PROJECT_ROOT)
print(STORAGE_ROOT)
PY
정상 출력은:

/user/gaad2403/Bachelor-Thesis
/user/gaad2403/bachelor_thesis
2. config.yaml이 이미 있다면
내용을 확인하세요.

sed -n '1,20p' config.yaml
아직 이렇게 되어 있다면:

project:
  root: /path/to/Bachelor-Thesis

storage:
  root: /path/to/bioacoustic_storage
실제 경로로 수정합니다.

sed -i \
  's|/path/to/Bachelor-Thesis|/user/gaad2403/Bachelor-Thesis|g; s|/path/to/bioacoustic_storage|/user/gaad2403/bachelor_thesis|g' \
  config.yaml
확인:

sed -n '1,12p' config.yaml
3. 그래도 예시 경로가 나온다면
현재 어떤 config 파일을 읽는지 확인하세요.

PYTHONPATH=src python - <<'PY'
from config import CONFIG_PATH
print(CONFIG_PATH)
PY
정상이라면:

/user/gaad2403/Bachelor-Thesis/config.yaml
이 나와야 합니다.

환경변수가 다른 파일을 가리키고 있을 수도 있습니다.

echo "$BIOACOUSTIC_CONFIG"
무언가 출력되면 해제하세요.

unset BIOACOUSTIC_CONFIG
그 뒤 다시 검사합니다.

4. 전체 경로 재검사
PYTHONPATH=src python - <<'PY'
from config import *

paths = {
    "PROJECT_ROOT": PROJECT_ROOT,
    "STORAGE_ROOT": STORAGE_ROOT,
    "ANURASET_AUDIO_DIR": ANURASET_AUDIO_DIR,
    "ANURASET_METADATA_PATH": ANURASET_METADATA_PATH,
    "ANURASET_TAXONOMY_PATH": ANURASET_TAXONOMY_PATH,
    "ESC50_AUDIO_DIR": ESC50_AUDIO_DIR,
    "ESC50_LABELS_PATH": ESC50_LABELS_PATH,
    "NATURELM_ANURASET_EMBED_DIR": NATURELM_ANURASET_EMBED_DIR,
    "BIRDNET_ANURASET_EMBED_DIR": BIRDNET_ANURASET_EMBED_DIR,
    "NATURELM_ESC50_EMBED_DIR": NATURELM_ESC50_EMBED_DIR,
    "BIRDNET_ESC50_EMBED_DIR": BIRDNET_ESC50_EMBED_DIR,
    "RESULTS_ROOT": RESULTS_ROOT,
}

for name, path in paths.items():
    print(f"{name}: {path}")
    print(f"  exists: {path.exists()}")
PY
핵심은 config.example.yaml은 GitHub용 예시이고, 실제 서버에서는 개인 경로가 들어간 config.yaml을 사용해야 한다는 점입니다.


(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ PYTHONPATH=src python - <<'PY'
> from config import *
> 
> paths = {
>     "PROJECT_ROOT": PROJECT_ROOT,
>     "STORAGE_ROOT": STORAGE_ROOT,
>     "ANURASET_AUDIO_DIR": ANURASET_AUDIO_DIR,
>     "ANURASET_METADATA_PATH": ANURASET_METADATA_PATH,
>     "ANURASET_TAXONOMY_PATH": ANURASET_TAXONOMY_PATH,
>     "ESC50_AUDIO_DIR": ESC50_AUDIO_DIR,
>     "ESC50_LABELS_PATH": ESC50_LABELS_PATH,
>     "NATURELM_ANURASET_EMBED_DIR": NATURELM_ANURASET_EMBED_DIR,
>     "BIRDNET_ANURASET_EMBED_DIR": BIRDNET_ANURASET_EMBED_DIR,
>     "NATURELM_ESC50_EMBED_DIR": NATURELM_ESC50_EMBED_DIR,
>     "BIRDNET_ESC50_EMBED_DIR": BIRDNET_ESC50_EMBED_DIR,
>     "RESULTS_ROOT": RESULTS_ROOT,
> }
> 
> for name, path in paths.items():
>     print(f"{name}: {path}")
>     print(f"  exists: {path.exists()}")
> PY
PROJECT_ROOT: /fs/dss/home/gaad2403/Bachelor-Thesis
  exists: True
STORAGE_ROOT: /fs/dss/home/gaad2403/bachelor_thesis
  exists: True
ANURASET_AUDIO_DIR: /fs/dss/home/gaad2403/bachelor_thesis/anuraset/anuraset/audio
  exists: True
ANURASET_METADATA_PATH: /fs/dss/home/gaad2403/bachelor_thesis/anuraset/anuraset/metadata.csv
  exists: True
ANURASET_TAXONOMY_PATH: /fs/dss/home/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv
  exists: True
ESC50_AUDIO_DIR: /fs/dss/home/gaad2403/bachelor_thesis/ESC-50-master/audio
  exists: True
ESC50_LABELS_PATH: /fs/dss/home/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv
  exists: True
NATURELM_ANURASET_EMBED_DIR: /fs/dss/home/gaad2403/bachelor_thesis/embeddings
  exists: True
BIRDNET_ANURASET_EMBED_DIR: /fs/dss/home/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet
  exists: True
NATURELM_ESC50_EMBED_DIR: /fs/dss/home/gaad2403/bachelor_thesis/embeddings_esc50
  exists: True
BIRDNET_ESC50_EMBED_DIR: /fs/dss/home/gaad2403/bachelor_thesis/embeddings_esc50_birdnet
  exists: True
RESULTS_ROOT: /fs/dss/home/gaad2403/Bachelor-Thesis/results
  exists: True

이 다음으로는?

좋습니다. 이제 YAML 설정과 경로 로더는 정상 작동합니다. 다음 단계는 대표 스크립트 하나를 YAML 방식으로 바꾸고 실행 전 검증하는 것입니다.

먼저 train_ualb_single.py부터 바꾸는 게 좋습니다.

1. 파일 백업
cp scripts/NatureLM/anuraset/ualb/train_ualb_single.py \
   scripts/NatureLM/anuraset/ualb/train_ualb_single.py.bak
2. 상단 경로 설정 수정
파일을 엽니다.

nano scripts/NatureLM/anuraset/ualb/train_ualb_single.py
기존의 다음 부분을 찾습니다.

import sys
import os
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")
아래처럼 바꾸세요.

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
현재 파일 위치가:

scripts/NatureLM/anuraset/ualb/train_ualb_single.py
이므로 parents[4]가 저장소 루트 Bachelor-Thesis를 가리킵니다.

3. config import 추가
기존 import 부분에 다음을 추가합니다.

from config import (
    NATURELM_ANURASET_EMBED_DIR,
    RESULTS_ROOT,
    TRAINING,
)
전체 상단은 대략 다음처럼 됩니다.

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from config import (
    NATURELM_ANURASET_EMBED_DIR,
    RESULTS_ROOT,
    TRAINING,
)

from NatureLM.anuraset.anuraset_dataset_unknown import AnuraSetEmbeddingDataset
from models.projection_head import ProjectionHead
from hmlc_loss_l0 import HMLCWithSpeciesLevel as HMLC
기존에 있던 import os는 사용하지 않는다면 삭제해도 됩니다.

4. 절대경로 교체
기존:

EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH  = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
SAVE_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_ualb_singlelabel_l0_unknown')
를 다음으로 바꾸세요.

EMBED_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
META_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"

SAVE_DIR = (
    RESULTS_ROOT
    / "NatureLM"
    / "anuraset"
    / "wp2_ualb_singlelabel_l0_unknown"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
5. 하이퍼파라미터도 YAML에서 읽기
기존:

BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PROJ_DIM     = 128
TEMPERATURE  = 0.07
VAL_SPLIT    = 0.1
를 다음으로 바꾸세요.

BATCH_SIZE = int(TRAINING["batch_size"])
EPOCHS = int(TRAINING["epochs"]["anuraset"])
LR = float(TRAINING["learning_rate"])
WEIGHT_DECAY = float(TRAINING["weight_decay"])
PROJ_DIM = int(TRAINING["projection_dim"])
TEMPERATURE = float(TRAINING["temperature"])
VAL_SPLIT = float(TRAINING["validation_split"])
6. 문법 검사
저장 후 실행합니다.

python -m py_compile scripts/NatureLM/anuraset/ualb/train_ualb_single.py
아무 출력이 없으면 문법상 정상입니다.

7. import와 경로만 시험
학습 전체를 돌리지 않고 설정만 확인하려면 다음 명령을 실행하세요.

PYTHONPATH=src python - <<'PY'
from config import (
    NATURELM_ANURASET_EMBED_DIR,
    RESULTS_ROOT,
    TRAINING,
)

embed = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
meta = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"
save_dir = (
    RESULTS_ROOT
    / "NatureLM"
    / "anuraset"
    / "wp2_ualb_singlelabel_l0_unknown"
)

print("Embedding:", embed, embed.exists())
print("Metadata:", meta, meta.exists())
print("Save directory:", save_dir)
print("Batch size:", TRAINING["batch_size"])
print("Epochs:", TRAINING["epochs"]["anuraset"])
PY
Embedding과 Metadata가 모두 True이면 됩니다.

8. 스크립트 import 자체 검사
이 파일은 import하는 순간 학습을 시작하도록 작성돼 있어서 단순 import 검사는 권하지 않습니다. 대신 첫 실행 전에 학습량을 잠시 줄여 시험할 수 있습니다.

config.yaml에서 일시적으로:

training:
  batch_size: 256
  epochs:
    anuraset: 1
    esc50: 450
로 바꾼 뒤 실행합니다.

python scripts/NatureLM/anuraset/ualb/train_ualb_single.py
1 epoch가 정상 완료되면 YAML 연결은 제대로 된 것입니다. 이후 다시:

epochs:
  anuraset: 100
으로 복구하세요.

9. 정상 작동하면 Git diff 확인
git diff -- scripts/NatureLM/anuraset/ualb/train_ualb_single.py
git status
아직은 전체 커밋하지 말고, 이 대표 파일이 실행되는 것까지 확인하는 편이 안전합니다.

그다음에는 같은 방식으로 나머지 파일을 자동 변환하는 스크립트를 만들면 됩니다.


train말고 eval을 돌려보고 싶어. 

좋아요. 학습 파일보다 NatureLM AnuraSet 일반 평가 파일부터 YAML 방식으로 바꾸고 실행해보는 게 더 안전합니다.

대상 파일:

scripts/NatureLM/anuraset/evaluation.py
1. 백업
cp scripts/NatureLM/anuraset/evaluation.py \
   scripts/NatureLM/anuraset/evaluation.py.bak
2. 파일 열기
nano scripts/NatureLM/anuraset/evaluation.py
기존 상단:

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')
를 아래로 바꾸세요.

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
이 파일은:

scripts/NatureLM/anuraset/evaluation.py
위치이므로 parents[3]이 저장소 루트입니다.

3. config import 추가
기존 import 아래에 추가하세요.

from config import (
    NATURELM_ANURASET_EMBED_DIR,
    NATURELM_INPUT_DIM,
    RESULTS_ROOT,
    ANURASET_TAXONOMY_PATH,
    EVALUATION,
)
상단은 대략 이렇게 됩니다.

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    NATURELM_ANURASET_EMBED_DIR,
    NATURELM_INPUT_DIM,
    RESULTS_ROOT,
    ANURASET_TAXONOMY_PATH,
    EVALUATION,
)

from NatureLM.anuraset.anuraset_dataset import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel
from hf1 import (
    joint_probe_singlelabel,
    ftm_only_preds_singlelabel,
    ground_truth_singlelabel,
    hierarchical_f1,
)
4. 경로 부분 교체
기존:

EMBED_PATH    = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH     = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
OUTPUT_DIR    = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp3_eval_total_hf1')
를 다음으로 바꾸세요.

EMBED_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
META_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"
TAXONOMY_PATH = ANURASET_TAXONOMY_PATH

OUTPUT_DIR = (
    RESULTS_ROOT
    / "NatureLM"
    / "anuraset"
    / "wp3_eval_total_hf1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
5. 모델 checkpoint 경로 교체
기존 MODEL_PATHS 전체를 아래로 바꾸세요.

MODEL_ROOT = RESULTS_ROOT / "NatureLM" / "anuraset"

MODEL_PATHS = {
    "UALB": (
        "projectionhead",
        MODEL_ROOT / "wp2_ualb_singlelabel_l0" / "best_model.pt",
    ),
    "FTM-only": (
        "hierarchical",
        MODEL_ROOT / "wp2_ftm_only_singlelabel" / "best_model.pt",
    ),
    "TT-only": (
        "hierarchical",
        MODEL_ROOT / "wp2_tt_only_singlelabel_self" / "best_model.pt",
    ),
    "FTM+TT": (
        "hierarchical",
        MODEL_ROOT / "wp2_hssn_singlelabel_self" / "best_model.pt",
    ),
}
6. 평가 설정도 YAML에서 읽기
기존:

INPUT_DIM    = 4096
PROBE_EPOCHS = 300
PROBE_LR     = 0.01
N_INTER_MULT = 5
SEED         = 42
를 다음으로 바꾸세요.

INPUT_DIM = NATURELM_INPUT_DIM
PROBE_EPOCHS = int(EVALUATION["probe_epochs"])
PROBE_LR = float(EVALUATION["probe_learning_rate"])
N_INTER_MULT = int(EVALUATION["inter_class_multiplier"])
SEED = int(EVALUATION["seed"])
DEVICE는 그대로 두면 됩니다.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
7. 문법 검사
python -m py_compile scripts/NatureLM/anuraset/evaluation.py
아무 출력이 없으면 정상입니다.

8. checkpoint 존재 여부 먼저 확인
PYTHONPATH=src python - <<'PY'
from config import RESULTS_ROOT

root = RESULTS_ROOT / "NatureLM" / "anuraset"

paths = {
    "UALB": root / "wp2_ualb_singlelabel_l0" / "best_model.pt",
    "FTM-only": root / "wp2_ftm_only_singlelabel" / "best_model.pt",
    "TT-only": root / "wp2_tt_only_singlelabel_self" / "best_model.pt",
    "FTM+TT": root / "wp2_hssn_singlelabel_self" / "best_model.pt",
}

for name, path in paths.items():
    print(name, path.exists(), path)
PY
일부가 False여도 평가 코드는 해당 모델을 건너뛸 수 있습니다.

9. 평가 실행
저장소 루트에서:

python scripts/NatureLM/anuraset/evaluation.py
필요하면 명시적으로:

PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation.py
정상적으로 끝나면 결과는 여기 저장됩니다.

results/NatureLM/anuraset/wp3_eval_total_hf1/wp3_full_eval_results.npy
한 가지 주의점이 있습니다. 현재 NatureLM 일반 평가 파일에서 FTM-only 분기에서는 majority_acc를 만들지 않는데, 나중에 출력이나 결과 저장에서 참조하면 오류가 날 수 있습니다. 실행 중 UnboundLocalError: majority_acc가 나오면 그 부분도 바로 고치면 됩니다.


뭐라는건지 하나도 모르겠다

네, 정상적으로 잘 돌아가고 있습니다.

지금 출력으로 보면:

YAML에서 데이터 경로를 제대로 읽었고

AnuraSet train/test 데이터도 정상 로드됐고

UALB, FTM-only, TT-only, FTM+TT 체크포인트도 모두 불러왔고

Baseline부터 평가 계산이 진행 중입니다

따라서 지금은 그대로 두면 됩니다. 마지막에 이런 식으로 끝나면 성공입니다.

Saved to: .../results/NatureLM/anuraset/wp3_eval_total_hf1/wp3_full_eval_results.npy
끝난 뒤 아래 명령으로 결과 파일이 생겼는지 확인하세요.

ls -lh results/NatureLM/anuraset/wp3_eval_total_hf1/
평가가 끝까지 완료되면, 이번 YAML 연결은 성공한 것입니다.


그러면 깃허브에는 어떻게 올려?

평가가 끝난 뒤, 개인 경로가 들어간 config.yaml은 빼고 예시 파일과 코드만 GitHub에 올리면 됩니다.

먼저 결과가 끝났는지 확인하세요.

ls -lh results/NatureLM/anuraset/wp3_eval_total_hf1/
그다음 Git 상태를 봅니다.

git status
올려야 할 핵심 파일은 보통 다음입니다.

config.example.yaml
src/config.py
scripts/NatureLM/anuraset/evaluation.py
.gitignore
README.md
개인용 config.yaml은 .gitignore에 들어 있어야 합니다.

grep -n '^config.yaml$' .gitignore
아무것도 안 나오면 추가하세요.

echo 'config.yaml' >> .gitignore
그다음 파일을 Git에 추가합니다.

git add config.example.yaml
git add src/config.py
git add scripts/NatureLM/anuraset/evaluation.py
git add .gitignore
git add README.md
변경 내용을 확인합니다.

git status
git diff --cached
문제가 없으면 커밋합니다.

git commit -m "Add YAML configuration and portable evaluation paths"
현재 브랜치가 backup-main이므로 그 브랜치에 올리려면:

git push origin backup-main
GitHub의 main 브랜치에 올리고 싶다면 현재 작업을 바로 main으로 푸시하지 말고, 먼저 브랜치를 확인하세요.

git branch
backup-main에서 작업 중이고 이 변경을 main에 합치려면:

git switch main
git pull origin main
git merge backup-main
git push origin main
다만 아직 여러 파일을 YAML 방식으로 바꾸는 중이라면, 우선 backup-main에만 푸시하는 것이 안전합니다.

GitHub에 올라가면 안 되는 파일도 확인하세요.

git ls-files config.yaml
git ls-files '*.npy' '*.pt' '*.pth'
아무것도 출력되지 않는 것이 정상입니다.

마지막으로 개인 절대경로가 새로 수정한 평가 파일에 남아 있는지 확인하세요.

grep -nE '/user/gaad2403|/fs/dss/home/gaad2403' \
scripts/NatureLM/anuraset/evaluation.py \
src/config.py \
config.example.yaml
config.example.yaml에는 /path/to/...만 있어야 하고, evaluation.py와 src/config.py에는 개인 경로가 없어야 합니다.


근데 그러면 이제 모든 코드에 대해서 다 바꿔야하는거 아니야? 마지막에 하면 되려나

맞아요. 최종적으로는 GitHub에 공개할 코드 중 실제로 실행 가능한 모든 .py 파일의 절대경로를 YAML 방식으로 바꾸는 게 맞습니다.

다만 지금 한 번에 96개 파일을 다 고치면 오류가 생겼을 때 원인을 찾기 어렵습니다. 순서는 이렇게 하는 게 안전합니다.

지금처럼 대표 평가 파일 1개로 YAML 구조가 실제로 작동하는지 확인

평가가 끝까지 정상 완료되는지 확인

그다음 나머지 코드에 일괄 적용

전체 문법 검사

핵심 스크립트 몇 개만 실제 실행

마지막에 커밋하고 GitHub에 올리기

즉, GitHub에 올리기 직전에 한꺼번에 바꾸는 것보다는, 지금 구조를 확정한 뒤 자동 변환하는 게 좋습니다.

현재는 평가 파일이 정상 실행 중이므로 YAML 구조 자체는 거의 검증됐습니다. 평가가 끝나면 다음 단계로 자동 변환 스크립트를 만들어서 아래 경로들을 한꺼번에 교체하면 됩니다.

/user/gaad2403/Bachelor-Thesis/data/...
→ PROJECT_ROOT / "data/..."

/user/gaad2403/Bachelor-Thesis/results/...
→ RESULTS_ROOT / ...

/user/gaad2403/bachelor_thesis/embeddings/...
→ NATURELM_ANURASET_EMBED_DIR / ...

/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/...
→ BIRDNET_ANURASET_EMBED_DIR / ...

/user/gaad2403/bachelor_thesis/embeddings_esc50/...
→ NATURELM_ESC50_EMBED_DIR / ...

/user/gaad2403/bachelor_thesis/embeddings_esc50_birdnet/...
→ BIRDNET_ESC50_EMBED_DIR / ...
다만 모든 결과 폴더 이름은 서로 다르기 때문에, SAVE_DIR과 MODEL_PATHS는 단순 문자열 치환보다는 다음처럼 RESULTS_ROOT 뒤의 상대경로만 남기도록 바꾸는 게 안전합니다.

SAVE_DIR = RESULTS_ROOT / "NatureLM" / "anuraset" / "wp2_ualb_singlelabel_l0"
평가가 끝나면 먼저 현재 YAML 관련 파일만 별도 커밋해두는 것도 좋습니다.

git add config.example.yaml src/config.py .gitignore
git add scripts/NatureLM/anuraset/evaluation.py
git commit -m "Add YAML configuration and migrate evaluation script"
그다음 나머지 파일을 자동 변환하고 새 커밋으로 분리하면, 문제가 생겨도 첫 단계로 쉽게 돌아갈 수 있습니다.

결론적으로 모든 코드를 바꿔야 하지만, 마지막 순간까지 미루지는 말고 지금 대표 파일 검증이 끝난 직후 일괄 변환하는 게 가장 좋습니다.


근데 다른 논문들도 올릴때 혹시 yaml을 올리나? 아니면 그냥 그대로 올리려나 내 path만 좀 지우고

둘 다 흔합니다. 논문 코드 저장소라고 해서 반드시 YAML을 써야 하는 것은 아닙니다.

보통은 세 가지 방식 중 하나입니다.

간단한 저장소
코드 안의 개인 절대경로만 지우고, 상대경로나 Path를 사용합니다.

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EMBED_PATH = PROJECT_ROOT / "embeddings" / "embeddings.npy"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "anuraset_taxonomy.csv"
데이터가 저장소 밖에 있는 경우
환경변수나 명령행 인자로 데이터 루트만 받습니다.

DATA_ROOT = Path(
    os.environ.get("BIOACOUSTIC_DATA_ROOT", "/path/to/data")
)
실험 조합과 설정이 많은 경우
YAML, JSON, Hydra 같은 설정 파일을 사용합니다.

현재 저장소는 NatureLM/BirdNET, AnuraSet/ESC-50, single/multi-label, normal/unknown/unseen처럼 경우가 많아서 YAML이 잘 맞기는 합니다. 하지만 논문 공개가 목적이고, 코드를 대대적으로 다시 설계하고 싶지 않다면 YAML까지 모든 파일에 적용할 필요는 없습니다.

제 판단으로는 지금은 이렇게 하는 것이 가장 현실적입니다.

개인 경로 /user/gaad2403/...와 /fs/dss/home/...는 모두 제거

저장소 내부 파일은 PROJECT_ROOT 기준 상대경로 사용

외부 데이터와 임베딩 경로만 config.yaml 또는 환경변수로 관리

하이퍼파라미터는 각 실험 코드에 그대로 두어도 됨

YAML에는 경로만 넣기

즉, YAML을 너무 크게 만들지 않고 아래 정도만 관리하면 충분합니다.

project_root: /path/to/Bachelor-Thesis
storage_root: /path/to/bioacoustic_storage

datasets:
  anuraset:
    audio_dir: anuraset/anuraset/audio
    metadata: anuraset/anuraset/metadata.csv

  esc50:
    audio_dir: ESC-50-master/audio

embeddings:
  naturelm_anuraset: embeddings
  birdnet_anuraset: embeddings_anuraset_birdnet
  naturelm_esc50: embeddings_esc50
  birdnet_esc50: embeddings_esc50_birdnet
각 코드에서는 이런 식으로 사용합니다.

EMBED_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
META_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "anuraset_taxonomy.csv"
SAVE_DIR = RESULTS_ROOT / "NatureLM" / "anuraset" / "wp2_ualb_singlelabel"
결론적으로 YAML을 쓰는 것은 좋은 선택이지만 필수는 아닙니다. 지금처럼 코드가 많은 경우에는 YAML에 경로만 모으고, 실험별 하이퍼파라미터와 결과 폴더명은 기존 코드에 남겨두는 방식이 가장 부담이 적고 공개용으로도 충분합니다.


자 그러면 아까 했던거 리셋하고 다시 해야겠네?

네. 아직 커밋하지 않았다면 리셋하고, YAML에는 경로만 두는 간단한 방식으로 다시 가는 게 좋습니다.

다만 config.example.yaml, config.yaml, src/config.py까지 모두 없애고 완전히 처음부터 시작할 필요는 없습니다. 아래처럼 정리하면 됩니다.

1. 방금 수정한 코드만 되돌리기
git reset --hard HEAD
새로 만든 추적되지 않은 파일은 남을 수 있으니 확인하세요.

git status --short
2. 남은 설정 파일 확인
ls -l config.yaml config.example.yaml src/config.py
경로 전용 YAML 방식은 그대로 쓸 수 있으므로 이 세 파일은 유지해도 됩니다.

다만 config.example.yaml 안에서 training, evaluation, models 같은 부분은 빼고, 아래처럼 경로만 남기는 걸 권합니다.

project:
  root: /path/to/Bachelor-Thesis

storage:
  root: /path/to/bioacoustic_storage

datasets:
  anuraset:
    audio_dir: anuraset/anuraset/audio
    metadata_path: anuraset/anuraset/metadata.csv
    taxonomy_path: data/anuraset_taxonomy.csv
    taxonomy_unknown_path: data/anuraset_taxonomy_unknown.csv
    unseen_split_path: data/unseen_species_split.json

  esc50:
    audio_dir: ESC-50-master/audio
    labels_path: data/esc50_bga_labels.csv
    unseen_split_path: data/unseen_category_split.json

embeddings:
  naturelm:
    anuraset_dir: embeddings
    esc50_dir: embeddings_esc50

  birdnet:
    anuraset_dir: embeddings_anuraset_birdnet
    esc50_dir: embeddings_esc50_birdnet

results:
  root: results
src/config.py도 경로만 읽도록 두면 됩니다. 하이퍼파라미터는 각 스크립트에 그대로 유지하세요.

3. 개인용 config.yaml 확인
project:
  root: /fs/dss/home/gaad2403/Bachelor-Thesis

storage:
  root: /fs/dss/home/gaad2403/bachelor_thesis
그리고 .gitignore에는:

config.yaml
이 있어야 합니다.

4. 다시 수정할 때는 경로만 바꾸기
예를 들어 evaluation.py에서는 하이퍼파라미터는 그대로 둡니다.

INPUT_DIM = 4096
PROBE_EPOCHS = 300
PROBE_LR = 0.01
N_INTER_MULT = 5
SEED = 42
경로만 YAML에서 가져옵니다.

from config import (
    NATURELM_ANURASET_EMBED_DIR,
    ANURASET_TAXONOMY_PATH,
    RESULTS_ROOT,
)

EMBED_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings.npy"
META_PATH = NATURELM_ANURASET_EMBED_DIR / "embeddings_metadata.csv"
TAXONOMY_PATH = ANURASET_TAXONOMY_PATH
OUTPUT_DIR = RESULTS_ROOT / "NatureLM" / "anuraset" / "wp3_eval_total_hf1"
즉, 리셋은 하되 YAML 구조 자체는 버리지 말고, 경로 설정만 관리하는 가벼운 형태로 정리하면 됩니다.


네, YAML 없이도 충분합니다. 논문 코드 공개가 목적이라면 가장 현실적인 방식은 다음입니다.

개인 절대경로만 지우고, README에 사용자가 수정해야 할 경로와 실행 파일을 명확히 적으면 됩니다.

예를 들어 각 코드 상단을 이렇게 바꾸세요.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/path/to/bioacoustic_data")

EMBED_PATH = DATA_ROOT / "embeddings" / "embeddings.npy"
META_PATH = DATA_ROOT / "embeddings" / "embeddings_metadata.csv"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "anuraset_taxonomy.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "NatureLM"
    / "anuraset"
    / "wp3_eval_total_hf1"
)
또는 더 간단하게 코드에 아래처럼 예시 경로만 남겨도 됩니다.

BASE_DIR = Path("/path/to/bioacoustic_data")
PROJECT_DIR = Path("/path/to/Bachelor-Thesis")
README에는 다음처럼 안내하면 됩니다.

Data and Path Setup
The original audio datasets, extracted embeddings, and trained checkpoints are not included in this repository.

Before running a script, update the path variables near the top of the corresponding Python file.

The expected external data structure is:

/path/to/bioacoustic_data/
├── anuraset/
│   └── anuraset/
│       ├── audio/
│       └── metadata.csv
├── ESC-50-master/
│   └── audio/
├── embeddings/
├── embeddings_anuraset_birdnet/
├── embeddings_esc50/
└── embeddings_esc50_birdnet/
Repository metadata files are stored under:

data/
├── anuraset_taxonomy.csv
├── anuraset_taxonomy_unknown.csv
├── esc50_bga_labels.csv
├── unseen_species_split.json
└── unseen_category_split.json
Embedding Extraction
NatureLM with AnuraSet:

python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
NatureLM with ESC-50:

python scripts/NatureLM/esc50/baseline/extract_embeddings.py
BirdNET with AnuraSet:

python scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
BirdNET with ESC-50:

python scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
Evaluation
NatureLM single-label AnuraSet evaluation:

PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation.py
BirdNET single-label AnuraSet evaluation:

PYTHONPATH=src python scripts/birdnet/anuraset/evaluation.py
NatureLM multi-label AnuraSet evaluation:

PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation.py
NatureLM unknown-inclusive multi-label evaluation:

PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py
Unseen-species evaluation:

PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation_unseen.py
Each script contains its own experiment-specific paths and hyperparameters. Review and update the path variables before execution.

이 방식이면 해야 할 일은 두 가지뿐입니다.

GitHub에 공개할 .py 파일에서 /user/gaad2403와 /fs/dss/home/gaad2403를 /path/to/... 형태로 바꾸기

README에 실행 파일과 예상 폴더 구조 적기

YAML 방식보다 덜 세련되지만, 논문 재현용 저장소로는 충분히 허용되는 방식입니다. 특히 스크립트가 많고 이미 실험이 끝난 상태라면 이쪽이 더 실용적입니다.


esc50_bga_labels.csv 저기에 없는데

맞아요. 제가 README 예시에서 잘못 넣었습니다.

현재 코드에서는 esc50_bga_labels.csv를 저장소의 data/ 폴더에서 읽도록 되어 있지만, 실제 파일이 없다면 README에 있다고 쓰면 안 됩니다. NatureLM·BirdNET ESC-50 추출 코드 모두 이 파일을 입력으로 요구합니다. 

 


먼저 실제 위치를 찾으세요.

find . /fs/dss/home/gaad2403/bachelor_thesis \
  -name 'esc50_bga_labels.csv' 2>/dev/null
검색 결과가 예를 들어 아래처럼 나오면:

/fs/dss/home/gaad2403/bachelor_thesis/ESC-50-master/meta/esc50_bga_labels.csv
각 추출 코드의 경로를 그 위치로 바꾸면 됩니다.

LABELS_PATH = Path(
    "/fs/dss/home/gaad2403/bachelor_thesis/ESC-50-master/meta/esc50_bga_labels.csv"
)
검색 결과가 아예 없다면, 이 파일은 build_bga_labels.py로 먼저 생성하는 구조일 가능성이 큽니다. 저장소 구조에 해당 파일이 확인됩니다.

sed -n '1,200p' scripts/NatureLM/esc50/baseline/build_bga_labels.py
그 파일 안의 입력·출력 경로를 확인한 뒤 실행하세요.

python scripts/NatureLM/esc50/baseline/build_bga_labels.py
README에는 당분간 아래처럼 쓰는 게 정확합니다.

ESC-50 hierarchy labels
ESC-50 does not include the BGA hierarchy used in this study.
The hierarchy label file must be generated before extracting embeddings.

python scripts/NatureLM/esc50/baseline/build_bga_labels.py
After generation, update LABELS_PATH in the ESC-50 scripts to the actual location of esc50_bga_labels.csv.

그리고 README의 이 부분은 삭제하거나 수정해야 합니다.

data/esc50_bga_labels.csv
실제 위치를 찾은 뒤 그 경로에 맞춰 README를 다시 정리하면 됩니다.


(naturelm-env) [gaad2403@hpcl001 Bachelor-Thesis]$ find . /fs/dss/home/gaad2403/bachelor_thesis \
>   -name 'esc50_bga_labels.csv' 2>/dev/null
./data/esc50_bga_labels.csv

그러면 파일은 정상적으로 저장소 안에 있습니다.

현재 위치:

Bachelor-Thesis/data/esc50_bga_labels.csv
코드에 적힌 경로:

LABELS_PATH = Path(
    "/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv"
)
도 실제 저장소 위치와 맞는 구조입니다. 서버에서 /user/...와 /fs/dss/home/...가 같은 홈 영역을 가리키는 차이만 있었던 것으로 보입니다.

README에는 아래처럼 적으면 됩니다.

Repository metadata files are stored under:

data/
├── anuraset_taxonomy.csv
├── anuraset_taxonomy_unknown.csv
├── esc50_bga_labels.csv
├── unseen_species_split.json
└── unseen_category_split.json
esc50_bga_labels.csv contains the BGA and intermediate-category hierarchy used to reorganize the original ESC-50 labels.

Git이 이 CSV를 실제로 추적하는지도 확인하세요.

git check-ignore -v data/esc50_bga_labels.csv
git ls-files data/esc50_bga_labels.csv
첫 번째 명령은 아무 출력이 없어야 하고, 두 번째 명령은 아래 경로를 출력해야 정상입니다.

data/esc50_bga_labels.csv
추적되지 않고 있다면:

git add data/esc50_bga_labels.csv
현재 .gitignore의 규칙이:

*.csv
!data/*.csv
이므로 data/ 바로 아래 CSV는 GitHub에 올라가도록 설정되어 있습니다.


좋아. 그러면 이제 readme를 한번 제대로 작성해보자. 실행에 대해서도 embedding, bga, 이런거 어떻게 하는지 알려주는게 낫지? 

네. README에는 최소한 아래 흐름이 보이게 하는 게 좋습니다.

데이터 준비

ESC-50 BGA 라벨 생성

NatureLM/BirdNET 임베딩 추출

학습

평가

결과 파일 위치

현재 코드 기준으로 바로 쓸 수 있게 정리한 초안입니다.

Hierarchy-Aware Representation Learning for Bioacoustic Data
This repository contains the code used to study hierarchy-aware representation learning for bioacoustic and environmental audio classification.

The experiments compare frozen audio embeddings from NatureLM-Audio and BirdNET using hierarchical learning objectives originally developed for computer vision.

Overview
Two datasets are used:

AnuraSet: Family → Genus → Species

ESC-50: BGA → Mid-category → Category

Two pretrained embedding models are compared:

NatureLM-Audio: 4096-dimensional embeddings

BirdNET: 1024-dimensional embeddings

The pretrained embedding models remain frozen. Only lightweight projection or hierarchy-classification heads are trained.

The evaluated methods are:

Baseline

UALB / Hierarchical Multi-Label Contrastive Learning

Focal Tree-Min (FTM-only)

Tree-Triplet (TT-only)

Combined Focal Tree-Min and Tree-Triplet (FTM+TT)

Experiments include single-label, multi-label, unknown-inclusive, and unseen-class settings.

Repository Structure
├── data/
│   ├── anuraset_taxonomy.csv
│   ├── anuraset_taxonomy_unknown.csv
│   ├── esc50_bga_labels.csv
│   ├── unseen_species_split.json
│   └── unseen_category_split.json
├── scripts/
│   ├── NatureLM/
│   ├── birdnet/
│   └── multilabel/
├── src/
│   ├── NatureLM/
│   ├── birdnet/
│   ├── multilabel/
│   └── models/
└── results/
Large audio files, embedding arrays, model checkpoints, and logs are not included in this repository.

Data Preparation
Download the original datasets separately.

The expected external data structure is:

/path/to/bioacoustic_data/
├── anuraset/
│   └── anuraset/
│       ├── audio/
│       └── metadata.csv
└── ESC-50-master/
    ├── audio/
    └── meta/
Before running the scripts, update the path variables near the top of the relevant Python files.

For example:

BASE_DIR = Path("/path/to/bioacoustic_data")
Repository metadata such as taxonomy files and predefined evaluation splits is stored under data/.

ESC-50 BGA Hierarchy
ESC-50 does not provide the soundscape hierarchy used in this study.

The original categories are reorganized into:

BGA → Mid-category → Category
where BGA contains:

Biophony

Geophony

Anthrophony

The hierarchy file is stored at:

data/esc50_bga_labels.csv
To regenerate it, update the paths in the following script and run:

python scripts/NatureLM/esc50/baseline/build_bga_labels.py
The generated CSV is then used by both NatureLM and BirdNET ESC-50 embedding extraction scripts.

Embedding Extraction
NatureLM × AnuraSet
python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
Expected output:

embeddings/
├── embeddings.npy
├── species_labels.npy
└── embeddings_metadata.csv
NatureLM × ESC-50
python scripts/NatureLM/esc50/baseline/extract_embeddings.py
Expected output:

embeddings_esc50/
├── embeddings.npy
└── embeddings_metadata.csv
BirdNET × AnuraSet
python scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
Expected output:

embeddings_anuraset_birdnet/
├── embeddings.npy
├── species_labels.npy
├── embeddings_metadata.csv
└── summary.txt
BirdNET × ESC-50
python scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
Expected output:

embeddings_esc50_birdnet/
├── embeddings.npy
├── embeddings_metadata.csv
└── summary.txt
NatureLM audio is converted to mono, resampled to 16 kHz, and pooled into one embedding per recording. BirdNET embeddings are averaged across segments.

Training
The training scripts are organized by embedding model, dataset, and method.

Examples:

UALB
PYTHONPATH=src python scripts/NatureLM/anuraset/ualb/train_ualb_single.py
FTM + Tree-Triplet
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py
FTM-only
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only.py
TT-only
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only.py
Equivalent scripts are available for BirdNET, ESC-50, multi-label, unknown-inclusive, and unseen-class experiments.

The main AnuraSet experiments use:

batch size: 256
epochs: 100
learning rate: 1e-3
weight decay: 1e-4
projection dimension: 128
hidden dimension: 1024
validation split: 0.1
The ESC-50 FTM+TT experiment uses 450 epochs.

Evaluation
NatureLM × AnuraSet Single-Label
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation.py
BirdNET × AnuraSet Single-Label
PYTHONPATH=src python scripts/birdnet/anuraset/evaluation.py
NatureLM × AnuraSet Multi-Label
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation.py
NatureLM × AnuraSet Unknown-Inclusive Multi-Label
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py
NatureLM × AnuraSet Unseen-Species
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation_unseen.py
Equivalent BirdNET and ESC-50 evaluation scripts are available in the corresponding directories.

Evaluation metrics include:

Accuracy

Macro-F1 and Micro-F1

Hierarchical Precision, Recall, and F1

Normalized Mutual Information

Embedding similarity Gap

AUROC for unseen-class detection

The evaluation scripts load the trained checkpoints for UALB, FTM-only, TT-only, and FTM+TT and compare them with the frozen baseline.

Main Findings
Current results show that no single hierarchy-aware method performs best in every setting.

The main observations are:

hierarchy-aware learning often improves fine-grained classification,

improvements are largest in the multi-label AnuraSet experiments,

FTM-only performs particularly well in multi-label classification,

UALB performs strongly in several single-label settings,

and larger embedding-space separation does not always produce better classification performance.

Requirements
Main dependencies include:

torch
torchaudio
numpy
pandas
scikit-learn
matplotlib
tqdm
tensorflow
birdnet
NatureLM-Audio must be available through:

NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
BirdNET experiments use:

birdnet.load("acoustic", "2.4", "tf")
Notes
Several scripts were originally developed on an HPC cluster and may contain environment-specific paths.

Before execution, check:

dataset paths,

embedding paths,

checkpoint paths,

result directories,

and Python module paths.

SLURM scripts are not required when running the Python files directly.

Publication Status
The manuscript is currently being finalized.

Working title:
Hierarchy-Aware Representation Learning for Bioacoustic Data

Citation and license information will be added after publication.

이 정도면 너무 길지 않으면서도, 방문자가 어떤 순서로 무엇을 실행해야 하는지 이해할 수 있습니다. 특히 BGA 생성과 네 가지 임베딩 추출 명령은 넣는 게 좋습니다.



results_tables(2)(1).pdf
PDF
이게 예시 결과 중 하나인데 이거 중간에 예시 넣어주면 좋지 않아? 

네. README 중간에 대표 결과표 하나를 넣는 게 좋습니다. 전체 10개 표를 다 넣기보다는, 연구의 핵심 메시지가 가장 잘 드러나는 NatureLM × AnuraSet multi-label 결과를 예시로 넣는 것을 권합니다.

이 표에서는 FTM-only가 species-level Macro-F1 54.03, Micro-F1 83.12, hF1 86.72로 가장 좋은 성능을 보여, “계층학습이 특히 multi-label 설정에서 효과적이었다”는 논문의 핵심을 잘 보여줍니다. 


README의 Main Findings 다음에 아래 섹션을 넣으면 자연스럽습니다.

Example Results
The table below shows a representative result from the NatureLM × AnuraSet multi-label experiment.

Method	Family Macro-F1	Genus Macro-F1	Species Macro-F1	Species Micro-F1	hF1
Baseline	51.29	60.50	36.73	56.36	62.07
UALB	63.47	67.23	42.23	64.77	73.96
FTM-only	61.72	70.04	54.03	83.12	86.72
TT-only	55.02	58.10	34.70	52.44	60.26
FTM+TT	51.99	59.77	35.34	53.91	60.16
FTM-only provides the strongest fine-grained and hierarchical classification performance in this setting. The TT-based methods produce larger embedding-space Gap values, but this does not consistently translate into better downstream classification performance.

Complete results for all datasets, embedding models, label settings, and unseen-class experiments are reported in the accompanying manuscript.

README 전체 흐름은 이렇게 잡으면 됩니다.

Overview
Repository Structure
Data Preparation
ESC-50 BGA Hierarchy
Embedding Extraction
Training
Evaluation
Main Findings
Example Results
Requirements
Publication Status
더 간단하게 하고 싶다면 표에는 다음 세 열만 남겨도 됩니다.

| Method | Species Macro-F1 | Species Micro-F1 | hF1 |
GitHub README에서는 이 축약형이 가장 보기 좋습니다.


자 그러면 이제 readme를 .md 파일로 줄래?

README.md 


라이브러리
/
README.md


# Hierarchy-Aware Representation Learning for Bioacoustic Data

This repository contains the code used to study hierarchy-aware representation learning for bioacoustic and environmental audio classification.

The experiments compare frozen embeddings from **NatureLM-Audio** and **BirdNET** using hierarchical learning objectives originally developed for computer vision.

## Overview

Two datasets and hierarchy types are used:

- **AnuraSet**: Family → Genus → Species
- **ESC-50**: BGA → Mid-category → Category

Two pretrained embedding models are compared:

- **NatureLM-Audio**: 4096-dimensional embeddings
- **BirdNET**: 1024-dimensional embeddings

The pretrained embedding models remain frozen. Only lightweight projection or hierarchy-classification heads are trained.

The evaluated methods are:

- Baseline
- UALB / Hierarchical Multi-Label Contrastive Learning
- Focal Tree-Min (`FTM-only`)
- Tree-Triplet (`TT-only`)
- Combined Focal Tree-Min and Tree-Triplet (`FTM+TT`)

Experiments include single-label, multi-label, unknown-inclusive, and unseen-class settings.

## Repository Structure

```text
├── data/
│   ├── anuraset_taxonomy.csv
│   ├── anuraset_taxonomy_unknown.csv
│   ├── esc50_bga_labels.csv
│   ├── unseen_species_split.json
│   └── unseen_category_split.json
├── scripts/
│   ├── NatureLM/
│   ├── birdnet/
│   └── multilabel/
├── src/
│   ├── NatureLM/
│   ├── birdnet/
│   ├── multilabel/
│   └── models/
└── results/
```

Large audio files, embedding arrays, model checkpoints, and logs are not included in this repository.

## Data Preparation

Download AnuraSet and ESC-50 separately.

Expected external data structure:

```text
/path/to/bioacoustic_data/
├── anuraset/
│   └── anuraset/
│       ├── audio/
│       └── metadata.csv
└── ESC-50-master/
    ├── audio/
    └── meta/
```

Before running a script, update the path variables near the top of the corresponding Python file.

Example:

```python
BASE_DIR = Path("/path/to/bioacoustic_data")
PROJECT_DIR = Path("/path/to/Bachelor-Thesis")
```

Repository metadata, taxonomy files, and evaluation splits are stored under `data/`.

## ESC-50 BGA Hierarchy

ESC-50 does not provide the soundscape hierarchy used in this study.

The original categories are reorganized as:

```text
BGA → Mid-category → Category
```

The highest-level BGA categories are:

- Biophony
- Geophony
- Anthrophony

The hierarchy file is stored at:

```text
data/esc50_bga_labels.csv
```

To regenerate it, update the paths in the script and run:

```bash
python scripts/NatureLM/esc50/baseline/build_bga_labels.py
```

The resulting CSV is used by both NatureLM and BirdNET ESC-50 scripts.

## Embedding Extraction

### NatureLM × AnuraSet

```bash
python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
```

Expected output:

```text
embeddings/
├── embeddings.npy
├── species_labels.npy
└── embeddings_metadata.csv
```

### NatureLM × ESC-50

```bash
python scripts/NatureLM/esc50/baseline/extract_embeddings.py
```

Expected output:

```text
embeddings_esc50/
├── embeddings.npy
└── embeddings_metadata.csv
```

### BirdNET × AnuraSet

```bash
python scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
```

Expected output:

```text
embeddings_anuraset_birdnet/
├── embeddings.npy
├── species_labels.npy
├── embeddings_metadata.csv
└── summary.txt
```

### BirdNET × ESC-50

```bash
python scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
```

Expected output:

```text
embeddings_esc50_birdnet/
├── embeddings.npy
├── embeddings_metadata.csv
└── summary.txt
```

NatureLM audio is converted to mono, resampled to 16 kHz, and pooled into one embedding per recording. BirdNET embeddings are averaged across segments.

## Training

Training scripts are organized by embedding model, dataset, and method.

Examples:

### UALB

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/ualb/train_ualb_single.py
```

### FTM + Tree-Triplet

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py
```

### FTM-only

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only.py
```

### TT-only

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only.py
```

Equivalent scripts are available for BirdNET, ESC-50, multi-label, unknown-inclusive, and unseen-class experiments.

## Evaluation

### NatureLM × AnuraSet Single-Label

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation.py
```

### BirdNET × AnuraSet Single-Label

```bash
PYTHONPATH=src python scripts/birdnet/anuraset/evaluation.py
```

### NatureLM × AnuraSet Multi-Label

```bash
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation.py
```

### NatureLM × AnuraSet Unknown-Inclusive Multi-Label

```bash
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py
```

### NatureLM × AnuraSet Unseen-Species

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation_unseen.py
```

Equivalent BirdNET and ESC-50 evaluation scripts are available in the corresponding directories.

Evaluation metrics include:

- Accuracy
- Macro-F1 and Micro-F1
- Hierarchical Precision, Recall, and F1
- Normalized Mutual Information
- Embedding similarity Gap
- AUROC for unseen-class detection

## Example Results

Representative results from the **NatureLM × AnuraSet multi-label** experiment are shown below.

| Method | Species Macro-F1 | Species Micro-F1 | hF1 |
|---|---:|---:|---:|
| Baseline | 36.73 | 56.36 | 62.07 |
| UALB | 42.23 | 64.77 | 73.96 |
| **FTM-only** | **54.03** | **83.12** | **86.72** |
| TT-only | 34.70 | 52.44 | 60.26 |
| FTM+TT | 35.34 | 53.91 | 60.16 |

FTM-only provides the strongest fine-grained and hierarchical classification performance in this setting.

Across experiments, hierarchy-aware learning often improves fine-grained classification, but no single method performs best in every setting. Larger embedding-space separation also does not always translate into better downstream classification.

## Requirements

Main dependencies include:

```text
torch
torchaudio
numpy
pandas
scikit-learn
matplotlib
tqdm
tensorflow
birdnet
```

NatureLM-Audio is loaded with:

```python
NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
```

BirdNET experiments use:

```python
birdnet.load("acoustic", "2.4", "tf")
```

## Notes

Several scripts were originally developed on an HPC cluster and may contain environment-specific paths.

Before execution, check:

- dataset paths,
- embedding paths,
- checkpoint paths,
- result directories,
- and Python module paths.

SLURM scripts are not required when running the Python files directly.

## Publication Status

The manuscript is currently being finalized.

**Working title:**  
*Hierarchy-Aware Representation Learning for Bioacoustic Data*

Citation and license information will be added after publication.
라이브러리
/
README.md


# Hierarchy-Aware Representation Learning for Bioacoustic Data

This repository contains the code used to study hierarchy-aware representation learning for bioacoustic and environmental audio classification.

The experiments compare frozen embeddings from **NatureLM-Audio** and **BirdNET** using hierarchical learning objectives originally developed for computer vision.

## Overview

Two datasets and hierarchy types are used:

- **AnuraSet**: Family → Genus → Species
- **ESC-50**: BGA → Mid-category → Category

Two pretrained embedding models are compared:

- **NatureLM-Audio**: 4096-dimensional embeddings
- **BirdNET**: 1024-dimensional embeddings

The pretrained embedding models remain frozen. Only lightweight projection or hierarchy-classification heads are trained.

The evaluated methods are:

- Baseline
- UALB / Hierarchical Multi-Label Contrastive Learning
- Focal Tree-Min (`FTM-only`)
- Tree-Triplet (`TT-only`)
- Combined Focal Tree-Min and Tree-Triplet (`FTM+TT`)

Experiments include single-label, multi-label, unknown-inclusive, and unseen-class settings.

## Repository Structure

```text
├── data/
│   ├── anuraset_taxonomy.csv
│   ├── anuraset_taxonomy_unknown.csv
│   ├── esc50_bga_labels.csv
│   ├── unseen_species_split.json
│   └── unseen_category_split.json
├── scripts/
│   ├── NatureLM/
│   ├── birdnet/
│   └── multilabel/
├── src/
│   ├── NatureLM/
│   ├── birdnet/
│   ├── multilabel/
│   └── models/
└── results/
```

Large audio files, embedding arrays, model checkpoints, and logs are not included in this repository.

## Data Preparation

Download AnuraSet and ESC-50 separately.

Expected external data structure:

```text
/path/to/bioacoustic_data/
├── anuraset/
│   └── anuraset/
│       ├── audio/
│       └── metadata.csv
└── ESC-50-master/
    ├── audio/
    └── meta/
```

Before running a script, update the path variables near the top of the corresponding Python file.

Example:

```python
BASE_DIR = Path("/path/to/bioacoustic_data")
PROJECT_DIR = Path("/path/to/Bachelor-Thesis")
```

Repository metadata, taxonomy files, and evaluation splits are stored under `data/`.

## ESC-50 BGA Hierarchy

ESC-50 does not provide the soundscape hierarchy used in this study.

The original categories are reorganized as:

```text
BGA → Mid-category → Category
```

The highest-level BGA categories are:

- Biophony
- Geophony
- Anthrophony

The hierarchy file is stored at:

```text
data/esc50_bga_labels.csv
```

To regenerate it, update the paths in the script and run:

```bash
python scripts/NatureLM/esc50/baseline/build_bga_labels.py
```

The resulting CSV is used by both NatureLM and BirdNET ESC-50 scripts.

## Embedding Extraction

### NatureLM × AnuraSet

```bash
python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
```

Expected output:

```text
embeddings/
├── embeddings.npy
├── species_labels.npy
└── embeddings_metadata.csv
```

### NatureLM × ESC-50

```bash
python scripts/NatureLM/esc50/baseline/extract_embeddings.py
```

Expected output:

```text
embeddings_esc50/
├── embeddings.npy
└── embeddings_metadata.csv
```

### BirdNET × AnuraSet

```bash
python scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
```

Expected output:

```text
embeddings_anuraset_birdnet/
├── embeddings.npy
├── species_labels.npy
├── embeddings_metadata.csv
└── summary.txt
```

### BirdNET × ESC-50

```bash
python scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
```

Expected output:

```text
embeddings_esc50_birdnet/
├── embeddings.npy
├── embeddings_metadata.csv
└── summary.txt
```

NatureLM audio is converted to mono, resampled to 16 kHz, and pooled into one embedding per recording. BirdNET embeddings are averaged across segments.

## Training

Training scripts are organized by embedding model, dataset, and method.

Examples:

### UALB

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/ualb/train_ualb_single.py
```

### FTM + Tree-Triplet

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py
```

### FTM-only

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only.py
```

### TT-only

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only.py
```

Equivalent scripts are available for BirdNET, ESC-50, multi-label, unknown-inclusive, and unseen-class experiments.

## Evaluation

### NatureLM × AnuraSet Single-Label

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation.py
```

### BirdNET × AnuraSet Single-Label

```bash
PYTHONPATH=src python scripts/birdnet/anuraset/evaluation.py
```

### NatureLM × AnuraSet Multi-Label

```bash
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation.py
```

### NatureLM × AnuraSet Unknown-Inclusive Multi-Label

```bash
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py
```

### NatureLM × AnuraSet Unseen-Species

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation_unseen.py
```

Equivalent BirdNET and ESC-50 evaluation scripts are available in the corresponding directories.

Evaluation metrics include:

- Accuracy
- Macro-F1 and Micro-F1
- Hierarchical Precision, Recall, and F1
- Normalized Mutual Information
- Embedding similarity Gap
- AUROC for unseen-class detection

## Example Results

Representative results from the **NatureLM × AnuraSet multi-label** experiment are shown below.

| Method | Species Macro-F1 | Species Micro-F1 | hF1 |
|---|---:|---:|---:|
| Baseline | 36.73 | 56.36 | 62.07 |
| UALB | 42.23 | 64.77 | 73.96 |
| **FTM-only** | **54.03** | **83.12** | **86.72** |
| TT-only | 34.70 | 52.44 | 60.26 |
| FTM+TT | 35.34 | 53.91 | 60.16 |

FTM-only provides the strongest fine-grained and hierarchical classification performance in this setting.

Across experiments, hierarchy-aware learning often improves fine-grained classification, but no single method performs best in every setting. Larger embedding-space separation also does not always translate into better downstream classification.

## Requirements

Main dependencies include:

```text
torch
torchaudio
numpy
pandas
scikit-learn
matplotlib
tqdm
tensorflow
birdnet
```

NatureLM-Audio is loaded with:

```python
NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
```

BirdNET experiments use:

```python
birdnet.load("acoustic", "2.4", "tf")
```

## Notes

Several scripts were originally developed on an HPC cluster and may contain environment-specific paths.

Before execution, check:

- dataset paths,
- embedding paths,
- checkpoint paths,
- result directories,
- and Python module paths.

SLURM scripts are not required when running the Python files directly.

## Publication Status

The manuscript is currently being finalized.

**Working title:**  
*Hierarchy-Aware Representation Learning for Bioacoustic Data*

Citation and license information will be added after publication.