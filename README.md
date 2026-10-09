# Hierarchical Scaffold-BPE experiment

논문의 BPE, Scaffold-BPE와 제안한 2-level Hierarchical Scaffold-BPE를 같은
byte-level/GPT-2 pretokenization 조건에서 비교하는 재현 실험 코드입니다.

## 공식 코드 상태

2026-09-28 기준 저자 [공식 저장소](https://github.com/Aaron-LHR/Scaffold-BPE)는
`Our code will be released to the public soon`이라고만 명시하며 구현을 공개하지
않았습니다. 이 프로젝트는 [AAAI 논문의 Algorithm 1/2](https://ojs.aaai.org/index.php/AAAI/article/download/34633/36788)를
독립적으로 구현합니다.

## 고정한 H-Scaffold 의미

학습 중 merge 직후 child token의 잔여 빈도 `f(a)`를 다음 queue head 빈도와
비교합니다.

- Level 0: `f(a) >= f(Qhead)`
- Level 1: `0.5 f(Qhead) <= f(a) < f(Qhead)`
- Level 2: `f(a) < 0.5 f(Qhead)`

Level 1/2는 visible vocabulary 크기에 포함하지 않습니다. 논문의 queue 재삽입에
따라 scaffold token이 이후 충분히 높은 순위가 되면 Level 0으로 재활성화됩니다.
인코딩은 전체 expanded vocabulary로 먼저 merge합니다. H-Scaffold는 남은 Level 2를
먼저 최소 Level 0/1 child sequence로 철거하고 Level 0/1 merge를 한 번 더 적용한 뒤,
남은 Level 1을 Level 0 sequence로 철거합니다. 이 규칙이 원안의 “Level 2를 더 빨리
demolish”를 재현 가능하게 만든 operational definition입니다.

## 실험 행렬

`3 tokenizers × 2 vocab sizes × 2 architectures = 12` scratch-training runs입니다.
architecture는 Llama 3.2 1B와 Llama 3.1 8B만 사용합니다. WikiText-103 train을
각각 5 epochs 학습하며 GPU 4/5/6/7에서 네 run씩 실행합니다.

## 설치 및 테스트

```bash
cd /home/jovyan/main-workspace/yerin/tokenizer
bash scripts/setup.sh
.venv/bin/pytest -q
bash scripts/run_smoke.sh
```

## 데이터 준비

데이터 revision은 `src/hscaffold_bpe/data.py`에 고정되어 있습니다.

```bash
# Tokenizer용 Wikipedia EN 1B whitespace words
.venv/bin/python -m hscaffold_bpe.data tokenizer \
  --output data/tokenizer/wikipedia_en_1b.txt \
  --word-budget 1000000000

.venv/bin/python -m hscaffold_bpe.cli count \
  data/tokenizer/wikipedia_en_1b.txt \
  data/tokenizer/wikipedia_en_1b.counts.pkl \
  --workers 32

# WikiText-103 / BLiMP / WMT17 평가 데이터
.venv/bin/python -m hscaffold_bpe.data evaluation --output-dir data/eval

```

## Tokenizer 학습과 평가

```bash
bash scripts/run_tokenizers.sh

.venv/bin/python -m hscaffold_bpe.cli evaluate \
  artifacts/tokenizers/hierarchical-32768.json \
  data/eval/wikitext103_test.txt \
  data/eval/wmt17_de_en_test_en.txt \
  data/eval/blimp_all.txt \
  --output artifacts/eval/hierarchical-32768.json
```

결과에는 fertility, bytes/token, encoding throughput, domain fertility 표준편차,
expanded vocabulary 크기와 Level별 개수가 들어갑니다.

세 trainer는 병렬 실행되고, expanded BPE encode는 Rust 기반 `tiktoken` backend를
사용합니다. demolish 결과는 별도 Python reference backend와 동일성 테스트를 거칩니다.

## LM 입력 생성과 학습

각 tokenizer로 WikiText-103 train/validation/test를 `.bin`으로 변환합니다.
vocabulary가 64K 이하이므로 token ID는 uint16입니다.

```bash
bash scripts/tokenize_wikitext.sh
bash scripts/launch_two_models.sh artifacts/tokens artifacts/lm
```

LM 결과는 token PPL과 tokenizer 간 비교 가능한 bits-per-byte를 함께 기록합니다.
모델은 pretrained checkpoint를 재사용하지 않고 Llama 3.2 1B와 Llama 3.1 8B
architecture를 새 vocabulary로 scratch 초기화합니다.

학습이 끝난 뒤 전체 tokenizer/LM test 평가와 HTML 보고서를 생성합니다.

```bash
bash scripts/evaluate_and_report.sh
# report/report.html
```

## CPU-only threshold 분석

기존 0.5 H-Scaffold 모델은 보존하고, 0.25와 0.75를 별도 경로에 학습한다.
두 vocabulary 크기에서 같은 WikiText-103 test, WMT17, BLiMP로 fertility를
비교하고, 최종 L2 전체 목록과 WikiText test에서 관측된 2차 재병합을 집계한다.

```bash
bash scripts/run_threshold_ablation.sh
# report/report.html
# artifacts/ablation/analysis.json 및 l2-t*-*.csv
```

`level2_threshold`는 child 잔여 빈도가 queue head 빈도의 몇 배 미만일 때
Level 2로 분류할지를 정한다. 이 분석에는 LM 재학습이 포함되지 않는다.
