# Direct VLM Document QA Evaluation

A small evaluation pipeline for answering questions directly from document images
with a vision-language model (VLM). The project measures answer quality, request
latency, token usage, retries, and failures without using a separate OCR stage.

## Objective

The goal is to evaluate a simple image-to-answer workflow:

```text
Document image + question -> GPT-5 -> structured answer -> offline evaluation
```

The experiment uses 500 questions sampled reproducibly from the validation split
of [DocumentVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA):

- 120 development questions for testing the prompt and pipeline.
- 380 document-disjoint held-out questions for final evaluation.

No training or fine-tuning is performed. Documents used in development are
excluded from the held-out set.

## Evaluation

The model receives a high-detail document image and one question through the
OpenAI Responses API. It returns a structured answer and confidence value. The
default configuration uses `gpt-5`, minimal reasoning effort, and a maximum of
1,000 output tokens.

The primary metric is **Average Normalized Levenshtein Similarity (ANLS)**, which
gives partial credit for small OCR-like character errors and assigns zero to
answers beyond its 0.5 distance threshold. Normalized Exact Match and token-level
F1 are retained as supporting metrics.

The evaluator also reports request latency, token usage, retries, API errors, and
missing predictions. Evaluation is offline once predictions have been generated.

## Final results

Results on the 380-question held-out subset:

| Metric | Result |
|---|---:|
| ANLS (primary) | **93.25%** |
| Normalized Exact Match | 87.89% |
| Token F1 | 93.13% |
| Mean request latency | 3.07 s |
| Median request latency | 2.96 s |
| P95 request latency | 5.07 s |
| Retries | 0 |
| Errors | 0 |
| Missing predictions | 0 |

All 380 questions produced successful predictions. Average usage was 818 input
tokens and 35.5 output tokens per question.

These are project results on a held-out subset of the official validation split,
not scores from the hidden DocumentVQA test set or official leaderboard. Latency
measures the API request rather than complete end-to-end program runtime.

## Project structure

```text
ocr-vlm-experiment/
├── src/ocr_vlm/           # Dataset, inference, and evaluation logic
├── scripts/               # Thin Python entry points
├── tests/                 # Offline unit tests
├── data/                  # Generated questions and document images
├── outputs/predictions/   # Generated model predictions
├── results/               # Generated aggregate metrics
├── .env.example
├── pyproject.toml
└── README.md
```

The `ocr-vlm-*` commands installed from `pyproject.toml` are the recommended way
to run the pipeline. The files in `scripts/` provide equivalent Python entry
points.

## Run guide

### 1. Set up the environment

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create the local environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your-key-here
```

The key is loaded automatically from `.env`. Dataset preparation and inference
require internet access, and inference incurs OpenAI API usage.

### 2. Prepare the dataset

```bash
ocr-vlm-prepare
```

This downloads the required DocumentVQA validation shards, selects the
document-isolated subsets, and writes:

```text
data/processed/questions/dev.jsonl
data/processed/questions/test.jsonl
data/raw/documentvqa/
```

### 3. Develop and validate

Run inference and evaluation on the 120-question development subset:

```bash
ocr-vlm-infer --overwrite
ocr-vlm-evaluate
```

For a cheaper initial smoke test, add `--max-questions 2` to both commands. Freeze
the model and prompt settings after development before evaluating the held-out
subset.

### 4. Run the held-out evaluation

```bash
ocr-vlm-infer \
  --questions data/processed/questions/test.jsonl \
  --output outputs/predictions/test.jsonl \
  --overwrite

ocr-vlm-evaluate \
  --questions data/processed/questions/test.jsonl \
  --predictions outputs/predictions/test.jsonl \
  --output results/test_metrics.json
```

Inference resumes an existing prediction file by default. Use `--overwrite` when
you intentionally want a fresh run.

### 5. Run the offline tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic records and mocked model responses; they do not download
the dataset or call the OpenAI API.
