# Dynamic Granularity Selection for Retrieval Augmented Generation

Medical RAG project based on Mix-of-Granularity (MoG). The system builds multi-level corpus chunks, trains a neural router to select useful retrieval granularity, retrieves evidence with BM25, generates answers with an LLM, and exposes the pipeline through FastAPI and a chatbot frontend.

## Project Structure

```text
src/
  data_processing/   corpus parsing, MoG corpus creation, retrieval-similarity scoring
  training/          soft-label generation and router training
  models/            router model
  retrieval/         BM25 and MoG retrieval system
  pipeline/          MedRAG answer-generation pipeline
  api/               FastAPI backend
  evaluation/        dataset loading and answer extraction
  prompts/           prompt templates
scripts/             inference and evaluation entry points
frontend/            chatbot UI
checkpoints/         trained router checkpoint
prediction_results/ saved prediction outputs
```

## Setup

```bash
pip install -r requirements.txt
```

Update `src/config.py` with local paths and required API keys before running generation or API inference.

## Main Commands

Build MoG corpus levels:

```bash
python src/data_processing/build_mog_corpus.py
```

Run retrieval and similarity scoring for one level:

```bash
python src/data_processing/retrieve_and_similarity.py --level half
python src/data_processing/retrieve_and_similarity.py --level 1
python src/data_processing/retrieve_and_similarity.py --level 2
python src/data_processing/retrieve_and_similarity.py --level 4
python src/data_processing/retrieve_and_similarity.py --level 8
```

Build soft labels:

```bash
python src/training/build_softlabels_top.py
# or
python src/training/build_softlabels_avg.py
```

Train router:

```bash
python src/training/train.py
```

Run inference:

```bash
python scripts/run_inference.py --dataset medqa --method mog --llm llama3
```

Run API:

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `frontend/index.html` to use the chatbot UI with the API.

## Methods

- `cot`: LLM-only baseline.
- `medrag`: BM25 retrieval without router.
- `mog`: router-guided multi-granularity retrieval.

## Key Artifacts

- Router checkpoint: `checkpoints/best_model_fold_5.pt`
- Evaluation data: `qa_datasets_rawdata/eval_data.json`
- Example output: `prediction_results/mog_medqa_llama3.json`