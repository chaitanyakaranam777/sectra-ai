# SENTRA AI — multimodal sentiment analysis (Phase 1: deep-learning core)

Five PyTorch models (CNN, BiLSTM, BERT, CNN+LSTM, CNN+BERT), a leakage-safe MVSA pipeline, a real training engine,
evaluation on a shared held-out split, Grad-CAM / token attribution, cached inference and a file-backed API for the dashboard.
The dashboard UI (`sentra-dashboard.html`) is unchanged and is wired to this API in the next phase.

## Install
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt            # CPU is enough; GPU notes in docs/GPU.md

## Run
    python scripts/import_mvsa.py --variant single --raw data/raw/MVSA_Single        # see docs/DATASET_IMPORT.md
    python scripts/train.py --model cnn_lstm --samples data/processed/mvsa_single/samples.csv --dataset mvsa_single
    python scripts/evaluate.py --samples data/processed/mvsa_single/samples.csv --runs artifacts/runs/*_v1
    uvicorn backend.main:app --reload                                                # http://localhost:8000/docs

## Test
    python -m pytest -q          # all tests use a synthetic MVSA-layout fixture, never real MVSA data

## Marks -> code
| Marks | Requirement | Where |
|---|---|---|
| 1 | Problem and dataset | `pipelines/ingestion/mvsa.py`, `docs/DATASET_IMPORT.md` |
| 1 | Preprocessing | `ml/data/*`, `pipelines/validation`, `pipelines/preprocessing`, `pipelines/run_pipeline.py` |
| 3 | Neural networks | `ml/architectures/*` (fusion in `fusion.py`) |
| 1 | Training / optimisation | `ml/training/*` |
| 2 | Evaluation | `ml/evaluation/*`, `scripts/evaluate.py` |
| 1 | Graphs | `history.json`, `test_metrics.json`, confusion matrices served by `backend/main.py` (charts: next phase) |
| 1 | Interpretation | `ml/explainability/*`, results of your real runs |

## Prediction API (added in step 2)
`POST /api/predict/text` (JSON), `/api/predict/image` (multipart `file`), `/api/predict/multimodal` (form `text` + `file`),
`POST /api/explain/{text|image|multimodal}`, `GET /api/predictions/history`. The best completed run of the requested modality
(by validation metric) is used unless `run_id` is given. Uploads: JPEG/PNG only, content-checked, 5 MB limit, never written to disk.
History uses SQLAlchemy; set `SENTRA_DATABASE_URL=postgresql+psycopg://...` for PostgreSQL (default: local SQLite file).
