# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MediAssist AI is a medical diagnosis support system combining ML-based symptom analysis with RAG-based medical literature search. The system provides diagnostic suggestions with evidence-based references while explicitly maintaining that final diagnosis must be made by qualified medical professionals.

## Tech Stack

**Backend:** Python 3.11+, FastAPI, SQLAlchemy 2.0 (async), Celery + Redis
**Frontend:** React 18, TypeScript, TailwindCSS, Zustand, TanStack Query, Three.js
**ML/AI:** PyTorch, scikit-learn, BioBERT (dmis-lab/biobert-v1.1), DenseNet121, Optuna (AutoML), LangChain
**Database:** PostgreSQL 15+, MongoDB 7, Redis 7.x, ChromaDB (Vector DB)
**Infrastructure:** Docker, Kubernetes, GitHub Actions

## Commands

```bash
# Development environment
docker-compose up -d postgres mongodb redis chromadb minio  # Start infrastructure only
docker-compose up -d              # Start all services
docker-compose logs -f            # View logs

# Install dependencies
make install                      # Install all (pip + npm)
pip install -r requirements.txt   # Python only
cd frontend && npm install        # Frontend only

# Run individual services (via Makefile)
make run-analysis                 # Analysis service (port 8003)
make run-auth                     # Auth service (port 8001)
make run-patient                  # Patient service (port 8002)
make run-report                   # Report service (port 8004)
make run-frontend                 # Frontend (port 3003)

# Or run directly with MongoDB env vars for auth
cd services/analysis && MONGODB_URI=mongodb://localhost:27017 MONGODB_DB=mediassist uvicorn app.main:app --reload --port 8003

# Run tests
PYTHONPATH=$PWD pytest -v                           # All tests (74 tests)
pytest tests/test_symptom_classifier.py -v          # Specific module
pytest tests/test_file.py -k test_name              # Single test
pytest --cov=models --cov=rag --cov-report=html     # With coverage report

# Code quality
make lint                         # Run ruff + mypy
make format                       # Run black + ruff fix
ruff check .                      # Linting only
black .                           # Formatting only (line-length: 100)
mypy --strict .                   # Type checking only

# Database migrations
make db-migrate                   # Run migrations (alembic upgrade head)
make db-rollback                  # Rollback one migration
```

## Architecture

### Service Ports
| Service | Port | Notes |
|---------|------|-------|
| Analysis Service | 8003 | Main backend (ML, RAG, AutoML) |
| Auth Service | 8001 | MongoDB-based auth |
| Patient Service | 8002 | Patient data management |
| Report Service | 8004 | MinIO integration |
| Frontend | 3003 / 3001 | Dev / Docker |
| PostgreSQL | 5433 | Host port (5432 internal) |
| MongoDB | 27017 | User storage |
| Redis | 6380 | Host port (6379 internal) |
| ChromaDB | 8005 | Host port (8000 internal) |
| MinIO | 9000-9001 | API / Console |

### Analysis Service Routers (`services/analysis/app/routers/`)
- `symptom.py` - Symptom analysis and disease prediction
- `image.py` - Chest X-ray analysis with DenseNet121 + Grad-CAM
- `rna.py` - RNA sequence disease prediction
- `automl.py` - Optuna-based hyperparameter optimization
- `rag.py` - Medical literature search and Q&A
- `graph.py` - Neo4j knowledge graph queries
- `auth.py` - MongoDB-based authentication

### ML Models (`models/`)

**SymptomClassifier** (`symptom_classifier/classifier.py`)
- BioBERT NER for symptom extraction + RandomForest ensemble
- Input: symptoms list + patient info → Output: disease predictions with ICD codes

**ImageAnalyzer** (`image_analyzer/model.py`)
- DenseNet121 for chest X-ray analysis + Grad-CAM heatmap visualization

**RNAPredictor** (`rna_predictor/model.py`)
- BERT-style transformer with CNN N-gram encoder
- Multi-task: RNA type classification + disease prediction + risk scoring

**AutoML** (`automl/`)
- Optuna optimizer with TPE/CMA-ES samplers, Hyperband pruning
- Ensemble generation (voting, hybrid)

### RAG System (`rag/`)

**MedicalRAG** (`medical_rag.py`)
- Embedding: BioBERT → ChromaDB (HNSW)
- Retrieval: Hybrid search (Dense + BM25) with cross-encoder reranking
- Generation: GPT-4 with medical prompts and citation formatting

### Authentication
- MongoDB-based user storage with SHA-256 password hashing
- Demo account: `demo@mediassist.ai` / `demo1234` (fallback when MongoDB unavailable)
- JWT tokens for session management

## Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | User login (MongoDB/demo fallback) |
| `/api/v1/auth/register` | POST | User registration |
| `/api/v1/symptom/analyze` | POST | Analyze symptoms and predict diseases |
| `/api/v1/symptom/extract` | POST | Extract symptoms from natural language (NER) |
| `/api/v1/image/analyze` | POST | Analyze chest X-ray images |
| `/api/v1/rna/analyze` | POST | Analyze RNA sequences for disease prediction |
| `/api/v1/rna/validate` | POST | Validate RNA sequence format |
| `/api/v1/automl/experiments` | POST | Create AutoML experiment |
| `/api/v1/automl/experiments/{id}` | GET | Get experiment status |
| `/api/v1/automl/experiments/{id}/best-model` | GET | Get best model from experiment |
| `/api/v1/rag/search` | POST | Search medical literature |
| `/api/v1/rag/query` | POST | RAG-based question answering |

API documentation available at http://localhost:8003/docs when service is running.

## Model Performance

| Model | Metric | Target | Achieved |
|-------|--------|--------|----------|
| Symptom Classifier | Accuracy | ≥ 85% | 87.5% |
| Symptom Classifier | F1-macro (AutoML) | - | 93.33% |
| RNA Predictor | Type Accuracy | ≥ 90% | 92% |
| RNA Predictor | Disease F1 | ≥ 0.75 | 0.84 |
| RNA Predictor | Accuracy (AutoML) | - | 89.55% |
| Image Analyzer | AUC-ROC | ≥ 0.90 | - |
| RAG System | Relevance | ≥ 0.70 | - |

## Medical Safety Requirements

All outputs must include disclaimer: "이 정보는 참고용이며 최종 진단은 의사가 결정합니다" (This information is for reference only; final diagnosis must be made by a physician).

- Never provide final diagnosis or prescriptions
- Always cite sources for medical information
- Red flag symptoms must trigger immediate warnings
- Explicitly state uncertainty when confidence is low

## Code Standards

- Type hints required (mypy strict mode)
- Line length: 100 (Black + Ruff)
- Test coverage ≥ 80%
- Commit messages: Conventional Commits format
