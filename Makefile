.PHONY: help install dev test lint format build up down logs clean

help:
	@echo "MediAssist AI - Available commands:"
	@echo ""
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Start development environment"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Clean up"

install:
	pip install -r requirements.txt
	cd frontend && npm install

dev:
	docker-compose up -d postgres redis chromadb minio

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check .
	mypy --strict .

format:
	black .
	ruff check --fix .

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Service-specific commands
run-auth:
	cd services/auth && uvicorn app.main:app --reload --port 8001

run-patient:
	cd services/patient && uvicorn app.main:app --reload --port 8002

run-analysis:
	cd services/analysis && uvicorn app.main:app --reload --port 8003

run-report:
	cd services/report && uvicorn app.main:app --reload --port 8004

run-frontend:
	cd frontend && npm run dev

# Database
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1
