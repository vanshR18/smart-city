# =============================================================================
# SmartCityAI — Makefile
# Shortcuts for every common dev task.
# Run `make help` to see all commands.
# =============================================================================

.PHONY: help up down dev test train deploy clean logs shell db-shell

# Colours for terminal output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
RESET  := \033[0m

help: ## Show this help
	@echo ""
	@echo "$(BLUE)SmartCityAI — Available Commands$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Infrastructure ────────────────────────────────────────────────────────────
up: ## Start all Docker services (build if needed)
	docker-compose up --build -d
	@echo "$(GREEN)✓ All services started$(RESET)"
	@echo "  Dashboard  → http://localhost:3000"
	@echo "  API Docs   → http://localhost:8000/docs"
	@echo "  MLflow UI  → http://localhost:5000"
	@echo "  pgAdmin    → http://localhost:5050"

down: ## Stop all Docker services
	docker-compose down
	@echo "$(YELLOW)All services stopped$(RESET)"

down-volumes: ## Stop services AND delete all data volumes
	docker-compose down -v
	@echo "$(YELLOW)All services and volumes removed$(RESET)"

restart: ## Restart all services
	docker-compose restart

logs: ## Tail logs from all services
	docker-compose logs -f

logs-backend: ## Tail backend logs only
	docker-compose logs -f backend

# ── Local development (no Docker for backend/frontend) ────────────────────────
dev-infra: ## Start only DB + Redis + MLflow in Docker (for local dev)
	docker-compose up -d db redis mlflow
	@echo "$(GREEN)✓ Infrastructure ready$(RESET)"
	@echo "  Postgres → localhost:5432"
	@echo "  Redis    → localhost:6379"
	@echo "  MLflow   → http://localhost:5000"

dev-backend: ## Run backend with hot-reload (needs dev-infra first)
	cd backend && uvicorn main:app --reload --port 8000

dev-frontend: ## Run frontend dev server with HMR
	cd frontend && npm run dev

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run all backend unit tests
	cd backend && \
		DATABASE_URL=postgresql://x:x@localhost/x \
		python -m pytest tests/ -v --tb=short

test-watch: ## Run tests in watch mode (re-runs on file change)
	cd backend && \
		DATABASE_URL=postgresql://x:x@localhost/x \
		python -m pytest tests/ -v --tb=short -f

test-coverage: ## Run tests with coverage report
	cd backend && \
		DATABASE_URL=postgresql://x:x@localhost/x \
		python -m pytest tests/ --cov=app --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report: backend/htmlcov/index.html$(RESET)"

# ── ML Training ───────────────────────────────────────────────────────────────
dataset: ## Download and prepare NLP training dataset
	cd backend && python scripts/download_dataset.py

train-nlp: ## Fine-tune DistilBERT NLP classifier
	cd backend && python scripts/train_nlp.py

train-nlp-fast: ## Quick training run (1 epoch, for testing)
	cd backend && python scripts/train_nlp.py --fast

train-timeseries: ## Train Prophet time-series models
	cd backend && python scripts/train_timeseries.py

train-all: dataset train-nlp train-timeseries ## Train all models end-to-end

retrain: ## Run MLOps retraining pipeline manually
	cd backend && python scripts/retrain_cron.py

retrain-force: ## Force retrain and promote (skip F1 gate)
	cd backend && python scripts/retrain_cron.py --force

retrain-dry-run: ## Check what data would be used for retraining
	cd backend && python scripts/retrain_cron.py --dry-run

mlflow-ui: ## Open MLflow UI locally
	cd backend && mlflow ui --port 5000
	@echo "$(GREEN)MLflow UI → http://localhost:5000$(RESET)"

# ── Data management ───────────────────────────────────────────────────────────
seed: ## Seed DB with 30 days of historical data
	curl -s -X POST "http://localhost:8000/simulate/seed-historical" | python -m json.tool

simulate: ## Fire 20 live events through the full pipeline
	curl -s -X POST "http://localhost:8000/simulate/batch?n=20" \
		| python -m json.tool | head -30

# ── Database ──────────────────────────────────────────────────────────────────
db-shell: ## Open psql shell in the DB container
	docker-compose exec db psql -U smartcity -d smartcitydb

db-backup: ## Backup the database to a timestamped file
	@mkdir -p backups
	docker-compose exec -T db \
		pg_dump -U smartcity smartcitydb \
		> backups/smartcitydb_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Backup saved to backups/$(RESET)"

# ── Shells ────────────────────────────────────────────────────────────────────
shell: ## Open a shell inside the backend container
	docker-compose exec backend bash

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

# ── Build & Deploy ────────────────────────────────────────────────────────────
build: ## Build Docker images without starting
	docker-compose build

build-no-cache: ## Force rebuild (ignores Docker layer cache)
	docker-compose build --no-cache

push: ## Build and push images to GitHub Container Registry
	docker build -t ghcr.io/$(GITHUB_USER)/smartcityai-backend:latest ./backend
	docker build -t ghcr.io/$(GITHUB_USER)/smartcityai-frontend:latest ./frontend
	docker push ghcr.io/$(GITHUB_USER)/smartcityai-backend:latest
	docker push ghcr.io/$(GITHUB_USER)/smartcityai-frontend:latest

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove Python cache files and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cache files removed$(RESET)"

clean-docker: ## Remove stopped containers, dangling images, unused volumes
	docker system prune -f
	@echo "$(GREEN)✓ Docker pruned$(RESET)"

# ── Health checks ─────────────────────────────────────────────────────────────
health: ## Check all service health endpoints
	@echo "$(BLUE)API Health:$(RESET)"
	@curl -sf http://localhost:8000/health | python -m json.tool || echo "Backend down"
	@echo "\n$(BLUE)MLflow:$(RESET)"
	@curl -sf http://localhost:5000/health && echo " OK" || echo "MLflow down"

status: ## Show running Docker services
	docker-compose ps