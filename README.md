# SmartCity AI — Emergency Response System

> End-to-end AI + Full Stack + MLOps project for real-time emergency detection,
> risk scoring, and live alerting across Lucknow city.

![CI](https://github.com/vanshR18/smart-city/actions/workflows/ci.yml/badge.svg)

---

## What This Builds

A production-grade system that:

- **Detects emergencies** from text (tweets/reports) using a fine-tuned DistilBERT NLP model
- **Detects incidents** from images/video using YOLOv8 object detection
- **Scores risk** using a weighted multi-signal fusion engine (CV + NLP + location + time)
- **Forecasts peak hours** using Facebook Prophet time-series models
- **Alerts in real-time** via Telegram bot and WebSocket dashboard
- **Self-heals** via an automated MLOps retraining pipeline with MLflow experiment tracking

```
[CCTV / Text / Sensors]
         ↓
[Data Pipeline — Redis Streams]
         ↓
[ML Models Layer]
  ├── DistilBERT  →  emergency classification + urgency score
  ├── YOLOv8      →  visual detection (accident, fire, crowd)
  └── Prophet     →  peak hour / high-risk zone prediction
         ↓
[Risk Scoring Engine]
  score = (CV × 0.5) + (NLP × 0.2) + (Location × 0.2) + (Time × 0.1)
         ↓
[FastAPI Backend + PostgreSQL/PostGIS]
         ↓
[React Dashboard]       [Telegram Alerts]
  Live Leaflet map       CRITICAL / HIGH
  Heatmap overlay        with explanation
  Analytics charts
         ↓
[MLOps Pipeline — MLflow + APScheduler]
  Auto-retrain every 24h
  Compare F1 → promote or reject
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| NLP | DistilBERT (HuggingFace Transformers) |
| CV | YOLOv8 (Ultralytics) + OpenCV |
| Time-series | Facebook Prophet |
| Risk Engine | Custom weighted fusion (Python) |
| MLOps | MLflow experiment tracking + model registry |
| Backend | FastAPI + SQLAlchemy + PostGIS |
| Real-time | Redis Streams + WebSocket |
| Alerts | Telegram Bot API |
| Frontend | React + Vite + Tailwind + Leaflet |
| Database | PostgreSQL 15 + PostGIS 3.3 |
| Deploy | Docker + Docker Compose + GitHub Actions |

---

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+
- Node.js 20+
- Git

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/smart-city.git
cd smart-city

# Configure secrets
cp .env.example .env
cp backend/.env.example backend/.env
# Edit backend/.env — add Telegram token (optional)
```

### 2. Start everything with Docker

```bash
make up
# Or: docker-compose up --build
```

Open:
- **Dashboard** → http://localhost:3000
- **API Docs** → http://localhost:8000/docs
- **MLflow UI** → http://localhost:5000
- **pgAdmin** → http://localhost:5050

### 3. Seed data and test

```bash
# Seed 30 days of historical data
make seed

# Fire 20 live events (watch the dashboard update in real-time)
make simulate
```

---

## Local Development (without Docker for code)

```bash
# Start only DB + Redis + MLflow
make dev-infra

# Terminal 1 — Backend (hot-reload)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make dev-backend

# Terminal 2 — Frontend (HMR)
make dev-frontend
```

---

## ML Model Training

```bash
# 1. Build NLP training dataset (210+ Lucknow emergency sentences)
make dataset

# 2. Fine-tune DistilBERT (5 epochs, ~20 min on CPU, ~5 min on GPU)
make train-nlp

# 3. Train Prophet time-series models
make train-timeseries

# 4. View all experiments in MLflow
make mlflow-ui
```

After training, the API automatically hot-reloads the new model — no restart needed.

---

## MLOps — Automated Retraining

The system retrains itself every 24 hours:

```
1. Pull recent events from PostgreSQL (new training signal)
2. Mix with synthetic baseline (stability)
3. Fine-tune DistilBERT
4. Compare new F1 vs production F1
5. Promote if improvement ≥ 1% → hot-reload inference engine
6. Reject otherwise → keep current model
7. Log full audit trail to MLflow
```

Trigger manually:
```bash
make retrain           # normal (respects F1 gate)
make retrain-force     # always promote
make retrain-dry-run   # preview data only
```

Or via API:
```bash
curl -X POST http://localhost:8000/mlops/retrain \
  -H "Content-Type: application/json" \
  -d '{"force": false, "num_epochs": 3}'
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | All service status |
| GET | `/events` | Recent incidents with risk scores |
| GET | `/stats` | Summary counts |
| POST | `/predict/text` | NLP classify + risk score |
| POST | `/predict/image` | YOLOv8 detect + risk score |
| POST | `/predict/video` | Multi-frame video analysis |
| POST | `/risk/score` | Raw signal → risk score |
| GET | `/risk/heatmap` | Area aggregation for map |
| GET | `/risk/time-profile` | 24h risk profile |
| GET | `/alerts` | Alert history |
| WS | `/ws/live` | Real-time event stream |
| POST | `/mlops/retrain` | Trigger retraining |
| GET | `/mlops/model-registry` | Current model version |
| GET | `/mlops/experiments` | MLflow run history |
| GET | `/mlops/data-drift` | Distribution drift report |
| POST | `/simulate/batch` | Generate N test events |

Full interactive docs: http://localhost:8000/docs

---

## Testing

```bash
make test            # all unit tests (83 tests)
make test-watch      # re-run on file change
make test-coverage   # with HTML coverage report
```

Tests cover:
- Phase 1: DB models, simulator, risk scoring formula
- Phase 2: NLP inference, label consistency, batch prediction
- Phase 3: CV detector, Risk Engine, time-series analyzer
- Phase 4: Alert engine, WebSocket manager, Telegram formatter

---

## Project Structure

```
smart-city/
├── backend/
│   ├── app/
│   │   ├── alerts/          # WebSocket, Telegram, alert engine
│   │   ├── cv/              # YOLOv8 detector, frame extractor
│   │   ├── mlops/           # Retraining pipeline, MLflow
│   │   ├── models/          # SQLAlchemy DB models
│   │   ├── nlp/             # DistilBERT training + inference
│   │   ├── risk_engine/     # Weighted fusion scorer
│   │   ├── routers/         # FastAPI route handlers
│   │   ├── simulator/       # Lucknow event generator
│   │   └── timeseries/      # Prophet models + analyzer
│   ├── scripts/             # Training + retraining scripts
│   ├── tests/               # 83 unit tests
│   └── main.py              # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── components/      # Map, alerts feed, analytics, header
│       └── hooks/           # WebSocket, event state management
├── .github/workflows/       # GitHub Actions CI/CD
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Deployment

### Railway (recommended for portfolio)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and create project
railway login
railway init

# Deploy backend
railway up --service backend

# Add environment variables
railway variables set TELEGRAM_BOT_TOKEN=...
railway variables set DATABASE_URL=...
```

### Render

1. Connect GitHub repo to Render
2. Create Web Service → select `backend/` → set build command `pip install -r requirements.txt`
3. Add environment variables in Render dashboard
4. Create Static Site → select `frontend/` → build command `npm run build` → publish `dist/`

### Manual Docker on any VPS

```bash
# On your server
git clone https://github.com/YOUR_USERNAME/smart-city.git
cd smart-city
cp .env.example .env      # fill in secrets
cp backend/.env.example backend/.env

docker-compose -f docker-compose.yml up -d
```

---

## Risk Scoring Formula

```
risk_score (0–100) =
  (cv_score       × 0.50) +   ← YOLOv8 detection confidence
  (nlp_score      × 0.20) +   ← DistilBERT urgency score
  (location_score × 0.20) +   ← historical hotspot risk
  (time_score     × 0.10)     ← Prophet peak-hour factor
  + event_type_severity_boost

Levels:  CRITICAL ≥75  |  HIGH ≥55  |  MEDIUM ≥35  |  LOW <35
```

Every score includes a full explanation: which signal dominated, why the area
is high-risk, whether it's peak hour, and the exact weighted breakdown.

---

## What Makes This Stand Out

1. **Multi-signal fusion** — not just one model, but CV + NLP + geospatial + temporal signals combined into one explainable score
2. **MLOps pipeline** — automated retraining with F1 gating, model versioning, and zero-downtime hot-reload
3. **Production patterns** — Redis Streams, WebSocket broadcast, PostGIS, Docker multi-stage builds, non-root containers
4. **Explainability** — every risk score tells you *why* (dominant signal, reasons, formula breakdown)
5. **Local context** — Lucknow-specific training data in Hindi-English (Hinglish), real area coordinates

---

## Author

Built for placement portfolio — demonstrating end-to-end ML systems engineering.
