# KeyExtract — TF-IDF Keyword Intelligence

> Surface the highest-signal terms from any document instantly.  
> Flask + Scikit-learn backend · Plain HTML/CSS/JS frontend · MLflow versioning · Render + Vercel deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start (Local)](#quick-start-local)
5. [Deploying the Backend on Render](#deploying-the-backend-on-render)
6. [Deploying the Frontend on Vercel](#deploying-the-frontend-on-vercel)
7. [MLOps — Model Versioning with MLflow + DagsHub](#mlops--model-versioning-with-mlflow--dagshub)
8. [API Reference](#api-reference)
9. [Environment Variables](#environment-variables)
10. [Contributing](#contributing)

---

## Overview

**KeyExtract** uses a pre-trained TF-IDF pipeline (CountVectorizer → TfidfTransformer) to rank and extract the most statistically distinctive keywords from any text. The backend is a lightweight Flask REST API; the frontend is a zero-dependency HTML file deployable as a static site.

Key features:
- ⚡ Near-zero latency inference (pure in-memory vectorisation)
- 🔢 Configurable keyword count (1–30)
- 📊 Score visualisation with animated bars and keyword chips
- 📋 One-click copy to clipboard
- 📡 Live API health indicator
- 🔁 MLflow model registry integration for versioned deployments

---

## Architecture

```
┌─────────────────────────────────┐       ┌──────────────────────────────┐
│   Frontend  (Vercel)            │       │   Backend  (Render)          │
│   index.html                    │──────▶│   Flask API  /api/extract    │
│   Pure HTML / CSS / JS          │  POST │   Gunicorn  (1 worker)       │
│   No framework, no bundler      │◀──────│   Scikit-learn + NLTK        │
└─────────────────────────────────┘  JSON └──────────────┬───────────────┘
                                                          │ optional
                                              ┌───────────▼──────────────┐
                                              │  MLflow  (DagsHub)        │
                                              │  Model registry           │
                                              │  Experiment tracking      │
                                              │  Artefact storage         │
                                              └──────────────────────────┘
```

---

## Project Structure

```
keyextract/
│
├── frontend/                   # Deployed to Vercel
│   ├── index.html              # Complete SPA (no framework)
│   └── vercel.json             # Vercel routing config
│
├── backend/                    # Deployed to Render
│   ├── extract.py              # Flask API (local + MLflow loaders)
│   ├── mlflow_log.py           # One-time script: register models in MLflow
│   ├── requirements.txt        # Python dependencies
│   ├── render.yaml             # Render deployment spec
│   ├── .env.example            # Environment variable template
│   ├── Count_Vector.pkl        # Pre-trained CountVectorizer
│   ├── TFIDF_Transformer.pkl   # Pre-trained TfidfTransformer
│   └── Feature_Names.pkl       # Vocabulary feature names
│
└── README.md
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/keyextract.git
cd keyextract
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
# Edit .env if needed — defaults work for local dev
```

Ensure the three `.pkl` files are in the `backend/` directory, then start the server:

```bash
python extract.py
# → Server running at http://localhost:5000
```

Verify the API is healthy:

```bash
curl http://localhost:5000/api/health
# {"models_loaded": true, "model_source": "local", "status": "healthy", ...}
```

### 3. Open the frontend

Simply open `frontend/index.html` in your browser. No build step required.

Click the **⚙ API endpoint** toggle and confirm it reads `http://localhost:5000`. The status pill in the header will turn green when the backend is reachable.

---

## Deploying the Backend on Render

[Render](https://render.com) offers a **free tier** web service that stays alive while it has traffic (spins down after 15 minutes of inactivity on free plan).

### Step-by-step

1. **Push your code to GitHub** (ensure `backend/` is at the repo root or adjust paths).

2. **Create a new Web Service** on [render.com](https://render.com):
   - Connect your GitHub repository
   - Set **Root Directory** to `backend`
   - Set **Build Command**: `pip install -r requirements.txt`
   - Set **Start Command**: `gunicorn extract:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - Select **Free** plan

3. **Add environment variables** in the Render dashboard:

   | Key | Value |
   |-----|-------|
   | `MODEL_SOURCE` | `local` |
   | `FLASK_DEBUG` | `0` |

4. **Deploy** — Render will build and deploy automatically.

5. **Copy your Render URL** (e.g. `https://keyextract-api.onrender.com`) and paste it into the frontend's API endpoint field.

> **Tip:** Render free services cold-start in ~30 seconds after inactivity. The frontend will show a timeout error on the first request — just click Extract again.

---

## Deploying the Frontend on Vercel

### Option A — Vercel CLI (recommended)

```bash
npm install -g vercel
cd frontend
vercel
# Follow prompts: framework = Other, root = .
# Vercel will detect vercel.json automatically
```

### Option B — GitHub integration

1. Push the `frontend/` folder to GitHub.
2. Go to [vercel.com](https://vercel.com) → **New Project** → import your repo.
3. Set **Root Directory** to `frontend`.
4. Click **Deploy**.

After deployment, open the live URL, expand **⚙ API endpoint**, and enter your Render backend URL.

---

## MLOps — Model Versioning with MLflow + DagsHub

This is where the MLOps workflow lives. We use **DagsHub** (free hosted MLflow server) for experiment tracking and model registry, so you get versioning without running your own infrastructure.

### Why this matters

| Without MLflow | With MLflow |
|----------------|-------------|
| `.pkl` files committed to git | Models stored in a registry with versions |
| No audit trail | Every run logs parameters, metrics, and artefacts |
| Manual deployment | Promote models to `Staging` → `Production` via UI or CLI |
| Hard to reproduce | Full lineage: data + code + environment + artefacts |

---

### Step 1 — Create a free DagsHub account

1. Sign up at [dagshub.com](https://dagshub.com)
2. Create a **new repository** (or connect your GitHub repo)
3. Go to **Integrations → MLflow** on your DagsHub repo page
4. Copy your **MLflow tracking URI** — it looks like:
   ```
   https://dagshub.com/<your-username>/<your-repo>.mlflow
   ```
5. Generate an **access token** at: `https://dagshub.com/user/settings/tokens`

---

### Step 2 — Set credentials locally

```bash
cd backend
cp .env.example .env
```

Edit `.env`:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
MLFLOW_EXPERIMENT_NAME=keyword-extractor
MLFLOW_MODEL_NAME=keyword-extractor-tfidf
```

Load the variables:

```bash
source .env         # Linux / macOS
# or on Windows:  set -a; . .env; set +a
```

---

### Step 3 — Log and register the model

```bash
python mlflow_log.py
```

This will:
- Upload `Count_Vector.pkl`, `TFIDF_Transformer.pkl`, `Feature_Names.pkl` as artefacts
- Wrap them in a `mlflow.pyfunc` model class
- Register the model as **`keyword-extractor-tfidf`** version 1

Open your DagsHub repo → **MLflow** → **Experiments** to see the run.

---

### Step 4 — Promote to Production

In the MLflow UI (on DagsHub):

1. Go to **Models** → `keyword-extractor-tfidf`
2. Click **Version 1** → **Stage** → **Transition to → Production**

---

### Step 5 — Enable MLflow loading on Render

In your Render dashboard, update the environment variables:

| Key | Value |
|-----|-------|
| `MODEL_SOURCE` | `mlflow` |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/<repo>.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | your DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | your DagsHub token |
| `MLFLOW_MODEL_NAME` | `keyword-extractor-tfidf` |
| `MLFLOW_MODEL_STAGE` | `Production` |

Trigger a **Manual Deploy** on Render. The backend will now pull the model from the registry on startup.

---

### Iterative versioning workflow

Whenever you retrain or update your models:

```bash
# 1. Replace/update your .pkl files in backend/
# 2. Re-run the logging script — this creates version 2, 3, etc.
python mlflow_log.py

# 3. In the MLflow UI, promote the new version to Production
# 4. Redeploy on Render (or it will pick up next restart)
```

No code changes needed in `extract.py` — it always loads the `Production` stage alias.

---

## API Reference

### `GET /api/health`

Returns the server and model status.

**Response 200**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_source": "local",
  "model_name": "local"
}
```

---

### `POST /api/extract`

Extract keywords from text.

**Request body**
```json
{
  "text": "Your document or passage here...",
  "numKeywords": 10
}
```

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `text` | string | ✅ | — | min 10 chars |
| `numKeywords` | integer | ❌ | 10 | 1–50 |

**Response 200** — object where keys are keywords and values are TF-IDF scores:
```json
{
  "neural":     0.4821,
  "network":    0.4103,
  "gradient":   0.3976,
  "learning":   0.3541,
  "backprop":   0.3102
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| 400 | No text / text too short / no keywords extracted |
| 500 | Models not loaded / internal server error |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SOURCE` | `local` | `local` = load .pkl from disk; `mlflow` = pull from registry |
| `PORT` | `5000` | HTTP port |
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode (never in production) |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `keyword-extractor` | Experiment name |
| `MLFLOW_MODEL_NAME` | `keyword-extractor-tfidf` | Registered model name |
| `MLFLOW_MODEL_STAGE` | `Production` | Model stage to load |
| `MLFLOW_TRACKING_USERNAME` | — | DagsHub / MLflow username |
| `MLFLOW_TRACKING_PASSWORD` | — | DagsHub token / MLflow password |

---

## Contributing

Pull requests are welcome. For major changes please open an issue first.

```bash
# Run locally
cd backend && python extract.py

# Test the extract endpoint
curl -X POST http://localhost:5000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning models learn patterns from training data using gradient descent optimisation.", "numKeywords": 5}'
```

---

*Built with Flask, Scikit-learn, NLTK, MLflow, and vanilla JS.*
