# KeyExtract — TF-IDF Keyword Intelligence

> Surface the highest-signal terms from any document instantly.  
> Flask + Scikit-learn backend · Plain HTML/CSS/JS frontend · MLflow versioning · Render + Vercel deployment.

**Live Demo:**
- 🌐 Frontend: [https://intelligentkeywordextractionengine-9x2zohixh.vercel.app](https://intelligentkeywordextractionengine-9x2zohixh.vercel.app)
- ⚙️ Backend API: [https://intelligent-keyword-extraction-engine-nlp.onrender.com](https://intelligent-keyword-extraction-engine-nlp.onrender.com)
- 🧪 MLflow Registry: [https://dagshub.com/bhaumikmango/Keywords-Extraction-NLP.mlflow](https://dagshub.com/bhaumikmango/Keywords-Extraction-NLP.mlflow)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start (Local)](#quick-start-local)
5. [Deploying the Backend on Render](#deploying-the-backend-on-render)
6. [Deploying the Frontend on Vercel](#deploying-the-frontend-on-vercel)
7. [MLOps — Model Versioning with MLflow + DagsHub](#mlops--model-versioning-with-mlflow--dagshub)
8. [Known Gotchas & Fixes](#known-gotchas--fixes)
9. [API Reference](#api-reference)
10. [Environment Variables](#environment-variables)

---

## Overview

**KeyExtract** uses a pre-trained TF-IDF pipeline (CountVectorizer → TfidfTransformer) to rank and extract the most statistically distinctive keywords from any text. The backend is a lightweight Flask REST API; the frontend is a zero-dependency HTML file deployable as a static site.

Key features:
- ⚡ Near-zero latency inference (pure in-memory vectorisation)
- 🔢 Configurable keyword count (1–30)
- 📊 Score visualisation with animated bars and keyword chips
- 📋 One-click copy to clipboard
- 📡 Live API health indicator with configurable endpoint
- 🔁 MLflow model registry integration for versioned deployments on DagsHub

---

## Architecture

```
┌─────────────────────────────────┐        ┌──────────────────────────────┐
│   Frontend  (Vercel)            │        │   Backend  (Render)          │
│   index.html                    │──────▶│   Flask API  /api/extract    │
│   Pure HTML / CSS / JS          │  POST  │   Gunicorn  (1 worker)       │
│   No framework, no bundler      │◀──────│   Scikit-learn + NLTK        │
└─────────────────────────────────┘  JSON  └──────────────┬───────────────┘
                                                          │
                                              ┌───────────▼──────────────┐
                                              │  MLflow  (DagsHub)       │
                                              │  Model registry          │
                                              │  Experiment tracking     │
                                              │  Artefact storage (.pkl) │
                                              └──────────────────────────┘
```

**Model loading flow on Render startup:**
1. Gunicorn starts with `--preload` flag
2. Flask app loads — `load_mlflow()` runs synchronously before workers fork
3. MLflow downloads artifacts from DagsHub (~8 files, ~5–10 seconds)
4. Model is loaded into memory
5. Gunicorn binds to port and workers are forked
6. Service goes live with model already warm

> **Why `--preload` matters:** Without it, Render's port scanner times out before the model finishes loading, causing the deploy to fail with "No open ports detected".

---

## Project Structure

```
keyextract/
│
├── frontend/                    # Deployed to Vercel
│   ├── index.html               # Complete SPA — zero dependencies
│   └── vercel.json              # Vercel static routing config
│
├── backend/                     # Deployed to Render
│   ├── extract.py               # Flask API (local + MLflow loaders)
│   ├── keyword_model.py         # MLflow PythonModel wrapper class
│   ├── mlflow_log.py            # One-time script: register models in MLflow
│   ├── requirements.txt         # Python dependencies
│   ├── render.yaml              # Render deployment spec
│   ├── .env.example             # Environment variable template
│   ├── Count_Vector.pkl         # Pre-trained CountVectorizer
│   ├── TFIDF_Transformer.pkl    # Pre-trained TfidfTransformer
│   └── Feature_Names.pkl        # Vocabulary feature names
│
└── README.md
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.10 or 3.11
- pip

> ⚠️ **Python 3.14 is NOT supported** — `pyarrow` (an MLflow dependency) has no pre-built wheel for 3.14 and fails to compile from source. Use 3.10 or 3.11.

### 1. Clone the repository

```bash
git clone https://github.com/bhaumikmango/Keywords-Extraction-NLP.git
cd Keywords-Extraction-NLP
```

### 2. Set up the backend

```bash
cd backend

python -m venv venv

# Activate — macOS/Linux:
source venv/bin/activate

# Activate — Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# For local dev the defaults work fine — MODEL_SOURCE=local
```

### 4. Start the server

Make sure the three `.pkl` files are in `backend/` alongside `extract.py`, then:

```bash
python extract.py
# → ✅  Local models loaded successfully
# → Starting server on 0.0.0.0:5000
```

### 5. Test the API

**Windows PowerShell:**
```powershell
# Health check
Invoke-WebRequest -Uri http://localhost:5000/api/health -Method GET | Select-Object -ExpandProperty Content

# Extract keywords
Invoke-WebRequest -Uri http://localhost:5000/api/extract -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"text": "Machine learning models learn patterns from large training datasets using gradient descent optimisation algorithms.", "numKeywords": 5}' `
  | Select-Object -ExpandProperty Content
```

**macOS / Linux / Git Bash:**
```bash
# Health check
curl http://localhost:5000/api/health

# Extract keywords
curl -X POST http://localhost:5000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning models learn patterns from training data using gradient descent.", "numKeywords": 5}'
```

### 6. Open the frontend

```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows — double-click the file, or:
start frontend/index.html
```

Click **⚙ API endpoint** and confirm it reads `http://localhost:5000`. The status pill turns 🟢 green when the backend is reachable.

> **macOS tip:** AirPlay Receiver uses port 5000. If you see "Address already in use", run `PORT=5001 python extract.py` and update the frontend endpoint to `http://localhost:5001`.

---

## Deploying the Backend on Render

[Render](https://render.com) free tier spins down after 15 minutes of inactivity. The first request after a cold start takes ~30 seconds.

### Step-by-step

1. Push your code to GitHub.

2. Go to [render.com](https://render.com) → **New +** → **Web Service** → connect your repo.

3. Configure the service:

   | Field | Value |
   |-------|-------|
   | **Root Directory** | `backend` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn extract:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --graceful-timeout 300 --preload` |
   | **Instance Type** | Free |

4. Add environment variables:

   | Key | Value |
   |-----|-------|
   | `MODEL_SOURCE` | `local` |
   | `FLASK_DEBUG` | `0` |
   | `PYTHON_VERSION` | `3.10.11` |

5. Click **Create Web Service** and watch the build logs.

> **Critical — start command flags:**
> - `--preload` loads the app before forking workers, so the model is ready before port binding
> - `--timeout 300` gives MLflow model downloads enough time to complete
> - Without these, Render will fail with "No open ports detected"

---

## Deploying the Frontend on Vercel

### Option A — Vercel dashboard

1. Go to [vercel.com](https://vercel.com) → **Add New Project** → import your repo.
2. Set **Root Directory** to `frontend`.
3. Leave all build settings blank (pure static file, no build step).
4. Click **Deploy**.

### Option B — Vercel CLI

```bash
npm install -g vercel
cd frontend
vercel
# Framework preset: Other
# Root: . (current directory)
```

### Connect to your Render backend

Once deployed, open your Vercel URL, expand **⚙ API endpoint** at the bottom of the input card, and paste your Render URL. The status pill will flip to 🟢 **api online**.

---

## MLOps — Model Versioning with MLflow + DagsHub

### Why this matters

| Without MLflow | With MLflow |
|----------------|-------------|
| `.pkl` files committed to git | Models stored in a registry with version history |
| No audit trail | Every run logs parameters, metrics, and artefacts |
| "Which model is live?" requires digging through git | Explicit version promotion |
| Rollback = manually swapping files | Change one env var on Render |
| Hard to reproduce results | Full lineage: code + config + artefacts |

---

### Step 1 — Create a DagsHub account

1. Sign up at [dagshub.com](https://dagshub.com) using **Connect with GitHub**
2. Create a repo or connect your existing GitHub repo
3. Your MLflow tracking URI is always:
   ```
   https://dagshub.com/<your-username>/<your-repo>.mlflow
   ```
   > There is no separate MLflow tab in the UI — just construct this URI from your username and repo name.
4. Generate an access token at `https://dagshub.com/user/settings/tokens`

### Step 2 — Configure `.env`

```bash
cd backend
cp .env.example .env
```

Fill in your values:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
MLFLOW_EXPERIMENT_NAME=keyword-extractor
MLFLOW_MODEL_NAME=keyword-extractor-tfidf
```

Load on Windows PowerShell:
```powershell
Get-Content .env | ForEach-Object {
  $var = $_.Split('=', 2)
  [System.Environment]::SetEnvironmentVariable($var[0], $var[1])
}
```

Load on macOS/Linux:
```bash
export $(cat .env | xargs)
```

### Step 3 — Register the model

```bash
python mlflow_log.py
```

Expected output:
```
INFO | Artifact paths:
INFO |   count_vector: .../Count_Vector.pkl
INFO |   tfidf_transformer: .../TFIDF_Transformer.pkl
INFO |   feature_names: .../Feature_Names.pkl
INFO | Raw .pkl artefacts logged
INFO | Model registered as 'keyword-extractor-tfidf'
✅  Done — model: keyword-extractor-tfidf
```

### Step 4 — Promote a version on DagsHub

1. Go to DagsHub → **Models** tab → click `keyword-extractor-tfidf`
2. Click the version you want to deploy
3. Click **"Promote model"** (top right)
4. In the dialog, type `Production` in the text field → click **Promote**

> **Note:** DagsHub's newer UI uses model aliases instead of classic MLflow stages. You won't see a "Staging/Production" dropdown — instead, type the alias name directly into the promotion dialog.

### Step 5 — Switch Render to load from MLflow

Update these variables in your Render dashboard:

| Key | Value |
|-----|-------|
| `MODEL_SOURCE` | `mlflow` |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/<repo>.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | your DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | your DagsHub token |
| `MLFLOW_MODEL_NAME` | `keyword-extractor-tfidf` |
| `MLFLOW_MODEL_STAGE` | version number, e.g. `6` |

Trigger a manual deploy. Watch for:
```
✅  MLflow model loaded: models:/keyword-extractor-tfidf/6
```

### Iterative versioning workflow

```bash
# 1. Update your .pkl files in backend/
# 2. Log a new version — creates version N+1 automatically
python mlflow_log.py

# 3. On DagsHub, promote the new version to Production
# 4. On Render, update MLFLOW_MODEL_STAGE to N+1
# 5. Trigger a manual deploy
```

To roll back: set `MLFLOW_MODEL_STAGE` to any previous version number and redeploy. No code changes required.

---

## Known Gotchas & Fixes

Real issues hit during deployment — documented so you don't repeat them.

### 1. Windows backslash baked into MLflow artifact metadata

**Problem:** `mlflow_log.py` run on Windows stores absolute Windows paths (`C:\Users\...`) in the logged model's metadata. On Linux (Render), these paths don't exist, causing:
```
No such file or directory: '/tmp/tmpXXX/C:\Users\...\Count_Vector.pkl'
```

**Fix:** Use `pathlib.Path.as_posix()` for all artifact paths, and `os.chdir(BASE)` before `log_model` so only the filename (not the full path) is stored:
```python
BASE = pathlib.Path(__file__).parent.resolve()
os.chdir(BASE)
artifacts = { "count_vector": PKL_CV.as_posix(), ... }
```

---

### 2. Render "No open ports detected" deploy failure

**Problem:** MLflow model download takes 10–15 seconds. Render's port scanner times out and kills the deploy before gunicorn binds.

**Fix:** Add `--preload` to the gunicorn start command. This loads the entire Flask app (including model download) in the master process *before* workers are forked and the port is bound:
```
gunicorn extract:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --graceful-timeout 300 --preload
```

---

### 3. MLflow code-based logging requires `set_model()`

**Problem:** Using `python_model="keyword_model.py"` raises:
```
MlflowException: ensure the model is set using mlflow.models.set_model()
```

**Fix:** Add this at the bottom of `keyword_model.py`:
```python
import mlflow.models
mlflow.models.set_model(KeywordExtractorModel())
```

---

### 4. `pyarrow` build failure on Python 3.14

**Problem:** Render uses the latest Python by default. `pyarrow` has no pre-built wheel for Python 3.14 and fails to compile from source with a CMake error.

**Fix:** Set `PYTHON_VERSION=3.10.11` in Render environment variables, and pin `pyarrow==17.0.0` in `requirements.txt` to force a pre-built binary wheel.

---

### 5. CloudPickle serialization error when logging model

**Problem:** Defining `KeywordExtractorModel` inline in `mlflow_log.py` causes:
```
PicklingError: args[0] from __newobj__ args has the wrong class
```

**Fix:** Move the class to a separate file `keyword_model.py` and import it:
```python
from keyword_model import KeywordExtractorModel
```

---

### 6. Windows PowerShell curl syntax

**Problem:** `curl` in PowerShell is an alias for `Invoke-WebRequest` and rejects Unix-style `-H` and `-d` flags.

**Fix:** Use the full PowerShell syntax:
```powershell
Invoke-WebRequest -Uri http://localhost:5000/api/extract -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"text": "...", "numKeywords": 5}' `
  | Select-Object -ExpandProperty Content
```
Or use Git Bash / WSL where standard `curl` works normally.

---

### 7. Background thread + Render restart loop

**Problem:** Loading the model in a background thread (to avoid blocking port binding) causes `models_loaded` to stay `false` — Render detects the port, marks the service live, then immediately restarts the process, killing the thread before it finishes.

**Fix:** Use synchronous loading with `--preload` instead of background threads. The model loads once in the gunicorn master process before any worker or port binding occurs.

---

## API Reference

### `GET /api/health`

Returns server and model status.

**Response 200**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_source": "mlflow",
  "model_name": "keyword-extractor-tfidf"
}
```

---

### `POST /api/extract`

Extract keywords from text using TF-IDF scoring.

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

**Response 200** — keywords with TF-IDF scores, sorted descending:
```json
{
  "neural":    0.4821,
  "gradient":  0.3976,
  "learning":  0.3541,
  "network":   0.3102,
  "backprop":  0.2988
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
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode — **never in production** |
| `PYTHON_VERSION` | — | Set to `3.10.11` on Render to avoid pyarrow build failures |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `keyword-extractor` | Experiment name |
| `MLFLOW_MODEL_NAME` | `keyword-extractor-tfidf` | Registered model name |
| `MLFLOW_MODEL_STAGE` | `Production` | Version number or alias to load (e.g. `6`) |
| `MLFLOW_TRACKING_USERNAME` | — | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | — | DagsHub access token |

---

*Built with Flask, Scikit-learn, NLTK, MLflow, DagsHub, and vanilla JS. Deployed on Render + Vercel.*
