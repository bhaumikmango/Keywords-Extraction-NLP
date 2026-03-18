"""
extract.py  —  KeyExtract Flask API
──────────────────────────────────────────────────────────────────────────────
Supports two loading strategies (chosen via env var MODEL_SOURCE):

  MODEL_SOURCE=local   (default)
      Loads Count_Vector.pkl / TFIDF_Transformer.pkl / Feature_Names.pkl
      directly from disk — good for quick local dev & Render free tier.

  MODEL_SOURCE=mlflow
      Downloads the registered model from your MLflow tracking server
      (set MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE).
      Ideal for versioned production deployments.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import re
import logging

import nltk
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_SOURCE    = os.getenv("MODEL_SOURCE", "local")         # "local" | "mlflow"
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI",  "mlruns")
MLFLOW_MODEL    = os.getenv("MLFLOW_MODEL_NAME",    "keyword-extractor-tfidf")
MLFLOW_STAGE    = os.getenv("MLFLOW_MODEL_STAGE",   "Production")   # or version number

# ── Global model state ─────────────────────────────────────────────────────
cv            = None
tfidf_trans   = None
feature_names = None
stop_words    = None
stemming      = None
lmtr          = None
mlflow_model  = None       # used only when MODEL_SOURCE=mlflow
models_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def _init_nlp():
    """Download NLTK data and return preprocessors."""
    for pkg in ("punkt", "stopwords", "wordnet", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    base_sw = set(stopwords.words("english"))
    extra   = {"fig","figure","image","sample","using","show","result",
               "large","also","one","two","three","four","five","six",
               "seven","eight","nine"}
    sw      = list(base_sw | extra)
    return sw, PorterStemmer(), WordNetLemmatizer()


def load_local():
    """Load models from local .pkl files."""
    global cv, tfidf_trans, feature_names, stop_words, stemming, lmtr, models_loaded

    try:
        logger.info("Loading models from local .pkl files…")
        with open("Count_Vector.pkl",      "rb") as f: cv            = pickle.load(f)
        with open("TFIDF_Transformer.pkl", "rb") as f: tfidf_trans   = pickle.load(f)
        with open("Feature_Names.pkl",     "rb") as f: feature_names = pickle.load(f)

        stop_words, stemming, lmtr = _init_nlp()
        models_loaded = True
        logger.info("✅  Local models loaded successfully")
        return True

    except FileNotFoundError as e:
        logger.error(f"Missing .pkl file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading local models: {e}")
        return False


def load_mlflow():
    """Load model from MLflow model registry."""
    global mlflow_model, stop_words, stemming, lmtr, models_loaded

    try:
        import mlflow
        import mlflow.pyfunc

        logger.info(f"mlflow version: {mlflow.__version__}")

        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        logger.info(f"Loading model '{MLFLOW_MODEL}' from MLflow @ {MLFLOW_URI}…")
        mlflow.set_tracking_uri(MLFLOW_URI)

        # Try stage name first, fall back to "latest"
        try:
            model_uri = f"models:/{MLFLOW_MODEL}/{MLFLOW_STAGE}"
            mlflow_model = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            model_uri = f"models:/{MLFLOW_MODEL}/latest"
            mlflow_model = mlflow.pyfunc.load_model(model_uri)

        stop_words, stemming, lmtr = _init_nlp()
        models_loaded = True
        logger.info(f"✅  MLflow model loaded: {model_uri}")
        return True

    except ImportError:
        logger.error(f"mlflow import error: {e} — falling back to local loading")
        return load_local()
    except Exception as e:
        logger.error(f"Error loading MLflow model: {e}")
        logger.warning("Falling back to local model loading…")
        return load_local()


# ── Choose loading strategy ────────────────────────────────────────────────
import threading

def _load_in_background():
    global models_loaded
    if MODEL_SOURCE == "mlflow":
        models_loaded = load_mlflow()
    else:
        models_loaded = load_local()

# Start loading in background so the port binds immediately
threading.Thread(target=_load_in_background, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) >= 3]
    tokens = [stemming.stem(w)    for w in tokens]
    tokens = [lmtr.lemmatize(w)   for w in tokens]
    return " ".join(tokens)


def _sort_coo(coo_matrix):
    return sorted(zip(coo_matrix.col, coo_matrix.data),
                  key=lambda x: (x[1], x[0]), reverse=True)


def _extract_topn(feat_names, sorted_items, topn=10):
    return {
        feat_names[idx]: round(float(score), 4)
        for idx, score in sorted_items[:topn]
        if idx < len(feat_names)
    }


def get_keywords_local(text: str, topn: int = 10) -> dict:
    proc = _preprocess(text)
    if not proc.strip():
        return {}
    vec    = tfidf_trans.transform(cv.transform([proc]))
    items  = _sort_coo(vec.tocoo())
    return _extract_topn(feature_names, items, topn)


def get_keywords_mlflow(text: str, topn: int = 10) -> dict:
    import pandas as pd
    df     = pd.DataFrame([{"text": text, "top_n": topn}])
    result = mlflow_model.predict(df)
    return result[0] if result else {}


def get_keywords(text: str, topn: int = 10) -> dict:
    if MODEL_SOURCE == "mlflow" and mlflow_model is not None:
        return get_keywords_mlflow(text, topn)
    return get_keywords_local(text, topn)


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":        "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "model_source":  MODEL_SOURCE,
        "model_name":    MLFLOW_MODEL if MODEL_SOURCE == "mlflow" else "local",
    })


@app.route("/api/extract", methods=["POST"])
def extract_endpoint():
    if not models_loaded:
        return jsonify({"error": "Models not loaded — check server logs"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    text         = data.get("text", "").strip()
    num_keywords = data.get("numKeywords", 10)

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 10:
        return jsonify({"error": "Text too short for meaningful keyword extraction"}), 400

    try:
        num_keywords = max(1, min(50, int(num_keywords)))
    except (ValueError, TypeError):
        num_keywords = 10

    try:
        logger.info(f"Extracting {num_keywords} kw from {len(text)}-char input")
        keywords = get_keywords(text, num_keywords)

        if not keywords:
            return jsonify({"error": "No keywords could be extracted from the text"}), 400

        logger.info(f"Returning {len(keywords)} keywords")
        resp = make_response(jsonify(keywords))
        resp.headers["Content-Type"] = "application/json"
        return resp

    except Exception as e:
        logger.exception("Unhandled error during extraction")
        return jsonify({"error": "Internal server error during keyword extraction"}), 500


@app.route("/api/extract", methods=["OPTIONS"])
def extract_preflight():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "*"
    return resp


# ── Error handlers ─────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    if models_loaded:
        logger.info(f"Starting server on 0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
    else:
        logger.error("Cannot start — models failed to load")
