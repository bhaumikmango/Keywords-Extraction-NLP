"""
mlflow_log.py
─────────────────────────────────────────────────────────────────────────────
One-time script to register your pre-trained TF-IDF pipeline into MLflow.

Run once locally (or in CI) to log the models, then point your server
at DagsHub or any MLflow tracking URI to pull them down for inference.

Usage:
    python mlflow_log.py

Environment variables (set in .env or shell):
    MLFLOW_TRACKING_URI   — e.g. https://dagshub.com/<user>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME  (DagsHub username)
    MLFLOW_TRACKING_PASSWORD  (DagsHub token)
"""

import os
import pickle
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── MLflow tracking URI ────────────────────────────────────────────────────
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")   # default: local ./mlruns
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT_NAME", "keyword-extractor")
MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "keyword-extractor-tfidf")

# ── Credentials (needed for DagsHub / remote URIs) ────────────────────────
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if username and password:
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# ── Custom MLflow PythonModel wrapper ─────────────────────────────────────
from keyword_model import KeywordExtractorModel

# ── Log run ───────────────────────────────────────────────────────────────
def main():
    logger.info(f"Logging to MLflow at: {TRACKING_URI}")

    with mlflow.start_run(run_name="initial-registration") as run:

        # Tag the run with metadata
        mlflow.set_tags({
            "model_type":   "tfidf-keyword-extractor",
            "framework":    "scikit-learn",
            "author":       os.getenv("USER", "unknown"),
        })

        # Log hyperparams / config
        mlflow.log_params({
            "vectorizer":    "CountVectorizer",
            "transformer":   "TfidfTransformer",
            "stopwords_ext": "fig,figure,image,sample,using,show,result,large,also,one-nine",
            "stemmer":       "PorterStemmer",
            "lemmatizer":    "WordNetLemmatizer",
        })

        # ── Log raw artefacts (pkl files) ──────────────────────────
        mlflow.log_artifact("Count_Vector.pkl",      artifact_path="model_artifacts")
        mlflow.log_artifact("TFIDF_Transformer.pkl", artifact_path="model_artifacts")
        mlflow.log_artifact("Feature_Names.pkl",     artifact_path="model_artifacts")
        logger.info("Raw .pkl artefacts logged")

        # ── Log as PythonModel ─────────────────────────────────────
        artifacts = {
            "count_vector":      "Count_Vector.pkl",
            "tfidf_transformer": "TFIDF_Transformer.pkl",
            "feature_names":     "Feature_Names.pkl",
        }

        mlflow.pyfunc.log_model(
            artifact_path  = "keyword_extractor",
            python_model   = KeywordExtractorModel(),
            artifacts      = artifacts,
            registered_model_name = MODEL_NAME,
            pip_requirements = [
                "scikit-learn==1.6.1",
                "nltk==3.9.1",
            ],
        )
        logger.info(f"Model registered as '{MODEL_NAME}'")

        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")
        logger.info(f"View at: {TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT).experiment_id}/runs/{run_id}")

    print("\n✅  Model logged and registered successfully.")
    print(f"   Tracking URI : {TRACKING_URI}")
    print(f"   Experiment   : {EXPERIMENT}")
    print(f"   Model name   : {MODEL_NAME}")


if __name__ == "__main__":
    main()
