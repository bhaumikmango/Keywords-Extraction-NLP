"""
mlflow_log.py  —  registers the TF-IDF pipeline in MLflow / DagsHub
Run once from the backend/ directory:
    python mlflow_log.py
"""

import os
import pathlib
import mlflow
import mlflow.pyfunc
import logging
from keyword_model import KeywordExtractorModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
USERNAME     = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
PASSWORD     = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT_NAME", "keyword-extractor")
MODEL_NAME   = os.environ.get("MLFLOW_MODEL_NAME",      "keyword-extractor-tfidf")

if USERNAME:
    os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
if PASSWORD:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# Always forward slashes
BASE      = pathlib.Path(__file__).parent.resolve()
PKL_CV    = BASE / "Count_Vector.pkl"
PKL_TFIDF = BASE / "TFIDF_Transformer.pkl"
PKL_FEAT  = BASE / "Feature_Names.pkl"

for p in [PKL_CV, PKL_TFIDF, PKL_FEAT]:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

artifacts = {
    "count_vector":      PKL_CV.as_posix(),
    "tfidf_transformer": PKL_TFIDF.as_posix(),
    "feature_names":     PKL_FEAT.as_posix(),
}

logger.info("Artifact paths:")
for k, v in artifacts.items():
    logger.info(f"  {k}: {v}")

with mlflow.start_run(run_name="registration") as run:

    mlflow.set_tags({"model_type": "tfidf-keyword-extractor", "framework": "scikit-learn"})
    mlflow.log_params({"vectorizer": "CountVectorizer", "transformer": "TfidfTransformer",
                       "stemmer": "PorterStemmer", "lemmatizer": "WordNetLemmatizer"})

    mlflow.log_artifact(PKL_CV.as_posix(),    artifact_path="model_artifacts")
    mlflow.log_artifact(PKL_TFIDF.as_posix(), artifact_path="model_artifacts")
    mlflow.log_artifact(PKL_FEAT.as_posix(),  artifact_path="model_artifacts")
    logger.info("Raw .pkl artefacts logged")

    os.chdir(BASE)

    mlflow.pyfunc.log_model(
        artifact_path         = "keyword_extractor",
        python_model          = KeywordExtractorModel(),
        artifacts             = artifacts,
        registered_model_name = MODEL_NAME,
        pip_requirements      = ["scikit-learn==1.6.1", "nltk==3.9.1"],
    )

    logger.info(f"Model registered as '{MODEL_NAME}'")

print(f"\n✅  Done — model: {MODEL_NAME}")