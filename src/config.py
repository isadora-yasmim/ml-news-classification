from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "news_classifier.joblib"

REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

TEXT_COLUMN = "text"
TARGET_COLUMN = "category"

RANDOM_STATE = 42

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "lowercase": True,
}

MODEL_TYPE = "linear_svm"