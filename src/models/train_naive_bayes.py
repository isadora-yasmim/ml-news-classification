from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")

METRICS_DIR = Path("reports/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.txt"
CLASSIFICATION_REPORT_PATH = (
    METRICS_DIR / "classification_report.txt"
)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    lowercase=True,
                ),
            ),
            (
                "model",
                MultinomialNB(),
            ),
        ]
    )

    return pipeline


def save_metrics(
    train_accuracy: float,
    test_accuracy: float,
    report: str,
) -> None:
    with open(BASELINE_METRICS_PATH, "w", encoding="utf-8") as file:
        file.write("=== BASELINE NAIVE BAYES ===\n\n")
        file.write(
            f"Acurácia no treino: "
            f"{train_accuracy:.4f}\n"
        )
        file.write(
            f"Acurácia no teste: "
            f"{test_accuracy:.4f}\n"
        )

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)


def main() -> None:
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    pipeline = create_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    train_predictions = pipeline.predict(X_train)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
    )

    save_metrics(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        report=report,
    )

    print("Modelo avaliado com sucesso!\n")

    print("=== BASELINE ===")
    print(
        f"Acurácia treino: "
        f"{train_accuracy:.4f}"
    )
    print(
        f"Acurácia teste: "
        f"{test_accuracy:.4f}"
    )

    print("\n=== CLASSIFICATION REPORT ===\n")
    print(report)

    print(
        "\nMétricas salvas em:"
    )
    print(BASELINE_METRICS_PATH)
    print(CLASSIFICATION_REPORT_PATH)


if __name__ == "__main__":
    main()