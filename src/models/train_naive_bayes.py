from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.utils.experiment_tracker import (
    calculate_metrics,
    create_experiment_id,
    save_classification_report,
    save_experiment,
    save_worst_categories,
)


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "lowercase": True,
}

MODEL_PARAMS = {}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("model", MultinomialNB(**MODEL_PARAMS)),
        ]
    )


def main() -> None:
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    model_name = "naive_bayes"
    experiment_type = "baseline"
    experiment_id = create_experiment_id(model_name, experiment_type)

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    report_path = save_classification_report(
        y_true=y_test,
        y_pred=test_predictions,
        experiment_id=experiment_id,
    )

    worst_categories_path = save_worst_categories(
        y_true=y_test,
        y_pred=test_predictions,
        model_name=model_name,
        experiment_id=experiment_id,
    )

    experiment_path = save_experiment(
        experiment_id=experiment_id,
        model_name=model_name,
        experiment_type=experiment_type,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        tfidf_params=TFIDF_PARAMS,
        model_params=MODEL_PARAMS,
        artifacts={
            "classification_report": str(report_path),
            "worst_categories": str(worst_categories_path),
        },
    )

    print("Baseline Naive Bayes treinado e registrado com sucesso!")
    print(f"Experimento salvo em: {experiment_path}")
    print(f"Accuracy teste: {test_metrics['accuracy']:.4f}")
    print(f"F1-score macro teste: {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()