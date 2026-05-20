from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("model", model),
        ]
    )


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=42,
        ),
        "linear_svm": LinearSVC(
            random_state=42,
        ),
    }


def evaluate_and_register_model(
    model_name: str,
    pipeline: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    experiment_type = "classic"
    experiment_id = create_experiment_id(model_name, experiment_type)

    print(f"\nTreinando modelo: {model_name}")

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

    model_params = pipeline.named_steps["model"].get_params()

    experiment_path = save_experiment(
        experiment_id=experiment_id,
        model_name=model_name,
        experiment_type=experiment_type,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        tfidf_params=TFIDF_PARAMS,
        model_params=model_params,
        artifacts={
            "classification_report": str(report_path),
            "worst_categories": str(worst_categories_path),
        },
    )

    print(f"Experimento salvo em: {experiment_path}")
    print(f"Accuracy teste: {test_metrics['accuracy']:.4f}")
    print(f"F1-score macro teste: {test_metrics['f1_macro']:.4f}")

    return {
        "experiment_id": experiment_id,
        "model": model_name,
        "accuracy": test_metrics["accuracy"],
        "precision_macro": test_metrics["precision_macro"],
        "recall_macro": test_metrics["recall_macro"],
        "f1_macro": test_metrics["f1_macro"],
        "f1_weighted": test_metrics["f1_weighted"],
        "experiment_path": str(experiment_path),
    }


def main() -> None:
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    results = []

    for model_name, model in get_models().items():
        pipeline = create_pipeline(model)

        result = evaluate_and_register_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    comparison_df = pd.DataFrame(results)

    print("\n=== COMPARAÇÃO DOS MODELOS CLÁSSICOS ===")
    print(comparison_df)

    print("\nModelos clássicos treinados, avaliados e versionados com sucesso!")


if __name__ == "__main__":
    main()