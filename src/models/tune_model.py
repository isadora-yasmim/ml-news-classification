from pathlib import Path
import json
import joblib

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")

REPORTS_PATH = Path("reports/experiments")
MODELS_PATH = Path("models")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def get_param_grid() -> dict:
    param_grid = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],
        "tfidf__max_features": [
            5000,
            10000,
            20000,
        ],
        "tfidf__stop_words": [
            None,
            "english",
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
        "model__C": [
            0.1,
            1.0,
            10.0,
        ],
    }

    return param_grid


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    return metrics


def save_experiment_results(
    grid_search: GridSearchCV,
    metrics: dict,
    output_path: Path,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        },
    }

    with open(output_path / "logistic_regression_gridsearch_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    with open(output_path / "logistic_regression_classification_report.txt", "w", encoding="utf-8") as file:
        file.write(metrics["classification_report"])

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(
        output_path / "logistic_regression_cv_results.csv",
        index=False,
    )


def save_best_model(model: Pipeline, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        output_path / "best_logistic_regression_model.joblib",
    )


def main() -> None:
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    pipeline = create_pipeline()
    param_grid = get_param_grid()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    print("Iniciando GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("\nGridSearchCV finalizado!")
    print("\nMelhores parâmetros:")
    print(grid_search.best_params_)

    print("\nMelhor F1-score macro na validação:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    metrics = evaluate_model(best_model, X_test, y_test)

    print("\n=== MÉTRICAS NO TESTE ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro: {metrics['recall_macro']:.4f}")
    print(f"F1-score macro: {metrics['f1_macro']:.4f}")
    print(f"F1-score weighted: {metrics['f1_weighted']:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(metrics["classification_report"])

    save_experiment_results(
        grid_search=grid_search,
        metrics=metrics,
        output_path=REPORTS_PATH,
    )

    save_best_model(
        model=best_model,
        output_path=MODELS_PATH,
    )

    print("\nResultados salvos em:")
    print(f"- {REPORTS_PATH / 'logistic_regression_gridsearch_results.json'}")
    print(f"- {REPORTS_PATH / 'logistic_regression_classification_report.txt'}")
    print(f"- {REPORTS_PATH / 'logistic_regression_cv_results.csv'}")
    print(f"- {MODELS_PATH / 'best_logistic_regression_model.joblib'}")


if __name__ == "__main__":
    main()