from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

EXPERIMENTS_PATH = Path("reports/experiments")
MODELS_PATH = Path("models")


BASE_TFIDF_PARAMS = {
    "lowercase": True,
}

BASE_MODEL_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**BASE_TFIDF_PARAMS)),
            ("model", LogisticRegression(**BASE_MODEL_PARAMS)),
        ]
    )


def get_param_grid() -> dict:
    return {
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


def save_gridsearch_cv_results(
    grid_search: GridSearchCV,
    experiment_id: str,
) -> Path:
    cv_results_dir = EXPERIMENTS_PATH / "cv_results"
    cv_results_dir.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(grid_search.cv_results_)

    output_path = cv_results_dir / f"{experiment_id}_cv_results.csv"
    cv_results.to_csv(output_path, index=False)

    return output_path


def save_best_model(
    model: Pipeline,
    experiment_id: str,
) -> Path:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    output_path = MODELS_PATH / f"{experiment_id}_model.joblib"
    joblib.dump(model, output_path)

    return output_path


def main() -> None:
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    model_name = "logistic_regression"
    experiment_type = "gridsearch"
    experiment_id = create_experiment_id(model_name, experiment_type)

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

    best_model = grid_search.best_estimator_

    train_predictions = best_model.predict(X_train)
    test_predictions = best_model.predict(X_test)

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

    cv_results_path = save_gridsearch_cv_results(
        grid_search=grid_search,
        experiment_id=experiment_id,
    )

    model_path = save_best_model(
        model=best_model,
        experiment_id=experiment_id,
    )

    experiment_path = save_experiment(
        experiment_id=experiment_id,
        model_name=model_name,
        experiment_type=experiment_type,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        tfidf_params=grid_search.best_params_,
        model_params=best_model.named_steps["model"].get_params(),
        artifacts={
            "classification_report": str(report_path),
            "worst_categories": str(worst_categories_path),
            "cv_results": str(cv_results_path),
            "model": str(model_path),
        },
        extra={
            "best_cv_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "scoring": "f1_macro",
            "cv": 3,
        },
    )

    print("\nGridSearchCV finalizado!")
    print(f"Experimento salvo em: {experiment_path}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Melhor F1-score macro na validação: {grid_search.best_score_:.4f}")
    print(f"Accuracy teste: {test_metrics['accuracy']:.4f}")
    print(f"F1-score macro teste: {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()