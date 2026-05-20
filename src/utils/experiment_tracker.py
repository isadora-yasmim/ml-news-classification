from datetime import datetime
from pathlib import Path
import json
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


EXPERIMENTS_DIR = Path("reports/experiments")


def create_experiment_id(
    model_name: str,
    experiment_type: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{timestamp}_{model_name}_{experiment_type}"


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def save_classification_report(
    y_true,
    y_pred,
    experiment_id: str,
) -> Path:
    reports_dir = EXPERIMENTS_DIR / "classification_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    output_path = reports_dir / f"{experiment_id}_classification_report.txt"
    output_path.write_text(report, encoding="utf-8")

    return output_path


def save_worst_categories(
    y_true,
    y_pred,
    model_name: str,
    experiment_id: str,
) -> Path:
    worst_dir = EXPERIMENTS_DIR / "worst_categories"
    worst_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for category, metrics in report.items():
        if category in ["accuracy", "macro avg", "weighted avg"]:
            continue

        rows.append(
            {
                "model": model_name,
                "category": category,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": metrics["support"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by="f1_score", ascending=True)

    output_path = worst_dir / f"{experiment_id}_worst_categories.csv"
    df.to_csv(output_path, index=False)

    return output_path


def save_experiment(
    experiment_id: str,
    model_name: str,
    experiment_type: str,
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    tfidf_params: dict[str, Any],
    model_params: dict[str, Any],
    artifacts: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "experiment_type": experiment_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "tfidf_params": tfidf_params,
        "model_params": model_params,
        "artifacts": artifacts or {},
        "extra": extra or {},
    }

    output_path = EXPERIMENTS_DIR / f"{experiment_id}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(experiment, file, indent=4, ensure_ascii=False)

    return output_path