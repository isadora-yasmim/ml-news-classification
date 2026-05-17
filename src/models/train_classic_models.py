from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")

METRICS_DIR = Path("reports/metrics")
FIGURES_DIR = Path("reports/figures")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_pipeline(model) -> Pipeline:
    return Pipeline(
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
            ("model", model),
        ]
    )


def save_classification_report(
    y_true,
    y_pred,
    model_name: str,
) -> None:
    report = classification_report(y_true, y_pred)

    output_path = METRICS_DIR / f"{model_name}_classification_report.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)


def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    model_name: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(16, 12))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)

    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_worst_categories(y_true, y_pred, model_name: str) -> pd.DataFrame:
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

    df_report = pd.DataFrame(rows)

    return df_report.sort_values(by="f1_score", ascending=True)


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    labels,
) -> dict:
    print(f"\nTreinando modelo: {model_name}")

    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    save_classification_report(y_test, test_predictions, model_name)
    save_confusion_matrix(y_test, test_predictions, labels, model_name)

    worst_categories = get_worst_categories(
        y_test,
        test_predictions,
        model_name,
    )

    worst_categories_path = (
        METRICS_DIR / f"{model_name}_worst_categories.csv"
    )
    worst_categories.to_csv(worst_categories_path, index=False)

    print(f"Acurácia no treino: {train_accuracy:.4f}")
    print(f"Acurácia no teste: {test_accuracy:.4f}")

    print("\nCategorias com pior desempenho:")
    print(worst_categories.head(10))

    return {
        "model": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    labels = sorted(y_train.unique())

    models = {
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

    results = []

    for model_name, model in models.items():
        pipeline = create_pipeline(model)

        result = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            labels=labels,
        )

        results.append(result)

    comparison_df = pd.DataFrame(results)
    comparison_path = METRICS_DIR / "classic_models_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n=== COMPARAÇÃO DOS MODELOS ===")
    print(comparison_df)

    print("\nModelos clássicos treinados e avaliados com sucesso!")


if __name__ == "__main__":
    main()