from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import (
    METRICS_DIR,
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_PATH,
    TEXT_COLUMN,
)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model(path: Path):
    return joblib.load(path)


def save_classification_report(
    y_true,
    y_pred,
    accuracy: float,
    path: Path,
) -> None:
    report = classification_report(y_true, y_pred)

    content = (
        "=== AVALIAÇÃO DO MODELO ===\n\n"
        f"Acurácia no teste: {accuracy:.4f}\n\n"
        "=== CLASSIFICATION REPORT ===\n\n"
        f"{report}"
    )

    path.write_text(content, encoding="utf-8")

    print(f"Classification report salvo em: {path}")


def save_confusion_matrix_csv(y_true, y_pred, path: Path) -> None:
    labels = sorted(y_true.unique())

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    matrix_df.to_csv(path, encoding="utf-8")

    print(f"Matriz de confusão CSV salva em: {path}")


def save_confusion_matrix_plot(y_true, y_pred, path: Path) -> None:
    labels = sorted(y_true.unique())

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(18, 18))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot(
        xticks_rotation=90,
        cmap="Blues",
        values_format="d",
    )

    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Matriz de confusão PNG salva em: {path}")


def evaluate_model() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df_test = load_data(TEST_PATH)

    X_test = df_test[TEXT_COLUMN]
    y_test = df_test[TARGET_COLUMN]

    model = load_model(MODEL_PATH)

    print("Iniciando avaliação...")
    print(f"Quantidade de registros de teste: {len(df_test)}")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Acurácia no teste: {accuracy:.4f}")

    save_classification_report(
        y_test,
        y_pred,
        accuracy,
        METRICS_DIR / "classification_report.txt",
    )

    save_confusion_matrix_csv(
        y_test,
        y_pred,
        METRICS_DIR / "confusion_matrix.csv",
    )

    save_confusion_matrix_plot(
        y_test,
        y_pred,
        METRICS_DIR / "confusion_matrix.png",
    )

    print("Avaliação concluída com sucesso!")


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()