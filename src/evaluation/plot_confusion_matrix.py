from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFUSION_MATRIX_PATH = Path("reports/metrics/confusion_matrix.csv")
FIGURES_PATH = Path("reports/figures")


def load_confusion_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def plot_confusion_matrix(df: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(18, 14))
    plt.imshow(df.values, aspect="auto")

    plt.title("Matriz de Confusão")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(
        ticks=range(len(df.columns)),
        labels=df.columns,
        rotation=90,
        fontsize=7,
    )

    plt.yticks(
        ticks=range(len(df.index)),
        labels=df.index,
        fontsize=7,
    )

    plt.colorbar()
    plt.tight_layout()

    output_file = FIGURES_PATH / "confusion_matrix_readable.png"
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_normalized_confusion_matrix(df: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    normalized_df = df.div(df.sum(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(18, 14))
    plt.imshow(normalized_df.values, aspect="auto")

    plt.title("Matriz de Confusão Normalizada")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(
        ticks=range(len(normalized_df.columns)),
        labels=normalized_df.columns,
        rotation=90,
        fontsize=7,
    )

    plt.yticks(
        ticks=range(len(normalized_df.index)),
        labels=normalized_df.index,
        fontsize=7,
    )

    plt.colorbar()
    plt.tight_layout()

    output_file = FIGURES_PATH / "confusion_matrix_normalized.png"
    plt.savefig(output_file, dpi=300)
    plt.close()


def main() -> None:
    df = load_confusion_matrix(CONFUSION_MATRIX_PATH)

    plot_confusion_matrix(df)
    plot_normalized_confusion_matrix(df)

    print("Matrizes de confusão geradas:")
    print(f"- {FIGURES_PATH / 'confusion_matrix_readable.png'}")
    print(f"- {FIGURES_PATH / 'confusion_matrix_normalized.png'}")


if __name__ == "__main__":
    main()