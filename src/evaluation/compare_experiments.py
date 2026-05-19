from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd


METRICS_PATH = Path("reports/metrics")
EXPERIMENTS_PATH = Path("reports/experiments")
FIGURES_PATH = Path("reports/figures")

OUTPUT_PATH = Path("reports/metrics_comparison.csv")


def normalize_model_name(file_name: str) -> str:
    name = file_name.lower()

    if "naive" in name or "baseline" in name:
        return "naive_bayes"

    if "logistic" in name:
        return "logistic_regression"

    if "svm" in name or "linear_svm" in name:
        return "linear_svm"

    if "gridsearch" in name or "grid_search" in name:
        return "logistic_regression_gridsearch"

    return file_name.replace(".txt", "").replace(".csv", "").replace(".json", "")


def load_gridsearch_json(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    test_metrics = data.get("test_metrics", {})

    return {
        "model": normalize_model_name(file_path.name),
        "source_file": file_path.name,
        "train_accuracy": None,
        "test_accuracy": test_metrics.get("accuracy"),
        "accuracy": test_metrics.get("accuracy"),
        "precision_macro": test_metrics.get("precision_macro"),
        "recall_macro": test_metrics.get("recall_macro"),
        "f1_macro": test_metrics.get("f1_macro"),
        "f1_weighted": test_metrics.get("f1_weighted"),
        "best_cv_score": data.get("best_cv_score"),
        "best_params": json.dumps(data.get("best_params", {}), ensure_ascii=False),
    }


def load_classic_models_csv(file_path: Path) -> list[dict]:
    df = pd.read_csv(file_path)

    results = []

    for _, row in df.iterrows():
        model_name = row["model"]

        results.append(
            {
                "model": model_name,
                "source_file": file_path.name,
                "train_accuracy": row.get("train_accuracy"),
                "test_accuracy": row.get("test_accuracy"),
                "accuracy": row.get("test_accuracy"),
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "f1_weighted": None,
                "best_cv_score": None,
                "best_params": "{}",
            }
        )

    return results


def load_baseline_txt(file_path: Path) -> dict:
    text = file_path.read_text(encoding="utf-8")

    train_match = re.search(r"Acurácia no treino:\s*([0-9.]+)", text)
    test_match = re.search(r"Acurácia no teste:\s*([0-9.]+)", text)

    train_accuracy = float(train_match.group(1)) if train_match else None
    test_accuracy = float(test_match.group(1)) if test_match else None

    return {
        "model": "naive_bayes",
        "source_file": file_path.name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy": test_accuracy,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "f1_weighted": None,
        "best_cv_score": None,
        "best_params": "{}",
    }


def parse_classification_report_txt(file_path: Path) -> dict:
    text = file_path.read_text(encoding="utf-8")

    model_name = normalize_model_name(file_path.name)

    accuracy_match = re.search(
        r"accuracy\s+([0-9.]+)\s+\d+",
        text,
    )

    macro_match = re.search(
        r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+",
        text,
    )

    weighted_match = re.search(
        r"weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+",
        text,
    )

    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    precision_macro = float(macro_match.group(1)) if macro_match else None
    recall_macro = float(macro_match.group(2)) if macro_match else None
    f1_macro = float(macro_match.group(3)) if macro_match else None

    f1_weighted = float(weighted_match.group(3)) if weighted_match else None

    return {
        "model": model_name,
        "source_file": file_path.name,
        "train_accuracy": None,
        "test_accuracy": accuracy,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "best_cv_score": None,
        "best_params": "{}",
    }


def should_ignore_csv(file_path: Path) -> bool:
    ignored_names = [
        "confusion_matrix",
        "worst_categories",
    ]

    return any(name in file_path.stem for name in ignored_names)


def load_all_results() -> pd.DataFrame:
    results = []

    print("\nBuscando arquivos de métricas...")

    json_files = list(EXPERIMENTS_PATH.glob("*.json")) if EXPERIMENTS_PATH.exists() else []
    csv_files = list(METRICS_PATH.glob("*.csv")) if METRICS_PATH.exists() else []
    txt_files = list(METRICS_PATH.glob("*.txt")) if METRICS_PATH.exists() else []

    for file_path in json_files:
        print(f"Lendo JSON: {file_path}")
        results.append(load_gridsearch_json(file_path))

    for file_path in csv_files:
        if should_ignore_csv(file_path):
            print(f"Ignorando CSV auxiliar: {file_path}")
            continue

        print(f"Lendo CSV: {file_path}")

        if file_path.name == "classic_models_comparison.csv":
            results.extend(load_classic_models_csv(file_path))

    for file_path in txt_files:
        print(f"Lendo TXT: {file_path}")

        if "baseline" in file_path.name:
            results.append(load_baseline_txt(file_path))
        elif "classification_report" in file_path.name:
            results.append(parse_classification_report_txt(file_path))

    if not results:
        raise FileNotFoundError(
            "Nenhum resultado válido encontrado em reports/metrics ou reports/experiments."
        )

    df = pd.DataFrame(results)

    metric_columns = [
        "train_accuracy",
        "test_accuracy",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "best_cv_score",
    ]

    for column in metric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = merge_duplicated_models(df)

    df = df.sort_values(
        by=["f1_macro", "accuracy"],
        ascending=[False, False],
        na_position="last",
    )

    return df


def merge_duplicated_models(df: pd.DataFrame) -> pd.DataFrame:
    merged_rows = []

    for model_name, group in df.groupby("model"):
        merged = {
            "model": model_name,
            "source_file": ", ".join(group["source_file"].dropna().astype(str)),
            "train_accuracy": group["train_accuracy"].dropna().max()
            if group["train_accuracy"].notna().any()
            else None,
            "test_accuracy": group["test_accuracy"].dropna().max()
            if group["test_accuracy"].notna().any()
            else None,
            "accuracy": group["accuracy"].dropna().max()
            if group["accuracy"].notna().any()
            else None,
            "precision_macro": group["precision_macro"].dropna().max()
            if group["precision_macro"].notna().any()
            else None,
            "recall_macro": group["recall_macro"].dropna().max()
            if group["recall_macro"].notna().any()
            else None,
            "f1_macro": group["f1_macro"].dropna().max()
            if group["f1_macro"].notna().any()
            else None,
            "f1_weighted": group["f1_weighted"].dropna().max()
            if group["f1_weighted"].notna().any()
            else None,
            "best_cv_score": group["best_cv_score"].dropna().max()
            if group["best_cv_score"].notna().any()
            else None,
            "best_params": next(
                (
                    params
                    for params in group["best_params"].dropna().astype(str)
                    if params not in ["{}", "", "nan"]
                ),
                "{}",
            ),
        }

        merged_rows.append(merged)

    return pd.DataFrame(merged_rows)


def save_metrics_comparison(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


def plot_metric_comparison(df: pd.DataFrame, metric: str) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    plot_df = df.dropna(subset=[metric])

    if plot_df.empty:
        print(f"Não foi possível gerar gráfico para {metric}: sem dados.")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["model"], plot_df[metric])

    plt.title(f"Comparação de {metric}")
    plt.xlabel("Modelo")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_file = FIGURES_PATH / f"{metric}_comparison.png"
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_metrics_summary(df: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]

    available_metrics = [
        metric for metric in metrics
        if metric in df.columns and df[metric].notna().any()
    ]

    if not available_metrics:
        print("Não foi possível gerar gráfico geral: sem métricas disponíveis.")
        return

    summary_df = df.set_index("model")[available_metrics]

    ax = summary_df.plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.set_title("Comparação geral das métricas por modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor da métrica")
    ax.legend(title="Métricas")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_file = FIGURES_PATH / "metrics_summary_comparison.png"
    plt.savefig(output_file, dpi=300)
    plt.close()


def print_best_model(df: pd.DataFrame) -> None:
    if df["f1_macro"].notna().any():
        best_model = df.sort_values(
            by="f1_macro",
            ascending=False,
        ).iloc[0]

        metric_used = "F1-score macro"
        metric_value = best_model["f1_macro"]
    else:
        best_model = df.sort_values(
            by="accuracy",
            ascending=False,
        ).iloc[0]

        metric_used = "Accuracy"
        metric_value = best_model["accuracy"]

    print("\n=== MELHOR MODELO ===")
    print(f"Modelo: {best_model['model']}")
    print(f"Métrica usada: {metric_used}")
    print(f"Valor: {metric_value:.4f}")

    print("\nMétricas disponíveis:")
    print(f"Accuracy: {best_model['accuracy']:.4f}" if pd.notna(best_model["accuracy"]) else "Accuracy: indisponível")
    print(f"Precision macro: {best_model['precision_macro']:.4f}" if pd.notna(best_model["precision_macro"]) else "Precision macro: indisponível")
    print(f"Recall macro: {best_model['recall_macro']:.4f}" if pd.notna(best_model["recall_macro"]) else "Recall macro: indisponível")
    print(f"F1-score macro: {best_model['f1_macro']:.4f}" if pd.notna(best_model["f1_macro"]) else "F1-score macro: indisponível")
    print(f"F1-score weighted: {best_model['f1_weighted']:.4f}" if pd.notna(best_model["f1_weighted"]) else "F1-score weighted: indisponível")


def main() -> None:
    df = load_all_results()

    print("\n=== COMPARAÇÃO CONSOLIDADA ===")
    print(
        df[
            [
                "model",
                "train_accuracy",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "source_file",
            ]
        ]
    )

    save_metrics_comparison(df)

    for metric in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]:
        plot_metric_comparison(df, metric)

    plot_metrics_summary(df)

    print_best_model(df)

    print("\nArquivos gerados:")
    print(f"- {OUTPUT_PATH}")
    print(f"- {FIGURES_PATH / 'accuracy_comparison.png'}")
    print(f"- {FIGURES_PATH / 'precision_macro_comparison.png'}")
    print(f"- {FIGURES_PATH / 'recall_macro_comparison.png'}")
    print(f"- {FIGURES_PATH / 'f1_macro_comparison.png'}")
    print(f"- {FIGURES_PATH / 'f1_weighted_comparison.png'}")
    print(f"- {FIGURES_PATH / 'metrics_summary_comparison.png'}")


if __name__ == "__main__":
    main()