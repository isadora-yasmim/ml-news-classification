from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT_DIR / "data" / "raw" / "News_Category_Dataset_v3.json"
REPORTS_DIR = ROOT_DIR / "reports"
EDA_REPORT_PATH = REPORTS_DIR / "eda_summary.md"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def create_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text"] = (
        df["headline"].fillna("") + " " +
        df["short_description"].fillna("")
    )

    return df


def generate_eda_summary(df: pd.DataFrame) -> str:
    category_counts = df["category"].value_counts()
    null_values = df.isnull().sum()
    duplicated_rows = df.duplicated().sum()

    summary = f"""# Relatório de Análise Exploratória

## Visão geral do dataset

- Quantidade de registros: {len(df)}
- Quantidade de colunas: {df.shape[1]}
- Quantidade de categorias: {df["category"].nunique()}
- Quantidade de linhas duplicadas: {duplicated_rows}

## Colunas disponíveis

{list(df.columns)}

## Valores nulos por coluna

{null_values.to_markdown()}

## Distribuição das categorias

{category_counts.to_markdown()}

## Conclusão

O dataset possui estrutura adequada para uma tarefa de classificação supervisionada de notícias.

As colunas principais para modelagem são:

- `headline`
- `short_description`
- `category`

A estratégia adotada para entrada textual é a concatenação de `headline` com `short_description`, formando a coluna `text`.

Essa abordagem tende a fornecer mais contexto ao modelo do que utilizar apenas o título da notícia.
"""

    return summary


def save_report(content: str, path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    path.write_text(content, encoding="utf-8")

    print(f"Relatório de EDA salvo em: {path}")


def main() -> None:
    df = load_data(DATASET_PATH)
    df = create_text_column(df)

    summary = generate_eda_summary(df)

    save_report(summary, EDA_REPORT_PATH)


if __name__ == "__main__":
    main()