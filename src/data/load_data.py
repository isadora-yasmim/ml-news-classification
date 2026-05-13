import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/News_Category_Dataset_v3.json")


def load_data():
    """
    Carrega o dataset bruto em formato JSON.
    """

    df = pd.read_json(DATA_PATH, lines=True)

    return df


def remove_null_values(df):
    """
    Remove registros com valores nulos
    nas colunas principais.
    """

    required_columns = [
        "headline",
        "short_description",
        "category"
    ]

    df = df.dropna(subset=required_columns)

    return df


def create_text_column(df):
    """
    Cria a coluna 'text' concatenando:
    headline + short_description
    """

    df["text"] = (
        df["headline"].astype(str)
        + " "
        + df["short_description"].astype(str)
    )

    return df


def create_label_column(df):
    """
    Cria a coluna 'label'
    usando a categoria.
    """

    df["label"] = df["category"]

    return df


def prepare_dataset():
    """
    Executa pipeline completa de preparação.
    """

    df = load_data()

    df = remove_null_values(df)

    df = create_text_column(df)

    df = create_label_column(df)

    return df[["text", "label"]]


if __name__ == "__main__":

    df = prepare_dataset()

    print("=" * 60)
    print("DATASET PREPARADO")
    print("=" * 60)

    print(f"\nQuantidade de registros: {len(df)}")

    print("\nExemplo:")
    print(df.head())

    print("\nCategorias:")
    print(df["label"].value_counts().head())