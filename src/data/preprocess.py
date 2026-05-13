from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")

TRAIN_INPUT_PATH = PROCESSED_DIR / "train.csv"
TEST_INPUT_PATH = PROCESSED_DIR / "test.csv"

TRAIN_OUTPUT_PATH = PROCESSED_DIR / "train_processed.csv"
TEST_OUTPUT_PATH = PROCESSED_DIR / "test_processed.csv"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={"label": "category"})

    df["text"] = df["text"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)

    df = df[["text", "category"]]

    df = df.drop_duplicates()
    df = df.dropna()

    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    train_df = load_data(TRAIN_INPUT_PATH)
    test_df = load_data(TEST_INPUT_PATH)

    train_processed = preprocess_data(train_df)
    test_processed = preprocess_data(test_df)

    save_data(train_processed, TRAIN_OUTPUT_PATH)
    save_data(test_processed, TEST_OUTPUT_PATH)

    print("Pré-processamento concluído com sucesso!")
    print(f"Train processado: {train_processed.shape}")
    print(f"Test processado: {test_processed.shape}")


if __name__ == "__main__":
    main()