from pathlib import Path

from sklearn.model_selection import train_test_split

from load_data import prepare_dataset
from preprocess import preprocess_dataframe


BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_dataset(df):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    return train_df, test_df


def save_dataset(train_df, test_df):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)


def main():
    df = prepare_dataset()
    df = preprocess_dataframe(df)

    train_df, test_df = split_dataset(df)

    save_dataset(train_df, test_df)

    print("Dataset processado com sucesso.")
    print(f"Treino: {train_df.shape}")
    print(f"Teste: {test_df.shape}")
    print(f"Arquivo de treino salvo em: {TRAIN_DATA_PATH}")
    print(f"Arquivo de teste salvo em: {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()