from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_tfidf_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    lowercase=True,
                ),
            )
        ]
    )

    return pipeline


def main() -> None:
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    pipeline = create_tfidf_pipeline()

    X_train_tfidf = pipeline.fit_transform(X_train)
    X_test_tfidf = pipeline.transform(X_test)

    print("Pipeline TF-IDF criado com sucesso!")

    print("\n=== TRAIN ===")
    print(f"Quantidade de textos: {X_train_tfidf.shape[0]}")
    print(f"Quantidade de features: {X_train_tfidf.shape[1]}")
    print(f"Quantidade de categorias: {y_train.nunique()}")

    print("\n=== TEST ===")
    print(f"Quantidade de textos: {X_test_tfidf.shape[0]}")
    print(f"Quantidade de features: {X_test_tfidf.shape[1]}")
    print(f"Quantidade de categorias: {y_test.nunique()}")


if __name__ == "__main__":
    main()