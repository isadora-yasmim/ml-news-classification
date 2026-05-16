from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_naive_bayes_pipeline() -> Pipeline:
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
            ),
            (
                "model",
                MultinomialNB(),
            ),
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

    pipeline = create_naive_bayes_pipeline()

    pipeline.fit(X_train, y_train)

    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)

    print("Modelo Naive Bayes treinado com sucesso!")
    print()
    print("=== BASELINE NAIVE BAYES ===")
    print(f"Acurácia no treino: {train_accuracy:.4f}")
    print(f"Acurácia no teste: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()