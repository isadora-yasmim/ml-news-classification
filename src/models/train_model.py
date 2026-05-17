from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    MODEL_DIR,
    MODEL_PATH,
    MODEL_TYPE,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TFIDF_PARAMS,
    TRAIN_PATH,
    RANDOM_STATE,
)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_model_pipeline(model_type: str) -> Pipeline:
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_type == "linear_svm":
        model = LinearSVC(
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(
            "Modelo inválido. Use 'logistic_regression' ou 'linear_svm'."
        )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("model", model),
        ]
    )

    return pipeline


def train_model() -> Pipeline:
    df_train = load_data(TRAIN_PATH)

    X_train = df_train[TEXT_COLUMN]
    y_train = df_train[TARGET_COLUMN]

    pipeline = create_model_pipeline(MODEL_TYPE)

    print("Iniciando treinamento...")
    print(f"Modelo selecionado: {MODEL_TYPE}")
    print(f"Quantidade de registros de treino: {len(df_train)}")

    pipeline.fit(X_train, y_train)

    print("Treinamento concluído com sucesso!")

    return pipeline


def save_model(model: Pipeline, path: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    print(f"Modelo salvo em: {path}")


def main() -> None:
    model = train_model()
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()