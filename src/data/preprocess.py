import re


def normalize_text(text):
    """
    Normaliza texto:
    - lowercase
    - remove espaços extras
    """

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def preprocess_dataframe(df):
    """
    Aplica pré-processamento
    na coluna de texto.
    """

    df["text"] = df["text"].apply(normalize_text)

    df = df[df["text"] != ""]

    return df