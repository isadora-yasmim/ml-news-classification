import pytest

from src.inference.predictor import MODEL_PATH, NewsPredictor


def test_model_file_exists() -> None:
    assert MODEL_PATH.exists()


def test_load_model() -> None:
    predictor = NewsPredictor()

    assert predictor.model is not None


def test_build_text_with_headline_and_short_description() -> None:
    text = NewsPredictor.build_text(
        headline="New technology impacts companies",
        short_description="Artificial intelligence tools are changing productivity.",
    )

    assert text == (
        "New technology impacts companies "
        "Artificial intelligence tools are changing productivity."
    )


def test_build_text_with_empty_text() -> None:
    with pytest.raises(ValueError, match="não podem estar vazios"):
        NewsPredictor.build_text(
            headline="",
            short_description="",
        )


def test_predict_news() -> None:
    predictor = NewsPredictor()

    result = predictor.predict(
        headline="New technology impacts companies",
        short_description="Artificial intelligence tools are changing productivity.",
    )

    assert "predicted_category" in result
    assert result["predicted_category"] is not None
    assert "confidence" in result


def test_predict_with_very_short_text() -> None:
    predictor = NewsPredictor()

    result = predictor.predict(
        headline="AI",
        short_description="Tech",
    )

    assert "predicted_category" in result
    assert result["predicted_category"] is not None