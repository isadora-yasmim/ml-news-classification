from fastapi import FastAPI, HTTPException

from src.api.schemas import NewsInput, PredictionOutput
from src.inference.predictor import NewsPredictor


app = FastAPI(
    title="ML News Classification API",
    description="API para classificação de notícias usando Machine Learning.",
    version="1.0.0",
)

predictor = NewsPredictor()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict_news(news: NewsInput) -> PredictionOutput:
    try:
        prediction = predictor.predict(
            headline=news.headline,
            short_description=news.short_description,
        )

        return PredictionOutput(**prediction)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))