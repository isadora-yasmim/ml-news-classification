from pydantic import BaseModel, Field


class NewsInput(BaseModel):
    headline: str = Field(..., min_length=1)
    short_description: str = Field(..., min_length=1)


class PredictionOutput(BaseModel):
    predicted_category: str
    confidence: float | None = None