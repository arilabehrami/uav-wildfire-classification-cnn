from pydantic import BaseModel


class ClassificationResponse(BaseModel):
    label: str
    fire_score: float
    threshold: float
    model_version: str