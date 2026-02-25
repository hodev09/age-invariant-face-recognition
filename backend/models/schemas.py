from pydantic import BaseModel


class ComparisonResponse(BaseModel):
    age1: int
    age2: int
    age_group1: str
    age_group2: str
    similarity_score: float
    confidence: float
    result: str  # "same_person" or "different_person"
    message: str


class RejectionResponse(BaseModel):
    age1: int
    age2: int
    age_group1: str
    age_group2: str
    result: str  # "rejected"
    message: str


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: str  # "ok"
    model_loaded: bool
