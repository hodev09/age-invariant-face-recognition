from models.schemas import (
    ComparisonResponse,
    RejectionResponse,
    ErrorResponse,
    HealthResponse,
)


class TestComparisonResponse:
    def test_serialization(self):
        resp = ComparisonResponse(
            age1=25, age2=30, age_group1="adult", age_group2="adult",
            similarity_score=0.85, confidence=0.77, result="same_person",
            message="Faces match",
        )
        data = resp.model_dump()
        assert data["age1"] == 25
        assert data["result"] == "same_person"
        assert set(data.keys()) == {
            "age1", "age2", "age_group1", "age_group2",
            "similarity_score", "confidence", "result", "message",
        }


class TestRejectionResponse:
    def test_serialization(self):
        resp = RejectionResponse(
            age1=2, age2=35, age_group1="infant", age_group2="adult",
            result="rejected", message="Cannot reliably compare",
        )
        data = resp.model_dump()
        assert data["result"] == "rejected"
        assert "age1" in data and "message" in data


class TestErrorResponse:
    def test_serialization(self):
        resp = ErrorResponse(error="Something went wrong")
        assert resp.model_dump() == {"error": "Something went wrong"}


class TestHealthResponse:
    def test_serialization(self):
        resp = HealthResponse(status="ok", model_loaded=True)
        assert resp.model_dump() == {"status": "ok", "model_loaded": True}
