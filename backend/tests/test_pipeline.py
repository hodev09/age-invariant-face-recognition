"""Tests for the pipeline service."""

import numpy as np
import pytest

from services.pipeline import compare_faces, PipelineResult, _decode_image
from ai_providers.base import AIProvider, FaceDetectionResult, AgeEstimationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_image_bytes() -> bytes:
    """Create minimal valid JPEG bytes using OpenCV."""
    import cv2

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class FakeProvider(AIProvider):
    """A fake AI provider for testing the pipeline orchestration."""

    def __init__(
        self,
        age1: int = 25,
        age_group1: str = "adult",
        age2: int = 30,
        age_group2: str = "adult",
    ):
        self._ages = [(age1, age_group1), (age2, age_group2)]
        self._call_index = 0

    async def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        face = image[:32, :32] if image.shape[0] >= 32 else image
        return FaceDetectionResult(face_image=face, bbox=(0, 0, 32, 32), confidence=0.99)

    async def estimate_age(self, face_image: np.ndarray) -> AgeEstimationResult:
        age, group = self._ages[self._call_index]
        self._call_index = min(self._call_index + 1, len(self._ages) - 1)
        return AgeEstimationResult(age=age, age_group=group)

    async def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(512)
        return vec / np.linalg.norm(vec)

    async def is_loaded(self) -> bool:
        return True


class FailDetectionProvider(AIProvider):
    """Provider that raises on detect_face."""

    async def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        raise ValueError("No face detected in the image")

    async def estimate_age(self, face_image: np.ndarray) -> AgeEstimationResult:
        return AgeEstimationResult(age=25, age_group="adult")

    async def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        return np.zeros(512)

    async def is_loaded(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# _decode_image tests
# ---------------------------------------------------------------------------

class TestDecodeImage:
    def test_valid_jpeg_bytes(self):
        img = _decode_image(_make_fake_image_bytes())
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3

    def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="Could not decode image"):
            _decode_image(b"not-an-image")

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="Could not decode image"):
            _decode_image(b"")


# ---------------------------------------------------------------------------
# compare_faces tests
# ---------------------------------------------------------------------------

class TestCompareFaces:
    @pytest.mark.asyncio
    async def test_successful_comparison_returns_pipeline_result(self):
        provider = FakeProvider(age1=25, age_group1="adult", age2=30, age_group2="adult")
        img_bytes = _make_fake_image_bytes()

        result = await compare_faces(img_bytes, img_bytes, provider)

        assert isinstance(result, PipelineResult)
        assert result.age1 == 25
        assert result.age2 == 30
        assert result.age_group1 == "adult"
        assert result.age_group2 == "adult"
        assert result.similarity_score is not None
        assert result.confidence is not None
        assert result.result in ("same_person", "different_person")

    @pytest.mark.asyncio
    async def test_identical_embeddings_same_person(self):
        """When the provider returns the same embedding for both faces, result should be same_person."""
        provider = FakeProvider()
        img_bytes = _make_fake_image_bytes()

        result = await compare_faces(img_bytes, img_bytes, provider)

        # FakeProvider uses the same seed, so embeddings are identical
        assert result.result == "same_person"
        assert result.similarity_score is not None
        assert result.similarity_score >= 0.35

    @pytest.mark.asyncio
    async def test_rejection_for_infant_vs_adult(self):
        provider = FakeProvider(age1=2, age_group1="infant", age2=35, age_group2="adult")
        img_bytes = _make_fake_image_bytes()

        result = await compare_faces(img_bytes, img_bytes, provider)

        assert result.result == "rejected"
        assert result.similarity_score is None
        assert result.confidence is None
        assert result.age1 == 2
        assert result.age2 == 35
        assert "Cannot reliably compare" in result.message

    @pytest.mark.asyncio
    async def test_rejection_for_infant_vs_senior(self):
        provider = FakeProvider(age1=1, age_group1="infant", age2=60, age_group2="senior")
        img_bytes = _make_fake_image_bytes()

        result = await compare_faces(img_bytes, img_bytes, provider)

        assert result.result == "rejected"
        assert result.similarity_score is None

    @pytest.mark.asyncio
    async def test_infant_vs_child_allowed(self):
        provider = FakeProvider(age1=3, age_group1="infant", age2=8, age_group2="child")
        img_bytes = _make_fake_image_bytes()

        result = await compare_faces(img_bytes, img_bytes, provider)

        assert result.result != "rejected"
        assert result.similarity_score is not None

    @pytest.mark.asyncio
    async def test_invalid_image_raises_value_error(self):
        provider = FakeProvider()

        with pytest.raises(ValueError, match="Could not decode image"):
            await compare_faces(b"bad-data", _make_fake_image_bytes(), provider)

    @pytest.mark.asyncio
    async def test_face_detection_failure_propagates(self):
        provider = FailDetectionProvider()
        img_bytes = _make_fake_image_bytes()

        with pytest.raises(ValueError, match="No face detected"):
            await compare_faces(img_bytes, img_bytes, provider)

    @pytest.mark.asyncio
    async def test_result_message_for_same_person(self):
        provider = FakeProvider()
        img_bytes = _make_fake_image_bytes()

        result = await compare_faces(img_bytes, img_bytes, provider)

        assert result.result == "same_person"
        assert result.message == "Faces belong to the same person"

    @pytest.mark.asyncio
    async def test_result_fields_have_correct_types(self):
        provider = FakeProvider()
        img_bytes = _make_fake_image_bytes()

        result = await compare_faces(img_bytes, img_bytes, provider)

        assert isinstance(result.age1, int)
        assert isinstance(result.age2, int)
        assert isinstance(result.age_group1, str)
        assert isinstance(result.age_group2, str)
        assert isinstance(result.similarity_score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.result, str)
        assert isinstance(result.message, str)


# ---------------------------------------------------------------------------
# Property-based test: Response completeness (Property 9)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st
from models.schemas import ComparisonResponse


# Strategy for generating valid ComparisonResponse instances
comparison_response_strategy = st.builds(
    ComparisonResponse,
    age1=st.integers(min_value=0, max_value=120),
    age2=st.integers(min_value=0, max_value=120),
    age_group1=st.sampled_from(["infant", "child", "teen", "adult", "senior"]),
    age_group2=st.sampled_from(["infant", "child", "teen", "adult", "senior"]),
    similarity_score=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    result=st.sampled_from(["same_person", "different_person"]),
    message=st.text(min_size=1, max_size=200),
)


# Feature: age-invariant-face-recognition, Property 9: Successful comparison response completeness
# **Validates: Requirements 7.1**
@given(response=comparison_response_strategy)
@settings(max_examples=100)
def test_property_response_completeness(response: ComparisonResponse):
    """For any successful comparison result, the JSON response SHALL contain all
    required fields: age1, age2, age_group1, age_group2, similarity_score,
    confidence, result, and message, with correct types."""

    # Serialize to JSON dict (simulates what FastAPI returns)
    json_data = response.model_dump()

    # All required fields must be present
    required_fields = [
        "age1", "age2", "age_group1", "age_group2",
        "similarity_score", "confidence", "result", "message",
    ]
    for field in required_fields:
        assert field in json_data, f"Missing required field: {field}"

    # Correct types
    assert isinstance(json_data["age1"], int)
    assert isinstance(json_data["age2"], int)
    assert isinstance(json_data["age_group1"], str)
    assert isinstance(json_data["age_group2"], str)
    assert isinstance(json_data["similarity_score"], float)
    assert isinstance(json_data["confidence"], float)
    assert isinstance(json_data["result"], str)
    assert isinstance(json_data["message"], str)

    # No extra fields beyond what's required
    assert set(json_data.keys()) == set(required_fields)
