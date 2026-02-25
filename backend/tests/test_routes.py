"""Unit tests for FastAPI routes (/compare-faces and /health).

Tests use a mocked AI provider to avoid requiring actual model files.
Requirements: 1.1, 2.3, 2.4, 4.5, 7.2, 7.3, 9.1, 9.2
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from ai_providers.base import AIProvider, AgeEstimationResult, FaceDetectionResult
from app import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg_bytes() -> bytes:
    """Create minimal valid JPEG bytes."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_png_bytes() -> bytes:
    """Create minimal valid PNG bytes."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class FakeProvider(AIProvider):
    """Fake provider returning configurable ages and identical embeddings."""

    def __init__(
        self,
        age1: int = 25,
        age_group1: str = "adult",
        age2: int = 30,
        age_group2: str = "adult",
        loaded: bool = True,
    ):
        self._ages = [(age1, age_group1), (age2, age_group2)]
        self._call_index = 0
        self._loaded = loaded

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
        return self._loaded


class NoFaceProvider(AIProvider):
    """Provider that raises 'No face detected'."""

    async def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        raise ValueError("No face detected in the image")

    async def estimate_age(self, face_image: np.ndarray) -> AgeEstimationResult:
        return AgeEstimationResult(age=25, age_group="adult")

    async def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        return np.zeros(512)

    async def is_loaded(self) -> bool:
        return True


class MultiFaceProvider(AIProvider):
    """Provider that raises 'Multiple faces detected'."""

    async def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        raise ValueError(
            "Multiple faces detected; please upload an image with exactly one face"
        )

    async def estimate_age(self, face_image: np.ndarray) -> AgeEstimationResult:
        return AgeEstimationResult(age=25, age_group="adult")

    async def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        return np.zeros(512)

    async def is_loaded(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fixture: async client
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# POST /compare-faces — valid images with mocked provider
# ---------------------------------------------------------------------------


class TestCompareFacesValid:
    """Validates: Req 1.1, 7.1 — valid comparison returns ComparisonResponse."""

    @pytest.mark.asyncio
    async def test_valid_jpeg_returns_200(self, client: AsyncClient):
        provider = FakeProvider(age1=25, age_group1="adult", age2=30, age_group2="adult")
        with patch("routes.compare.get_provider", return_value=provider):
            resp = await client.post(
                "/compare-faces",
                files={
                    "image1": ("face1.jpg", _make_jpeg_bytes(), "image/jpeg"),
                    "image2": ("face2.jpg", _make_jpeg_bytes(), "image/jpeg"),
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["age1"] == 25
        assert data["age2"] == 30
        assert data["age_group1"] == "adult"
        assert data["age_group2"] == "adult"
        assert "similarity_score" in data
        assert "confidence" in data
        assert data["result"] in ("same_person", "different_person")
        assert "message" in data

    @pytest.mark.asyncio
    async def test_valid_png_returns_200(self, client: AsyncClient):
        provider = FakeProvider()
        with patch("routes.compare.get_provider", return_value=provider):
            resp = await client.post(
                "/compare-faces",
                files={
                    "image1": ("face1.png", _make_png_bytes(), "image/png"),
                    "image2": ("face2.png", _make_png_bytes(), "image/png"),
                },
            )
        assert resp.status_code == 200
        assert resp.json()["result"] in ("same_person", "different_person")


# ---------------------------------------------------------------------------
# POST /compare-faces — invalid file types and sizes
# ---------------------------------------------------------------------------


class TestCompareFacesInvalidFiles:
    """Validates: Req 1.2, 1.3 — invalid format/size returns 400."""

    @pytest.mark.asyncio
    async def test_unsupported_extension_returns_400(self, client: AsyncClient):
        resp = await client.post(
            "/compare-faces",
            files={
                "image1": ("notes.txt", b"hello world", "text/plain"),
                "image2": ("face.jpg", _make_jpeg_bytes(), "image/jpeg"),
            },
        )
        assert resp.status_code == 400
        assert "Unsupported file format" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_unsupported_extension_second_image(self, client: AsyncClient):
        resp = await client.post(
            "/compare-faces",
            files={
                "image1": ("face.jpg", _make_jpeg_bytes(), "image/jpeg"),
                "image2": ("doc.pdf", b"pdf-content", "application/pdf"),
            },
        )
        assert resp.status_code == 400
        assert "Unsupported file format" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_oversized_file_returns_400(self, client: AsyncClient):
        oversized = b"\x00" * (10 * 1024 * 1024 + 1)  # 10 MB + 1 byte
        resp = await client.post(
            "/compare-faces",
            files={
                "image1": ("big.jpg", oversized, "image/jpeg"),
                "image2": ("face.jpg", _make_jpeg_bytes(), "image/jpeg"),
            },
        )
        assert resp.status_code == 400
        assert "10 MB" in resp.json()["error"]


# ---------------------------------------------------------------------------
# POST /compare-faces — no-face and multi-face scenarios
# ---------------------------------------------------------------------------


class TestCompareFacesFaceDetectionErrors:
    """Validates: Req 2.3, 2.4 — detection errors return 422."""

    @pytest.mark.asyncio
    async def test_no_face_detected_returns_422(self, client: AsyncClient):
        provider = NoFaceProvider()
        with patch("routes.compare.get_provider", return_value=provider):
            resp = await client.post(
                "/compare-faces",
                files={
                    "image1": ("face1.jpg", _make_jpeg_bytes(), "image/jpeg"),
                    "image2": ("face2.jpg", _make_jpeg_bytes(), "image/jpeg"),
                },
            )
        assert resp.status_code == 422
        assert "No face detected" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_multiple_faces_detected_returns_422(self, client: AsyncClient):
        provider = MultiFaceProvider()
        with patch("routes.compare.get_provider", return_value=provider):
            resp = await client.post(
                "/compare-faces",
                files={
                    "image1": ("face1.jpg", _make_jpeg_bytes(), "image/jpeg"),
                    "image2": ("face2.jpg", _make_jpeg_bytes(), "image/jpeg"),
                },
            )
        assert resp.status_code == 422
        assert "Multiple faces detected" in resp.json()["error"]


# ---------------------------------------------------------------------------
# POST /compare-faces — age rule rejection
# ---------------------------------------------------------------------------


class TestCompareFacesAgeRuleRejection:
    """Validates: Req 4.5, 7.3 — rejection returns ages + message."""

    @pytest.mark.asyncio
    async def test_infant_vs_adult_returns_rejection(self, client: AsyncClient):
        provider = FakeProvider(age1=2, age_group1="infant", age2=35, age_group2="adult")
        with patch("routes.compare.get_provider", return_value=provider):
            resp = await client.post(
                "/compare-faces",
                files={
                    "image1": ("baby.jpg", _make_jpeg_bytes(), "image/jpeg"),
                    "image2": ("adult.jpg", _make_jpeg_bytes(), "image/jpeg"),
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"] == "rejected"
        assert data["age1"] == 2
        assert data["age2"] == 35
        assert data["age_group1"] == "infant"
        assert data["age_group2"] == "adult"
        assert "Cannot reliably compare" in data["message"]
        # Rejection response must NOT contain similarity_score or confidence
        assert "similarity_score" not in data
        assert "confidence" not in data

    @pytest.mark.asyncio
    async def test_infant_vs_senior_returns_rejection(self, client: AsyncClient):
        provider = FakeProvider(age1=1, age_group1="infant", age2=60, age_group2="senior")
        with patch("routes.compare.get_provider", return_value=provider):
            resp = await client.post(
                "/compare-faces",
                files={
                    "image1": ("baby.jpg", _make_jpeg_bytes(), "image/jpeg"),
                    "image2": ("senior.jpg", _make_jpeg_bytes(), "image/jpeg"),
                },
            )
        assert resp.status_code == 200
        assert resp.json()["result"] == "rejected"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Validates: Req 9.1, 9.2 — health check returns status and model_loaded."""

    @pytest.mark.asyncio
    async def test_health_with_loaded_provider(self, client: AsyncClient):
        provider = FakeProvider(loaded=True)
        with patch("routes.health.get_provider", return_value=provider):
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_health_with_unloaded_provider(self, client: AsyncClient):
        provider = FakeProvider(loaded=False)
        with patch("routes.health.get_provider", return_value=provider):
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False

    @pytest.mark.asyncio
    async def test_health_when_provider_raises(self, client: AsyncClient):
        with patch("routes.health.get_provider", side_effect=RuntimeError("boom")):
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False
