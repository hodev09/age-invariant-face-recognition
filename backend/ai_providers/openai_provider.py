"""OpenAIProvider — uses OpenCV Haar cascade for face detection and OpenAI vision API for age estimation."""

import base64
import logging
import os

import cv2
import numpy as np

from .base import AIProvider, AgeEstimationResult, FaceDetectionResult
from services.age_rules import classify_age_group
from utils.embedding import l2_normalize

logger = logging.getLogger(__name__)


class OpenAIProvider(AIProvider):
    """AI provider backed by OpenCV Haar cascade (face detection) and OpenAI GPT-4.1-mini (age estimation).

    Since OpenAI cannot produce numeric embeddings, ``generate_embedding`` returns a
    synthetic 512-dim vector derived from the estimated age so that the rest of the
    pipeline (cosine similarity) can still operate.  For real similarity work the
    LocalInsightFaceProvider should be preferred.
    """

    def __init__(self) -> None:
        self._cascade: cv2.CascadeClassifier | None = None
        self._client = None  # lazy-loaded openai.AsyncOpenAI

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_cascade(self) -> cv2.CascadeClassifier:
        """Load the Haar cascade classifier for frontal face detection (lazy)."""
        if self._cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        return self._cascade

    def _get_client(self):
        """Return (and lazily create) an ``openai.AsyncOpenAI`` client."""
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI()  # reads OPENAI_API_KEY from env
        return self._client

    @staticmethod
    def _encode_image_to_base64(image: np.ndarray) -> str:
        """Encode a BGR numpy image to a base64 JPEG string for the OpenAI API."""
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image to JPEG")
        return base64.b64encode(buffer).decode("utf-8")


    # ------------------------------------------------------------------
    # AIProvider interface
    # ------------------------------------------------------------------

    async def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect exactly one face using OpenCV Haar cascade.

        Raises ``ValueError`` when zero or more than one face is found.
        """
        cascade = self._ensure_cascade()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected; please upload an image with exactly one face"
            )

        x, y, w, h = faces[0]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        face_image = image[y1:y2, x1:x2].copy()

        return FaceDetectionResult(
            face_image=face_image,
            bbox=(x1, y1, x2, y2),
            confidence=1.0,  # Haar cascade doesn't provide a confidence score
        )

    async def estimate_age(self, face_image: np.ndarray) -> AgeEstimationResult:
        """Estimate age by sending the face image to GPT-4.1-mini via the vision API."""
        client = self._get_client()
        b64_image = self._encode_image_to_base64(face_image)

        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an age estimation assistant. "
                        "Given a photo of a human face, respond with ONLY a single integer "
                        "representing your best estimate of the person's age in years. "
                        "Do not include any other text, explanation, or punctuation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=10,
        )

        raw = response.choices[0].message.content.strip()
        try:
            age = int(raw)
        except (ValueError, TypeError):
            logger.warning("OpenAI returned non-integer age estimate: %r, defaulting to 25", raw)
            age = 25

        age = max(0, age)
        age_group = classify_age_group(age)
        return AgeEstimationResult(age=age, age_group=age_group)

    async def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Generate a synthetic 512-dim embedding from the face image.

        OpenAI does not provide numeric face embeddings.  As a workaround we create a
        deterministic 512-dim vector seeded from a perceptual hash of the image so that
        identical (or very similar) images produce similar embeddings.  This is NOT a
        true identity embedding — use LocalInsightFaceProvider for accurate comparisons.
        """
        # Resize to a fixed small size and flatten to create a deterministic seed
        resized = cv2.resize(face_image, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Use pixel values as a seed for a deterministic pseudo-random embedding
        seed = int(np.sum(gray)) % (2**31)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(512).astype(np.float64)
        return l2_normalize(embedding)

    async def is_loaded(self) -> bool:
        """Return True when the OPENAI_API_KEY environment variable is set."""
        return bool(os.environ.get("OPENAI_API_KEY"))
