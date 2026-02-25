from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceDetectionResult:
    """Result of face detection containing the cropped face image, bounding box, and confidence."""

    face_image: np.ndarray  # Cropped face as BGR numpy array
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


@dataclass
class AgeEstimationResult:
    """Result of age estimation containing numeric age and categorical age group."""

    age: int
    age_group: str  # "infant", "child", "teen", "adult", "senior"


class AIProvider(ABC):
    """Abstract base class for AI providers that perform face analysis operations."""

    @abstractmethod
    async def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect exactly one face in the image.

        Args:
            image: Input image as a BGR numpy array.

        Returns:
            FaceDetectionResult with cropped face, bounding box, and confidence.

        Raises:
            ValueError: If zero or more than one face is detected.
        """
        ...

    @abstractmethod
    async def estimate_age(self, face_image: np.ndarray) -> AgeEstimationResult:
        """Estimate age from a cropped face image.

        Args:
            face_image: Cropped face image as a BGR numpy array.

        Returns:
            AgeEstimationResult with numeric age and age group.
        """
        ...

    @abstractmethod
    async def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Generate a 512-dimensional L2-normalized face embedding.

        Args:
            face_image: Cropped face image as a BGR numpy array.

        Returns:
            A 512-dimensional L2-normalized numpy array.
        """
        ...

    @abstractmethod
    async def is_loaded(self) -> bool:
        """Check if the AI models are loaded and ready for inference."""
        ...
