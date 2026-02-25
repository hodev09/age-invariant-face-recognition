"""Pipeline service that orchestrates the full face comparison flow."""

from dataclasses import dataclass

import cv2
import numpy as np

from ai_providers.base import AIProvider
from services.age_rules import check_age_rules
from services.similarity import compute_similarity


@dataclass
class PipelineResult:
    """Result of the face comparison pipeline."""

    age1: int
    age2: int
    age_group1: str
    age_group2: str
    similarity_score: float | None  # None if rejected
    confidence: float | None
    result: str  # "same_person", "different_person", "rejected"
    message: str


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    if not image_bytes:
        raise ValueError(
            "Could not decode image. Please upload a valid image file."
        )
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None:
        raise ValueError(
            "Could not decode image. Please upload a valid image file."
        )
    return image


async def compare_faces(
    image1_bytes: bytes,
    image2_bytes: bytes,
    provider: AIProvider,
) -> PipelineResult:
    """Run the full face comparison pipeline.

    Steps:
        1. Decode both images from bytes
        2. Detect a single face in each image
        3. Estimate age for each detected face
        4. Check age-based comparison rules
        5. If allowed, generate embeddings and compute similarity
        6. Assemble and return PipelineResult
    """
    # 1. Decode images
    image1 = _decode_image(image1_bytes)
    image2 = _decode_image(image2_bytes)

    # 2. Detect faces
    face1 = await provider.detect_face(image1)
    face2 = await provider.detect_face(image2)

    # 3. Estimate ages
    age_result1 = await provider.estimate_age(face1.face_image)
    age_result2 = await provider.estimate_age(face2.face_image)

    # 4. Check age rules
    allowed, rejection_message = check_age_rules(
        age_result1.age_group, age_result2.age_group
    )

    if not allowed:
        return PipelineResult(
            age1=age_result1.age,
            age2=age_result2.age,
            age_group1=age_result1.age_group,
            age_group2=age_result2.age_group,
            similarity_score=None,
            confidence=None,
            result="rejected",
            message=rejection_message,  # type: ignore[arg-type]
        )

    # 5. Generate embeddings and compute similarity
    embedding1 = await provider.generate_embedding(face1.face_image)
    embedding2 = await provider.generate_embedding(face2.face_image)

    similarity_score, result_label, confidence = compute_similarity(
        embedding1, embedding2
    )

    # 6. Assemble result
    return PipelineResult(
        age1=age_result1.age,
        age2=age_result2.age,
        age_group1=age_result1.age_group,
        age_group2=age_result2.age_group,
        similarity_score=similarity_score,
        confidence=confidence,
        result=result_label,
        message=(
            "Faces belong to the same person"
            if result_label == "same_person"
            else "Faces belong to different people"
        ),
    )
