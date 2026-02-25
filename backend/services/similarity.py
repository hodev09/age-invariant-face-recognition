import numpy as np

from utils.embedding import l2_normalize

THRESHOLD = 0.35
MAX_DISTANCE = 0.65  # 1.0 - THRESHOLD


def compute_similarity(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> tuple[float, str, float]:
    """Compute cosine similarity between two embeddings and classify the result.

    Returns (similarity_score, result_label, confidence).
    result_label: "same_person" if score >= 0.35, "different_person" otherwise.
    confidence: distance from threshold normalized to [0, 1].
    """
    e1 = l2_normalize(embedding1)
    e2 = l2_normalize(embedding2)

    similarity = float(np.dot(e1, e2))

    result_label = "same_person" if similarity >= THRESHOLD else "different_person"
    confidence = min(abs(similarity - THRESHOLD) / MAX_DISTANCE, 1.0)

    return similarity, result_label, confidence
