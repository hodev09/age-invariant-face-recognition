import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length using L2 normalization.

    Returns the zero vector unchanged if the input is a zero vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm
