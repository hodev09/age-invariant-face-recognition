import numpy as np
import pytest
from services.similarity import compute_similarity, THRESHOLD


def _unit_vector(dim: int, index: int = 0) -> np.ndarray:
    """Create a unit vector with 1.0 at the given index."""
    vec = np.zeros(dim)
    vec[index] = 1.0
    return vec


def _random_unit_vector(rng: np.random.Generator, dim: int = 512) -> np.ndarray:
    vec = rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


class TestComputeSimilarity:
    def test_identical_embeddings_same_person(self):
        emb = _unit_vector(512)
        score, label, confidence = compute_similarity(emb, emb)
        assert np.isclose(score, 1.0, atol=1e-6)
        assert label == "same_person"
        assert confidence == 1.0

    def test_opposite_embeddings_different_person(self):
        emb1 = _unit_vector(512, 0)
        emb2 = -emb1
        score, label, confidence = compute_similarity(emb1, emb2)
        assert np.isclose(score, -1.0, atol=1e-6)
        assert label == "different_person"

    def test_orthogonal_embeddings_different_person(self):
        emb1 = _unit_vector(512, 0)
        emb2 = _unit_vector(512, 1)
        score, label, confidence = compute_similarity(emb1, emb2)
        assert np.isclose(score, 0.0, atol=1e-6)
        assert label == "different_person"

    def test_threshold_boundary_same_person(self):
        """Score exactly at 0.35 should be classified as same_person."""
        # Construct two unit vectors with dot product = 0.35
        emb1 = np.zeros(512)
        emb1[0] = 1.0
        emb2 = np.zeros(512)
        emb2[0] = 0.35
        emb2[1] = np.sqrt(1.0 - 0.35**2)
        score, label, confidence = compute_similarity(emb1, emb2)
        assert np.isclose(score, 0.35, atol=1e-6)
        assert label == "same_person"
        assert np.isclose(confidence, 0.0, atol=1e-6)

    def test_just_below_threshold_different_person(self):
        """Score just below 0.35 should be classified as different_person."""
        emb1 = np.zeros(512)
        emb1[0] = 1.0
        emb2 = np.zeros(512)
        emb2[0] = 0.34
        emb2[1] = np.sqrt(1.0 - 0.34**2)
        score, label, confidence = compute_similarity(emb1, emb2)
        assert score < THRESHOLD
        assert label == "different_person"

    def test_confidence_at_threshold_is_zero(self):
        """Confidence should be 0 when score equals the threshold."""
        emb1 = np.zeros(512)
        emb1[0] = 1.0
        emb2 = np.zeros(512)
        emb2[0] = 0.35
        emb2[1] = np.sqrt(1.0 - 0.35**2)
        _, _, confidence = compute_similarity(emb1, emb2)
        assert np.isclose(confidence, 0.0, atol=1e-6)

    def test_confidence_at_max_similarity_is_one(self):
        """Confidence should be 1.0 when score is 1.0."""
        emb = _unit_vector(512)
        _, _, confidence = compute_similarity(emb, emb)
        assert confidence == 1.0

    def test_confidence_bounded_zero_to_one(self):
        """Confidence should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            e1 = _random_unit_vector(rng)
            e2 = _random_unit_vector(rng)
            _, _, confidence = compute_similarity(e1, e2)
            assert 0.0 <= confidence <= 1.0

    def test_normalizes_non_unit_inputs(self):
        """Should handle non-normalized inputs by normalizing them first."""
        emb = np.full(512, 2.0)
        score, label, _ = compute_similarity(emb, emb)
        assert np.isclose(score, 1.0, atol=1e-6)
        assert label == "same_person"

    def test_returns_float_types(self):
        emb = _unit_vector(512)
        score, label, confidence = compute_similarity(emb, emb)
        assert isinstance(score, float)
        assert isinstance(label, str)
        assert isinstance(confidence, float)


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def _make_unit_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length; regenerate if zero."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


# Feature: age-invariant-face-recognition, Property 7: Cosine similarity range for unit vectors
# **Validates: Requirements 6.1**
@given(
    v1=arrays(
        np.float64,
        (512,),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
    v2=arrays(
        np.float64,
        (512,),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=200)
def test_cosine_similarity_range_for_unit_vectors(v1, v2):
    """For any two L2-normalized 512-dim vectors, cosine similarity is in [-1, 1]."""
    u1 = _make_unit_vector(v1)
    u2 = _make_unit_vector(v2)
    score, _, _ = compute_similarity(u1, u2)
    # Allow a tiny floating-point tolerance (IEEE 754 rounding in high-dim dot products)
    assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9, f"Similarity {score} out of range [-1, 1]"


# Feature: age-invariant-face-recognition, Property 8: Threshold classification consistency
# **Validates: Requirements 6.2, 6.3**
@given(score=st.floats(min_value=-1.0, max_value=1.0))
@settings(max_examples=200)
def test_threshold_classification_consistency(score):
    """For any similarity score, classification is 'same_person' iff score >= 0.35."""
    # Build two unit vectors whose dot product equals `score`.
    # e1 = [1, 0, 0, ...], e2 = [score, sqrt(1 - score^2), 0, ...]
    e1 = np.zeros(512)
    e1[0] = 1.0

    e2 = np.zeros(512)
    e2[0] = score
    remainder = 1.0 - score * score
    if remainder > 0:
        e2[1] = np.sqrt(remainder)

    _, label, _ = compute_similarity(e1, e2)

    if score >= THRESHOLD:
        assert label == "same_person", f"Score {score} >= {THRESHOLD} but got '{label}'"
    else:
        assert label == "different_person", f"Score {score} < {THRESHOLD} but got '{label}'"
