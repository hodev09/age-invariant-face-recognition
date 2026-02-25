import numpy as np
import pytest
from utils.embedding import l2_normalize


class TestL2Normalize:
    def test_normalizes_to_unit_length(self):
        vec = np.array([3.0, 4.0])
        result = l2_normalize(vec)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_zero_vector_returns_zero(self):
        vec = np.zeros(512)
        result = l2_normalize(vec)
        np.testing.assert_array_equal(result, np.zeros(512))

    def test_512_dim_vector(self):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(512)
        result = l2_normalize(vec)
        assert result.shape == (512,)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_already_unit_vector(self):
        vec = np.zeros(512)
        vec[0] = 1.0
        result = l2_normalize(vec)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)
        np.testing.assert_allclose(result, vec, atol=1e-6)

    def test_negative_values(self):
        vec = np.array([-3.0, -4.0])
        result = l2_normalize(vec)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)
        assert np.allclose(result, np.array([-0.6, -0.8]), atol=1e-6)

    def test_preserves_direction(self):
        rng = np.random.default_rng(99)
        vec = rng.standard_normal(512)
        result = l2_normalize(vec)
        # Normalized vector should be a positive scalar multiple of original
        scale = vec / result
        assert np.allclose(scale, scale[0], atol=1e-6)

# --- Property-based tests ---
# Feature: age-invariant-face-recognition, Property 6: L2 normalization invariant
# For any non-zero 512-dimensional vector, after L2 normalization the resulting
# vector's L2 norm SHALL equal 1.0 within a tolerance of 1e-6.
# **Validates: Requirements 5.2, 5.3**

from hypothesis import given, settings, assume
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st


@given(vec=arrays(np.float64, (512,), elements=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)))
@settings(max_examples=200)
def test_l2_normalization_invariant(vec):
    """For any non-zero 512-dim vector, L2 norm after normalization equals 1.0."""
    assume(np.linalg.norm(vec) > 0)
    result = l2_normalize(vec)
    norm = np.linalg.norm(result)
    assert np.isclose(norm, 1.0, atol=1e-6), (
        f"Expected L2 norm ~1.0, got {norm}"
    )
