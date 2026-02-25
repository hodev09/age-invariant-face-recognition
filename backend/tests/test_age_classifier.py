"""Unit tests for age group classifier.

Tests boundary values and representative ages for each group.
"""

import pytest

from services.age_rules import classify_age_group


class TestClassifyAgeGroup:
    """Unit tests for classify_age_group."""

    # Boundary values — each maps to the group whose range starts at that value
    @pytest.mark.parametrize("age,expected", [
        (0, "infant"),
        (5, "child"),
        (13, "teen"),
        (20, "adult"),
        (50, "senior"),
    ])
    def test_boundary_values(self, age, expected):
        assert classify_age_group(age) == expected

    # Upper boundary of each range
    @pytest.mark.parametrize("age,expected", [
        (4, "infant"),
        (12, "child"),
        (19, "teen"),
        (49, "adult"),
    ])
    def test_upper_boundaries(self, age, expected):
        assert classify_age_group(age) == expected

    # Representative mid-range values
    @pytest.mark.parametrize("age,expected", [
        (2, "infant"),
        (8, "child"),
        (16, "teen"),
        (35, "adult"),
        (75, "senior"),
    ])
    def test_mid_range_values(self, age, expected):
        assert classify_age_group(age) == expected

    def test_very_old_age(self):
        assert classify_age_group(120) == "senior"

    def test_returns_exactly_one_of_five_groups(self):
        valid_groups = {"infant", "child", "teen", "adult", "senior"}
        for age in range(0, 121):
            assert classify_age_group(age) in valid_groups


# --- Property 3: Age group classification correctness ---
# Feature: age-invariant-face-recognition, Property 3: Age group classification correctness
# For any non-negative integer age, classify_age_group(age) SHALL return exactly
# one of the five groups, and the returned group SHALL match the defined ranges:
# 0–4 → "infant", 5–12 → "child", 13–19 → "teen", 20–49 → "adult", 50+ → "senior".
# Edge cases: boundary values 0, 5, 13, 20, 50 SHALL map to the group whose range
# starts at that value.
# **Validates: Requirements 3.2, 3.3**

from hypothesis import given, strategies as st, settings

VALID_GROUPS = {"infant", "child", "teen", "adult", "senior"}

AGE_RANGES = {
    "infant": (0, 4),
    "child": (5, 12),
    "teen": (13, 19),
    "adult": (20, 49),
    "senior": (50, 120),
}


@given(age=st.integers(min_value=0, max_value=120))
@settings(max_examples=200)
def test_returns_exactly_one_valid_group(age):
    """For any non-negative age, the result is exactly one of the five groups."""
    result = classify_age_group(age)
    assert result in VALID_GROUPS


@given(age=st.integers(min_value=0, max_value=120))
@settings(max_examples=200)
def test_age_maps_to_correct_range(age):
    """The returned group matches the defined range for the given age."""
    result = classify_age_group(age)
    low, high = AGE_RANGES[result]
    assert low <= age <= high, (
        f"age={age} mapped to '{result}' but expected range is [{low}, {high}]"
    )


@given(boundary=st.sampled_from([0, 5, 13, 20, 50]))
@settings(max_examples=100)
def test_boundary_values_map_to_starting_group(boundary):
    """Boundary values map to the group whose range starts at that value."""
    expected = {0: "infant", 5: "child", 13: "teen", 20: "adult", 50: "senior"}
    result = classify_age_group(boundary)
    assert result == expected[boundary], (
        f"boundary={boundary} mapped to '{result}' but expected '{expected[boundary]}'"
    )
