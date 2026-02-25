"""Unit tests and property tests for the age rule engine.

Tests the check_age_rules function against the rule matrix defined in the design.
"""

import pytest

from services.age_rules import check_age_rules

_REJECTION_MSG = "Cannot reliably compare infant/childhood images with adult images"

ALL_GROUPS = ["infant", "child", "teen", "adult", "senior"]


class TestCheckAgeRules:
    """Unit tests covering every cell in the rule matrix."""

    # Allowed pairings
    @pytest.mark.parametrize("g1,g2", [
        ("infant", "infant"),
        ("infant", "child"),
        ("child", "infant"),
        ("child", "child"),
        ("child", "teen"),
        ("child", "adult"),
        ("child", "senior"),
        ("teen", "child"),
        ("teen", "teen"),
        ("teen", "adult"),
        ("teen", "senior"),
        ("adult", "child"),
        ("adult", "teen"),
        ("adult", "adult"),
        ("adult", "senior"),
        ("senior", "child"),
        ("senior", "teen"),
        ("senior", "adult"),
        ("senior", "senior"),
    ])
    def test_allowed_pairings(self, g1, g2):
        allowed, msg = check_age_rules(g1, g2)
        assert allowed is True
        assert msg is None

    # Rejected pairings
    @pytest.mark.parametrize("g1,g2", [
        ("infant", "teen"),
        ("infant", "adult"),
        ("infant", "senior"),
        ("teen", "infant"),
        ("adult", "infant"),
        ("senior", "infant"),
    ])
    def test_rejected_pairings(self, g1, g2):
        allowed, msg = check_age_rules(g1, g2)
        assert allowed is False
        assert msg == _REJECTION_MSG


# --- Property-based tests ---

from hypothesis import given, strategies as st, settings

_ALL_GROUPS = ["infant", "child", "teen", "adult", "senior"]
_NON_INFANT_GROUPS = ["child", "teen", "adult", "senior"]
_INFANT_REJECTED_GROUPS = ["teen", "adult", "senior"]


# --- Property 4: Infant rejection rule ---
# Feature: age-invariant-face-recognition, Property 4: Age rule engine — infant rejection
# For any pair of age groups where one is "infant" and the other is "teen",
# "adult", or "senior", check_age_rules SHALL return rejected with the message
# "Cannot reliably compare infant/childhood images with adult images".
# **Validates: Requirements 4.2**

@given(other=st.sampled_from(_INFANT_REJECTED_GROUPS))
@settings(max_examples=100)
def test_infant_first_rejected(other):
    """infant paired with teen/adult/senior (infant first) is always rejected."""
    allowed, msg = check_age_rules("infant", other)
    assert allowed is False
    assert msg == _REJECTION_MSG


@given(other=st.sampled_from(_INFANT_REJECTED_GROUPS))
@settings(max_examples=100)
def test_infant_second_rejected(other):
    """infant paired with teen/adult/senior (infant second) is always rejected."""
    allowed, msg = check_age_rules(other, "infant")
    assert allowed is False
    assert msg == _REJECTION_MSG


# --- Property 5: Non-infant allowance rule ---
# Feature: age-invariant-face-recognition, Property 5: Age rule engine — non-infant allowance
# For any pair of age groups where neither is "infant", check_age_rules SHALL
# return allowed.
# **Validates: Requirements 4.4**

@given(
    g1=st.sampled_from(_NON_INFANT_GROUPS),
    g2=st.sampled_from(_NON_INFANT_GROUPS),
)
@settings(max_examples=100)
def test_non_infant_always_allowed(g1, g2):
    """Any pairing where neither group is infant is always allowed."""
    allowed, msg = check_age_rules(g1, g2)
    assert allowed is True
    assert msg is None


# --- Property 12: Age rule symmetry ---
# Feature: age-invariant-face-recognition, Property 12: Age rule symmetry
# For any two age groups g1 and g2, check_age_rules(g1, g2) SHALL return the
# same result as check_age_rules(g2, g1).
# **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

@given(
    g1=st.sampled_from(_ALL_GROUPS),
    g2=st.sampled_from(_ALL_GROUPS),
)
@settings(max_examples=100)
def test_age_rule_symmetry(g1, g2):
    """check_age_rules is symmetric: swapping arguments gives the same result."""
    result_forward = check_age_rules(g1, g2)
    result_reverse = check_age_rules(g2, g1)
    assert result_forward == result_reverse
