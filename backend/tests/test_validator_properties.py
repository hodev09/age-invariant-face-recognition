"""Property-based tests for file upload validator.

Uses Hypothesis to verify correctness properties across many random inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from utils.validator import validate_upload, ALLOWED_EXTENSIONS, MAX_FILE_SIZE


# --- Property 1: File format validation ---
# Feature: age-invariant-face-recognition, Property 1: File format validation
# For any file with an extension not in {jpg, jpeg, png, webp}, the validator
# SHALL reject it; for any file with an extension in {jpg, jpeg, png, webp},
# the validator SHALL NOT reject it for format reasons.
# **Validates: Requirements 1.2**

@given(ext=st.sampled_from(sorted(ALLOWED_EXTENSIONS)))
@settings(max_examples=100)
def test_allowed_extensions_accepted(ext):
    """Any file with an allowed extension should pass format validation."""
    filename = f"photo.{ext}"
    result = validate_upload(filename, b"data")
    assert result == b"data"


@given(ext=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Ll",))))
@settings(max_examples=100)
def test_disallowed_extensions_rejected(ext):
    """Any file with an extension not in the allowed set should be rejected."""
    assume(ext.lower() not in ALLOWED_EXTENSIONS)
    filename = f"photo.{ext}"
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_upload(filename, b"data")


# --- Property 2: File size validation ---
# Feature: age-invariant-face-recognition, Property 2: File size validation
# For any file with size greater than 10 MB, the validator SHALL reject it;
# for any file with size less than or equal to 10 MB, the validator SHALL NOT
# reject it for size reasons.
# **Validates: Requirements 1.3**

@given(size=st.integers(min_value=MAX_FILE_SIZE + 1, max_value=MAX_FILE_SIZE + 1024))
@settings(max_examples=100)
def test_oversized_files_rejected(size):
    """Any file exceeding 10 MB should be rejected."""
    data = b"\x00" * size
    with pytest.raises(ValueError, match="File size exceeds 10 MB limit"):
        validate_upload("photo.jpg", data)


@given(size=st.integers(min_value=0, max_value=MAX_FILE_SIZE))
@settings(max_examples=100)
def test_valid_size_files_accepted(size):
    """Any file at or under 10 MB should pass size validation."""
    data = b"\x00" * size
    result = validate_upload("photo.jpg", data)
    assert len(result) == size
