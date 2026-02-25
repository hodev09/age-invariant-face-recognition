"""Age group classification and age-based comparison rules."""


def classify_age_group(age: int) -> str:
    """Classify a numeric age into an age group.

    Mapping:
        0–4   → "infant"
        5–12  → "child"
        13–19 → "teen"
        20–49 → "adult"
        50+   → "senior"

    Boundary values map to the group whose range starts at that value.
    """
    if age <= 4:
        return "infant"
    elif age <= 12:
        return "child"
    elif age <= 19:
        return "teen"
    elif age <= 49:
        return "adult"
    else:
        return "senior"

# Age groups that cannot be paired with "infant"
_INFANT_REJECTED = frozenset({"teen", "adult", "senior"})

_REJECTION_MESSAGE = "Cannot reliably compare infant/childhood images with adult images"


def check_age_rules(age_group1: str, age_group2: str) -> tuple[bool, str | None]:
    """Check whether two age groups are eligible for comparison.

    The only rejected pairings are infant vs teen, adult, or senior.
    All other combinations (including infant-infant and infant-child) are allowed.

    Returns:
        (True, None) when the comparison is allowed.
        (False, rejection_message) when the comparison is rejected.
    """
    if age_group1 == "infant" and age_group2 in _INFANT_REJECTED:
        return (False, _REJECTION_MESSAGE)
    if age_group2 == "infant" and age_group1 in _INFANT_REJECTED:
        return (False, _REJECTION_MESSAGE)
    return (True, None)

