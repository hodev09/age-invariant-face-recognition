ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_upload(filename: str, file_bytes: bytes) -> bytes:
    """Validate file type and size, return bytes on success.

    Raises ValueError with a descriptive message on failure.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {ext!r}. Allowed: jpg, jpeg, png, webp"
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("File size exceeds 10 MB limit")

    return file_bytes
