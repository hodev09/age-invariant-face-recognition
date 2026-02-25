import pytest
from utils.validator import validate_upload, ALLOWED_EXTENSIONS, MAX_FILE_SIZE


class TestValidateUpload:
    def test_valid_jpg(self):
        result = validate_upload("photo.jpg", b"fake image data")
        assert result == b"fake image data"

    def test_valid_jpeg(self):
        result = validate_upload("photo.jpeg", b"data")
        assert result == b"data"

    def test_valid_png(self):
        result = validate_upload("photo.png", b"data")
        assert result == b"data"

    def test_valid_webp(self):
        result = validate_upload("photo.webp", b"data")
        assert result == b"data"

    def test_rejects_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            validate_upload("photo.bmp", b"data")

    def test_rejects_no_extension(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            validate_upload("photo", b"data")

    def test_rejects_oversized_file(self):
        big_data = b"x" * (MAX_FILE_SIZE + 1)
        with pytest.raises(ValueError, match="File size exceeds 10 MB limit"):
            validate_upload("photo.jpg", big_data)

    def test_accepts_exact_max_size(self):
        data = b"x" * MAX_FILE_SIZE
        result = validate_upload("photo.png", data)
        assert len(result) == MAX_FILE_SIZE

    def test_case_insensitive_extension(self):
        result = validate_upload("photo.JPG", b"data")
        assert result == b"data"
