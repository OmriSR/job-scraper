"""Tests for CV PDF extraction."""

import tempfile
from pathlib import Path

import pytest

from matchai.cv.extractor import _clean_whitespace, extract_text_from_pdf


class TestCleanWhitespace:
    def test_normalizes_multiple_spaces(self):
        text = "hello    world"
        assert _clean_whitespace(text) == "hello world"

    def test_normalizes_multiple_newlines(self):
        text = "hello\n\n\n\nworld"
        assert _clean_whitespace(text) == "hello\n\nworld"

    def test_converts_crlf_to_lf(self):
        text = "hello\r\nworld"
        assert _clean_whitespace(text) == "hello\nworld"

    def test_strips_leading_trailing(self):
        text = "  hello world  "
        assert _clean_whitespace(text) == "hello world"


class TestExtractTextFromPdf:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(Path("/nonexistent/file.pdf"))

    def test_not_a_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a pdf")
            path = Path(f.name)

        with pytest.raises(ValueError, match="not a PDF"):
            extract_text_from_pdf(path)

        path.unlink()
