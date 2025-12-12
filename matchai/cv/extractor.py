import re
from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text with normalized whitespace.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the file is not a valid PDF.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")

    text_parts = []

    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    raw_text = "\n".join(text_parts)
    cleaned_text = _clean_whitespace(raw_text)

    return cleaned_text


def _clean_whitespace(text: str) -> str:
    """Normalize whitespace in extracted text."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text
