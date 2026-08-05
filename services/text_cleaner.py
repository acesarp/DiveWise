import re


class TextCleaner:
    """Provides text cleaning utilities for PDF content."""

    @staticmethod
    def clean(text: str) -> str:
        """
        Cleans extracted PDF text while preserving its meaning.

        Args:
            text: Raw text extracted from a PDF.

        Returns:
            Cleaned text.
        """

        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove tabs
        text = text.replace("\t", " ")

        # Remove non-breaking spaces
        text = text.replace("\xa0", " ")

        # Collapse multiple spaces
        text = re.sub(r"[ ]{2,}", " ", text)

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Trim spaces around new lines
        text = re.sub(r" *\n *", "\n", text)

        return text.strip()