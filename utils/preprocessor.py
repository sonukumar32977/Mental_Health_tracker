"""
preprocessor.py - Text cleaning and normalization utilities.
Prepares raw journal entry text for NLP model inference.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize raw input text before model inference.

    Steps:
        1. Strip leading/trailing whitespace.
        2. Normalize unicode characters (e.g., smart quotes -> straight quotes).
        3. Remove URLs.
        4. Remove excessive whitespace and newlines.
        5. Limit to 512 characters (most transformer models have a 512 token limit).

    Args:
        text (str): Raw user-provided journal entry text.

    Returns:
        str: Cleaned and normalized text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip leading/trailing whitespace
    text = text.strip()

    # Normalize unicode (e.g., accented characters, smart quotes)
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Replace multiple newlines/carriage returns with a single space
    text = re.sub(r"[\r\n]+", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove non-printable / control characters (keep standard punctuation)
    text = re.sub(r"[^\x20-\x7E]", "", text)

    # Final strip
    text = text.strip()

    return text


def truncate_text(text: str, max_chars: int = 1500) -> str:
    """
    Truncate text to a maximum number of characters, preserving whole words.

    Args:
        text (str): Cleaned text string.
        max_chars (int): Maximum allowed character count.

    Returns:
        str: Truncated text string.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Trim to last complete word
    last_space = truncated.rfind(" ")
    if last_space != -1:
        truncated = truncated[:last_space]

    return truncated + "..."


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline: clean then truncate.

    Args:
        text (str): Raw journal entry from the user.

    Returns:
        str: Fully processed text ready for inference.
    """
    cleaned = clean_text(text)
    processed = truncate_text(cleaned)
    return processed
