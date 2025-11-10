import logging
from typing import Literal
from textwrap import shorten
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def summarize_text(text: str, max_length: int = 300) -> str:
    """
    Simple heuristic-based summarization for concise mode.
    (Optional: Can be replaced with LLM-based summarization.)
    """
    try:
        if len(text) <= max_length:
            return text
        return shorten(text, width=max_length, placeholder="...")
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return text


def format_response(response: str, mode: Literal["concise", "detailed"] = "detailed") -> str:
    """
    Format the LLM response according to the selected mode.

    Args:
        response: The raw AI-generated response text.
        mode: 'concise' or 'detailed'.

    Returns:
        Formatted text string.
    """
    try:
        if mode == "concise":
            return summarize_text(response, max_length=300)
        else:
            return response
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return response


def clean_text(text: str) -> str:
    """
    Sanitize and normalize user input.
    """
    try:
        return text.strip().replace("\n", " ").replace("\r", " ")
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text


def chunk_large_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    """
    Break long text into overlapping chunks for embedding.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return [text]
