import json
import logging
import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger("ImageEmbed")

SUPPORTED_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def get_genai_client() -> genai.Client:
    """Initialize and return the Gemini API client using GEMINI_API_KEY."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def generate_image_embedding_from_bytes( image_bytes: bytes, mime_type: str = "image/jpeg" ) -> list[float]:
    """
    Generate Gemini multimodal vector embedding for raw image bytes.

    :param image_bytes: Raw binary content of the image.
    :param mime_type: Content type string (e.g., 'image/jpeg', 'image/png').
    :return: 768-dimensional float embedding vector.
    """
    if not image_bytes:
        raise ValueError("image_bytes cannot be empty.")

    clean_mime_type = mime_type.split(";")[0].strip().lower()
    if not clean_mime_type.startswith("image/"):
        clean_mime_type = "image/jpeg"

    client = get_genai_client()

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=clean_mime_type,
            )
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=768,
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini API returned an empty response for image embedding.")

    return list(response.embeddings[0].values)


def generate_image_embedding(image_path: Union[str, Path]) -> list[float]:
    """
    Generate Gemini multimodal vector embedding for a single image file on disk.

    :param image_path: File system path to the image file.
    :return: 768-dimensional float embedding vector.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type = SUPPORTED_MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
    image_bytes = path.read_bytes()

    return generate_image_embedding_from_bytes(image_bytes, mime_type=mime_type)



