import math
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def generate_text_embedding(text: str) -> list[float]:
    """
    Generate Gemini embedding vector (768-dim) for a text string.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    client = get_genai_client()

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text.strip(),
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    if not result.embeddings:
        raise RuntimeError("Gemini API returned an empty response for text embedding.")

    return list(result.embeddings[0].values)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Compute cosine similarity score between two float vectors.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)