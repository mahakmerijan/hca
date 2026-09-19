"""
Shared Gemini client factory.

Priority:
  1. GOOGLE_API_KEY env var  → google.genai direct API (no GCP creds needed, works on Render)
  2. Otherwise               → Vertex AI (needs GOOGLE_APPLICATION_CREDENTIALS / ADC)
"""

import os

try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    _GENAI_AVAILABLE = False


def make_genai_client():
    """Return a google-genai Client using the best available credentials."""
    if not _GENAI_AVAILABLE:
        return None

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        return genai.Client(api_key=api_key)

    return genai.Client(
        vertexai=True,
        project=os.getenv("VERTEX_PROJECT", "ai-ml-integrations"),
        location=os.getenv("VERTEX_LOCATION", "us-central1"),
    )
