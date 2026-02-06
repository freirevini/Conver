"""LLM service package - Vertex AI integration for gemini-2.5-pro."""

from app.services.llm.vertex_client import (
    VertexAIClient,
    VertexConfig,
    get_vertex_client,
)

__all__ = [
    "VertexAIClient",
    "VertexConfig",
    "get_vertex_client",
]
