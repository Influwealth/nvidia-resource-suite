"""
nvidia-resource-suite.nim

NVIDIA NIM (Inference Microservices) client library.
Supports: LLM, Embedding, Vision, Audio, RAG, Guardrails.

All calls go to the NIM API at integrate.api.nvidia.com/v1
using an OpenAI-compatible interface.
"""

from .client import NIMClient
from .llm import LLMClient
from .embedding import EmbeddingClient
from .vision import VisionClient
from .audio import AudioClient
from .rag import RAGPipeline
from .guardrails import GuardrailsClient

__all__ = [
    "NIMClient",
    "LLMClient",
    "EmbeddingClient",
    "VisionClient",
    "AudioClient",
    "RAGPipeline",
    "GuardrailsClient",
]
