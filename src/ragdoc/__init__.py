"""ragdoc: agentic multimodal RAG with GraphRAG over documents, media, code and services."""

from ragdoc.config import Settings, load_settings
from ragdoc.types import AnswerResult, Chunk, Citation

__all__ = ["AnswerResult", "Chunk", "Citation", "Settings", "load_settings"]
__version__ = "0.3.0"
