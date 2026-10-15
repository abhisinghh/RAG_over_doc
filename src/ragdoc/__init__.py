"""ragdoc: agentic retrieval-augmented generation over documents."""

from ragdoc.config import Settings, load_settings
from ragdoc.types import AnswerResult, Chunk, Citation

__all__ = ["AnswerResult", "Chunk", "Citation", "Settings", "load_settings"]
__version__ = "0.2.0"
