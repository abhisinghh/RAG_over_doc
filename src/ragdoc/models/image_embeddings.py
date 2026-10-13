"""Joint text-image embeddings (CLIP / SigLIP via sentence-transformers).

Figures are embedded as images; queries are embedded as text into the same space, so a
question like "architecture diagram with a load balancer" can find a figure even when its
VLM description misses the wording. Requires ``pip install ragdoc[local]``.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np


class ClipEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        from PIL import Image

        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        vectors = self._model.encode(pil, batch_size=16, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
