"""Mengubah teks menjadi vektor (embedding) dengan model lokal lewat fastembed."""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TextEmbedding(EMBED_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Kembalikan matriks (jumlah teks x 384). Vektor dinormalkan agar panjangnya 1."""
    vectors = np.array(list(_model().embed(texts, batch_size=32)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
