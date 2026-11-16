"""Komponen LlamaIndex: model embedding, LLM, dan vector store di PostgreSQL + pgvector."""
from __future__ import annotations

import warnings

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore

from .settings import Settings


def build_embed_model(settings: Settings) -> FastEmbedEmbedding:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FastEmbedEmbedding(model_name=settings.embed_model)


def build_llm(settings: Settings) -> OpenAILike:
    return OpenAILike(
        model=settings.llm_model,
        api_base=settings.llm_base_url,
        api_key=settings.openrouter_api_key,
        is_chat_model=True,
        temperature=0.2,
        max_tokens=1200,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_vector_store(settings: Settings) -> PGVectorStore:
    return PGVectorStore.from_params(
        connection_string=settings.sync_url,
        async_connection_string=settings.async_url,
        table_name=settings.table_name,
        embed_dim=settings.embed_dim,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def build_index(settings: Settings) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        build_vector_store(settings), embed_model=build_embed_model(settings)
    )
