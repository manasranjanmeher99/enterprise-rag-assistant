"""Shared embedding model and vector store handles (created once per process)."""
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    import chromadb

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    vector = get_embedder().encode(config.QUERY_PREFIX + query, normalize_embeddings=True)
    return vector.tolist()
