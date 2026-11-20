"""Semantic retrieval over the vector store."""
from dataclasses import dataclass

import config
from store import embed_query, get_collection


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    location: str
    score: float  # cosine similarity, higher is better


def retrieval_query(question: str, history: list[dict]) -> str:
    """Short follow-ups ("and for contractors?") lack context, so prepend the previous question."""
    if history and len(question.split()) < 6:
        previous = [m["content"] for m in history if m["role"] == "user"]
        if previous:
            return f"{previous[-1]} {question}"
    return question


def retrieve(
    query: str,
    top_k: int = config.TOP_K,
    min_score: float = config.MIN_SIMILARITY,
    sources: list[str] | None = None,
) -> list[RetrievedChunk]:
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    kwargs = {}
    if sources:
        kwargs["where"] = {"source": {"$in": sources}}

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
        **kwargs,
    )

    chunks = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        score = 1.0 - distance
        if score >= min_score:
            chunks.append(
                RetrievedChunk(chunk_id, text, meta["source"], meta.get("location", ""), score)
            )
    return chunks
