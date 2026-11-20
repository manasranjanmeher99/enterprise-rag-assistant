"""Index documents into the vector store.

Usage:
    python ingest.py                     # index everything in data/docs
    python ingest.py a.pdf b.docx        # index specific files
    python ingest.py --reset             # wipe the index first, then index data/docs
"""
import argparse
import hashlib
from pathlib import Path

import config
from loaders import Chunk, chunk_segments, load_file
from store import embed_documents, get_collection

BATCH_SIZE = 128


def _chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.source}|{chunk.location}|{chunk.index}|{chunk.text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def index_file(path: Path) -> int:
    """(Re)index one file. Returns the number of chunks stored."""
    path = Path(path)
    chunks = chunk_segments(load_file(path))
    collection = get_collection()
    collection.delete(where={"source": path.name})  # replace any older version of this file
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c.text for c in batch]
        collection.upsert(
            ids=[_chunk_id(c) for c in batch],
            documents=texts,
            embeddings=embed_documents(texts),
            metadatas=[
                {"source": c.source, "location": c.location, "chunk_index": c.index}
                for c in batch
            ],
        )
    return len(chunks)


def clear_index() -> None:
    collection = get_collection()
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)


def find_documents(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("paths", nargs="*", help="files to index (default: everything in data/docs)")
    parser.add_argument("--reset", action="store_true", help="clear the index before indexing")
    args = parser.parse_args()

    if args.reset:
        clear_index()
        print("Index cleared.")

    files = [Path(p) for p in args.paths] if args.paths else find_documents(config.DOCS_DIR)
    if not files:
        print(f"No documents found. Put files in {config.DOCS_DIR} or pass paths.")
        return

    total = 0
    for path in files:
        try:
            count = index_file(path)
        except Exception as exc:  # keep going if one file is broken
            print(f"  ERROR  {path.name}: {exc}")
            continue
        if count == 0:
            print(f"  WARN   {path.name}: no text extracted (scanned PDF? it needs OCR first)")
        else:
            print(f"  OK     {path.name}: {count} chunks")
        total += count
    print(f"Done. {total} chunks indexed; {get_collection().count()} in the store.")


if __name__ == "__main__":
    main()
