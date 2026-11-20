"""Central configuration. Override any value with environment variables or a .env file."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma"))
COLLECTION_NAME = "company_knowledge"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Embeddings. bge-small accepts up to 512 tokens per chunk. If you switch to a model
# that doesn't use a query instruction (e.g. all-MiniLM-L6-v2), set QUERY_PREFIX to "".
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)

# Chunking is measured in words (about 1.3 tokens per word), so 350 words stays well
# under the embedding model's 512-token limit.
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "350"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
# Chunks scoring below this cosine similarity are dropped. If nothing is left, the
# assistant says it couldn't find an answer without calling the LLM. Tune with evaluate.py.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.40"))

# Generation
# Any chat model available on your Groq account (see the model list in the Groq console).
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "1024"))
HISTORY_MESSAGES = int(os.getenv("HISTORY_MESSAGES", "6"))
