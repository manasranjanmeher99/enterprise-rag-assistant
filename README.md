# Enterprise RAG Assistant — Company Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that lets users ask questions about company documents and receive answers grounded in retrieved passages, with numbered citations pointing to the source file and page or section.

> **Demo note:** The included `data/docs/leave_policy.md` is fictional sample content for development and testing. Do not treat it as a real company policy.

## Features

- Upload and index PDF, DOCX, TXT, and Markdown (`.md`) documents.
- Extract PDF page numbers and document section headings where available.
- Split extracted text into overlapping chunks.
- Generate embeddings with Sentence Transformers (`BAAI/bge-small-en-v1.5`).
- Store and search embeddings in a persistent ChromaDB collection using cosine similarity.
- Retrieve the most relevant passages and filter low-similarity results.
- Generate answers with the Groq API and require citations such as `[1]`.
- Show retrieved source passages and allow users to download the original indexed file.
- Keep short conversation history for follow-up questions.
- Run a small retrieval/answer evaluation set with `evaluate.py`.

## Tech stack

| Area | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM provider | Groq API |
| Default chat model | `openai/gpt-oss-120b` |
| Embeddings | Sentence Transformers — `BAAI/bge-small-en-v1.5` |
| Vector database | ChromaDB (persistent local collection) |
| UI | Streamlit |
| Document parsing | pypdf, python-docx |

**Vector database clarification:** This repository currently uses **ChromaDB**, not FAISS. The code in `store.py`, `retriever.py`, and `ingest.py` is implemented for ChromaDB. A FAISS migration would require code changes and is not part of this version.

## Repository structure

```text
enterprise-rag-assistant/
│
├── app.py                       # Streamlit chat UI and document upload
├── config.py                    # Central settings and environment variables
├── ingest.py                    # CLI document indexing / re-indexing
├── loaders.py                   # PDF, DOCX, TXT, MD extraction and chunking
├── store.py                     # Sentence Transformer embeddings + ChromaDB
├── retriever.py                 # Similarity search and source filtering
├── generator.py                 # Groq prompt, answer streaming, citation parsing
├── evaluate.py                  # Retrieval and optional answer evaluation
│
├── data/
│   └── docs/
│       └── leave_policy.md       # Fictional sample knowledge-base document
│
├── eval_questions.json          # Example evaluation questions and expected results
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment configuration (safe to commit)
├── .gitignore                   # Excludes secrets, local DB, and Python artifacts
└── README.md
```

At runtime, ChromaDB creates `data/chroma/`. This generated vector database is intentionally excluded from Git. The `.venv/` environment and `.env` secrets file are also excluded.

## Requirements

- Python 3.10 or newer.
- A Groq account and API key.
- Internet access on first run to install packages and download the embedding model.

## Setup — Windows PowerShell

Open PowerShell in the project directory:

```powershell
# Create and enter the project folder if you have not already
cd path\to\enterprise-rag-assistant

# Create a virtual environment
py -3.10 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, run this for the current terminal only:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Create your local environment file
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your actual Groq key:

```env
GROQ_API_KEY=your_actual_groq_api_key
LLM_MODEL=openai/gpt-oss-120b
```

Get a key from the [Groq Console](https://console.groq.com/keys). **Never commit `.env` or expose your API key.**

## Setup — macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Then add your Groq API key to `.env`.

## Add documents

Place supported files in `data/docs/`:

- PDF: `.pdf`
- Word: `.docx`
- Plain text: `.txt`
- Markdown: `.md`

You can also upload files from the Streamlit sidebar after starting the app. Only add documents that you are authorized to use. The included sample policy is fictional.

## Index documents

From the project root, with the virtual environment activated:

```powershell
# Index all supported documents in data/docs
python ingest.py

# Rebuild the index from scratch, then index all documents
python ingest.py --reset

# Index specific files
python ingest.py data\docs\leave_policy.md
```

The first indexing run downloads the embedding model if it is not already cached. The local vector database is stored in `data/chroma/`.

## Run the application

```powershell
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser. Use the sidebar to upload/index documents, filter search to selected documents, re-index, clear the index, or start a new conversation.

## Example questions

Using the included fictional leave-policy document:

1. How many days of annual leave do full-time employees get?
2. When do I need a doctor's note for sick leave?
3. How many days a week can I work from home?
4. How many unused annual leave days can be carried over?
5. What is the stock option vesting schedule? *(The sample document does not contain this information; the assistant should say it cannot find the answer.)*

The expected annual leave answer in the included Markdown sample is **24 days**. The chatbot should cite `leave_policy.md` and the `Annual Leave` section.

## Evaluation

Run retrieval checks without calling the Groq API:

```powershell
python evaluate.py
```

Run answer and citation checks as well (this makes Groq API calls):

```powershell
python evaluate.py --answers
```

Evaluation questions and expected sources/keywords are in `eval_questions.json`. Treat this as a starter test set; add representative questions from your own documents before relying on evaluation results.

## Configuration

Settings are defined in `config.py` and can be overridden in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required Groq API key |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model identifier available to your account |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence Transformers embedding model |
| `MIN_SIMILARITY` | `0.40` | Minimum cosine similarity for a retrieved chunk |
| `TOP_K` | `5` | Maximum number of chunks retrieved |
| `CHUNK_WORDS` | `350` | Approximate chunk size in words |
| `CHUNK_OVERLAP_WORDS` | `60` | Word overlap between chunks |
| `MAX_ANSWER_TOKENS` | `1024` | Maximum generated answer tokens |
| `HISTORY_MESSAGES` | `6` | Number of previous chat messages retained |

If you change the embedding model, rebuild the vector index with `python ingest.py --reset` so stored document vectors and query vectors use the same model.

## How the RAG pipeline works

1. **Load:** `loaders.py` extracts text from each supported file. PDF text is tracked per page; DOCX and Markdown content can retain section headings.
2. **Chunk:** Text is divided into chunks of approximately 350 words with 60 words of overlap.
3. **Embed:** Sentence Transformers converts each chunk into a vector representation.
4. **Store:** `ingest.py` writes vectors, text, and source metadata to persistent ChromaDB.
5. **Retrieve:** `retriever.py` embeds the user's question and searches for the top matching chunks. Results below the similarity threshold are discarded.
6. **Generate:** `generator.py` sends the question and numbered retrieved sources to Groq. The prompt instructs the model to answer only from those sources and cite factual statements.
7. **Display:** `app.py` streams the response and displays cited source passages with file and page/section metadata.

## Current limitations and security considerations

- **Not production-ready access control:** The current app does not implement per-user or per-document authorization. Anyone using the same running app can potentially search all indexed documents. Do not deploy it with confidential company data until access controls are implemented and tested.
- **External LLM processing:** Retrieved text is sent to Groq to generate answers. Review your organization's data-handling requirements and Groq's applicable terms before using private documents.
- **Local vector storage:** ChromaDB is stored on the machine running the app. Protect the host and its files.
- **Scanned PDFs:** `pypdf` extracts embedded text but does not perform OCR. Scanned/image-only PDFs may yield no text and need OCR preprocessing.
- **Citation scope:** Citations refer to retrieved source chunks and their available page/section metadata. They are not a guarantee that every generated claim is correct; verify important answers against the original documents.
- **Duplicate filenames:** The current indexing logic identifies documents by filename. Files with the same name in different folders can overwrite each other's indexed entries.
- **Similarity threshold:** `MIN_SIMILARITY` is a configurable starting value, not a calibrated confidence score. Tune it using representative evaluation questions.
- **Sample data:** The included `leave_policy.md` is fictional demo content. Replace it with documents you are permitted to process.

## Possible future improvements

- Migrate ChromaDB to FAISS if a FAISS-based local index is required.
- Add per-user authentication and document-level access filters.
- Add OCR for scanned PDFs.
- Add hybrid retrieval (keyword + vector search) and a reranker.
- Add automated ingestion and document versioning.
- Improve evaluation with retrieval metrics (Recall@k / MRR) and answer-groundedness checks.

