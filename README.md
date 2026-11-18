# Basic RAG (Retrieval-Augmented Generation)

A small, readable RAG pipeline in Python: chunk documents, embed them with a local
sentence-transformer model, retrieve the top matches by cosine similarity, and have an LLM
answer **only** from the retrieved context with source citations.

## How it works

```
docs/*.md ──► chunk (500 chars, 100 overlap) ──► embed (all-MiniLM-L6-v2) ──► index/ (NumPy)
                                                                                  │
question ──► embed ──► cosine similarity, top-3 chunks ◄──────────────────────────┘
                                │
                                ▼
              prompt = context + question ──► LLM ──► grounded answer with [file#chunk] citations
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

```bash
python rag.py ingest
python rag.py ask "Why is chunk overlap useful?"
python rag.py chat
```

Without an API key the project runs in retrieval-only mode and prints the matching chunks.
To get generated answers, set `ANTHROPIC_API_KEY` (optionally `RAG_LLM_MODEL`).

Add your own `.txt` or `.md` files to `docs/`, then re-run `ingest`.

## Tests

```bash
pytest -q
```

## Possible next steps

- Swap the NumPy store for a vector database (Chroma, FAISS, or Azure Cosmos DB vector search).
- Add PDF loading and per-chunk metadata (page numbers).
- Add an evaluation script: a small question set with the expected source, measuring recall@k.
- Add a reranker or hybrid keyword + vector search.
- Wrap it in a FastAPI or Streamlit interface.
