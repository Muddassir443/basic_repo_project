from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DOCS_DIR = Path("docs")
INDEX_DIR = Path("index")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-sonnet-4-5")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


def load_documents(folder: Path) -> dict[str, str]:
    """Read every .txt and .md file in `folder` into {filename: text}."""
    docs = {}
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs[path.name] = path.read_text(encoding="utf-8")
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows, preferring to cut at whitespace."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(docs: dict[str, str]) -> list[Chunk]:
    chunks = []
    for name, text in docs.items():
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(source=name, chunk_id=i, text=piece))
    return chunks


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]):
        self.vectors = vectors
        self.chunks = chunks

    def search(self, query_vec: np.ndarray, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ query_vec
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: Path = INDEX_DIR) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: Path = INDEX_DIR) -> "VectorStore":
        vectors = np.load(folder / "vectors.npy")
        data = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        return cls(vectors, [Chunk(**d) for d in data])


PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the provided documents."
Cite sources in square brackets like [file.md#2].

Context:
{context}

Question: {question}"""


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(f"[{c.source}#{c.chunk_id}] {c.text}" for c, _ in hits)
    return PROMPT.format(context=context, question=question)


def generate(question: str, hits: list[tuple[Chunk, float]]) -> str | None:
    """Call the LLM. Returns None if no API key is set (retrieval-only mode)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    return msg.content[0].text


def cmd_ingest() -> None:
    docs = load_documents(DOCS_DIR)
    if not docs:
        sys.exit(f"No .txt or .md files found in {DOCS_DIR}/")
    chunks = build_chunks(docs)
    vectors = Embedder().encode([c.text for c in chunks])
    VectorStore(vectors, chunks).save()
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents -> {INDEX_DIR}/")


def answer(question: str, embedder: Embedder, store: VectorStore) -> None:
    hits = store.search(embedder.encode([question])[0])
    print("\nRetrieved context:")
    for c, score in hits:
        print(f"  [{c.source}#{c.chunk_id}] score={score:.3f}  {c.text[:90]}...")
    reply = generate(question, hits)
    if reply is None:
        print("\n(ANTHROPIC_API_KEY not set: showing retrieval only, no generated answer.)")
    else:
        print(f"\nAnswer:\n{reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic RAG demo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="build the vector index from ./docs")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    sub.add_parser("chat", help="interactive question loop")
    args = parser.parse_args()

    if args.cmd == "ingest":
        return cmd_ingest()

    if not (INDEX_DIR / "vectors.npy").exists():
        sys.exit("No index found. Run: python rag.py ingest")

    embedder = Embedder()
    store = VectorStore.load()
    if args.cmd == "ask":
        answer(args.question, embedder, store)
    else:
        while (q := input("\nQuestion (blank to quit): ").strip()):
            answer(q, embedder, store)


if __name__ == "__main":
    main()
