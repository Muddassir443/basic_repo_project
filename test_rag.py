import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import Chunk, VectorStore, build_prompt, chunk_text


def test_chunks_respect_size_and_overlap():
    text = "word " * 400
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)


def test_search_returns_closest_vector_first():
    vectors = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=float)
    chunks = [Chunk("a.txt", 0, "A"), Chunk("b.txt", 0, "B"), Chunk("c.txt", 0, "C")]
    hits = VectorStore(vectors, chunks).search(np.array([1.0, 0.0]), k=2)
    assert hits[0][0].text == "A"
    assert hits[1][0].text == "C"


def test_prompt_contains_context_and_citation_tag():
    hits = [(Chunk("doc.md", 3, "RAG grounds answers."), 0.9)]
    prompt = build_prompt("What is RAG?", hits)
    assert "[doc.md#3] RAG grounds answers." in prompt
    assert "What is RAG?" in prompt
