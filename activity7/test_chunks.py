"""
test_chunks.py — Test and visualise chunking strategies without needing Qdrant.
Usage: python test_chunks.py
"""

import re

SOURCE_TEXT = """
Qdrant is a vector database designed for similarity search and retrieval.

Chunking is the process of dividing a document into smaller pieces before embedding.
If a chunk is too large, it may contain too much unrelated information.
If a chunk is too small, it may lose the surrounding context needed for the answer.

Overlap helps preserve meaning when a sentence or idea crosses a boundary.
Metadata such as source, section, and strategy makes debugging easier.
"""


def fixed_size_chunk(text: str, chunk_size: int = 140, overlap: int = 30) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def paragraph_chunk(text: str) -> list:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def embed_text(text: str) -> list:
    vocab = ["qdrant", "chunking", "embedding", "overlap",
             "metadata", "retrieval", "context", "vector"]
    lowered = text.lower()
    vector = [float(lowered.count(w)) for w in vocab]
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector] if norm else [0.0] * len(vocab)


def cosine_similarity(a: list, b: list) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def print_section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def main():
    print_section("Fixed-Size Chunks (size=140, overlap=30)")
    fixed = fixed_size_chunk(SOURCE_TEXT)
    for i, c in enumerate(fixed):
        print(f"\n  Chunk {i}  ({len(c)} chars)")
        print(f"  {c}")

    print_section("Paragraph Chunks")
    para = paragraph_chunk(SOURCE_TEXT)
    for i, c in enumerate(para):
        print(f"\n  Chunk {i}")
        print(f"  {c}")

    # Similarity test
    query = "Why does overlap help when chunking a document?"
    q_vec = embed_text(query)

    print_section(f"Query: \"{query}\"")

    all_chunks = [("fixed_size", i, c) for i, c in enumerate(fixed)] + \
                 [("paragraph",  i, c) for i, c in enumerate(para)]

    scored = [(cosine_similarity(q_vec, embed_text(c)), s, i, c)
              for s, i, c in all_chunks]
    scored.sort(reverse=True)

    print(f"\n  Top 3 matches:\n")
    for rank, (score, strategy, idx, text) in enumerate(scored[:3], 1):
        print(f"  #{rank}  [{strategy.upper()}] chunk {idx}  score={score:.4f}")
        print(f"       {text[:120]}{'...' if len(text) > 120 else ''}\n")


if __name__ == "__main__":
    main()
