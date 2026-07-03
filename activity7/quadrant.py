import os
import re
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

COLLECTION_NAME = "activity7_memory"
VECTOR_SIZE = 8

SOURCE_TEXT = """
Qdrant is a vector database designed for similarity search and retrieval.

Chunking is the process of dividing a document into smaller pieces before embedding.
If a chunk is too large, it may contain too much unrelated information.
If a chunk is too small, it may lose the surrounding context needed for the answer.

Overlap helps preserve meaning when a sentence or idea crosses a boundary.
Metadata such as source, section, and strategy makes debugging easier.
"""

def fixed_size_chunk(text: str, chunk_size: int = 140, overlap: int = 30) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def paragraph_chunk(text: str) -> list[str]:
    regex_pattern = r"\n\s*\n"
    return [p.strip() for p in re.split(regex_pattern, text) if p.strip()]


def embed_text(text: str) -> list[float]:
    vocab = [
        "qdrant", "chunking", "embedding", "overlap",
        "metadata", "retrieval", "context", "vector"
    ]

    lowered = text.lower()
    vector = [float(lowered.count(word)) for word in vocab]

    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return [0.0 for _ in vector]

    return [v / norm for v in vector]


def store_chunks(client, collection_name: str, chunks: list[str], strategy: str):
    points = []

    for index, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=index + (0 if strategy == "fixed_size" else 1000),
                vector=embed_text(chunk),
                payload={
                    "text": chunk,
                    "strategy": strategy,
                    "chunk_index": index,
                    "source": "sample_doc",
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def retrieve_best_match(client, collection_name: str, query_vector: list[float]):
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return result.points[0] if result.points else None


def main():
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    fixed_chunks = fixed_size_chunk(SOURCE_TEXT)
    paragraph_chunks = paragraph_chunk(SOURCE_TEXT)

    print("--- Fixed-size Chunks ---")
    for i, c in enumerate(fixed_chunks):
        print(i, c)

    print("\n--- Paragraph Chunks ---")
    for i, c in enumerate(paragraph_chunks):
        print(i, c)

    store_chunks(client, COLLECTION_NAME, fixed_chunks, "fixed_size")
    store_chunks(client, COLLECTION_NAME, paragraph_chunks, "paragraph")

    query_text = "Why does overlap help when chunking a document?"
    query_vector = embed_text(query_text)

    match = retrieve_best_match(client, COLLECTION_NAME, query_vector)

    print(f"\nQuery: {query_text}")

    if match:
        payload = match.payload
        print(f"\nBest Match Strategy Found: {payload['strategy'].upper()}")
        print(f"Chunk Index: {payload['chunk_index']}")
        print(f"Text:\n{payload['text']}")


if __name__ == "__main__":
    main()