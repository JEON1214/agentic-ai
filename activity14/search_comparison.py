import os
from activity14.bridging_search_tool import search_documents
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "course_memory")

client = QdrantClient(url=QDRANT_URL)

TEST_QUERIES = [
    {
        "query": "What distance metric does Qdrant use?",
        "expected_chunk": "Cosine",
        "topics": ["qdrant", "setup"],
    },
    {
        "query": "What is the travel budget for flights?",
        "expected_chunk": "$2000",
        "topics": ["budget", "travel"],
    },
    {
        "query": "What is ReAct?",
        "expected_chunk": "ReAct stands for Reasoning",
        "topics": ["react"],
    },
]


def original_search_documents(query: str) -> str:
    """Baseline: use Qdrant's simple text query endpoint (no expansion / rerank)."""
    try:
        resp = client.query(collection_name=COLLECTION_NAME, query_text=query, limit=1)
        if not resp:
            return ""
        first = resp[0]
        # resp items may be QueryResponse with .payload
        text = first.payload.get("text_segment") or first.payload.get("text") or ""
        return text
    except Exception as e:
        return f"Baseline search error: {e}"


def compare_search(queries):
    for item in queries:
        q = item["query"]
        baseline = original_search_documents(q)
        improved = search_documents(q)
        expected = item["expected_chunk"].lower()
        baseline_hit = expected in baseline.lower() if baseline else False
        improved_hit = expected in improved.lower() if improved else False

        print(f"Q: {q}")
        print(f"  Baseline hit: {baseline_hit} | Improved hit: {improved_hit}")
        print(f"  Baseline chunk: {baseline[:120]}" + ("..." if len(baseline)>120 else ""))
        print(f"  Improved chunk: {improved[:120]}" + ("..." if len(improved)>120 else ""))
        print()


if __name__ == '__main__':
    compare_search(TEST_QUERIES)
