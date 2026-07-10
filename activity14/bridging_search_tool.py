import os
import json
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "course_memory")

# Initialize clients lazily
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_qdrant_client = QdrantClient(url=QDRANT_URL)


def _expand_query_if_short(query: str) -> str:
    if not query or len(query.split()) >= 20:
        return query
    try:
        prompt = f"""Rewrite the following short query as a detailed, paragraph-style\nstatement suitable for similarity search. Include key terms and context.\n\nOriginal query: {query}\n\nExpanded passage:"""
        resp = _gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return resp.text.strip() if getattr(resp, "text", None) else query
    except Exception:
        return query


def _hybrid_rerank(query: str, dense_results: List[object], alpha: float = 0.7) -> str:
    query_terms = set(query.lower().split())
    scored = []
    for point in dense_results:
        try:
            dense_score = float(getattr(point, "score", 0.0))
            payload = getattr(point, "payload", {}) or point.payload
            text = payload.get("text_segment", "") or payload.get("text", "") or ""
            doc_terms = set(text.split())
            overlap = len(query_terms & doc_terms)
            tf_sum = sum(text.count(term) for term in query_terms)
            keyword_score = (overlap / len(query_terms)) * min(1.0, tf_sum / 10.0) if query_terms else 0.0
            hybrid = alpha * dense_score + (1 - alpha) * keyword_score
            scored.append((hybrid, text))
        except Exception:
            continue

    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def search_documents(query: str) -> str:
    """Qdrant-backed retrieval tool. Always returns a clean string."""
    if not query or not query.strip():
        return "No query provided."

    # Expand short queries to counter embedding asymmetry
    expanded = _expand_query_if_short(query)

    if not _gemini_client:
        return "Embedding client unavailable (no API key)."

    try:
        emb_resp = _gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=expanded,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = emb_resp.embeddings[0].values
    except Exception as e:
        return f"Embedding failed: {e}"

    try:
        results = _qdrant_client.query(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=10,
        )
    except Exception as e:
        return f"Qdrant connection error: {e}"

    candidates = [r.payload.get("text_segment", "") or r.payload.get("text", "") for r in results]
    if not candidates:
        return "No relevant information found in the knowledge base."

    # Hybrid re-rank and LLM re-rank could be combined; use hybrid first
    best = _hybrid_rerank(query, results, alpha=0.7)
    if best:
        return best

    # Fallback to first candidate
    return candidates[0] if candidates else "No text payload found." 
