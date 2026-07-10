"""Improved Week 3 RAG backend for Activity 13."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ.setdefault("GOOGLE_API_KEY", "")

app = FastAPI(title="Activity 13 RAG Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from activity14.project_react_loop import react_loop
except Exception:
    try:
        from project_react_loop import react_loop
    except Exception:
        react_loop = None

@app.post('/qa')
async def qa(payload: dict):
    query = payload.get('query', '').strip()
    if not query:
        return JSONResponse({'error': 'Please provide a query.'}, status_code=400)

    if react_loop is None:
        return JSONResponse({'error': 'ReAct loop unavailable.'}, status_code=500)

    transcript = react_loop(query, system_prompt='You are a helpful assistant with tools.')
    answers = [e.get('content') for e in transcript if e.get('phase') == 'ANSWER']
    answer = answers[-1] if answers else ''
    return {'answer': answer, 'transcript': transcript}


class VectorStore:
    def __init__(self) -> None:
        self.db: dict[str, list[dict]] = {}

    def add(self, col: str, texts: List[str], metas: List[dict]) -> None:
        if col not in self.db:
            self.db[col] = []
        for text, meta in zip(texts, metas):
            self.db[col].append({"text": text, "meta": meta})

    def search(self, col: str, query: str, top_k: int = 5) -> List[dict]:
        if col not in self.db:
            return []

        query_terms = {
            term for term in re.findall(r"\w+", query.lower()) if term not in STOP_WORDS
        }
        scored = []
        for item in self.db[col]:
            text = item["text"]
            text_terms = {
                term for term in re.findall(r"\w+", text.lower()) if term not in STOP_WORDS
            }
            overlap = len(query_terms & text_terms)
            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**item, "score": float(score)} for score, item in scored[:top_k]]


store = VectorStore()


TIMESTAMP_RE = re.compile(r"^\s*\d+:\d+\s+(?:\d+\s+\w+(?:,\s*\d+\s+\w+)?\s+)?", re.MULTILINE)


def strip_ts(text: str) -> str:
    return TIMESTAMP_RE.sub("", text).strip()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ts(text)).strip()


def chunk_transcript(text: str) -> List[str]:
    lines = re.split(r"(?=\d+:\d+\s)", text)
    cleaned = [clean(line) for line in lines if len(clean(line)) > 25]

    if len(cleaned) < 5:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        cleaned = [clean(sentence) for sentence in sentences if len(clean(sentence)) > 25]

    if len(cleaned) < 5:
        words = text.split()
        cleaned = [" ".join(words[i : i + 25]) for i in range(0, len(words), 25)]
        cleaned = [c for c in cleaned if len(c) > 25]

    if not cleaned:
        return [clean(text)[:400]] if clean(text) else []

    window = 2 if len(cleaned) <= 12 else 3
    chunks = []
    for i in range(0, len(cleaned), window):
        chunk = " ".join(cleaned[i : i + window]).strip()
        if len(chunk) > 30:
            chunks.append(chunk)

    if len(chunks) < 8:
        chunks = [c for c in cleaned if len(c) > 30]

    return chunks if chunks else cleaned


GEMINI_CLIENT = None
try:
    import google.genai as genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        GEMINI_CLIENT = genai.Client(api_key=api_key)
except Exception:
    GEMINI_CLIENT = None


def call_with_backoff(fn, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1 or "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)
    return None


def gemini_call(prompt: str) -> Optional[str]:
    if not GEMINI_CLIENT:
        return None

    def _call() -> Optional[str]:
        resp = GEMINI_CLIENT.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return getattr(resp, "text", None).strip() if getattr(resp, "text", None) else None

    try:
        return call_with_backoff(_call)
    except Exception:
        return None


STOP_WORDS = {
    "what", "why", "how", "is", "are", "the", "a", "an", "in", "of", "to", "and", "for", "this", "that",
    "used", "use", "does", "do", "was", "were", "it", "its", "be", "been", "being", "have", "has", "had",
    "with", "from", "about", "can", "could", "would", "will", "at", "on", "or", "but", "not", "so", "than"
}


def is_transcript_meta_question(query: str) -> bool:
    q = query.lower()
    meta_phrases = (
        "this transcript",
        "this upload",
        "this file",
        "this document",
        "this content",
        "what is this",
        "tell me about",
        "summarize",
        "summary",
        "overview",
        "what topics",
        "what is covered",
        "what is discussed",
        "about the uploaded",
        "about this",
    )
    return any(phrase in q for phrase in meta_phrases)


def focused_answer(query: str, chunks: List[str]) -> str:
    context = " ".join(chunks)
    full_text = clean(context)
    sents = re.split(r"(?<=[.!?])\s+|\n+", full_text)
    sents = [s.strip() for s in sents if len(s.strip()) > 25]
    if not sents:
        sents = [full_text[:400]] if full_text else []

    q_words = set(re.findall(r"\w+", query.lower())) - STOP_WORDS
    scored = []
    for s in sents:
        s_words = set(re.findall(r"\w+", s.lower())) - STOP_WORDS
        overlap = len(s_words & q_words)
        bonus = 1 if any(keyword in s.lower() for keyword in ["primitive", "int", "char", "double", "boolean", "method", "class", "object"]) else 0
        if overlap > 0 or bonus > 0:
            scored.append((overlap + bonus, s))

    if not scored:
        return chunks[0] if chunks else "I don't know."

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def llm_score(prompt: str) -> float:
    result = gemini_call(prompt)
    if not result:
        return 0.5
    numbers = re.findall(r"\d+\.?\d*", result)
    if numbers:
        score = float(numbers[0])
        if score > 1.0:
            score = score / 10.0 if score <= 10.0 else score / 100.0
        return round(min(max(score, 0.0), 1.0), 3)
    return 0.5


def rag_triad(query: str, context: str, answer: str) -> dict:
    ar_prompt = f"""Rate how well this answer addresses the question on a scale of 0.0 to 1.0. Only respond with a number between 0.0 and 1.0.\n\nQuestion: {query}\nAnswer: {answer}\n\nScore:"""
    cr_prompt = f"""Rate how relevant this context is to answering the question on a scale of 0.0 to 1.0. Only respond with a number between 0.0 and 1.0.\n\nQuestion: {query}\nContext: {context}\n\nScore:"""
    gr_prompt = f"""Rate how well this answer is grounded in (supported by) the context on a scale of 0.0 to 1.0. Only respond with a number between 0.0 and 1.0.\n\nContext: {context}\nAnswer: {answer}\n\nScore:"""

    answer_relevance = llm_score(ar_prompt)
    context_relevance = llm_score(cr_prompt)
    groundedness = llm_score(gr_prompt)

    scores = [answer_relevance, context_relevance, groundedness]
    non_zero = [s for s in scores if s > 0]
    rag_score = round(len(non_zero) / sum(1 / s for s in non_zero), 3) if non_zero else 0.0

    return {
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "groundedness": groundedness,
        "rag_score": rag_score,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...), collection: str = Form(...), user_id: str = Form(...)):
    raw = (await file.read()).decode(errors="ignore")
    chunks = chunk_transcript(raw)

    if not chunks:
        return JSONResponse({"error": "No readable text found."}, status_code=400)

    metas = [{"user_id": user_id, "source": file.filename, "collection": collection} for _ in chunks]
    store.add(collection, chunks, metas)

    return JSONResponse({"status": "uploaded", "chunks_indexed": len(chunks)})


@app.post("/chat")
async def chat(payload: dict):
    query = payload.get("query", "").strip()
    collection = payload.get("collection", "")
    user_id = payload.get("user_id", "")

    if not query:
        return {"answer": "Please ask a question.", "sources": [], "triad": None}

    results = store.search(collection, query, top_k=5)
    results = [r for r in results if r["meta"].get("user_id") == user_id]
    results = [r for r in results if len(r["text"]) > 15]

    user_chunks = [
        clean(item["text"])
        for item in store.db.get(collection, [])
        if item["meta"].get("user_id") == user_id and len(clean(item["text"])) > 15
    ]
    topic_words = {word for word in re.findall(r"\w+", query.lower()) if word not in STOP_WORDS}

    if not results:
        if is_transcript_meta_question(query) and user_chunks:
            top_chunks = user_chunks[:3]
        else:
            return {"answer": "I don't know — this topic is not covered in the uploaded transcript.", "sources": [], "triad": None}
    else:
        if not topic_words:
            return {"answer": "I don't know — this topic is not covered in the uploaded transcript.", "sources": [], "triad": None}

        seen = set()
        unique = []
        for r in results:
            key = r["text"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        top_chunks = [clean(r["text"]) for r in unique[:3]]

    context = " ".join(top_chunks)

    prompt = f"""You are a strict assistant. You ONLY answer based on the context below. If the context does not contain information to answer the question, reply EXACTLY: \"I don't know — this topic is not covered in the uploaded transcript.\" Do NOT use any outside knowledge. Do NOT make up answers.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    answer = gemini_call(prompt) or focused_answer(query, top_chunks)
    if answer and (answer.lower().startswith("i don't know") or "not covered" in answer.lower()):
        answer = "I don't know — this topic is not covered in the uploaded transcript."

    if answer != "I don't know — this topic is not covered in the uploaded transcript.":
        overlap = len(topic_words & {word for chunk in top_chunks for word in re.findall(r"\w+", chunk.lower()) if word not in STOP_WORDS})
        if overlap == 0 and not is_transcript_meta_question(query):
            answer = "I don't know — this topic is not covered in the uploaded transcript."
        else:
            answer = re.sub(r"\s+", " ", answer).strip()
            if answer.count(".") == 0 and len(answer) > 120:
                answer = answer[:300].strip() + "..."

    triad = rag_triad(query, context, answer)

    return {
        "answer": answer,
        "sources": [{"text": r["text"], "score": r["score"]} for r in unique[:3]] if results else [{"text": chunk, "score": 0.0} for chunk in top_chunks],
        "triad": triad,
    }


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
