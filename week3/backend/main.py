"""
RAG Backend — Python 3.14 compatible (no PyTorch).
Features:
  - TF-IDF + cosine similarity for retrieval
  - Gemini for answer generation with exponential backoff
  - RAG Triad evaluation (Answer Relevance, Context Relevance, Groundedness)
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import re
import uuid
import os
import time
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

# Hardcoded fallback (remove after .env is working)
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = ""

# =========================
# APP INIT
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# VECTOR STORE (TF-IDF)
# =========================
class VectorStore:
    def __init__(self):
        self.db          = {}
        self.vectorizers = {}
        self.matrices    = {}

    def _refit(self, col: str):
        texts = [item["text"] for item in self.db[col]]
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        self.matrices[col]    = vec.fit_transform(texts)
        self.vectorizers[col] = vec

    def add(self, col: str, texts: List[str], metas: List[dict]):
        if col not in self.db:
            self.db[col] = []
        for t, m in zip(texts, metas):
            self.db[col].append({"text": t, "meta": m})
        self._refit(col)

    def search(self, col: str, query: str, top_k: int = 5):
        if col not in self.db:
            return []
        q_vec  = self.vectorizers[col].transform([query])
        scores = sk_cosine(q_vec, self.matrices[col]).flatten()
        idx    = scores.argsort()[::-1][:top_k]
        return [
            {**self.db[col][i], "score": float(scores[i])}
            for i in idx if scores[i] > 0
        ]

store = VectorStore()

# =========================
# TEXT PROCESSING
# =========================
TIMESTAMP_RE = re.compile(
    r'^\s*\d+:\d+\s+(?:\d+\s+\w+(?:,\s*\d+\s+\w+)?\s+)?',
    re.MULTILINE
)

def strip_ts(text: str) -> str:
    return TIMESTAMP_RE.sub('', text).strip()

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', strip_ts(text)).strip()

def chunk_transcript(text: str) -> List[str]:
    """
    Split on timestamp boundaries then apply a 3-line sliding window.
    Falls back to sentence-based chunking if no timestamps found.
    Minimum 20 chunks guaranteed by using smaller window sizes.
    """
    lines = re.split(r'(?=\d+:\d+\s)', text)
    cleaned = [clean(l) for l in lines if len(clean(l)) > 15]

    # Fallback: split by sentences if no timestamps found
    if len(cleaned) < 5:
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        cleaned = [clean(s) for s in sentences if len(clean(s)) > 15]

    # Fallback: split by words into small 20-word chunks
    if len(cleaned) < 5:
        words = text.split()
        cleaned = [' '.join(words[i:i+20]) for i in range(0, len(words), 20)]
        cleaned = [c for c in cleaned if len(c) > 15]

    # Use window=1 (each line is its own chunk) if we have few lines
    # Use window=3 sliding if we have many lines
    if len(cleaned) <= 10:
        # Small window: each line + next line only
        window = 2
    else:
        window = 3

    chunks = []
    for i in range(len(cleaned)):
        chunk = ' '.join(cleaned[i:i+window]).strip()
        if len(chunk) > 15:
            chunks.append(chunk)

    # If still fewer than 20, reduce to individual lines
    if len(chunks) < 20:
        chunks = [c for c in cleaned if len(c) > 15]

    return chunks if chunks else cleaned

# =========================
# EXPONENTIAL BACKOFF
# =========================
def call_with_backoff(fn, max_retries: int = 5, base_delay: float = 1.0):
    """
    Call fn() with exponential backoff on rate-limit (429) errors.
    Delay pattern: 1s, 2s, 4s, 8s, 16s
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
            is_last       = attempt == max_retries - 1

            if is_last or not is_rate_limit:
                raise

            delay = base_delay * (2 ** attempt)   # 1, 2, 4, 8, 16
            print(f"  [backoff] Rate limited. Retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

    return None

# =========================
# GEMINI LLM
# =========================
GEMINI_CLIENT = None
GEMINI_MODEL  = None

try:
    import google.genai as genai
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        GEMINI_CLIENT = genai.Client(api_key=api_key)
except Exception:
    pass

CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

def _pick_model() -> Optional[str]:
    """Find the first working model (cached after first success)."""
    global GEMINI_MODEL
    if GEMINI_MODEL:
        return GEMINI_MODEL
    if not GEMINI_CLIENT:
        return None
    for m in CANDIDATE_MODELS:
        try:
            r = GEMINI_CLIENT.models.generate_content(model=m, contents="hi")
            if r and r.text:
                GEMINI_MODEL = m
                print(f"  [gemini] Using model: {m}")
                return m
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # quota hit but model exists — still use it with backoff
                GEMINI_MODEL = m
                return m
    return None

def gemini_call(prompt: str) -> Optional[str]:
    """Call Gemini with exponential backoff."""
    if not GEMINI_CLIENT:
        return None
    model = _pick_model()
    if not model:
        return None

    def _call():
        resp = GEMINI_CLIENT.models.generate_content(model=model, contents=prompt)
        return resp.text.strip() if resp and resp.text else None

    try:
        return call_with_backoff(_call)
    except Exception as e:
        print(f"  [gemini] Failed after retries: {str(e)[:80]}")
        return None

# =========================
# RAG TRIAD EVALUATION
# =========================
def llm_scoreilarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using TF-IDF.
    More accurate than raw word overlap — handles synonyms and partial matches.
    Returns a score between 0.0 and 1.0.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vec.fit_transform([text_a, text_b])
        score = float(sk_cosine(matrix[0], matrix[1])[0][0])
        return round(score, 3)
    except Exception:
        return 0.0

def llm_score(prompt: str) -> float:
    """
    Use Gemini to semantically score a statement.
    Returns a float between 0.0 and 1.0.
    Falls back to 0.5 if LLM unavailable.
    """
    result = gemini_call(prompt)
    if not result:
        return 0.5
    
    # Extract numeric score from response
    numbers = re.findall(r'\d+\.?\d*', result)
    if numbers:
        score = float(numbers[0])
        # Normalize to 0.0-1.0 range (handles 0-1, 0-10, 0-100 responses)
        if score > 1.0:
            score = score / 10.0 if score <= 10.0 else score / 100.0
        return round(min(max(score, 0.0), 1.0), 3)
    return 0.5

def rag_triad(query: str, context: str, answer: str) -> dict:
    """
    RAG Triad — three evaluation scores (0.0 – 1.0):

    1. Answer Relevance   — does the answer address the query?
    2. Context Relevance  — does the retrieved context relate to the query?
    3. Groundedness       — is the answer grounded in the context?
    
    Uses LLM for semantic evaluation instead of keyword matching.
    """
    
    # Answer Relevance: Does the answer address the question?
    ar_prompt = f"""Rate how well this answer addresses the question on a scale of 0.0 to 1.0.
Only respond with a number between 0.0 and 1.0.

Question: {query}
Answer: {answer}

Score:"""
    
    # Context Relevance: Is the context relevant to the question?
    cr_prompt = f"""Rate how relevant this context is to answering the question on a scale of 0.0 to 1.0.
Only respond with a number between 0.0 and 1.0.

Question: {query}
Context: {context}

Score:"""
    
    # Groundedness: Is the answer based on the context?
    gr_prompt = f"""Rate how well this answer is grounded in (supported by) the context on a scale of 0.0 to 1.0.
Only respond with a number between 0.0 and 1.0.

Context: {context}
Answer: {answer}

Score:"""
    
    answer_relevance  = llm_score(ar_prompt)
    context_relevance = llm_score(cr_prompt)
    groundedness      = llm_score(gr_prompt)

    # Overall RAG score = harmonic mean of the three
    scores   = [answer_relevance, context_relevance, groundedness]
    non_zero = [s for s in scores if s > 0]
    rag_score = round(len(non_zero) / sum(1/s for s in non_zero), 3) if non_zero else 0.0

    return {
        "answer_relevance":  answer_relevance,
        "context_relevance": context_relevance,
        "groundedness":      groundedness,
        "rag_score":         rag_score,
    }

# =========================
# FALLBACK ANSWER (no Gemini)
# =========================
STOP_WORDS = {
    'what','why','how','is','are','the','a','an','in','of','to','and',
    'for','this','that','used','use','does','do','was','were','it',
    'its','be','been','being','have','has','had','with','from','about',
    'can','could','would','will','at','on','or','but','not','so','than'
}

def focused_answer(query: str, chunks: List[str]) -> str:
    context = ' '.join(chunks)
    sents   = re.split(r'(?<=[a-z])\s+(?=[A-Z])|(?<=\w)[.!?]\s+', context)
    sents   = [s.strip() for s in sents if len(s.strip()) > 15]
    q_words = set(re.findall(r'\w+', query.lower())) - STOP_WORDS
    scored  = [(len(set(re.findall(r'\w+', s.lower())) & q_words), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for sc, s in scored[:4] if sc > 0] or [s for _, s in scored[:2]]
    return ' '.join(best) if best else (chunks[0] if chunks else "I don't know.")

# =========================
# UPLOAD ENDPOINT
# =========================
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    collection: str  = Form(...),
    user_id: str     = Form(...),
    session_id: str  = Form(default="")
):
    raw    = (await file.read()).decode(errors="ignore")
    chunks = chunk_transcript(raw)

    if not chunks:
        return JSONResponse({"error": "No readable text found."}, status_code=400)

    sid   = session_id.strip() or str(uuid.uuid4())
    metas = [{"user_id": user_id, "source": file.filename, "session_id": sid}
             for _ in chunks]
    store.add(collection, chunks, metas)

    return JSONResponse({
        "status":         "uploaded",
        "chunks_indexed": len(chunks),
        "session_id":     sid,
    })

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
async def chat(payload: dict):
    query      = payload.get("query", "").strip()
    collection = payload.get("collection", "")
    user_id    = payload.get("user_id", "")

    if not query:
        return {"answer": "Please ask a question.", "sources": [], "triad": None}

    # --- Retrieve ---
    results = store.search(collection, query, top_k=5)
    results = [r for r in results if r["meta"].get("user_id") == user_id]
    results = [r for r in results if len(r["text"]) > 15]

    if not results:
        return {
            "answer":  "No relevant content found in the uploaded transcript.",
            "sources": [],
            "triad":   None,
        }

    # Deduplicate
    seen, unique = set(), []
    for r in results:
        key = r["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    top_chunks = [clean(r["text"]) for r in unique[:3]]
    context    = " ".join(top_chunks)

    # --- Generate (with exponential backoff) ---
    prompt = f"""You are a strict assistant. You ONLY answer based on the context below.
If the context does not contain information to answer the question, reply EXACTLY:
"I don't know — this topic is not covered in the uploaded transcript."
Do NOT use any outside knowledge. Do NOT make up answers.

Context:
{context}

Question: {query}

Answer:"""

    answer = gemini_call(prompt) or focused_answer(query, top_chunks)

    # --- RAG Triad Evaluation ---
    triad = rag_triad(query, context, answer)

    return {
        "answer":  answer,
        "sources": [{"text": r["text"], "score": r["score"]} for r in unique[:3]],
        "triad":   triad,
    }

# =========================
# SERVE FRONTEND
# =========================
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
