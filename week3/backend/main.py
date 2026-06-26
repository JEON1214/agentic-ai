"""
RAG Backend — Python 3.14 compatible (no PyTorch).
Uses TF-IDF + cosine similarity for retrieval, Gemini for answer generation.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import re
import uuid
import os
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

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
        scores = cosine_similarity(q_vec, self.matrices[col]).flatten()
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
    """Remove timestamp prefix from a single line."""
    return TIMESTAMP_RE.sub('', text).strip()

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', strip_ts(text)).strip()

def chunk_transcript(text: str) -> List[str]:
    """
    Split transcript into one chunk per timestamp line.
    Then group every N lines into a small window so TF-IDF
    has enough vocabulary variance to distinguish questions.
    Window size = 3 lines, step = 1 (sliding window).
    do not answer outside the scope of the sent transcripts or what is in your memory
    """
    # Split on timestamp boundaries
    lines = re.split(r'(?=\d+:\d+\s)', text)
    cleaned = []
    for line in lines:
        c = clean(line)
        if len(c) > 15:          # skip very short fragments
            cleaned.append(c)

    if not cleaned:
        # fallback: split by words into 40-word chunks
        words = text.split()
        cleaned = [' '.join(words[i:i+40]) for i in range(0, len(words), 40)]
        cleaned = [c for c in cleaned if len(c) > 20]

    # Sliding window: group 3 consecutive lines per chunk
    chunks = []
    window = 3
    for i in range(len(cleaned)):
        group = cleaned[i : i + window]
        chunk = ' '.join(group).strip()
        if len(chunk) > 20:
            chunks.append(chunk)

    return chunks if chunks else cleaned

# =========================
# GEMINI LLM
# =========================
GEMINI_CLIENT = None
try:
    import google.genai as genai
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        GEMINI_CLIENT = genai.Client(api_key=api_key)
except Exception:
    pass

def gemini_answer(query: str, context: str) -> str:
    """Use Gemini to answer the query based on context."""
    if not GEMINI_CLIENT:
        return None
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context below.
Be concise and direct. Do not repeat the context verbatim.

Context:
{context}

Question: {query}

Answer:"""
    for model in ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        try:
            resp = GEMINI_CLIENT.models.generate_content(model=model, contents=prompt)
            if resp and resp.text:
                return resp.text.strip()
        except Exception:
            continue
    return None

# =========================
# FOCUSED ANSWER BUILDER (fallback)
# =========================
STOP = {
    'what','why','how','is','are','the','a','an','in','of','to','and',
    'for','this','that','used','use','does','do','was','were','it',
    'its','be','been','being','have','has','had','with','from','about',
    'can','could','would','will','at','on','or','but','not','so','than'
}

def focused_answer(query: str, chunks: List[str]) -> str:
    """Pick the single most relevant chunk, then extract best sentences."""
    # Use the top chunk (already ranked by TF-IDF)
    context = ' '.join(chunks)

    # Split into sentences on common boundaries
    sents = re.split(r'(?<=[a-z])\s+(?=[A-Z])|(?<=\w)[.!?]\s+', context)
    sents = [s.strip() for s in sents if len(s.strip()) > 15]

    q_words = set(re.findall(r'\w+', query.lower())) - STOP

    scored = []
    for s in sents:
        w = set(re.findall(r'\w+', s.lower()))
        scored.append((len(q_words & w), s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top 3 sentences with at least 1 keyword match
    best = [s for sc, s in scored[:4] if sc > 0]

    if not best:
        best = [s for _, s in scored[:2]]  # fallback

    return ' '.join(best) if best else chunks[0]

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

    sid = session_id.strip() or str(uuid.uuid4())
    metas = [{"user_id": user_id, "source": file.filename, "session_id": sid}
             for _ in chunks]

    store.add(collection, chunks, metas)

    return JSONResponse({
        "status": "uploaded",
        "chunks_indexed": len(chunks),
        "session_id": sid
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
        return {"answer": "Please ask a question.", "sources": []}

    results = store.search(collection, query, top_k=5)
    results = [r for r in results if r["meta"].get("user_id") == user_id]
    results = [r for r in results if len(r["text"]) > 15]

    if not results:
        return {"answer": "No relevant content found in the uploaded transcript.", "sources": []}

    # Deduplicate
    seen, unique = set(), []
    for r in results:
        key = r["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    top_chunks = [clean(r["text"]) for r in unique[:3]]
    context    = " ".join(top_chunks)

    # Try Gemini first, fall back to keyword extraction
    answer = gemini_answer(query, context)
    if not answer:
        answer = focused_answer(query, top_chunks)

    return {
        "answer": answer,
        "sources": [{"text": r["text"], "score": r["score"]} for r in unique[:3]]
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
