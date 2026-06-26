Simple RAG Chat (week3)

Backend: FastAPI (serve frontend and provide /upload and /chat endpoints)
Frontend: Single-file React SPA (CDN) at / (served by backend)

Setup:

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

3. Open `http://localhost:8000` in your browser.

Notes:
- The vector store is an in-memory store (no external vector DB) implemented in `backend/main.py`.
- Upload a text file (plain .txt) to a collection and then ask questions in the chat box.
- The agent returns the exact string `i don't know if prompt is not relevant to data.` when it finds no sufficiently similar facts.
