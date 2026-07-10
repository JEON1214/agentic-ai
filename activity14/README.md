# Activity 13: Improved Week 3 RAG Chat

This folder contains a cleaner, runnable version of the Week 3 retrieval-augmented chat experience.

## What is included
- FastAPI backend with `/upload` and `/chat` endpoints
- In-memory TF-IDF retrieval over uploaded transcript chunks
- Optional Gemini integration when an API key is available
- Simple browser-based frontend for uploading files and chatting

## Setup

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

4. Open http://localhost:8000 in your browser

## Notes
- If no Gemini API key is configured, the app falls back to a local answer strategy.
- Upload a plain text transcript first, then chat with it using a collection name and user ID.
