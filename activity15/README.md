# Activity 15

This folder is the polished integration point for the chatbot project.

## What is included
- A FastAPI backend based on the Week 3 chatbot flow
- A browser-based frontend for upload and chat
- A clean starting point for integrating Activity 13 and Activity 14 features

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
