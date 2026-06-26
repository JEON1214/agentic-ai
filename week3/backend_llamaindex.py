"""
RAG Pipeline using llama-index-style architecture with compatible packages.
Uses chromadb for vector storage and sentence-transformers for embeddings.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import re

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# EMBEDDING MODEL (llama-index style abstraction)
# ============================================================================

class EmbeddingModel:
    """Wrapper for sentence-transformers following llama-index patterns."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_tensor=False)
        result = embedding[0].tolist() if hasattr(embedding[0], 'tolist') else embedding[0]
        return result


# ============================================================================
# VECTOR STORE (llama-index style abstraction using chromadb)
# ============================================================================

class VectorStore:
    """Chromadb-backed vector store following llama-index patterns."""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collections = {}
    
    def create_collection(self, name: str, get_or_create: bool = True):
        """Create or get a collection."""
        if name not in self.collections:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            self.collections[name] = collection
        return self.collections[name]
    
    def add_documents(self, collection_name: str, texts: List[str], 
                     embeddings: List[List[float]], metadatas: List[dict]):
        """Add documents to collection."""
        collection = self.create_collection(collection_name)
        ids = [str(uuid.uuid4()) for _ in texts]
        
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return ids
    
    def search(self, collection_name: str, query_embedding: List[float], 
              top_k: int = 3) -> List[dict]:
        """Search for similar documents."""
        collection = self.create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results following llama-index conventions
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                    'score': 1 - (results['distances'][0][i] / 2) if results['distances'] else 0
                })
        return formatted


# ============================================================================
# LLM INTEGRATION (llama-index style)
# ============================================================================

try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
GEMINI_CLIENT = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except:
        GEMINI_CLIENT = None


class LLMModel:
    """Wrapper for Gemini LLM following llama-index patterns."""
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        if GEMINI_CLIENT is None:
            return None
        
        try:
            models_to_try = ['gemini-3.1-flash-lite']
            for model_name in models_to_try:
                try:
                    response = GEMINI_CLIENT.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if response and hasattr(response, 'text') and response.text:
                        return response.text
                except:
                    continue
        except:
            pass
        
        return None


# ============================================================================
# RAG PIPELINE (llama-index style composition)
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline following llama-index architecture."""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.llm = LLMModel()
    
    def ingest(self, texts: List[str], collection_name: str = "documents", 
              metadatas: Optional[List[dict]] = None):
        """Ingest documents into the pipeline."""
        if metadatas is None:
            metadatas = [{"source": "user_upload"} for _ in texts]
        
        # Embed documents
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Store in vector DB
        ids = self.vector_store.add_documents(
            collection_name, texts, embeddings, metadatas
        )
        
        return {"status": "success", "ids": ids, "count": len(ids)}
    
    def retrieve(self, query: str, collection_name: str = "documents", 
                top_k: int = 3) -> List[dict]:
        """Retrieve relevant documents."""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(collection_name, query_embedding, top_k)
        return results
    
    def generate_answer(self, query: str, context: List[dict]) -> Optional[str]:
        """Generate answer from context using LLM."""
        # Format context
        context_text = "\n\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(context)])
        
        prompt = f"""Based on the following context, answer the user's question concisely.

Question: {query}

Context:
{context_text}

Answer:"""
        
        return self.llm.generate(prompt)
    
    def query(self, query: str, collection_name: str = "documents") -> dict:
        """End-to-end query: retrieve + generate."""
        # Retrieve
        context = self.retrieve(query, collection_name, top_k=3)
        
        if not context:
            return {"answer": "No relevant documents found.", "context": []}
        
        # Generate (with fallback)
        answer = self.generate_answer(query, context)
        
        if not answer:
            # Fallback: concatenate context
            answer = " ".join([c['text'][:100] + "..." for c in context[:2]])
        
        return {"answer": answer, "context": context}


# ============================================================================
# INITIALIZE PIPELINE
# ============================================================================

pipeline = RAGPipeline()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/upload")
async def upload(file: UploadFile = File(...), collection: str = Form("documents"), 
                user_id: str = Form("user1")):
    """Upload and ingest a document."""
    try:
        content = (await file.read()).decode(errors='ignore')
        
        # Simple chunking (500 chars)
        chunks = []
        chunk_size = 500
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        
        # Prepare metadata
        metadatas = [
            {
                "source": file.filename,
                "user_id": user_id,
                "session_id": str(uuid.uuid4()),
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        
        # Ingest
        result = pipeline.ingest(chunks, collection, metadatas)
        session_id = metadatas[0].get("session_id") if metadatas else str(uuid.uuid4())
        
        return JSONResponse({
            "status": "success",
            "collection": collection,
            "session_id": session_id,
            "chunks_indexed": result["count"]
        })
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/chat")
async def chat(query_data: dict):
    """Query the RAG pipeline."""
    try:
        query = query_data.get("query")
        collection = query_data.get("collection", "documents")
        
        if not query:
            return JSONResponse({"answer": "", "context": []})
        
        # Query pipeline
        result = pipeline.query(query, collection)
        
        return JSONResponse(result)
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/health")
async def health():
    """Health check."""
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
