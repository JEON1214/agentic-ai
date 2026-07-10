import os
import uuid
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

try:
    from google import genai
except Exception:
    genai = None

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "course_memory")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

SAMPLE_TEXTS = [
    "ReAct stands for Reasoning + Acting. It interleaves thought and tool calls.",
    "Qdrant is a vector database designed for similarity search and retrieval.",
    "Chunking splits documents into smaller pieces to preserve context and meaning.",
    "The travel budget is $2000 for flights and $500 for accommodations.",
    "Primitive types include int, char, long, double, and boolean in many languages.",
]

client = QdrantClient(url=QDRANT_URL)

# Choose embedding strategy
if GEMINI_API_KEY and genai:
    gemini = genai.Client(api_key=GEMINI_API_KEY)
    use_gemini = True
else:
    gemini = None
    use_gemini = False

if use_gemini:
    # get embedding size
    try:
        resp = gemini.models.embed_content(model="gemini-embedding-2", contents=SAMPLE_TEXTS[0])
        VECTOR_SIZE = len(resp.embeddings[0].values)
    except Exception:
        use_gemini = False

if not use_gemini:
    VECTOR_SIZE = 8

print(f"Using vector size: {VECTOR_SIZE} (gemini: {use_gemini})")

# create collection if needed
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    print(f"Collection {COLLECTION_NAME} exists; deleting and recreating for seed.")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

points = []
for i, text in enumerate(SAMPLE_TEXTS):
    if use_gemini and gemini:
        try:
            r = gemini.models.embed_content(model="gemini-embedding-2", contents=text)
            vec = r.embeddings[0].values
        except Exception as e:
            print('Embedding failed, falling back to local embed:', e)
            vec = [float((hash(text) >> j) & 0xFF) / 255.0 for j in range(VECTOR_SIZE)]
    else:
        vec = [float((hash(text) >> j) & 0xFF) / 255.0 for j in range(VECTOR_SIZE)]

    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text_segment": text,
                "source": "seed",
                "chunk_index": i,
            },
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Seeded {len(points)} points into collection '{COLLECTION_NAME}' at {QDRANT_URL}.")
