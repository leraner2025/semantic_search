# models.py

import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Initialize Vertex AI once
PROJECT_ID = "x1245"
LOCATION = "us-centr"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load embedding model globally
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
embedding_model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)

def gemini_embed_single(text: str) -> np.ndarray:
    embedding = embedding_model.get_embeddings([text])[0]
    return np.array(embedding.values, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a) + 1e-8
    b_norm = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (a_norm * b_norm))
