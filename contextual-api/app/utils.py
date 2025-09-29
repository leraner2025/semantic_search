import numpy as np
from google.cloud import bigquery, bigquery_storage
from vertexai.language_models import TextEmbeddingModel
import vertexai
from config import PROJECT_ID, LOCATION, GEMINI_EMBEDDING_MODEL

vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)

def gemini_embed_single(text: str) -> np.ndarray:
    embedding = embedding_model.get_embeddings([text])[0]
    return np.array(embedding.values, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

def fetch_all_cui_embeddings(project_id: str, table_fqn: str):
    bq_client = bigquery.Client(project=project_id)
    bqs_client = bigquery_storage.BigQueryReadClient()
    query = f"SELECT REF_cui, REF_Embedding FROM `{table_fqn}`"
    df = bq_client.query(query).result().to_dataframe(bqstorage_client=bqs_client)
    return {row["REF_cui"]: np.array(row["REF_Embedding"], dtype=np.float32).ravel() for _, row in df.iterrows()}
