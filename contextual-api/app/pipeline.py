from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import json

from google.cloud import bigquery, bigquery_storage
import vertexai
from vertexai.language_models import TextEmbeddingModel
from config.settings import PROJECT_ID, LOCATION, GEMINI_EMBEDDING_MODEL, BQ_CUI_TABLE

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)

# -------------------- Utility Functions --------------------

def gemini_embed_single(text: str) -> np.ndarray:
    embedding = embedding_model.get_embeddings([text])[0]
    return np.array(embedding.values, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a) + 1e-8
    b_norm = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (a_norm * b_norm))

def fetch_all_cui_embeddings(project_id: str, table_fqn: str) -> Dict[str, np.ndarray]:
    bq_client = bigquery.Client(project=project_id)
    bqs_client = bigquery_storage.BigQueryReadClient()

    query = f"SELECT REF_cui, REF_Embedding FROM `{table_fqn}`"
    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe(bqstorage_client=bqs_client)

    embeddings = {
        row["REF_cui"]: np.array(row["REF_Embedding"], dtype=np.float32).ravel()
        for _, row in df.iterrows()
    }
    return embeddings

# -------------------- DocAI Parsing --------------------

def load_docai_json(response_content: str) -> Tuple[str, List[dict]]:
    doc = json.loads(response_content)
    output = doc.get("output", {})
    return output.get("text", ""), output.get("pages", [])

def extract_text_from_anchor_with_context(text_anchor: Dict, full_text: str) -> str:
    if not text_anchor or "textSegments" not in text_anchor:
        return ""
    segments = text_anchor["textSegments"]
    texts = [
        full_text[int(seg.get("startIndex", 0)):int(seg.get("endIndex", 0))]
        for seg in segments
        if int(seg.get("startIndex", 0)) < int(seg.get("endIndex", 0)) <= len(full_text)
    ]
    return " ".join(texts).strip()

def merge_text_from_docai_blocks(full_text: str, pages: List[dict]) -> List[str]:
    merged_texts = []
    for page in pages:
        for block in page.get("blocks", []):
            layout = block.get("layout", {})
            anchor = layout.get("textAnchor", {})
            block_text = extract_text_from_anchor_with_context(anchor, full_text)

            header = block.get("sectionHeader", {}).get("text", "")
            if header and header not in block_text:
                block_text = f"{header}\n{block_text}"
            if block_text:
                merged_texts.append(block_text)
    return merged_texts

# -------------------- Embedding Flow --------------------

@dataclass
class Entity:
    entity_id: str
    text: str
    emb: Optional[np.ndarray] = None
    sim_to_query: Optional[float] = None
    top_cuis: Optional[List[Tuple[str, float]]] = None

    @property
    def cui(self):
        return self.top_cuis[0][0] if self.top_cuis else None

def entities_from_docai_blocks(merged_texts: List[str]) -> List[Entity]:
    return [Entity(entity_id=str(i), text=text) for i, text in enumerate(merged_texts)]

def assign_best_matching_cui(entities, cui_embeddings, top_n=3, min_sim: float = 0.6):
    for entity in entities:
        if entity.emb is None:
            continue
        scores = [
            (cui, cosine_sim(entity.emb, cui_emb))
            for cui, cui_emb in cui_embeddings.items()
            if cosine_sim(entity.emb, cui_emb) >= min_sim
        ]
        entity.top_cuis = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n] or None

def attach_embeddings(entities: List[Entity], project_id: str, cui_table: str, query_text: str, top_k_cuis: int = 100):
    query_emb = gemini_embed_single(query_text)
    for e in entities:
        e.emb = gemini_embed_single(e.text)

    cui_embeddings = fetch_all_cui_embeddings(project_id, cui_table)
    cui_scores = {cui: cosine_sim(query_emb, emb) for cui, emb in cui_embeddings.items()}
    top_cuis = sorted(cui_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_cuis]
    filtered_cui_embeddings = {cui: cui_embeddings[cui] for cui, _ in top_cuis}

    assign_best_matching_cui(entities, filtered_cui_embeddings, top_n=3)

def compute_query_similarities(entities: List[Entity], query_emb: np.ndarray, query_text: str):
    for e in entities:
        if e.emb is not None:
            sim = cosine_sim(query_emb, e.emb)
            if query_text.lower() in e.text.lower():
                sim += 0.05
            e.sim_to_query = sim

def multihop_from_root(root_entity: Entity, entities: List[Entity], num_neighbors: int = 4) -> List[Entity]:
    neighbors = [
        (e, cosine_sim(root_entity.emb, e.emb))
        for e in entities
        if e.entity_id != root_entity.entity_id and e.emb is not None
    ]
    return [e for e, _ in sorted(neighbors, key=lambda x: x[1], reverse=True)[:num_neighbors]]

# -------------------- End-to-End Pipeline --------------------

def run_docai_embedding_pipeline(response_content: str, query_text: str, project_id: str, cui_table: str) -> List[Entity]:
    full_text, pages = load_docai_json(response_content)
    merged_texts = merge_text_from_docai_blocks(full_text, pages)
    entities = entities_from_docai_blocks(merged_texts)

    attach_embeddings(entities, project_id, cui_table, query_text=query_text)
    query_emb = gemini_embed_single(query_text)
    compute_query_similarities(entities, query_emb, query_text)

    query_sorted = sorted(
        [e for e in entities if e.sim_to_query is not None],
        key=lambda x: x.sim_to_query, reverse=True
    )
    if not query_sorted:
        return []

    root = query_sorted[0]
    direct_4 = query_sorted[1:5]
    multihop_4 = multihop_from_root(root, entities, num_neighbors=4)

    combined = {e.entity_id: e for e in direct_4 + multihop_4}
    final_4 = sorted(combined.values(), key=lambda x: x.sim_to_query or 0, reverse=True)[:4]

    return [root] + final_4
