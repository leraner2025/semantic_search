# Consolidated Final
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import json

from google.cloud import bigquery, bigquery_storage
import vertexai
from vertexai.language_models import TextEmbeddingModel

# CONFIG
PROJECT_ID = PROJECT_ID
LOCATION = LOCATION
BQ_CUI_TABLE = BQ_CUI_TABLE

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# Load embedding model once globally
embedding_model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)


# Utility Functions

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
    
    query = f"""
        SELECT REF_cui, REF_Embedding
        FROM `{table_fqn}`
    """
    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe(bqstorage_client=bqs_client)

    embeddings = {}
    for _, row in df.iterrows():
        embeddings[row["REF_cui"]] = np.array(row["REF_Embedding"], dtype=np.float32).ravel()
    return embeddings


# DocAI Parsing

def load_docai_json(response_content: str) -> Tuple[str, List[dict]]:
    doc = json.loads(response_content)
    output = doc.get("output", {})
    full_text = output.get("text", "")
    pages = output.get("pages", [])
    return full_text, pages

def extract_text_from_anchor_with_context(text_anchor: Dict, full_text: str) -> str:
    if not text_anchor or "textSegments" not in text_anchor:
        return ""
    segments = text_anchor["textSegments"]
    texts = []
    for segment in segments:
        start = int(segment.get("startIndex", 0))
        end = int(segment.get("endIndex", 0))
        if start >= end or end > len(full_text):
            continue
        texts.append(full_text[start:end])
    return " ".join(texts).strip()

def merge_text_from_docai_blocks(full_text: str, pages: List[dict]) -> List[str]:
    merged_texts = []
    for page in pages:
        for block in page.get("blocks", []):
            layout = block.get("layout", {})
            anchor = layout.get("textAnchor", {})
            block_text = extract_text_from_anchor_with_context(anchor, full_text) 
            
            # Removed unused vars for clarity:
            # para_type = layout.get("detectedLanguages", [{}])[0].get("languageCode", "")
            # block_type = block.get("detectedBreakType", "")

            header = block.get("sectionHeader", {}).get("text", "")
            if header and header not in block_text:
                block_text = f"{header}\n{block_text}"
            if block_text:
                merged_texts.append(block_text)
    return merged_texts


# Embedding Flow

@dataclass
class Entity:
    entity_id: str
    text: str
    emb: Optional[np.ndarray] = None
    sim_to_query: Optional[float] = None
    top_cuis: Optional[List[Tuple[str, float]]] = None  # [(CUI, score)]

    @property
    def cui(self):
        # Return top CUI for backward compatibility
        return self.top_cuis[0][0] if self.top_cuis else None

def entities_from_docai_blocks(merged_texts: List[str]) -> List[Entity]:
    return [Entity(entity_id=str(i), text=text) for i, text in enumerate(merged_texts)]

def assign_best_matching_cui(entities: List[Entity], cui_embeddings: Dict[str, np.ndarray], top_n: int = 3):
    for entity in entities:
        if entity.emb is None:
            continue
        scores = []
        for cui, cui_emb in cui_embeddings.items():
            score = cosine_sim(entity.emb, cui_emb)
            scores.append((cui, score))
        top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        entity.top_cuis = top_matches

def attach_embeddings(
    entities: List[Entity],
    project_id: str,
    cui_table: str,
    query_text: str,
    top_k_cuis: int = 100,
):
    query_emb = gemini_embed_single(query_text)

    for e in entities:
        e.emb = gemini_embed_single(e.text)

    cui_embeddings = fetch_all_cui_embeddings(project_id, cui_table)

    # Pre-filter CUIs using query similarity
    cui_scores = {
        cui: cosine_sim(query_emb, emb)
        for cui, emb in cui_embeddings.items()
    }
    top_cuis = sorted(cui_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_cuis]
    filtered_cui_embeddings = {cui: cui_embeddings[cui] for cui, _ in top_cuis}

    # print(f"[INFO] Using top {top_k_cuis} CUIs (filtered from {len(cui_embeddings)})")

    assign_best_matching_cui(entities, filtered_cui_embeddings, top_n=3)
    
def compute_query_similarities(entities: List[Entity], query_emb: np.ndarray, query_text: str):
    for e in entities:
        if e.emb is not None:
            sim = cosine_sim(query_emb, e.emb)
            # Boost if block contains query term literally (lightly)
            if query_text.lower() in e.text.lower():
                sim += 0.05  # Boost by small amount
            e.sim_to_query = sim

def multihop_from_root(root_entity: Entity, entities: List[Entity], num_neighbors: int = 4) -> List[Entity]:
    neighbors = []
    for e in entities:
        if e.entity_id == root_entity.entity_id or e.emb is None:
            continue
        sim = cosine_sim(root_entity.emb, e.emb)
        neighbors.append((e, sim))
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in neighbors[:num_neighbors]]


# End-to-End Pipeline

def run_docai_embedding_pipeline(
    response_content: str,
    query_text: str,
    project_id: str,
    cui_table: str
) -> List[Entity]:
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


# # Sample usage (make sure you set response_content and PROJECT_ID, BQ_CUI_TABLE accordingly)

# response_content = response.content.decode("utf-8")  # from requests or HTTP call
# query = input("Enter your search query: ").strip()
# results = run_docai_embedding_pipeline(response_content, query, PROJECT_ID, BQ_CUI_TABLE)

# for idx, e in enumerate(results, 1):
#     print(f"[{idx}] {e.text.replace(chr(10), ' ').strip()}")
#     if e.top_cuis:
#         cuis = [cui for cui, _ in e.top_cuis]
#         print("Top CUIs: " + ", ".join(cuis))
#     else:
#         print("Top CUIs: None")
#     print()

# # # Save results as JSON (not printed)
# # final_output = [
# #     {
#         "entity_id": e.entity_id,
#         "text": e.text,
#         "top_cuis": [{"cui": cui, "score": score} for cui, score in (e.top_cuis or [])],
#         "embedding": e.emb.tolist() if e.emb is not None else None,
#         "similarity_to_query": e.sim_to_query
#     }
#     for e in results
# ]

# with open("final_entities.json", "w", encoding="utf-8") as f:
#     json.dump(final_output, f, indent=2)
