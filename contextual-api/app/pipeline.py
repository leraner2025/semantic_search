from models import Entity
from utils import gemini_embed_single, cosine_sim, fetch_all_cui_embeddings
from docai_parser import load_docai_json, merge_text_from_docai_blocks

def entities_from_docai_blocks(merged_texts):
    return [Entity(entity_id=str(i), text=text) for i, text in enumerate(merged_texts)]

def assign_best_matching_cui(entities, cui_embeddings, top_n=3, min_sim=0.6):
    for entity in entities:
        if entity.emb is None:
            continue
        scores = [(cui, cosine_sim(entity.emb, emb)) for cui, emb in cui_embeddings.items()]
        entity.top_cuis = sorted([s for s in scores if s[1] >= min_sim], key=lambda x: x[1], reverse=True)[:top_n]

def compute_query_similarities(entities, query_emb, query_text):
    for e in entities:
        if e.emb is not None:
            sim = cosine_sim(query_emb, e.emb)
            if query_text.lower() in e.text.lower():
                sim += 0.05
            e.sim_to_query = sim

def multihop_from_root(root_entity, entities, num_neighbors=4):
    neighbors = [(e, cosine_sim(root_entity.emb, e.emb)) for e in entities if e.entity_id != root_entity.entity_id and e.emb is not None]
    return [e for e, _ in sorted(neighbors, key=lambda x: x[1], reverse=True)[:num_neighbors]]

def run_docai_embedding_pipeline(response_content, query_text, project_id, cui_table):
    full_text, pages = load_docai_json(response_content)
    merged_texts = merge_text_from_docai_blocks(full_text, pages)
    entities = entities_from_docai_blocks(merged_texts)

    query_emb = gemini_embed_single(query_text)
    for e in entities:
        e.emb = gemini_embed_single(e.text)

    cui_embeddings = fetch_all_cui_embeddings(project_id, cui_table)
    top_cuis = sorted({cui: cosine_sim(query_emb, emb) for cui, emb in cui_embeddings.items()}.items(), key=lambda x: x[1], reverse=True)[:100]
    filtered_cui_embeddings = {cui: cui_embeddings[cui] for cui, _ in top_cuis}

    assign_best_matching_cui(entities, filtered_cui_embeddings)
    compute_query_similarities(entities, query_emb, query_text)

    query_sorted = sorted([e for e in entities if e.sim_to_query is not None], key=lambda x: x.sim_to_query, reverse=True)
    if not query_sorted:
        return []

    root = query_sorted[0]
    final_entities = [root] + multihop_from_root(root, entities)
    return final_entities
