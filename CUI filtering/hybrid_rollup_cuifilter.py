import requests
import numpy as np
import pandas as pd
from collections import defaultdict
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery, aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Initialize Vertex AI
aiplatform.init(project=project_id, location="us-central1")
gemini_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

# GCP Auth Header Setup
headers = None
def gcp_update_header():
    global headers
    tmp = subprocess.run(['gcloud', 'auth', 'print-identity-token'], stdout=subprocess.PIPE, universal_newlines=True)
    if tmp.returncode != 0:
        raise Exception("Cannot get GCP access token")
    identity_token = tmp.stdout.strip()
    headers = {
        "Authorization": f"Bearer {identity_token}",
        "Content-Type": "application/json"
    }
gcp_update_header()

# Step 1: NER API
def call_ner_api(text):
    payload = {"query_texts": [text], "top_k": 3}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"NER API error: {e}")
        return {}

# Step 2: Extract CUIs
def extract_cuis(ner_response, input_text):
    return list(set(ner_response.get(input_text, [])))

# Step 3: Retrieve embeddings for CUIs from BigQuery
def get_cui_embeddings(client, project_id, dataset, embedding_table, cuis):
    if not cuis:
        return {}
    cuis_str = ",".join([f"'{c}'" for c in cuis])
    query = f"""
    SELECT REF_CUI, REF_Embedding
    FROM `{project_id}.{dataset}.{embedding_table}`
    WHERE REF_CUI IN UNNEST([{cuis_str}])
    """
    results = client.query(query).result()
    return {row.REF_CUI: row.REF_Embedding for row in results}

# Step 4: Load UMLS hierarchy
def load_umls_relations(mrrel_path):
    child_to_parents = defaultdict(set)
    parent_to_children = defaultdict(set)
    with open(mrrel_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if parts[3] == 'PAR':
                child, parent = parts[0], parts[4]
                child_to_parents[child].add(parent)
                parent_to_children[parent].add(child)
    return child_to_parents, parent_to_children

# Step 5: Compute IC scores
def compute_ic(parent_to_children):
    all_cuis = set(parent_to_children.keys()) | {c for children in parent_to_children.values() for c in children}
    total_cuis = len(all_cuis)
    def count_descendants(cui):
        visited = set()
        queue = [cui]
        while queue:
            current = queue.pop()
            for child in parent_to_children.get(current, []):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return len(visited)
    ic_scores = {}
    for cui in all_cuis:
        desc_count = count_descendants(cui)
        ic_scores[cui] = -np.log((desc_count + 1) / total_cuis)
    return ic_scores

# Step 6: Get ancestors
def get_ancestors(cui, child_to_parents):
    ancestors = set()
    queue = [cui]
    while queue:
        current = queue.pop()
        for parent in child_to_parents.get(current, []):
            if parent not in ancestors:
                ancestors.add(parent)
                queue.append(parent)
    return ancestors

# Step 7: Find Lowest Informative Ancestor
def find_lia(cui, ic_scores, child_to_parents, threshold):
    candidates = get_ancestors(cui, child_to_parents) | {cui}
    informative = [c for c in candidates if ic_scores.get(c, 0) >= threshold]
    if informative:
        return min(informative, key=lambda c: ic_scores[c])
    return cui

# Step 8: Roll up CUIs
def rollup_cuis(cui_list, ic_scores, child_to_parents, threshold):
    rolled_up = set()
    for cui in cui_list:
        rolled_up_cui = find_lia(cui, ic_scores, child_to_parents, threshold)
        rolled_up.add(rolled_up_cui)
    return list(rolled_up)

# Step 9: Embed user query
def embed_query(text):
    embedding = gemini_model.get_embeddings([text])[0]
    return np.array(embedding.values).reshape(1, -1)

# Step 10: Filter CUIs by similarity
def filter_by_similarity(query_text, cui_embeddings_dict, threshold=0.6):
    query_vec = embed_query(query_text)
    cui_ids = list(cui_embeddings_dict.keys())
    cui_vecs = np.array([cui_embeddings_dict[cui] for cui in cui_ids])
    sims = cosine_similarity(query_vec, cui_vecs)[0]
    return [cui for cui, sim in zip(cui_ids, sims) if sim >= threshold]

# Step 11: Full hybrid pipeline
def get_final_cuis(text, mrrel_path, bq_client, project_id, dataset, embedding_table, similarity_threshold=0.6):
    ner_response = call_ner_api(text)
    cui_list = extract_cuis(ner_response, text)
    if not cui_list:
        return []

    child_to_parents, parent_to_children = load_umls_relations(mrrel_path)
    ic_scores = compute_ic(parent_to_children)
    ic_threshold = np.median(list(ic_scores.values()))
    rolled_up_cuis = rollup_cuis(cui_list, ic_scores, child_to_parents, ic_threshold)
    
    cui_embeddings = get_cui_embeddings(bq_client, project_id, dataset, embedding_table, rolled_up_cuis)
    final_cuis = filter_by_similarity(text, cui_embeddings, threshold=similarity_threshold)
    return final_cuis
from google.cloud import bigquery

bq_client = bigquery.Client()
mrrel_path = "MRREL.RRF"
query_text = "Patient has Type 2 Diabetes and hypertension"
final_cuis = get_final_cuis(
    text=query_text,
    mrrel_path=mrrel_path,
    bq_client=bq_client,
    project_id=project_id,
    dataset="your_dataset",
    embedding_table="your_embedding_table",
    similarity_threshold=0.65
)
print("Final CUIs:", final_cuis)
