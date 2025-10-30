import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery, aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
import subprocess
import time
import pandas as pd
from collections import defaultdict

# Initialize Vertex AI
aiplatform.init(project=project_id, location="us-central1")
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
gemini_model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)

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
    url = url
    payload = {"query_texts": [text], "top_k": 3}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()         
        cui_count = len(data.get(text, []))
        print(f"API call successful. CUIs returned: {cui_count}")
        return data
    except Exception as e:
        print(f"NER API error: {e}")
        return {}

# Step 2: Extract CUIs
def extract_cuis(ner_response, input_text):
    cuis = list(set(ner_response.get(input_text, [])))
    return cuis
# Step 3: Load UMLS hierarchy
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

# Step 4: Compute IC scores
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

# Step 5: Get ancestors
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

# Step 6: Find Lowest Informative Ancestor
def find_lia(cui, ic_scores, child_to_parents, threshold):
    candidates = get_ancestors(cui, child_to_parents) | {cui}
    informative = [c for c in candidates if ic_scores.get(c, 0) >= threshold]
    if informative:
        return min(informative, key=lambda c: ic_scores[c])
    return cui

# Step 7: Roll up CUIs
def rollup_cuis(cui_list, ic_scores, child_to_parents, threshold):
    rolled_up = set()
    for cui in cui_list:
        rolled_up_cui = find_lia(cui, ic_scores, child_to_parents, threshold)
        rolled_up.add(rolled_up_cui)
    return list(rolled_up)

# Step 8: Full pipeline for query → rolled-up CUIs
def get_rolled_up_cuis_for_query(text, mrrel_path):
    ner_response = call_ner_api(text)
    cui_list = extract_cuis(ner_response, text)
    if not cui_list:
        return []

    child_to_parents, parent_to_children = load_umls_relations(mrrel_path)
    ic_scores = compute_ic(parent_to_children)
    threshold = np.median(list(ic_scores.values()))
    rolled_up_cuis = rollup_cuis(cui_list, ic_scores, child_to_parents, threshold)
    return rolled_up_cuis

mrrel_path = "MRREL.RRF"  # Path to your UMLS file
query_text = "Patient has Type 2 Diabetes and hypertension"
final_cuis = get_rolled_up_cuis_for_query(query_text, mrrel_path)
print("Rolled-up CUIs:", final_cuis)
