# ----------------------------
# Step 1: Import required libraries
# ----------------------------
import requests
import math
import pandas as pd
from google.cloud import bigquery
import subprocess
import json
import re  # For sentence splitting

# ----------------------------
# Step 2: Authenticate with GCP and set headers
# ----------------------------
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

# ----------------------------
# Step 3: NER API – CUIs fetched from input text
# Supports batch of sentences
# ----------------------------
def get_cuis_from_sentences(sentences, url):
    """
    Takes a list of sentences and sends them in one request to the NER API.
    Returns a list of unique CUIs.
    """
    payload = {"query_texts": sentences, "top_k": 3}
    all_cuis = []
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        for sentence in sentences:
            cuis = data.get(sentence, [])
            all_cuis.extend(cuis)
        all_cuis = list(set(all_cuis))  # deduplicate
        print(f"[NER] Total unique CUIs retrieved from batch: {len(all_cuis)}")
        return all_cuis
    except requests.exceptions.RequestException as e:
        print(f"[NER] Request failed: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"[NER] JSON decode error: {e}")
        return []

# ----------------------------
# Step 4: Initialize BigQuery client
# ----------------------------
client = bigquery.Client()

# ----------------------------
# Step 5: Load UMLS hierarchy – prepare parent and child maps
# ----------------------------
class MRRELCache:
    def __init__(self, project_id, dataset, mrrel_table):
        self.project_id = project_id
        self.dataset = dataset
        self.mrrel_table = mrrel_table
        self.parent_map = {}
        self.child_map = {}
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        query = f"""
            SELECT CUI1 AS child_cui, CUI2 AS parent_cui
            FROM `{self.project_id}.{self.dataset}.{self.mrrel_table}`
            WHERE REL = 'PAR'
        """
        df = client.query(query).to_dataframe()
        self.parent_map = df.groupby("child_cui")["parent_cui"].apply(list).to_dict()
        self.child_map = df.groupby("parent_cui")["child_cui"].apply(list).to_dict()
        print(f"[Hierarchy] Loaded {len(self.parent_map)} child CUIs")
        self.loaded = True

# ----------------------------
# Step 6: Get full semantic tree (ancestors + descendants)
# ----------------------------
def get_semantic_tree(cui, parent_map, child_map):
    tree_cuis = set()
    stack = [cui]

    # Traverse upward
    while stack:
        current = stack.pop()
        for parent in parent_map.get(current, []):
            if parent not in tree_cuis:
                tree_cuis.add(parent)
                stack.append(parent)

    # Traverse downward
    stack = [cui]
    while stack:
        current = stack.pop()
        for child in child_map.get(current, []):
            if child not in tree_cuis:
                tree_cuis.add(child)
                stack.append(child)

    tree_cuis.add(cui)
    return tree_cuis

# ----------------------------
# Step 7: Compute IC
# ----------------------------
def compute_ic(cuis, descendants_table, project_id, dataset):
    if not cuis:
        return {}, 0
    cuis_str = ",".join([f"'{c}'" for c in cuis])
    query = f"""
        SELECT CUI, NarrowerConceptCount AS num_descendants
        FROM `{project_id}.{dataset}.{descendants_table}`
        WHERE CUI IN ({cuis_str})
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        return {}, 0
    total_cuis = len(cuis)
    df["IC"] = df["num_descendants"].apply(lambda x: -math.log((x + 1)/total_cuis))
    ic_map = dict(zip(df["CUI"], df["IC"]))
    threshold = df["IC"].median()
    print(f"[IC] Computed IC for {len(ic_map)} CUIs. Median threshold: {threshold:.4f}")
    return ic_map, threshold

# ----------------------------
# Helper: Split text into single-sentence chunks
# ----------------------------
def chunk_into_sentences(text):
    """
    Splits text into individual sentences using ., ?, or ! as delimiters.
    Returns list of sentences.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences

# ----------------------------
# Step 8: Apply IC threshold
# ----------------------------
def semantic_rollup(text, ner_url, mrrel_cache, descendants_table, project_id, dataset):
    # Split text into sentences
    sentences = chunk_into_sentences(text)
    print(f"[Chunking] {len(sentences)} sentences found.")

    # Call NER API in batch
    cuis = get_cuis_from_sentences(sentences, ner_url)

    if not cuis:
        print("[Rollup] No CUIs found for text.")
        return pd.DataFrame()

    # Load UMLS hierarchy
    mrrel_cache.load()

    # Expand each CUI to full semantic tree
    all_cuis = set()
    for cui in cuis:
        all_cuis.update(get_semantic_tree(cui, mrrel_cache.parent_map, mrrel_cache.child_map))

    # Compute IC
    ic_map, threshold = compute_ic(list(all_cuis), descendants_table, project_id, dataset)
    if not ic_map:
        print("[Rollup] No IC computed.")
        return pd.DataFrame()

    # Filter CUIs with IC >= threshold
    final_cuis = {cui for cui in all_cuis if ic_map.get(cui, 0) >= threshold}

    # Prepare output DataFrame
    output = [{"CUI": c, "IC": ic_map[c]} for c in final_cuis if c in ic_map]
    df = pd.DataFrame(output)
    df = df.sort_values("IC", ascending=False).reset_index(drop=True)
    print(f"[Rollup] Total filtered CUIs: {len(df)}")
    return df

# ----------------------------
# Step 9: Main execution — read formatted query JSON from Code 1
# ----------------------------
if __name__ == "__main__":
    # Load output from Code 1
    with open("pipeline_output.json") as f:
        pipeline_data = json.load(f)

    # Use the formatted query for NER
    text_for_ner = pipeline_data["cui_processing_input"]["text_for_ner"]
    print(f"[Input] Using formatted query for NER: {text_for_ner}")

    # Define GCP / NER API parameters
    project_id = project_id
    dataset = dataset
    descendants_table = descendants_table
    mrrel_table = mrrel_table
    url = url  # NER API endpoint

    # Initialize hierarchy cache
    mrrel_cache = MRRELCache(project_id, dataset, mrrel_table)

    # Run semantic rollup
    result_df = semantic_rollup(text_for_ner, url, mrrel_cache, descendants_table, project_id, dataset)

    # Save results
    if not result_df.empty:
        print(result_df)
        result_df.to_csv("filtered_cuis.csv", index=False)
        print("[Output] Saved filtered CUIs with IC.")
    else:
        print("[Output] No CUIs passed the threshold.")
