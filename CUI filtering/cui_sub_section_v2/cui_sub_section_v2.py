# ----------------------------
# Step 1: Import required libraries
# ----------------------------
import requests
import math
import pandas as pd
from google.cloud import bigquery
import subprocess
import json

# ----------------------------
# Step 2: Authenticate with GCP and set headers
# ----------------------------
headers = None

def gcp_update_header():
    # Fetch identity token using gcloud CLI and set headers for authenticated requests
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
# ----------------------------
def get_cuis_from_text(text, url):
    payload = {"query_texts": [text], "top_k": 3}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        cuis = data.get(text, [])
        print(f"[NER] CUIs retrieved: {len(cuis)}")
        return cuis
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
        # Load parent-child relationships where REL = 'PAR'
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
# Step 6: Get full semantic tree (ancestors + descendants) for each CUI
# ----------------------------
def get_semantic_tree(cui, parent_map, child_map):
    tree_cuis = set()
    stack = [cui]

    # Traverse upward to get ancestors
    while stack:
        current = stack.pop()
        for parent in parent_map.get(current, []):
            if parent not in tree_cuis:
                tree_cuis.add(parent)
                stack.append(parent)

    # Traverse downward to get descendants
    stack = [cui]
    while stack:
        current = stack.pop()
        for child in child_map.get(current, []):
            if child not in tree_cuis:
                tree_cuis.add(child)
                stack.append(child)

    # Include the original CUI
    tree_cuis.add(cui)
    return tree_cuis

# ----------------------------
# Step 7: Bring descendant count and calculate IC for all CUIs
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
# Step 8: Apply IC threshold to full semantic tree and filter CUIs
# ----------------------------
def semantic_rollup(text, ner_url, mrrel_cache, descendants_table, project_id, dataset):
    # Step 8a: Get CUIs from NER API
    cuis = get_cuis_from_text(text, ner_url)
    if not cuis:
        print("[Rollup] No CUIs found for text.")
        return pd.DataFrame()

    # Step 8b: Load UMLS hierarchy
    mrrel_cache.load()

    # Step 8c: Expand each CUI to its full semantic tree
    all_cuis = set()
    for cui in cuis:
        all_cuis.update(get_semantic_tree(cui, mrrel_cache.parent_map, mrrel_cache.child_map))

    # Step 8d: Compute IC for all CUIs in semantic trees
    ic_map, threshold = compute_ic(list(all_cuis), descendants_table, project_id, dataset)
    if not ic_map:
        print("[Rollup] No IC computed.")
        return pd.DataFrame()

    # Step 8e: Filter CUIs with IC ≥ threshold
    final_cuis = {cui for cui in all_cuis if ic_map.get(cui, 0) >= threshold}

    # Step 8f: Prepare final output DataFrame
    output = [{"CUI": c, "IC": ic_map[c]} for c in final_cuis if c in ic_map]
    df = pd.DataFrame(output)
    df = df.sort_values("IC", ascending=False).reset_index(drop=True)
    print(f"[Rollup] Total filtered CUIs: {len(df)}")
    return df

# ----------------------------
# Step 9: Example usage and save output
# ----------------------------
if __name__ == "__main__":
    # Define your GCP and API parameters here
    project_id = project_id 
    dataset = dataset
    descendants_table = descendants_table
    mrrel_table = mrrel_table
    url = url

    # Initialize hierarchy cache
    mrrel_cache = MRRELCache(project_id, dataset, mrrel_table)

    # Run semantic rollup for sample text
    sample_text = "MRI of head"
    result_df = semantic_rollup(sample_text, url, mrrel_cache, descendants_table, project_id, dataset)

    # Save results to CSV
    if not result_df.empty:
        print(result_df)
        result_df.to_csv("filtered_cuis.csv", index=False)
        print("[Output] Saved filtered CUIs with IC.")
    else:
        print("[Output] No CUIs passed the threshold.")
