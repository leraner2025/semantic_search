# STEP 1: Install required packages
!pip install google-cloud-bigquery pandas

# STEP 2: Import libraries
from google.cloud import bigquery
import pandas as pd
from collections import defaultdict

# STEP 3: Set up GCP project
PROJECT_ID = "your-gcp-project-id"  # Replace with your actual project ID
bq_client = bigquery.Client(project=PROJECT_ID)

# STEP 4: Load CUI hierarchy from BigQuery
QUERY = """
SELECT Parent_CUI, Child_CUI
FROM `your_dataset.your_table`
WHERE Child_CUI IS NOT NULL
"""
df = bq_client.query(QUERY).to_dataframe()

# STEP 5: Forward-fill merged parent rows
df['Parent_CUI'] = df['Parent_CUI'].ffill()

# STEP 6: Build parent-to-children mapping
cui_tree = defaultdict(list)
for _, row in df.iterrows():
    parent = str(row['Parent_CUI']).strip()
    child = str(row['Child_CUI']).strip()
    cui_tree[parent].append(child)

# STEP 7: Recursive function to find deepest children
def find_deepest_children(cui, tree):
    if cui not in tree:
        return [cui]  # It's a leaf
    leaves = []
    for child in tree[cui]:
        leaves.extend(find_deepest_children(child, tree))
    return leaves

# STEP 8: Input list of CUIs to search
input_cuis = ["CUI001", "CUI010", "CUI020"]  # Replace with your actual CUIs

# STEP 9: Find deepest children for each input CUI
all_deepest = []
for cui in input_cuis:
    deepest = find_deepest_children(cui, cui_tree)
    all_deepest.extend(deepest)

# STEP 10: Deduplicate and display result
unique_deepest = sorted(set(all_deepest))
print("Final deepest CUIs:")
print(unique_deepest)
