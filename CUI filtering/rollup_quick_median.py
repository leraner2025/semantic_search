In biq query 
#query for returing the single tabel CREATE OR REPLACE TABLE `your_dataset.cui_ancestor_ic` AS
WITH RECURSIVE ancestors AS (
  SELECT child_cui, parent_cui
  FROM `your_dataset.mrrel_table`
  WHERE rel = 'PAR'
  UNION ALL
  SELECT a.child_cui, r.parent_cui
  FROM ancestors a
  JOIN `your_dataset.mrrel_table` r
  ON a.parent_cui = r.child_cui
  WHERE r.rel = 'PAR'
),
total_cui_count AS (
  SELECT COUNT(DISTINCT cui) AS total_cuis
  FROM `your_dataset.descendants_table`
),
ic_scores AS (
  SELECT
    a.child_cui AS cui,
    a.parent_cui AS ancestor_cui,
    d.descendant_count,
    -LOG((d.descendant_count + 1) / total.total_cuis) AS ic_score
  FROM
    ancestors a
  JOIN
    `your_dataset.descendants_table` d
  ON
    a.parent_cui = d.cui
  CROSS JOIN
    total_cui_count total
)
SELECT * FROM ic_scores;








in python : 
Make sure to update:

#your_project → your actual GCP project ID

#your_dataset → your BigQuery dataset name

#mrrel_table → your table with parent-child relationships

#descendants_table → your table with descendant counts




from google.cloud import bigquery
client = bigquery.Client(project="your_project")

job = client.query(create_table_query)
job.result()  # Waits for the query to finish

print("Table `cui_ancestor_ic` created successfully.")

create_table_query = """
CREATE OR REPLACE TABLE `your_project.your_dataset.cui_ancestor_ic` AS
WITH RECURSIVE ancestors AS (
  SELECT child_cui, parent_cui
  FROM `your_project.your_dataset.mrrel_table`
  WHERE rel = 'PAR'
  UNION ALL
  SELECT a.child_cui, r.parent_cui
  FROM ancestors a
  JOIN `your_project.your_dataset.mrrel_table` r
  ON a.parent_cui = r.child_cui
  WHERE r.rel = 'PAR'
),
total_cui_count AS (
  SELECT COUNT(DISTINCT cui) AS total_cuis
  FROM `your_project.your_dataset.descendants_table`
),
ic_scores AS (
  SELECT
    a.child_cui AS cui,
    a.parent_cui AS ancestor_cui,
    d.descendant_count,
    -LOG((d.descendant_count + 1) / total.total_cuis) AS ic_score
  FROM
    ancestors a
  JOIN
    `your_project.your_dataset.descendants_table` d
  ON
    a.parent_cui = d.cui
  CROSS JOIN
    total_cui_count total
)
SELECT * FROM ic_scores
"""




Consuming this table in the over pipeline


import requests
import subprocess
from google.cloud import bigquery

# === CONFIG ===
project_id = "your-gcp-project-id"
location = "us-central1"
precomputed_table = "your_dataset.cui_ancestor_ic"
ner_api_url = "https://your-ner-api-endpoint"

# === GCP Auth Header ===
def gcp_update_header():
    tmp = subprocess.run(['gcloud', 'auth', 'print-identity-token'], stdout=subprocess.PIPE, universal_newlines=True)
    if tmp.returncode != 0:
        raise Exception("Cannot get GCP access token")
    identity_token = tmp.stdout.strip()
    return {
        "Authorization": f"Bearer {identity_token}",
        "Content-Type": "application/json"
    }

headers = gcp_update_header()
client = bigquery.Client(project=project_id)

# === Step 1: Call NER API ===
def call_ner_api(text):
    payload = {"query_texts": [text], "top_k": 3}
    try:
        response = requests.post(ner_api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return list(set(data))  # API returns a flat list of CUIs
    except Exception as e:
        print(f"NER API error: {e}")
        return []

# === Step 2: Filter CUIs Above Median IC from Precomputed Table ===
def get_above_median_ancestors(cui_list):
    cui_str = ",".join(f"'{c}'" for c in cui_list)
    query = f"""
    WITH filtered AS (
      SELECT * FROM `{precomputed_table}`
      WHERE cui IN ({cui_str})
    ),
    median_calc AS (
      SELECT cui, APPROX_QUANTILES(ic_score, 2)[OFFSET(1)] AS median_ic
      FROM filtered
      GROUP BY cui
    )
    SELECT f.cui, f.ancestor_cui
    FROM filtered f
    JOIN median_calc m
    ON f.cui = m.cui
    WHERE f.ic_score > m.median_ic
    """
    result = client.query(query).result()
    return [(row.cui, row.ancestor_cui) for row in result]

# === Step 3: Main Pipeline ===
def process_query(text):
    cuis = call_ner_api(text)
    filtered_pairs = get_above_median_ancestors(cuis)
    final_cuis = set(cuis)  # include original CUIs
    final_cuis.update([ancestor for _, ancestor in filtered_pairs])
    return list(final_cuis)

# === Example Usage ===
if __name__ == "__main__":
    input_text = "Patient has chest pain and shortness of breath"
    result_cuis = process_query(input_text)
    print("Final CUIs above median IC:", result_cuis)
