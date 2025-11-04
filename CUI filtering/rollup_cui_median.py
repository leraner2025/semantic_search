import requests
import subprocess
from google.cloud import bigquery

# === CONFIG ===
project_id = "your-gcp-project-id"
location = "us-central1"
mrrel_table = "your_dataset.mrrel_table"
descendants_table = "your_dataset.descendants_table"
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

# === Step 2: Filter CUIs Above Median IC (Local Universe) ===
def get_filtered_cuis_above_median_local(cui):
    query = f"""
    WITH RECURSIVE ancestors AS (
      SELECT child_cui AS cui, parent_cui
      FROM `{mrrel_table}`
      WHERE child_cui = @cui AND rel = 'PAR'
      UNION ALL
      SELECT a.parent_cui AS cui, r.parent_cui
      FROM ancestors a
      JOIN `{mrrel_table}` r
      ON a.parent_cui = r.child_cui
      WHERE r.rel = 'PAR'
    ),
    descendants AS (
      SELECT parent_cui AS cui, child_cui
      FROM `{mrrel_table}`
      WHERE parent_cui = @cui AND rel = 'PAR'
      UNION ALL
      SELECT d.child_cui AS cui, r.child_cui
      FROM descendants d
      JOIN `{mrrel_table}` r
      ON d.child_cui = r.parent_cui
      WHERE r.rel = 'PAR'
    ),
    local_universe AS (
      SELECT DISTINCT parent_cui AS cui FROM ancestors
      UNION DISTINCT
      SELECT DISTINCT child_cui AS cui FROM descendants
      UNION DISTINCT
      SELECT @cui AS cui
    ),
    ic_scores AS (
      SELECT
        a.parent_cui AS cui,
        -LOG((d.descendant_count + 1) / (SELECT COUNT(*) FROM local_universe)) AS ic_score
      FROM
        ancestors a
      JOIN
        `{descendants_table}` d
      ON
        a.parent_cui = d.cui
    ),
    median_calc AS (
      SELECT APPROX_QUANTILES(ic_score, 2)[OFFSET(1)] AS median_ic FROM ic_scores
    )
    SELECT
      cui
    FROM
      ic_scores, median_calc
    WHERE
      ic_score > median_ic
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cui", "STRING", cui)
        ]
    )
    result = client.query(query, job_config=job_config).result()
    return [row.cui for row in result]

# === Step 3: Main Pipeline ===
def process_query(text):
    cuis = call_ner_api(text)
    final_cuis = set()
    for cui in cuis:
        filtered = get_filtered_cuis_above_median_local(cui)
        final_cuis.update(filtered)
    return list(final_cuis)

# === Example Usage ===
if __name__ == "__main__":
    input_text = "Patient has chest pain and shortness of breath"
    result_cuis = process_query(input_text)
    print("Final CUIs above median IC (local):", result_cuis)
