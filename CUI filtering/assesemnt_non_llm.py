# ----------------------------
# Install Required Packages
# ----------------------------
!pip install --upgrade google-cloud-aiplatform google-cloud-bigquery scikit-learn pandas

# ----------------------------
# Imports
# ----------------------------
from vertexai.language_models import TextEmbeddingModel
from google.cloud import bigquery
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
PROJECT_ID = "your-gcp-project-id"  # 🔁 Replace with your GCP project ID
LOCATION = "us-central1"            # 🔁 Replace with your region
QUERY_TEXT = "MRI of head"
BIGQUERY_TABLE = "your_dataset.cui_embeddings"  # 🔁 Replace with your BigQuery table path
CUI_ID_COLUMN = "cui"
CUI_NAME_COLUMN = "name"
CUI_EMBEDDING_COLUMN = "embedding"

# 🔁 Replace with your target CUI list
TARGET_CUIS = [
    "C0407663", "C1540653", "C0203763", "C0993821", "C0411858"
]

# ----------------------------
# Initialize Vertex AI
# ----------------------------
from vertexai import init
init(project=PROJECT_ID, location=LOCATION)

# ----------------------------
# Generate Query Embedding
# ----------------------------
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
query_embedding = embedding_model.get_embeddings([QUERY_TEXT])[0].values
query_vec = np.array(query_embedding)

# ----------------------------
# Load Filtered CUIs from BigQuery
# ----------------------------
client = bigquery.Client(project=PROJECT_ID)

# Format CUI list for SQL
cui_list_sql = ", ".join([f"'{cui}'" for cui in TARGET_CUIS])

query = f"""
SELECT
  {CUI_ID_COLUMN} AS cui,
  {CUI_NAME_COLUMN} AS name,
  {CUI_EMBEDDING_COLUMN} AS embedding
FROM `{BIGQUERY_TABLE}`
WHERE {CUI_ID_COLUMN} IN ({cui_list_sql})
"""
df = client.query(query).to_dataframe()

# ----------------------------
# Prepare Embedding Arrays
# ----------------------------
def parse_embedding(embedding):
    return np.array(embedding)

df["embedding"] = df["embedding"].apply(parse_embedding)
cui_vecs = np.vstack(df["embedding"].values)

# ----------------------------
# Compute Cosine Similarities
# ----------------------------
similarities = cosine_similarity([query_vec], cui_vecs)[0]
df["similarity"] = similarities

# ----------------------------
# Unified Coverage Score
# ----------------------------
coverage_score = round(np.mean(similarities), 2)

# ----------------------------
# Unified Explanation
# ----------------------------
explanation = f"""
🧠 Unified Coverage Evaluation for Query: '{QUERY_TEXT}'

Intent:
- This query expresses a diagnostic procedure (MRI) targeting the head region (brain, skull, cranial vessels).

Evaluation:
- CUIs evaluated: {len(df)}
- Average semantic similarity between query and CUIs: {coverage_score:.2f}

Justification:
- The CUIs collectively reflect the core intent of the query if the average similarity is high.
- A score above 0.75 suggests strong semantic alignment.
- A score between 0.6–0.75 suggests partial alignment, possibly missing specificity.
- A score below 0.6 suggests weak justification — CUIs may be off-topic or fragmented.

Conclusion:
- Based on embedding similarity, the CUIs {'strongly' if coverage_score > 0.75 else 'partially' if coverage_score > 0.6 else 'do not'} justify the user query.
"""

print(explanation)
