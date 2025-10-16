# STEP 1: Install required packages
!pip install google-cloud-bigquery google-cloud-aiplatform pandas scikit-learn matplotlib openpyxl

# STEP 2: Import libraries
from google.cloud import bigquery
from google.cloud import aiplatform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# STEP 3: Set up GCP project and location
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# STEP 4: Load CUI data from BigQuery (including embeddings)
QUERY = """
SELECT Parent_CUI, Child_CUI, Description, Embedding
FROM `your_dataset.your_table`
WHERE Description IS NOT NULL AND Embedding IS NOT NULL
"""
bq_client = bigquery.Client(project=PROJECT_ID)
df = bq_client.query(QUERY).to_dataframe()

# Convert embedding string to list if needed
if isinstance(df['Embedding'].iloc[0], str):
    import ast
    df['Embedding'] = df['Embedding'].apply(ast.literal_eval)

# STEP 5: Embed the query using Gemini
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

def get_embeddings(texts):
    model = aiplatform.TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)
    response = model.get_embeddings(texts)
    return [r.values for r in response]

query = "MRI"
query_embedding = get_embeddings([query])[0]
query_vec = np.array(query_embedding).reshape(1, -1)

# STEP 6: Apply initial similarity threshold
cui_matrix = np.array(df['Embedding'].tolist())
similarities = cosine_similarity(query_vec, cui_matrix)[0]
df['Similarity'] = similarities
filtered_df = df[df['Similarity'] >= 0.50].reset_index(drop=True)

print(f"CUIs passing initial threshold (≥ 0.50): {len(filtered_df)}")

# STEP 7: Leave-One-Out Refinement with Impact Display
def leave_one_out_filter(df, query_vec):
    embeddings = np.array(df['Embedding'].tolist())
    full_centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    full_score = cosine_similarity(query_vec, full_centroid)[0][0]

    print(f"Full Coverage Score with filtered CUIs: {full_score:.4f}")

    impacts = []
    retained_indices = []

    for i in range(len(embeddings)):
        reduced = np.delete(embeddings, i, axis=0)
        centroid = np.mean(reduced, axis=0).reshape(1, -1)
        score = cosine_similarity(query_vec, centroid)[0][0]
        delta = round(full_score - score, 4)
        impacts.append({
            'Index': i,
            'Child_CUI': df.iloc[i]['Child_CUI'],
            'Parent_CUI': df.iloc[i]['Parent_CUI'],
            'Description': df.iloc[i]['Description'],
            'Coverage Drop': delta
        })
        if delta > 0:
            retained_indices.append(i)

    impact_df = pd.DataFrame(impacts)
    print("Leave-One-Out Impact Table:")
    print(impact_df)

    # Plot impact
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(impacts)), [row['Coverage Drop'] for row in impacts], color='salmon')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Leave-One-Out Impact on Coverage")
    plt.xlabel("CUI Index")
    plt.ylabel("Coverage Drop When Removed")
    plt.grid(True)
    plt.show()

    refined_df = df.iloc[retained_indices].reset_index(drop=True)
    refined_score = cosine_similarity(query_vec, np.mean(np.array(refined_df['Embedding'].tolist()), axis=0).reshape(1, -1))[0][0]
    return refined_df, refined_score

refined_df, refined_score = leave_one_out_filter(filtered_df, query_vec)
refined_parent_cuis = refined_df['Parent_CUI'].unique().tolist()
print(f"Refined CUIs: {len(refined_df)}")
print(f"Coverage After Refinement: {refined_score:.4f}")

# STEP 8: Show final selected CUIs as DataFrame
print("\nFinal Selected CUIs:")
print(refined_df[['Parent_CUI', 'Child_CUI', 'Description']])

# STEP 9: Export final Parent CUIs to Excel
output_df = pd.DataFrame({'Parent_CUI': refined_parent_cuis})
output_df.to_excel("final_selected_parent_cuis.xlsx", index=False)
print("Saved to final_selected_parent_cuis.xlsx")
