# STEP 1: Install required packages
!pip install pandas scikit-learn sentence-transformers openpyxl matplotlib

# STEP 3: Load and parse Excel
import pandas as pd
import ast
df = pd.read_excel("cui_with_embeddings.xlsx")
# Ensure required columns
required_cols = ['Parent_CUI', 'Child_CUI', 'Description']
assert all(col in df.columns for col in required_cols), "Missing required columns"

# STEP 4: Generate embeddings for descriptions
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
df['Embedding'] = df['Description'].apply(lambda x: model.encode(x).tolist())

# STEP 5: Embed the query
query = "MRI"
query_embedding = model.encode([query])[0]
query_vec = np.array(query_embedding).reshape(1, -1)

# STEP 6: Apply initial similarity threshold
from sklearn.metrics.pairwise import cosine_similarity

cui_matrix = np.array(df['Embedding'].tolist())
similarities = cosine_similarity(query_vec, cui_matrix)[0]
df['Similarity'] = similarities
filtered_df = df[df['Similarity'] >= 0.50].reset_index(drop=True)

print(f"\n🔍 CUIs passing initial threshold (≥ 0.50): {len(filtered_df)}")

# STEP 7: Leave-One-Out Refinement with Impact Display
import matplotlib.pyplot as plt

def leave_one_out_filter(df, query_vec):
    embeddings = np.array(df['Embedding'].tolist())
    full_centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    full_score = cosine_similarity(query_vec, full_centroid)[0][0]

    print(f"\nFull Coverage Score with filtered CUIs: {full_score:.4f}\n")

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
    print("📊 Leave-One-Out Impact Table:")
    display(impact_df)

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
print(f"\n Refined CUIs: {len(refined_df)} → Coverage After Refinement: {refined_score:.4f}")

# STEP 8: Export final Parent CUIs
output_df = pd.DataFrame({'Parent_CUI': refined_parent_cuis})
output_df.to_excel("refined_parent_cuis_0.5.xlsx", index=False)
files.download("refined_parent_cuis_0.5.xlsx")
