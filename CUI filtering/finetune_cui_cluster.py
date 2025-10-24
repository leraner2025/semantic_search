import subprocess
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
import scann
from datetime import datetime

# Initialize Vertex AI embedding model
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

# NER API
def call_ner_api(text, url):
    payload = {"query_texts": [text], "top_k": 3}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        cuis = data.get(text, [])
        print(f"NER API call successful. CUIs found: {len(cuis)}")
        return cuis
    except Exception as e:
        print(f"NER API error: {e}")
        return []

# Get embeddings from BigQuery
def get_cui_embeddings(client, project_id, dataset, embedding_table, cuis):
    if not cuis:
        return {}
    query = f"""
        SELECT CUI, Embedding
        FROM `{project_id}.{dataset}.{embedding_table}`
        WHERE CUI IN UNNEST(@cuis)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("cuis", "STRING", cuis)]
    )
    results = client.query(query, job_config=job_config).result()
    embeddings = {row.CUI: row.Embedding for row in results}
    print(f"Retrieved embeddings for {len(embeddings)} CUIs.")
    return embeddings

# ScaNN clustering
def cluster_embeddings(embeddings, config):
    if len(embeddings) < 2:
        return [0] * len(embeddings)
    normed_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    searcher = scann.scann_ops_pybind.builder(normed_embeddings, config["num_neighbors"], "dot_product") \
        .tree(config["num_leaves"], config["num_leaves_to_search"], config["training_sample_size"]) \
        .score_ah(config["quantization_bins"], anisotropic_quantization_threshold=config["quantization_threshold"]) \
        .reorder(config["reorder"]) \
        .build()
    clusters = [-1] * len(embeddings)
    cluster_id = 0
    for i in range(len(embeddings)):
        if clusters[i] != -1:
            continue
        neighbors, distances = searcher.search_batched(normed_embeddings[i:i+1], final_num_neighbors=config["num_neighbors"])
        neighbors = neighbors[0]
        distances = distances[0]
        for n, dist in zip(neighbors[1:], distances[1:]):
            if dist >= 1 - config["distance_threshold"] and clusters[n] == -1:
                clusters[n] = cluster_id
        clusters[i] = cluster_id
        cluster_id += 1
    return clusters

# Redundancy filtering
def pick_most_informative_cuis(cluster_cuis, cluster_embeddings, similarity_threshold=0.9):
    sim_matrix = cosine_similarity(cluster_embeddings)
    n = len(cluster_cuis)
    if n == 1:
        return cluster_cuis, list(cluster_embeddings), [], []
    avg_similarities = sim_matrix.sum(axis=1) / (n - 1)
    to_remove = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= similarity_threshold:
                if avg_similarities[i] >= avg_similarities[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    final_indices = [i for i in range(n) if i not in to_remove]
    final_cuis = [cluster_cuis[i] for i in final_indices]
    final_embeddings = [cluster_embeddings[i] for i in final_indices]
    redundant_indices = [i for i in range(n) if i in to_remove]
    redundant_cuis = [cluster_cuis[i] for i in redundant_indices]
    redundant_embeddings = [cluster_embeddings[i] for i in redundant_indices]
    return final_cuis, final_embeddings, redundant_cuis, redundant_embeddings

# CUI selection
def select_representatives(cui_list, embeddings, text_embedding=None, config=None,
                           similarity_threshold=0.9, min_similarity_to_text=0.3, min_secondary_text_sim=0.25):
    labels = cluster_embeddings(embeddings, config)
    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append((cui_list[i], embeddings[i]))
    final_cuis_with_flags = []
    for cluster in cluster_map.values():
        if not cluster:
            continue
        cluster_cuis = [cui for cui, _ in cluster]
        cluster_embs = np.array([embedding for _, embedding in cluster])
        primary_cuis, primary_embs, redundant_cuis, redundant_embs = pick_most_informative_cuis(
            cluster_cuis, cluster_embs, similarity_threshold
        )
        kept_primaries = []
        for cui, emb in zip(primary_cuis, primary_embs):
            if text_embedding is not None:
                sim = cosine_similarity([emb], [text_embedding])[0][0]
                if sim >= min_similarity_to_text:
                    kept_primaries.append((cui, emb))
                    final_cuis_with_flags.append((cui, "primary"))
            else:
                kept_primaries.append((cui, emb))
                final_cuis_with_flags.append((cui, "primary"))
        for cui, emb in zip(redundant_cuis, redundant_embs):
            if text_embedding is not None:
                sim_to_text = cosine_similarity([emb], [text_embedding])[0][0]
                if sim_to_text >= min_secondary_text_sim:
                    too_similar = False
                    for _, sel_emb in kept_primaries:
                        sim_to_sel = cosine_similarity([emb], [sel_emb])[0][0]
                        if sim_to_sel >= similarity_threshold:
                            too_similar = True
                            break
                    if not too_similar:
                        final_cuis_with_flags.append((cui, "secondary"))
            else:
                final_cuis_with_flags.append((cui, "secondary"))
    return final_cuis_with_flags, len(set(labels))

# Run pipeline with config
def run_pipeline_with_config(text, project_id, dataset, embedding_table, ner_url, config):
    client = bigquery.Client()
    ner_cuis = call_ner_api(text, ner_url)
    if not ner_cuis:
        return None
    cui_embeddings = get_cui_embeddings(client, project_id, dataset, embedding_table, ner_cuis)
    if not cui_embeddings:
        return None
    text_embedding = gemini_model.get_embeddings([text])[0].values
    cui_list = list(cui_embeddings.keys())
    embedding_matrix = np.array([cui_embeddings[cui] for cui in cui_list])
    selected_cuis_with_flags, cluster_count = select_representatives(
        cui_list, embedding_matrix, text_embedding, config
    )
    selected_embeddings = [emb for _, emb in selected_cuis_with_flags]
    if selected_embeddings:
        centroid = np.mean(selected_embeddings, axis=0).reshape(1, -1)
        query_vector = np.array(text_embedding).reshape(1, -1)
        coverage_score = cosine_similarity(query_vector, centroid)[0][0]
    else:
        coverage_score = 0.0
    return {
        "config": config,
        "coverage_score": round(coverage_score, 4),
        "cluster_count": cluster_count,
        "selected_cuis": len(selected_cuis_with_flags),
        "selected_cuis_with_flags": selected_cuis_with_flags
    }
# Tuning loop
def run_tuning_loop(text, project_id, dataset, embedding_table, ner_url):
    scann_configs = [
        {
            "name": "High Accuracy",
            "num_neighbors": 20,
            "num_leaves": 300,
            "num_leaves_to_search": 100,
            "training_sample_size": 25000,
            "quantization_bins": 4,
            "quantization_threshold": 0.1,
            "reorder": 200,
            "distance_threshold": 0.2
        },
{
            "name": "Balanced",
            "num_neighbors": 10,
            "num_leaves": 200,
            "num_leaves_to_search": 50,
            "training_sample_size": 25000,
            "quantization_bins": 2,
            "quantization_threshold": 0.2,
            "reorder": 100,
            "distance_threshold": 0.2
        },
        {
            "name": "Fast",
            "num_neighbors": 5,
            "num_leaves": 100,
            "num_leaves_to_search": 30,
            "training_sample_size": 25000,
            "quantization_bins": 2,
            "quantization_threshold": 0.3,
            "reorder": 50,
            "distance_threshold": 0.2
        }
    ]

    results = []
    for config in scann_configs:
        print(f"\nRunning config: {config['name']}")
        result = run_pipeline_with_config(text, project_id, dataset, embedding_table, ner_url, config)
        if result:
            results.append(result)
            print(f"Coverage Score: {result['coverage_score']}, Clusters: {result['cluster_count']}, CUIs: {result['selected_cuis']}")
        else:
            print("Pipeline failed for this config.")

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"scann_tuning_results_{timestamp}.csv", index=False)
    print("\nTuning complete. Results saved to CSV.")
    print(df)

    # Automatically pick best config
    best_result = max(results, key=lambda x: x["coverage_score"])
    print(f"\n✅ Best config: {best_result['config']['name']} with score {best_result['coverage_score']}")
    return best_result["config"], best_result["selected_cuis_with_flags"]

if __name__ == "__main__":
    # Assign your GCP details here
    project_id = "your_project_id"
    dataset = "your_dataset"
    embedding_table = "your_embedding_table"
    ner_url = "your_ner_api_url"

    text = "MRI of head"

    # Step 1: Run tuning and get best config
    best_config, final_cuis_with_flags = run_tuning_loop(text, project_id, dataset, embedding_table, ner_url)

    # Step 2: Save final CUIs
    df_final = pd.DataFrame(final_cuis_with_flags, columns=["CUI", "Type"])
    df_final.to_csv("final_cuis_best_config.csv", index=False)
    print("\n🎯 Final CUIs selected using best config:")
    print(df_final)
