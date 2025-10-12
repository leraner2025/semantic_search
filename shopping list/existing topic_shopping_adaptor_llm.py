#save the below code till usabaility as shopping_adaptor.py,which could be then used as adaptor 
!pip install faiss-cpu numpy google-cloud-storage google-cloud-aiplatform

shopping_adaptor
import json
import numpy as np
import faiss
from vertexai.preview.language_models import TextGenerationModel, TextEmbeddingModel
from google.cloud import storage

# === Load Topics from GCS ===
def load_json_from_gcs(bucket_name, file_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = blob.download_as_text()
    return json.loads(content)

# === Embed Query using Gemini Text Embedding 001 ===
def embed_query(query, model):
    embedding = model.get_embeddings([query])[0].values
    norm = np.linalg.norm(embedding) + 1e-10
    return np.array(embedding, dtype=np.float32) / norm

# === Semantic Search using FAISS ===
def semantic_search(query_embedding, topic_embeddings, topics, top_k=3):
    dim = topic_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(topic_embeddings)
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [topics[i] for i in indices[0]]

# === Extract Intent and Entities using Gemini Pro ===
def extract_intent_entities(query, llm):
    prompt = f"""
You are a medical assistant. Analyze the following question and extract:
1. Intent (e.g., get_test_results, get_admission_date, get_medication_history)
2. Entities (e.g., MRI, warfarin, liver enzymes)

Question: {query}

Respond in JSON format:
{{"intent": "...", "entities": ["..."]}}
"""
    response = llm.predict(prompt, temperature=0.2, max_output_tokens=200)
    try:
        return json.loads(response.text)
    except:
        return {"intent": "", "entities": []}

# === Generate Answer using Gemini Pro ===
def generate_answer(query, intent, entities, matched_chunks, llm):
    context = "\n".join([c["chunk"] for c in matched_chunks])
    prompt = f"""
You are a medical assistant. Based on the following patient history, answer the question.

Intent: {intent}
Entities: {', '.join(entities)}

Patient History:
{context}

Question:
{query}

Answer:
"""
    response = llm.predict(prompt, temperature=0.3, max_output_tokens=300)
    return response.text.strip()

# === Main Execution ===
if __name__ == "__main__":
    # === GCS Config ===
    BUCKET_NAME = "your-bucket-name"
    FILE_PATH = "path/to/topics.json"

    # === Load Topics ===
    topics = load_json_from_gcs(BUCKET_NAME, FILE_PATH)
    topic_embeddings = np.array([t["embedding"] for t in topics], dtype=np.float32)

    # === Initialize Models ===
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    llm = TextGenerationModel.from_pretrained("gemini-pro")

    # === Get User Query ===
    query = input("Enter your medical question: ")

    # === Embed + Search + Extract + Answer ===
    query_embedding = embed_query(query, embedding_model)
    matched_chunks = semantic_search(query_embedding, topic_embeddings, topics)
    meta = extract_intent_entities(query, llm)
    answer = generate_answer(query, meta["intent"], meta["entities"], matched_chunks, llm)

    # === Output ===
    print("\n--- Answer ---")
    print(answer)









# using in the main pipeline as adaptor
from shopping_adaptor import MedicalAdapter

adapter = MedicalAdapter(bucket_name="your-bucket-name", file_path="path/to/topics.json")
response = adapter.answer_query("When was the last MRI done?")

print("Answer:", response["answer"])
print("Intent:", response["intent"])
print("Entities:", response["entities"])
