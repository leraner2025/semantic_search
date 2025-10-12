#assuming we already have the topics json data 
!pip install faiss-cpu numpy flask google-cloud-aiplatform sentence-transformers
import json
import numpy as np
import faiss
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import TextEmbeddingModel

# === Load Topics from JSON ===
with open("topics.json", "r") as f:
    topics = json.load(f)

topic_embeddings = np.array([t["embedding"] for t in topics], dtype=np.float32)
topic_chunks = [t["chunk"] for t in topics]

# === FAISS Setup ===
dim = topic_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(topic_embeddings)

# === Vertex AI Models ===
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
llm = TextGenerationModel.from_pretrained("gemini-pro")

# === Embed Query ===
def embed_query(query):
    embedding = embedding_model.get_embeddings([query])[0].values
    norm = np.linalg.norm(embedding) + 1e-10
    return np.array(embedding, dtype=np.float32) / norm

# === Semantic Search ===
def semantic_search(query_embedding, top_k=3):
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [topics[i] for i in indices[0]]

# === Intent + Entity Extraction ===
def extract_intent_entities(query):
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

# === Answer Generation ===
def generate_answer(query, intent, entities, chunks):
    context = "\n".join([c["chunk"] for c in chunks])
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

# === Run Locally ===
if __name__ == "__main__":
    query = input("Enter your medical question: ")
    query_embedding = embed_query(query)
    matched_chunks = semantic_search(query_embedding)
    meta = extract_intent_entities(query)
    answer = generate_answer(query, meta["intent"], meta["entities"], matched_chunks)

    print("\n--- Answer ---")
    print(answer)
