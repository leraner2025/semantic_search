#shopping_adaptor.py
!pip install faiss-cpu numpy flask google-cloud-aiplatform sentence-transformers
import json
import numpy as np
import faiss
from vertexai.preview.language_models import TextGenerationModel, TextEmbeddingModel

class TopicAdapter:
    def __init__(self, topic_json_path: str, embedding_model_name="textembedding-gecko@001", llm_model_name="gemini-pro"):
        # Load topics
        with open(topic_json_path, "r") as f:
            self.topics = json.load(f)

        self.topic_embeddings = np.array([t["embedding"] for t in self.topics], dtype=np.float32)
        self.topic_chunks = [t["chunk"] for t in self.topics]

        # FAISS setup
        dim = self.topic_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.topic_embeddings)

        # Vertex AI models
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        self.llm = TextGenerationModel.from_pretrained(llm_model_name)

    def _embed_query(self, query: str) -> np.ndarray:
        embedding = self.embedding_model.get_embeddings([query])[0].values
        norm = np.linalg.norm(embedding) + 1e-10
        return np.array(embedding, dtype=np.float32) / norm

    def _semantic_search(self, query_embedding: np.ndarray, top_k: int = 3) -> list:
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [self.topics[i] for i in indices[0]]

    def _generate_answer(self, query: str, chunks: list) -> str:
        context = "\n".join([c["chunk"] for c in chunks])
        prompt = f"""
You are a medical assistant. Based on the following patient history, answer the question.

Patient History:
{context}

Question:
{query}

Answer:
"""
        response = self.llm.predict(prompt, temperature=0.3, max_output_tokens=300)
        return response.text.strip()

    def answer_query(self, query: str, top_k: int = 3) -> dict:
        query_embedding = self._embed_query(query)
        matched_chunks = self._semantic_search(query_embedding, top_k)
        answer = self._generate_answer(query, matched_chunks)
        return {
            "query": query,
            "matched_chunks": [c["chunk"] for c in matched_chunks],
            "answer": answer
        }
from shopping_adaptor import TopicAdapter

adapter = TopicAdapter("topics.json")
query = "What tests were done for neurological symptoms?"
result = adapter.answer_query(query)

print("🧠 Answer:", result["answer"])
print("📚 Matched Chunks:")
for chunk in result["matched_chunks"]:
    print("-", chunk)
