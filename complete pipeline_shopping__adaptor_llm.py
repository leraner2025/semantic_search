#shopping_list_adaptor.py
!pip install faiss-cpu numpy torch nltk sentence-transformers bertopic hdbscan umap-learn gensim google-cloud-aiplatform

import os, random, numpy as np, torch, nltk, logging, re, json
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from umap import UMAP
import faiss
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from vertexai.preview.language_models import TextGenerationModel

# Setup
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
nltk.download("punkt")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_entity_contexts(chunks, entities_per_chunk, use_multi_sentence=True):
    entity_context_pairs = []
    for idx, ents in enumerate(entities_per_chunk):
        chunk = clean_text(chunks[idx])
        sentences = sent_tokenize(chunk)
        for ent in ents:
            ent_lower = ent.lower()
            matched = False
            for i, sent in enumerate(sentences):
                if ent_lower in sent:
                    context = " ".join(sentences[max(0, i - 1): i + 2]) if use_multi_sentence else sent.strip()
                    enriched = f"The concept '{ent_lower}' appears in the following context: {context}"
                    entity_context_pairs.append((ent_lower, enriched.strip()))
                    matched = True
                    break
            if not matched:
                fallback = f"The concept '{ent_lower}' appears in the following context: {chunk}"
                entity_context_pairs.append((ent_lower, fallback.strip()))
    return entity_context_pairs

class TopicAdapter:
    def __init__(self, chunks, entities_per_chunk, model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.chunks = chunks
        self.entities_per_chunk = entities_per_chunk
        self.embedding_model = SentenceTransformer(model_name)
        self.llm = TextGenerationModel.from_pretrained("gemini-pro")
        self.topic_metadata = []
        self.topic_embeddings = None
        self.search_index = None
        self._run_grid_search_and_build()

    def _run_grid_search_and_build(self):
        umap_model = UMAP(n_neighbors=5, n_components=3, min_dist=0.1, metric="cosine", random_state=SEED)
        hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean", prediction_data=True)

        entity_context_pairs = extract_entity_contexts(self.chunks, self.entities_per_chunk)
        contextual_texts = [ctx for _, ctx in entity_context_pairs]
        contextual_embeddings = self.embedding_model.encode(contextual_texts, normalize_embeddings=False)

        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=KeyBERTInspired(),
            calculate_probabilities=True,
            verbose=False,
        )

        topics, _ = topic_model.fit_transform(contextual_texts, embeddings=contextual_embeddings)

        topic_to_entities = defaultdict(set)
        topic_to_embeddings = defaultdict(list)

        for i, topic in enumerate(topics):
            if topic == -1:
                continue
            ent, _ = entity_context_pairs[i]
            topic_to_entities[topic].add(ent)
            topic_to_embeddings[topic].append(contextual_embeddings[i])

        topic_metadata = []
        for topic_id in topic_to_entities:
            emb = topic_to_embeddings[topic_id]
            if len(emb) == 0:
                continue
            centroid = np.mean(emb, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-10
            topic_metadata.append({
                "topic_id": topic_id,
                "entities": list(topic_to_entities[topic_id]),
                "embedding": centroid.astype(np.float32)
            })

        self.topic_metadata = topic_metadata
        self.topic_embeddings = np.array([m["embedding"] for m in topic_metadata], dtype=np.float32)

        dim = self.topic_embeddings.shape[1]
        self.search_index = faiss.IndexFlatIP(dim)
        self.search_index.add(self.topic_embeddings)

    def _generate_llm_summary(self, query: str, matched_topics: list) -> str:
        topic_descriptions = [
            f"Topic {topic['topic_id']} includes entities: {', '.join(topic['entities'])}"
            for topic in matched_topics
        ]
        context = "\n".join(topic_descriptions)

        prompt = f"""
You are a medical assistant. Based on the following topics, answer the question.

Topics:
{context}

Question:
{query}

Answer:
"""
        response = self.llm.predict(prompt, temperature=0.3, max_output_tokens=300)
        return response.text.strip()

    def answer_query(self, query: str, top_k: int = 3) -> dict:
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        query_emb /= np.linalg.norm(query_emb) + 1e-10
        query_emb = query_emb.astype(np.float32).reshape(1, -1)

        scores, indices = self.search_index.search(query_emb, top_k)
        matched_topics = []
        for i, idx in enumerate(indices[0]):
            score = float(scores[0][i])
            if score < -1e+10:
                continue
            meta = self.topic_metadata[idx]
            matched_topics.append({
                "topic_id": meta["topic_id"],
                "score": score,
                "entities": meta["entities"]
            })

        answer = self._generate_llm_summary(query, matched_topics)

        return {
            "query": query,
            "matched_topics": matched_topics,
            "answer": answer
        }



#usage as the shopping_list_adaptor
from shopping_list_adaptor import TopicAdapter

chunks = [
    "Patient was admitted with neurological symptoms and underwent an MRI scan.",
    "Blood tests showed elevated liver enzymes and signs of infection.",
    "Patient was prescribed warfarin and discharged on 2025-09-15."
]

entities_per_chunk = [
    ["MRI", "neurological symptoms"],
    ["liver enzymes", "infection"],
    ["warfarin", "discharge"]
]

adapter = TopicAdapter(chunks, entities_per_chunk)
result = adapter.answer_query("What tests were done for neurological symptoms?")

print("Answer:", result["answer"])
print("📚 Matched Topics:")
for topic in result["matched_topics"]:
    print(f"  - Topic {topic['topic_id']} (score: {topic['score']:.3f})")
