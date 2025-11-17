# smart_medical_hybrid_adapter.py

import json
import math
from typing import List, Dict, Any
import vertexai
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from datetime import datetime, timedelta
import re


# ----------------------------------------------------------------------
# Utility: cosine similarity
# ----------------------------------------------------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)


# ======================================================================
# HYBRID SMART ADAPTER WITH RELATIVE TEMPORAL SUPPORT
# ======================================================================
class SmartMedicalHybridAdapter:

    def __init__(self, project: str, location: str):
        vertexai.init(project=project, location=location)

        # LLMs
        self.gen_llm = TextGenerationModel.from_pretrained("gemini-pro")
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

        # Load topics
        self.topics = self._load_demo_topics()

        # Precompute topic embeddings
        self.topic_embeddings = self._embed_topics(self.topics)

    # ------------------------------------------------------------------
    # Demo topics (replace with BigQuery)
    # ------------------------------------------------------------------
    def _load_demo_topics(self):
        return [
            {"title": "MRI Results", "updated": "2024-10-02"},
            {"title": "CT Scan", "updated": "2024-09-18"},
            {"title": "Neurological Assessment", "updated": "2024-11-01"},
            {"title": "Medication History", "updated": "2024-04-20"},
            {"title": "Blood Pressure Monitoring", "updated": "2024-10-10"},
            {"title": "Surgical History", "updated": "2022-02-15"},
        ]

    # ------------------------------------------------------------------
    # EMBEDDING topics once for semantic search
    # ------------------------------------------------------------------
    def _embed_topics(self, topics):
        texts = [t["title"] for t in topics]
        emb = self.embedding_model.get_embeddings(texts)
        return [e.values for e in emb]

    # ------------------------------------------------------------------
    # STEP 1 — Expand Query
    # ------------------------------------------------------------------
    def _expand_query(self, query: str) -> str:
        prompt = f"""
Expand this medical question into a clearer and more detailed form 
WITHOUT changing its meaning.

Original:
{query}

Expanded:
"""
        response = self.gen_llm.predict(prompt=prompt, temperature=0.2)
        return response.text.strip()

    # ------------------------------------------------------------------
    # STEP 2 — Extract Intent + Entities + Temporal (with relative)
    # ------------------------------------------------------------------
    def _extract_intent_entities_temporal(self, query: str) -> Dict[str, Any]:

        prompt = f"""
Extract structured metadata from the user's question.

Required fields:

1. intent — a machine-friendly description of what the patient wants.
2. entities — medical terms mentioned.
3. temporal — normalized time expression.

Temporal rules:
- If date found → type: "point"
- If range found → type: "range"
- If relative time phrase ("last month", "past few days") → type: "relative"
- If none → type: "none"

Return JSON ONLY:

{{
  "intent": "",
  "entities": [],
  "temporal": {{
    "type": "none | point | range | relative",
    "value": ""
  }}
}}

Question:
{query}
"""
        response = self.gen_llm.predict(prompt=prompt, temperature=0.1)

        try:
            return json.loads(response.text)
        except:
            return {
                "intent": "",
                "entities": [],
                "temporal": {"type": "none", "value": ""}
            }

    # ------------------------------------------------------------------
    # Helper: Convert relative expressions → date range
    # ------------------------------------------------------------------
    def _resolve_relative_time(self, expr: str):

        today = datetime.today()

        expr = expr.lower().strip()

        # -------------------------
        # Standard expressions
        # -------------------------
        if "last week" in expr:
            return today - timedelta(days=7), today

        if "last month" in expr:
            return today - timedelta(days=30), today

        if "last year" in expr:
            return today - timedelta(days=365), today

        if "past week" in expr:
            return today - timedelta(days=7), today

        if "past month" in expr:
            return today - timedelta(days=30), today

        if "past year" in expr:
            return today - timedelta(days=365), today

        if "recent" in expr:
            return today - timedelta(days=90), today

        # -------------------------
        # Phrases like “past 3 days”
        # -------------------------
        match = re.search(r"past (\d+) days", expr)
        if match:
            n = int(match.group(1))
            return today - timedelta(days=n), today

        match = re.search(r"last (\d+) days", expr)
        if match:
            n = int(match.group(1))
            return today - timedelta(days=n), today

        # Default fallback
        return None, None

    # ------------------------------------------------------------------
    # STEP 3 — Semantic Embedding Matching
    # ------------------------------------------------------------------
    def _semantic_match_topics(self, expanded_query: str, top_k: int = 5):

        q_emb = self.embedding_model.get_embeddings([expanded_query])[0].values

        scored = []
        for topic, t_emb in zip(self.topics, self.topic_embeddings):
            score = cosine_similarity(q_emb, t_emb)
            scored.append((topic, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # STEP 4 — LLM Reranking
    # ------------------------------------------------------------------
    def _rerank_topics_llm(self, expanded_query: str, candidates: List[Dict]):

        titles = [c["title"] for c in candidates]

        prompt = f"""
Rank the following topics by relevance to answering the user query.

Return JSON list ONLY:
["topic1", "topic2", ...]

User Query:
{expanded_query}

Candidate Topics:
{titles}
"""

        response = self.gen_llm.predict(prompt=prompt, temperature=0.1)

        try:
            ranked_titles = json.loads(response.text)
        except:
            return candidates

        ranked = []
        for title in ranked_titles:
            for t in candidates:
                if t["title"] == title:
                    ranked.append(t)

        # Add any missing
        for t in candidates:
            if t not in ranked:
                ranked.append(t)

        return ranked

    # ------------------------------------------------------------------
    # STEP 5 — Temporal Filtering (handles relative)
    # ------------------------------------------------------------------
    def _apply_temporal_filter(self, topics: List[Dict], temporal: Dict[str, str]):

        t_type = temporal["type"]
        val = temporal["value"]

        if t_type == "none":
            return topics

        # Convert string date to datetime
        def to_dt(d):
            return datetime.strptime(d, "%Y-%m-%d")

        # -------------------------------
        # POINT date
        # -------------------------------
        if t_type == "point" and len(val) == 10:
            return [t for t in topics if t["updated"] == val]

        # -------------------------------
        # RANGE date
        # -------------------------------
        if t_type == "range" and "to" in val:
            start, end = val.split(" to ")
            start, end = to_dt(start), to_dt(end)

            return [
                t for t in topics
                if start <= to_dt(t["updated"]) <= end
            ]

        # -------------------------------
        # RELATIVE date (convert to range)
        # -------------------------------
        if t_type == "relative":
            start, end = self._resolve_relative_time(val)
            if start is None:
                return topics  # fallback

            return [
                t for t in topics
                if start <= to_dt(t["updated"]) <= end
            ]

        return topics

    # ------------------------------------------------------------------
    # STEP 6 — Final Answer
    # ------------------------------------------------------------------
    def _generate_answer(
        self, expanded_query: str, intent: str, entities: List[str],
        temporal: Dict[str, str], topics: List[Dict]
    ) -> str:

        context = "\n".join([f"- {t['title']} (updated {t['updated']})" for t in topics])

        prompt = f"""
You are a medical assistant AI.

Use ONLY the topics listed below to answer the user’s question.
Do NOT hallucinate missing details.

Intent: {intent}
Entities: {entities}
Temporal: {temporal}

Relevant Topics:
{context}

User Question:
{expanded_query}

Answer:
"""
        response = self.gen_llm.predict(prompt=prompt, temperature=0.2)
        return response.text.strip()

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def answer_query(self, query: str) -> Dict[str, Any]:

        # Step 1
        expanded = self._expand_query(query)

        # Step 2
        meta = self._extract_intent_entities_temporal(expanded)

        # Step 3
        sem_matches = [t for t, _ in self._semantic_match_topics(expanded)]

        # Step 4
        reranked = self._rerank_topics_llm(expanded, sem_matches)

        # Step 5
        final_topics = self._apply_temporal_filter(reranked, meta["temporal"])

        # Step 6
        answer = self._generate_answer(
            expanded, meta["intent"], meta["entities"], meta["temporal"], final_topics
        )

        return {
            "original_query": query,
            "expanded_query": expanded,
            "intent": meta["intent"],
            "entities": meta["entities"],
            "temporal": meta["temporal"],
            "matched_topics": final_topics,
            "answer": answer
        }
