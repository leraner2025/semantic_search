# smart_medical_adapter.py
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"

import json
from vertexai.language_models import TextGenerationModel
import vertexai

class SmartMedicalAdapter:
    def __init__(self, project: str, location: str):
        vertexai.init(project=project, location=location)
        self.llm = TextGenerationModel.from_pretrained("gemini-pro")
        self.topics = self._load_demo_topics()

    def _load_demo_topics(self):
        return [
            {"title": "MRI Results"},
            {"title": "CT Scan"},
            {"title": "Liver Function"},
            {"title": "Kidney Function"},
            {"title": "Blood Pressure Monitoring"},
            {"title": "Heart Rate Analysis"},
            {"title": "Medication History"},
            {"title": "Allergy Profile"},
            {"title": "Surgical History"},
            {"title": "Admission Details"},
            {"title": "Discharge Summary"},
            {"title": "Neurological Assessment"},
            {"title": "Cognitive Function"},
            {"title": "Respiratory Evaluation"},
            {"title": "Diabetes Management"},
            {"title": "Thyroid Function"},
            {"title": "Cholesterol Levels"},
            {"title": "Immunization Records"},
            {"title": "Family Medical History"},
            {"title": "Genetic Risk Factors"},
            {"title": "Cancer Screening"},
            {"title": "Bone Density"},
            {"title": "Vision Test"},
            {"title": "Hearing Test"},
            {"title": "Mental Health Assessment"},
            {"title": "Sleep Study"},
            {"title": "Physical Therapy Notes"},
            {"title": "Nutrition Assessment"},
            {"title": "Substance Use History"},
            {"title": "Follow-up Appointments"}
        ]

    def _expand_query(self, query: str) -> str:
        prompt = f"""
You are a medical assistant. Expand the following question to make it more detailed and specific.

Original Question:
{query}

Expanded Question:
"""
        response = self.llm.predict(prompt=prompt, temperature=0.3, max_output_tokens=200)
        return response.text.strip()

    def _extract_intent_entities(self, query: str) -> dict:
        prompt = f"""
You are a medical assistant. Analyze the following question and extract:
1. Intent (e.g., get_test_results, get_admission_date, get_medication_history)
2. Entities (e.g., MRI, warfarin, liver enzymes)

Question: {query}

Respond in JSON format:
{{"intent": "...", "entities": ["..."]}}
"""
        response = self.llm.predict(prompt=prompt, temperature=0.2, max_output_tokens=200)
        try:
            return json.loads(response.text)
        except:
            return {"intent": "", "entities": []}

    def _filter_topics(self, entities: list) -> list:
        if not entities:
            return self.topics
        return [
            topic for topic in self.topics
            if any(entity.lower() in topic["title"].lower() for entity in entities)
        ]

    def _generate_answer(self, query: str, intent: str, entities: list, matched_topics: list) -> str:
        context = "\n".join([t["title"] for t in matched_topics])
        prompt = f"""
You are a medical assistant. Based on the following patient topics, answer the question.

Intent: {intent}
Entities: {', '.join(entities)}

Patient Topics:
{context}

Question:
{query}

Answer:
"""
        response = self.llm.predict(prompt=prompt, temperature=0.3, max_output_tokens=400)
        return response.text.strip()

    def answer_query(self, query: str) -> dict:
        expanded_query = self._expand_query(query)
        meta = self._extract_intent_entities(expanded_query)
        matched_topics = self._filter_topics(meta["entities"])
        answer = self._generate_answer(expanded_query, meta["intent"], meta["entities"], matched_topics)
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "intent": meta["intent"],
            "entities": meta["entities"],
            "matched_topics": [t["title"] for t in matched_topics],
            "answer": answer
        }

adapter = SmartMedicalAdapter(project=PROJECT_ID, location=LOCATION)
response = adapter.answer_query("What tests were done for neurological symptoms?")
print("📚 Matched Topics:", response["matched_topics"])
print("✅ Answer:", response["answer"])
