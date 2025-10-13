# smart_medical_adapter.py

import json
from vertexai.preview.language_models import TextGenerationModel

class SmartMedicalAdapter:
    def __init__(self):
        self.llm = TextGenerationModel.from_pretrained("gemini-pro")
        self.topics = self._load_demo_topics()

    def _load_demo_topics(self):
        return [
            {
                "title": "MRI Results",
                "description": "Patient underwent an MRI scan to investigate neurological symptoms. Results showed mild inflammation in the frontal lobe."
            },
            {
                "title": "Liver Function",
                "description": "Blood tests revealed elevated liver enzymes. Follow-up tests were scheduled to monitor liver function."
            },
            {
                "title": "Medication History",
                "description": "Patient was prescribed warfarin and lisinopril. Previous medications included aspirin and metformin."
            },
            {
                "title": "Admission Details",
                "description": "Patient was admitted on March 12, 2023, following complaints of dizziness and blurred vision."
            },
            {
                "title": "CT Scan",
                "description": "CT scan of the brain showed no abnormalities. Scan was performed after a minor head injury."
            }
        ]

    def _expand_query(self, query: str) -> str:
        prompt = f"""
You are a medical assistant. Expand the following question to make it more detailed and specific.

Original Question:
{query}

Expanded Question:
"""
        response = self.llm.predict(prompt, temperature=0.3, max_output_tokens=200)
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
        response = self.llm.predict(prompt, temperature=0.2, max_output_tokens=200)
        try:
            return json.loads(response.text)
        except:
            return {"intent": "", "entities": []}

    def _filter_topics(self, entities: list) -> list:
        if not entities:
            return self.topics
        return [
            topic for topic in self.topics
            if any(entity.lower() in topic["title"].lower() or entity.lower() in topic["description"].lower()
                   for entity in entities)
        ]

    def _generate_answer(self, query: str, intent: str, entities: list, matched_topics: list) -> str:
        context = "\n".join([f"{t['title']}: {t['description']}" for t in matched_topics])
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
        response = self.llm.predict(prompt, temperature=0.3, max_output_tokens=400)
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
            "matched_topics": matched_topics,
            "answer": answer
        }
