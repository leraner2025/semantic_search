import json
import re
from datetime import datetime
from google.cloud import aiplatform
import google.generativeai as genai

# ==================================================
# Prompts
# ==================================================

QUERY_EXPANSION_PROMPT = """
SYSTEM INSTRUCTION:
You are a medical AI assistant.
Expand the user's query medically, expanding abbreviations and clarifying all medical entities mentioned.
Do NOT add unrelated medical information or speculate.

Current Query: {query}

Return ONLY JSON:
{{
    "Expanded_Query": "string"
}}
"""

INTENT_ENTITY_PROMPT = """
You are a clinical AI assistant.

Expanded Query: {expanded_query}
Datetime: {datetime}

TASK:
1. Identify the primary medical intent(s) of the query. Examples include:
   RetrieveLabResults, RetrieveVitalSigns, RetrieveImaging, RetrieveMedications, RetrieveProcedures, RetrieveAllergies, RetrieveDiagnoses
2. Extract entities explicitly mentioned in the query (labs, vitals, imaging, procedures, medications, timeframe).
3. Produce a formatted query:
   - SINGLE string if only one sentence/intent.
   - LIST of strings if multiple sentences/intents.
   - The formatted query should clearly combine intent + entities + timeframe in a readable clinical style.

STRICT SAFETY RULES (MANDATORY):
- Never add or infer medical details the user did not explicitly state.
- Do not hallucinate symptoms, severity, conditions, timelines, or impacts.
- Use the classification guidelines ONLY to categorize user-provided content.
- If a detail is not present, do NOT include it.

RULES:
- Do NOT hallucinate or add extra medical info.
- Always return strictly JSON with fields: intent, entities, formatted_query
- Entities should be objects: {{ "name": "...", "category": "...", "timeframe": "..." }}

Return JSON ONLY:
{{
  "intent": "string OR list",
  "entities": [
    {{
      "name": "string",
      "category": "string",
      "timeframe": "string"
    }}
  ],
  "formatted_query": "string OR list"
}}
"""

# ==================================================
# Smart Medical Pipeline (Vertex AI)
# ==================================================

class SmartMedicalPipeline:
    def __init__(self, project: str, location: str = "us-central1", model_name="gemini-2.0-flash"):
        # Initialize Vertex AI
        aiplatform.init(project=project, location=location)
        # Configure generative AI for Vertex AI environment
        genai.configure(project=project, location=location)
        self.model = genai.GenerativeModel(model_name)

    def _call_model(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2048
            }
        )
        return response.text.strip()

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    return {}
        return {}

    def expand_query(self, query: str) -> str:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        out = self._call_model(prompt)
        data = self._extract_json(out)
        return data.get("Expanded_Query", query)

    def extract_intents_entities(self, expanded_query: str) -> dict:
        prompt = INTENT_ENTITY_PROMPT.format(
            expanded_query=expanded_query,
            datetime=datetime.utcnow().isoformat()
        )
        out = self._call_model(prompt)
        data = self._extract_json(out)

        # formatted_query single vs multi-sentence logic
        fq = data.get("formatted_query", "")
        if isinstance(fq, str):
            if "." in fq or "\n" in fq:
                sentences = [s.strip() for s in re.split(r"\.|\n|;", fq) if s.strip()]
                data["formatted_query"] = sentences if len(sentences) > 1 else fq
        return data

    def run_pipeline(self, query: str) -> dict:
        expanded = self.expand_query(query)
        extracted = self.extract_intents_entities(expanded)

        print("\n=== Expanded Query ===")
        print(expanded)
        print("\n=== Intent(s) ===")
        print(extracted.get("intent"))
        print("\n=== Entities ===")
        print(extracted.get("entities"))
        print("\n=== Formatted Query ===")
        print(extracted.get("formatted_query"))

        return {
            "original_query": query,
            "expanded_query": expanded,
            "intent": extracted.get("intent"),
            "entities": extracted.get("entities"),
            "formatted_query": extracted.get("formatted_query"),
        }

# ==================================================
# Example usage
# ==================================================

PROJECT_ID = "your-gcp-project-id"
pipeline = SmartMedicalPipeline(project=PROJECT_ID)

query = """Patient vital signs over the past 6 months.
Also pull CBC, CMP, and TSH results during the same interval.
"""

result = pipeline.run_pipeline(query)
