import json
import re
from datetime import datetime
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# ==================================================
# Prompts
# ==================================================

QUERY_EXPANSION_PROMPT = """
SYSTEM INSTRUCTION:
You are a medical AI assistant.
Expand the user's input query into a full detailed **concise clinical description**.
Clarify abbreviations and vague terms. Include relevant medical concepts, conditions, or exposures.
but do NOT add unrelated information.
Do NOT include verbs like "Document", "Analyze", or other instructional phrases.
Do not hallucinate unrelated conditions. Use standard clinical associations.


Use knowledge of:
- Vital signs, labs, medications, procedures
- Medical history and family history patterns
- Clinical associations (e.g., smoking → COPD, lung cancer)
- Standard ways clinicians describe requests

Current Query: {query}

EXAMPLES:

1. Input: "patient vitals last 6mo"
   Expanded: "Review and analyze the patient's vital signs, including blood pressure, heart rate, respiratory rate, temperature, and oxygen saturation, recorded over the past six months."

2. Input: "family history of smoking"
   Expanded: "Review the patient's family history of tobacco use, including health conditions potentially associated with exposure to smoking, such as COPD, lung cancer, and cardiovascular disease."

Return ONLY JSON:
{{
    "Expanded_Query": "string"
}}
"""

INTENT_ENTITY_PROMPT = """
SYSTEM INSTRUCTION:
You are a clinical AI assistant.
Identify all actionable clinical intents from the input query and expand them into a structured JSON.
For each intent, provide:
- intent_title: concise name of the clinical task
- description: short explanation of the intent
- nature: high-level category (e.g., History Review, Lab / Vital Monitoring, Symptom Assessment)
- sub_natures: finer breakdown of the nature
- entities: clinical entities mentioned or implied (diseases, conditions, organs, labs, vitals, exposures)
- requested_data: data or measurements that should be retrieved for this intent

GUIDELINES:
1. Use clinical knowledge to determine **related labs, conditions, and risks** for any term in the query.
   - Reference the types of knowledge captured in **LOINC and SNOMED CT**, but do not display codes.
   - Example: Family history of smoking → COPD, lung cancer, cardiovascular disease.
   - Example: Renal function → Creatinine, BUN, eGFR.
2. For organ/system-specific terms or labs, generate a separate intent per organ/system.
3. For family history or exposure terms, generate a History Review intent including relevant diseases or risks as entities.
4. For symptoms, generate a Symptom Assessment intent including potential relevant conditions as entities.
5. Include all possible intents in the query. Do not ignore any relevant information.
6. Do not hallucinate unrelated conditions or labs. Only include those implied by the query or medically associated.
7. JSON format only. No extra text.

Expanded Query: {expanded_query}
Datetime: {datetime}

EXAMPLES:
1. Input: "family history of smoking"
Output:
{{
  "intents": [
    {{
      "intent_title": "Retrieve Family History of Smoking",
      "description": "Analyze the patient's family history of tobacco use, including associated health conditions and exposures.",
      "nature": "History Review",
      "sub_natures": ["Family History", "Health Risks"],
      "entities": ["COPD", "Lung Cancer", "Cardiovascular Disease"],
      "requested_data": [
        "Chronic obstructive pulmonary disease (COPD)",
        "Lung Cancer",
        "Cardiovascular Disease Risk Factors"
      ]
    }}
  ]
}}

2. Input: "renal function and kidney labs over past year"
Output:
{{
  "intents": [
    {{
      "intent_title": "Review Renal Function and Kidney Health",
      "description": "Assess the patient's renal function tests over the past year to monitor kidney health.",
      "nature": "Lab / Vital Monitoring",
      "sub_natures": ["Renal Function", "Kidney Health"],
      "entities": ["Creatinine", "BUN", "eGFR", "Renal Function"],
      "requested_data": [
        "Serum Creatinine",
        "Blood Urea Nitrogen (BUN)",
        "Estimated Glomerular Filtration Rate (eGFR)"
      ]
    }}
  ]
}}
"""

# ==================================================
# Smart Medical Pipeline (Vertex AI)
# ==================================================

class SmartMedicalPipeline:
    def __init__(self, project: str, location: str = "us-central1", model_name="gemini-2.0-flash"):
        # Initialize Vertex AI
        aiplatform.init(project=project, location=location)
        self.model = GenerativeModel(model_name)

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
            # fallback: find first JSON object in text
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
        if "intents" not in data:
            data["intents"] = []
        return data

    def run_pipeline(self, query: str) -> dict:
        expanded = self.expand_query(query)
        extracted = self.extract_intents_entities(expanded)

        print("\n=== Expanded Query ===")
        print(expanded)
        print("\n=== Extracted Intents ===")
        print(json.dumps(extracted, indent=2))

        return {
            "original_query": query,
            "expanded_query": expanded,
            "intents": extracted.get("intents", [])
        }

# ==================================================
# Example usage
# ==================================================

PROJECT_ID = PROJECT_ID

pipeline = SmartMedicalPipeline(project=PROJECT_ID)

query = "medical history of food allergy"

result = pipeline.run_pipeline(query)
