import json
import re
from datetime import datetime
from vertexai.preview.language_models import TextGenerationModel
import vertexai

# ==================================================
# SETUP PROJECT & LOCATION
# ==================================================
PROJECT_ID = "your-gcp-project-id"   # <-- replace with your project id
LOCATION = "us-central1"             # <-- replace with your location

vertexai.init(project=PROJECT_ID, location=LOCATION)

# ==================================================
# PROMPTS (unchanged)
# ==================================================

QUERY_EXPANSION_PROMPT = """
SYSTEM INSTRUCTION:
You are a medical AI assistant.
Expand the user's query medically, expanding abbreviations and clarifying all medical entities mentioned.
Do NOT add unrelated clinical information or speculate.

Current Query: {query}

Return ONLY JSON:
{{
    "Expanded_Query": "string"
}}
"""

INTENT_BREAKDOWN_PROMPT = """
SYSTEM INSTRUCTION:
You are a clinical reasoning assistant that interprets physician queries and transforms them
into structured clinical intents with detailed clinical sub-components.

EXPANDED QUERY:
{expanded_query}

DATETIME:
{datetime}

GOAL:
For ANY clinical topic (e.g., medications, labs, vitals, imaging, allergies, diagnoses,
medical history, surgeries, encounters, care plans, treatments, orders, symptoms, results),
produce a highly structured, clinically detailed interpretation that is clear, domain-specific,
and reflects appropriate medical sub-components.

REQUIREMENTS:

1. PRIMARY INTENT:
   Identify the single clearest clinical intent the query is seeking.

2. EXPANDED INTENT:
   Provide a concise, high-clarity restatement of the intent that describes what full
   clinical scope it covers. This expanded intent should briefly reference the categories of
   information relevant to the intent, but must NOT add fictional patient details.

3. SUB-COMPONENTS:
   Break the expanded intent into clinically meaningful categories and sub-components.
   These must reflect real medical documentation structures and clinical concepts.
   Do NOT add any patient-specific values. Only structural categories.

4. FORMATTED_QUERY:
   Produce a bullet-style list:
     - Each main bullet is a sub-component category.
     - Include the sub-items in parentheses or after a colon.
     - Avoid full sentence framing.
     - Do NOT hallucinate patient-specific info.

OUTPUT FORMAT:
Return ONLY JSON:

{{
  "primary_intent": "string",
  "expanded_intent": "string",
  "sub_components": {{
      "CategoryName1": ["item1", "item2"],
      "CategoryName2": ["item1", "item2"]
  }},
  "formatted_query": [
      "bullet1",
      "bullet2",
      "bullet3"
  ]
}}
"""

# ==================================================
# SMART MEDICAL PIPELINE USING GEMINI-FLASH
# ==================================================

class SmartMedicalPipeline:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = TextGenerationModel.from_pretrained(model_name)

    def _call_model(self, prompt: str) -> str:
        response = self.model.predict(
            prompt,
            temperature=0.0,
            max_output_tokens=4096
        )
        return response.text.strip()

    def _extract_json(self, text: str) -> dict:
        """Safely extract the largest valid JSON object from the output."""
        try:
            return json.loads(text)
        except:
            pass

        start_index = None
        brace_count = 0
        for i, ch in enumerate(text):
            if ch == "{":
                if start_index is None:
                    start_index = i
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0 and start_index is not None:
                    candidate = text[start_index:i+1]
                    try:
                        return json.loads(candidate)
                    except:
                        pass
        return {}

    def expand_query(self, query: str) -> str:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        raw = self._call_model(prompt)
        data = self._extract_json(raw)

        # Normalize keys
        canonical = {k.lower().replace("_", "").replace(" ", ""): v for k,v in data.items()}
        for key in ["expandedquery", "expanded_query", "expanded query"]:
            k = key.replace("_", "").replace(" ", "")
            if k in canonical:
                return canonical[k]

        return query

    def breakdown_intent(self, expanded_query: str) -> dict:
        prompt = INTENT_BREAKDOWN_PROMPT.format(
            expanded_query=expanded_query,
            datetime=datetime.utcnow().isoformat()
        )
        raw = self._call_model(prompt)
        data = self._extract_json(raw)

        # Normalize formatted_query into bullet-style list
        fq = data.get("formatted_query", [])
        if isinstance(fq, str):
            data["formatted_query"] = [s.strip() for s in re.split(r"[;\n]", fq) if s.strip()]

        # Generate bullet-style formatted query from sub-components if not provided
        if "sub_components" in data and not data.get("formatted_query"):
            bullets = []
            for cat, items in data["sub_components"].items():
                if items:
                    bullet = f"- {cat}: {', '.join(items)}"
                    bullets.append(bullet)
            data["formatted_query"] = bullets

        return data

    def run_pipeline(self, query: str) -> dict:
        expanded = self.expand_query(query)
        breakdown = self.breakdown_intent(expanded)

        print("\n=== Expanded Query ===")
        print(expanded)

        print("\n=== Primary Intent ===")
        print(breakdown.get("primary_intent"))

        print("\n=== Expanded Intent ===")
        print(breakdown.get("expanded_intent"))

        print("\n=== Sub-Components ===")
        print(json.dumps(breakdown.get("sub_components"), indent=2))

        print("\n=== Formatted Query ===")
        for s in breakdown.get("formatted_query", []):
            print(s)

        return {
            "original_query": query,
            "expanded_query": expanded,
            "primary_intent": breakdown.get("primary_intent"),
            "expanded_intent": breakdown.get("expanded_intent"),
            "sub_components": breakdown.get("sub_components"),
            "formatted_query": breakdown.get("formatted_query")
        }

# ==================================================
# EXAMPLE USAGE
# ==================================================

if __name__ == "__main__":
    pipeline = SmartMedicalPipeline()

    query = "medical history including current medications"
    result = pipeline.run_pipeline(query)
