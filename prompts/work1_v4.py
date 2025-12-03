import json
import re
from datetime import datetime, timezone
from google.cloud import aiplatform

# ==================================================
# PROMPTS
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

INTENT_BREAKDOWN_PROMPT_STRICT = """
SYSTEM INSTRUCTION:
You are a medical AI assistant that produces structured clinical interpretations.
STRICTLY ONLY use content directly relevant to the expanded query.
Do NOT default to any topic unless the query explicitly mentions it.

EXPANDED QUERY:
{expanded_query}

DATETIME:
{datetime}

INSTRUCTIONS:

1. PRIMARY INTENT:
   Identify the single clearest clinical intent matching the query. 
   Do not invent unrelated categories.

2. EXPANDED INTENT:
   Provide a concise clinical scope description.
   Include only categories clearly suggested by the query.

3. SUB-COMPONENTS:
   List meaningful sub-components relevant to the query.
   Only include examples if explicitly relevant.
   Do NOT add default examples like allergies or food unless the query mentions them.

4. FORMATTED_QUERY:
   Bullet-style list of sub-components.
   Do not add unrelated topics.

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
      "bullet2"
  ]
}}
"""

# ==================================================
# PIPELINE CLASS FOR VERTEX AI
# ==================================================

class SmartMedicalPipelineVertex:
    def __init__(self, project: str, location: str, model: str = "text-bison@001"):
        self.project = project
        self.location = location
        self.model = model
        aiplatform.init(project=project, location=location)
        self.client = aiplatform.gapic.PredictionServiceClient()
        self.endpoint = None  # Optional: if you deploy a custom endpoint

    def _call_model(self, prompt: str) -> str:
        # Using the default generative text model
        response = aiplatform.TextGenerationModel.from_pretrained(self.model).predict(
            prompt,
            max_output_tokens=1024,
            temperature=0.0
        )
        return response.text

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except:
            # Fallback: extract largest JSON object
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
        canonical = {k.lower().replace("_", "").replace(" ", ""): v for k, v in data.items()}
        for key in ["expandedquery", "expanded_query", "expanded query"]:
            k = key.replace("_", "").replace(" ", "")
            if k in canonical:
                return canonical[k]
        return query

    def breakdown_intent(self, expanded_query: str) -> dict:
        prompt = INTENT_BREAKDOWN_PROMPT_STRICT.format(
            expanded_query=expanded_query,
            datetime=datetime.now(timezone.utc).isoformat()
        )
        raw = self._call_model(prompt)
        data = self._extract_json(raw)

        # Normalize formatted_query into list
        fq = data.get("formatted_query", [])
        if isinstance(fq, str):
            data["formatted_query"] = [s.strip() for s in re.split(r"[;\n]", fq) if s.strip()]

        # Generate bullet-style formatted query from sub-components if missing
        if "sub_components" in data and not data.get("formatted_query"):
            bullets = []
            for cat, items in data["sub_components"].items():
                if items:
                    bullets.append(f"- {cat}: {', '.join(items)}")
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
    PROJECT_ID = "your-gcp-project-id"
    LOCATION = "us-central1"
    pipeline = SmartMedicalPipelineVertex(project=PROJECT_ID, location=LOCATION)

    query = "chest pain"
    result = pipeline.run_pipeline(query)
