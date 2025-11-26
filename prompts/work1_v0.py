import json
import re
from google import genai
from google.genai import types
from datetime import datetime

# ==================================================
# Prompts
# ==================================================

QUERY_EXPANSION_PROMPT = """
SYSTEM INSTRUCTION:
You are a highly advanced medical AI assistant specialized in **context-aware query expansion**.
Your task is to enhance the user's query, expanding medical abbreviations and inferring missing context to produce the most precise and complete **Expanded_Query**.

Current Query: {query}

Return ONLY a JSON object like:
{{
    "Expanded_Query": "string"
}}
"""

INTENT_ENTITY_PROMPT = """
SYSTEM INSTRUCTION:
You are a medical AI assistant. Your task is to:
1. Identify all intents embedded in a single query.
2. For each intent, extract relevant entities with their categories and any temporal information.
3. Preserve context and hierarchy: primary concerns first, secondary concerns next.
4. Produce a **formatted query** as a list of meaningful sentences, integrating intent descriptions and entities.

Datetime: {datetime}
Expanded Query: {expanded_query}

Output JSON Schema:

{{
  "intents": [
      {{
        "intent_name": "string",
        "description": "string"
      }}
  ],
  "entities": [
    {{
      "name": "string",
      "category": "string",
      "related_entity": "string",
      "timeframe": "string"
    }}
  ],
  "formatted_query": "string or list of sentences"
}}
"""

# ==================================================
# Smart Medical Pipeline
# ==================================================

class SmartMedicalPipeline:
    def __init__(self, project_id: str, location: str = "us-central1", model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name

    def _call_model(self, prompt: str) -> str:
        part = types.Part(text=prompt)
        contents = [types.Content(role="user", parts=[part])]
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )
        return response.text.strip()

    def _extract_json(self, text: str) -> dict:
        """Safely extract JSON and prepare formatted_query as sentence-level list."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                data = json.loads(match.group(0)) if match else {}
            except Exception:
                data = {}

        data.setdefault("intents", [])
        data.setdefault("entities", [])
        fq = data.get("formatted_query", "")

        # Keep only real entities
        filtered_entities = []
        for e in data.get("entities", []):
            if any(e.get(k) for k in ["name", "category", "related_entity"]):
                e.setdefault("timeframe", "")
                filtered_entities.append(e)
        data["entities"] = filtered_entities

        # Convert formatted_query to sentence-level list
        if isinstance(fq, str):
            sentences = [s.strip() for s in re.split(r'\.|\n|;', fq) if s.strip()]
            # Prepend intent info for hierarchy
            hierarchical_sentences = []
            for intent in data.get("intents", []):
                hierarchical_sentences.append(f"Intent: {intent.get('intent_name','')} - {intent.get('description','')}")
            hierarchical_sentences.extend(sentences)
            data["formatted_query"] = hierarchical_sentences
        elif isinstance(fq, list):
            data["formatted_query"] = fq
        else:
            data["formatted_query"] = []

        return data

    def expand_query(self, query: str) -> str:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        output = self._call_model(prompt)
        try:
            data = json.loads(output)
            expanded_query = data.get("Expanded_Query", query)
        except:
            expanded_query = query
        return expanded_query

    def extract_intents_entities(self, expanded_query: str) -> dict:
        prompt = INTENT_ENTITY_PROMPT.format(
            datetime=datetime.utcnow().isoformat(),
            expanded_query=expanded_query
        )
        output = self._call_model(prompt)
        data = self._extract_json(output)
        return data

    def run_pipeline(self, query: str) -> dict:
        expanded_query = self.expand_query(query)
        extraction = self.extract_intents_entities(expanded_query)

        # Print step-by-step outputs
        print("\n=== Expanded Query ===")
        print(expanded_query)
        print("\n=== Intents & Entities ===")
        print("Intents:", extraction["intents"])
        print("Entities:", extraction["entities"])
        print("Formatted Query (sentence-level list):", extraction["formatted_query"])

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "intents": extraction["intents"],
            "entities": extraction["entities"],
            "formatted_query": extraction["formatted_query"]
        }

# ==================================================
# Example Usage
# ==================================================

if __name__ == "__main__":
    PROJECT_ID = PROJECT_ID  # <-- Replace with your GCP project ID
    pipeline = SmartMedicalPipeline(project_id=PROJECT_ID, location="us-central1")

    # Query 1
    query1 = "get bp and labs for last 6 months"
    result1 = pipeline.run_pipeline(query1)

    # Query 2
    query2 = ("Patient reports severe chest pain since yesterday and difficulty breathing. "
              "Also mentions occasional mild headaches and wants to ask about a small rash on the arm.")
    result2 = pipeline.run_pipeline(query2)
