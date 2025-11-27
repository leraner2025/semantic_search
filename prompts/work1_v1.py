import json
import re
from google import genai
from google.genai import types
from datetime import datetime

# ==================================================
# Query Expansion Prompt (unchanged except user history removed)
# ==================================================

QUERY_EXPANSION_PROMPT = """
SYSTEM INSTRUCTION:
You expand the user's query into a clearer medical question.
Expand abbreviations and implicit medical mentions but NEVER add new symptoms,
diagnoses, severities, timelines, or conditions that the user did not explicitly state.

Current Query: {query}

Return ONLY:
{{
    "Expanded_Query": "string"
}}
"""

# ==================================================
# INTENT / ENTITY EXTRACTION PROMPT
# ==================================================

INTENT_ENTITY_PROMPT = """
SYSTEM INSTRUCTION:
You are a clinical AI assistant that extracts ONLY:
1. Primary Health Concerns
2. Secondary Health Concerns
3. Medical entities explicitly stated in the text
4. A clinically formatted version of the query as a list of short sentences

STRICT SAFETY RULES (MANDATORY):
- Never add or infer medical details the user did not explicitly state.
- Do not hallucinate symptoms, severity, conditions, timelines, or impacts.
- Use the classification guidelines ONLY to categorize user-provided content.
- If a detail is not present, do NOT include it.

VALID INTENTS (ONLY these two):
- "Identify Primary Health Concerns"
- "Identify Secondary Health Concerns"

=====================================
CLASSIFICATION GUIDELINES  
(Used ONLY to classify, NEVER to generate new content)
=====================================

PRIMARY HEALTH CONCERNS INCLUDE ONLY WHAT USER EXPLICITLY STATES:
- Central / main problem
- Specific condition/diagnosis (only if user states it)
- Major symptoms (e.g., “severe chest pain”, if stated)
- Explicit reasons for visit (if stated)
- Stated locations of concern
- Explicit severity descriptors (“severe”, “intense”, etc.)
- Explicit acute onset (“started suddenly”, “since yesterday”)
- Explicit comparisons to baseline (“worse than usual”)
- Explicit functional impairments (“cannot walk”, etc.)
- Explicit worsening of chronic illness
- Explicit timelines (“past 2 days”, “since morning”)

SECONDARY HEALTH CONCERNS:
- Additional symptoms (“also mild headache”)
- Co-occurring conditions explicitly mentioned
- Relevant medical history if user states it
- Mild or background symptoms explicitly stated
- Non-primary inquiries (“I also wanted to ask about this rash”)
- Wellness questions if explicitly stated

DO NOT invent ANYTHING.  
Only classify what is literally written by the user.

=====================================
FORMATTED QUERY RULE:
The formatted_query must contain ONLY medically relevant statements explicitly expressed by the user,
rewritten as clear clinical sentences.

- No meta text (e.g., "the user wants to know", "the user is asking").
- No procedural statements.
- No classification rule text.
- If the user expresses NO patient symptoms, diagnoses, or medical details,
  formatted_query must be an empty list [].

=====================================

Expanded Query: {expanded_query}
Datetime: {datetime}

Return ONLY this JSON:

{{
  "intent": ["list of intents"],
  "entities": [
    {{
      "name": "string",
      "category": "string",
      "timeframe": "string"
    }}
  ],
  "formatted_query": ["sentence1", "sentence2", ...]
}}
"""

# ==================================================
# Smart Medical Pipeline
# ==================================================

class SmartMedicalPipeline:
    def __init__(self, project_id: str, location: str = "us-central1",
                 model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name

    def _call_model(self, prompt: str) -> str:
        part = types.Part(text=prompt)
        contents = [types.Content(role="user", parts=[part])]
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )
        return response.text.strip()

    def _extract_json(self, text: str) -> dict:
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}

        # Defaults
        data.setdefault("intent", [])
        data.setdefault("entities", [])
        data.setdefault("formatted_query", [])

        # Ensure formatted_query becomes list of sentences
        fq = data.get("formatted_query", [])
        if isinstance(fq, str):
            data["formatted_query"] = [
                s.strip() for s in re.split(r'\.|\n|;', fq) if s.strip()
            ]
        elif not isinstance(fq, list):
            data["formatted_query"] = []

        return data

    def expand_query(self, query: str) -> str:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        output = self._call_model(prompt)
        data = self._extract_json(output)
        return data.get("Expanded_Query", query)

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

        print("\n=== Expanded Query ===")
        print(expanded_query)

        print("\n=== Intents ===")
        print(extraction["intent"])

        print("\n=== Entities ===")
        print(extraction["entities"])

        print("\n=== Formatted Query ===")
        print(extraction["formatted_query"])

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "intent": extraction["intent"],
            "entities": extraction["entities"],
            "formatted_query": extraction["formatted_query"]
        }


# ==================================================
# Example Usage
# ==================================================
if __name__ == "__main__":
    PROJECT_ID = PROJECT_ID  # Replace with actual GCP project ID
    pipeline = SmartMedicalPipeline(project_id=PROJECT_ID)

    query = "I have severe chest pain since yesterday and also some mild headaches."
    result = pipeline.run_pipeline(query)
