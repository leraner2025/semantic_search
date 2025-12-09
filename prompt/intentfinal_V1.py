import json
import re
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel


QUERY_EXPANSION_PROMPT = """
You are an expert medical AI assistant specializing in clinical query expansion.

TASK: Expand the user's query into a comprehensive, detailed clinical description.

INSTRUCTIONS:
1. Expand ALL medical abbreviations to full terms (e.g., HTN → Hypertension, DM → Diabetes Mellitus, SOB → Shortness of Breath)
2. Clarify vague medical terms with specific clinical language
3. Add relevant medical context based on standard clinical practice
4. Identify implicit clinical concepts that should be explicit
5. DO NOT add assumptions beyond reasonable clinical interpretation
6. DO NOT include action verbs like "analyze", "review", "check" unless in original query
7. DO NOT hallucinate information not implied by the query
8. Maintain the original query's intent and scope

EXAMPLES:
- "Pt with DM" → "Patient with Diabetes Mellitus"
- "Check vitals" → "Vital signs measurement including blood pressure, heart rate, temperature, respiratory rate, oxygen saturation"
- "Family hx of heart disease" → "Cardiovascular disease in family including coronary artery disease, myocardial infarction, heart failure"
- "SOB on exertion" → "Shortness of breath on exertion"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "expanded_query": "comprehensive expanded clinical description",
  "abbreviations_expanded": ["list of abbreviations that were expanded"],
  "concepts_added": ["list of medical concepts that were made explicit"]
}}

User Input: {query}
"""

INTENT_EXTRACTION_PROMPT = """
You are an expert clinical intent extraction engine specialized in medical information retrieval and taxonomic classification.

FIRST: Verify this is a CLINICAL query. If the query is NOT related to healthcare, medical conditions, symptoms, treatments, or clinical information, return:
{
  "is_clinical": false,
  "reason": "Query is not clinical in nature",
  "intents": []
}

Examples of NON-CLINICAL queries to reject:
- Weather, sports, entertainment, general knowledge
- Technical/IT support unrelated to medical systems
- Travel, food, shopping (unless medically relevant)

If the query IS clinical, proceed.

TASK: Extract ALL clinical intents from the expanded query and generate specific, granular queries optimized for CUI extraction.

CRITICAL: EVERY intent MUST have a populated final_queries array (minimum 5 queries).

--------------------------------------------
      INTENT STRUCTURE & TAXONOMY RULES
--------------------------------------------

Each intent must have:

1. intent_title  
2. description (explain clinically why this intent exists)
3. nature (primary clinical characteristic)
4. sub_nature (multi-level hierarchical classification)
5. final_queries (highly specific, concise, atomic, CUI-friendly)

--------------------------------------------
         SUB-NATURE HIERARCHY RULES
--------------------------------------------

Each sub_nature requires **deep multi-level decomposition**, typically 3–5 levels:

Level 1: Nature  
Level 2: Broad clinical domain  
Level 3: More specific domain  
Level 4: Narrow clinical category  
Level 5: Atomic elements (deepest level)

Allowed structure:

{
  "category": "Level 2 >> Level 3 >> Level 4",
  "elements": ["atomic level items"],
  "entities": ["medical entity types"]
}

Multiple sub_nature entries must cover different dimensions (anatomical, temporal, severity, qualitative, contextual, functional, etc.).

--------------------------------------------
        CLASSIFICATION EXAMPLES (kept concise)
--------------------------------------------

Genetic / family history breakdown includes:
- Disease domain → specific disease categories → variants  
- Family relationship → degree of relation → specific relation  
- Disease impact → complication type → organ involvement → specific complications  

Symptom breakdown includes dimensions:
- Anatomical (location → body region → specific site)  
- Sensory (quality category → descriptors)  
- Temporal (onset → duration → pattern)  
- Severity (scale → level)  
- Modifying factors (trigger/relief/associated factors)

Diagnostic data:
- Investigation type → clinical domain → test category → specific components

--------------------------------------------
            NATURE DETERMINATION RULES
--------------------------------------------

Nature describes the **primary characteristic** of the clinical information need.

To determine nature, analyze:

1. When (past, present, future context)
2. Why (clinical purpose)
3. How central this information is
4. Where it originates (subjective, objective, record-based, measured)

Nature requirements:
- Must be clinically meaningful
- Must accurately reflect the primary purpose
- Must be broad enough to include all sub-natures
- Must be specific enough to distinguish from other intents

--------------------------------------------
            MULTIPLE INTENTS HANDLING
--------------------------------------------

Separate intents if:
- They represent different clinical information needs  
- They have different purposes (e.g., current symptom vs. family history)

Keep unified only if one part is contextual to the other.

--------------------------------------------
        FINAL QUERY GENERATION RULES
--------------------------------------------

Final queries MUST come ONLY from **atomic elements** (deepest level).  
DO NOT generate from intermediate levels.

Queries must be:
- Concise (2–5 words)  
- Specific (maps to 2–15 CUIs)  
- Clinically relevant  
- Derived from atomic elements  
- Non-redundant  
- Searchable in clinical ontologies

Examples of correct formatting:
- “Type 2 Diabetes in father”
- “Chest pain substernal”
- “Heart rate resting”
- “LDL cholesterol”

Remove unnecessary words:
- “history of”, “assessment”, “level”, “associated with”, articles (the/a/an)
- “patient”, “user”, “subject”

--------------------------------------------
        QUERY COMBINATION STRATEGY
--------------------------------------------

Generate combinations only when:
- Clinically meaningful  
- Commonly documented  
- Diagnostic or relevant  

Avoid exhaustive permutations.

Typical number of queries:
- Simple intents: 5–15  
- Moderate: 15–25  
- Complex: 25–40  

Priority rules:
1. First-degree relatives > extended  
2. Common presentations > rare variants  
3. Standard assessments > exhaustive lists  
4. Documented characteristics > theoretical ones  

--------------------------------------------
        VALIDATION BEFORE RETURNING
--------------------------------------------

Verify:
- Clinical nature evaluated
- Every intent has final_queries (≥5)
- Queries come from atomic elements
- Queries are concise and clinically meaningful
- JSON is valid

--------------------------------------------
                  OUTPUT FORMAT
--------------------------------------------
IMPORTANT CHANGE: **Do NOT return original_query**
Return ONLY valid JSON:

{
  "is_clinical": true,
  "reason": "",
  "original_query": "{original_query}",
  "expanded_query": "{expanded_query}",
  "total_intents_detected": number,
  "intents": [
    {
      "intent_title": "string",
      "description": "string",
      "nature": "string",
      "sub_nature": [
        {
          "category": "string",
          "elements": ["..."],
          "entities": ["..."]
        }
      ],
      "final_queries": [
        "query 1",
        "query 2",
        "query 3",
        "query 4",
        "query 5"
      ]
    }
  ]
}

User Input: {expanded_query}
Timestamp: {timestamp}
"""

# Pipeline


class ContextualIntentPipeline:
    def __init__(self, project: str, location: str = "us-central1",
                 model: str = MODEL_VERSION,
                 enable_refinement: bool = False):

        aiplatform.init(project=project, location=location)
        self.model = GenerativeModel(model)
        self.expansion_cache: Dict[str, Dict] = {}
        self.intent_cache: Dict[str, Dict] = {}

    def _call_model(self, prompt: str, temperature: float = 1, max_tokens: int = 4096) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                    "seed": 23
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error calling model: {str(e)}")
            return "{}"

    def _safe_json(self, text: str) -> Dict[str, Any]:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return {}

    def _validate_json_structure(self, data: Dict, required_keys: List[str]) -> bool:
        return all(key in data for key in required_keys)

    # STEP 1: Query Expansion
    def expand_query(self, query: str) -> Dict[str, Any]:
        if query in self.expansion_cache:
            return self.expansion_cache[query]

        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        raw_response = self._call_model(prompt)
        data = self._safe_json(raw_response)

        if not self._validate_json_structure(data, ["expanded_query"]):
            return {
                "expanded_query": query,
                "abbreviations_expanded": [],
                "concepts_added": []
            }

        self.expansion_cache[query] = data
        return data

    # STEP 2: Intent Extraction
    def extract_intents(self, original_query: str, expanded_query: str) -> Dict[str, Any]:
        if expanded_query in self.intent_cache:
            return self.intent_cache[expanded_query]

        prompt = INTENT_EXTRACTION_PROMPT.format(
            original_query=original_query,
            expanded_query=expanded_query,
            timestamp=datetime.utcnow().isoformat()
        )

        raw_response = self._call_model(prompt, max_tokens=4096)
        data = self._safe_json(raw_response)

        if not data.get("is_clinical", True):
            return {
                "intents": [],
                "total_intents_detected": 0,
                "is_clinical": False,
                "rejected_reason": data.get("reason", "Query is not clinical"),
                "original_query": original_query,
                "expanded_query": expanded_query
            }

        intents = data.get("intents", [])
        validated_intents = [
            intent for intent in intents
            if "final_queries" in intent and intent["final_queries"]
        ]

        data["intents"] = validated_intents
        data["total_intents_detected"] = len(validated_intents)

        self.intent_cache[expanded_query] = data
        return data

    # PIPELINE EXEC
    def run(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        start_time = datetime.utcnow()

        if verbose:
            print(f"Original Query: {query}")

        expansion_result = self.expand_query(query)
        expanded_query = expansion_result["expanded_query"]

        if verbose:
            print(f"Expanded Query: {expanded_query}")

        intent_result = self.extract_intents(query, expanded_query)

        if not intent_result.get("is_clinical", True):
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            if verbose:
                print("Status: NON-CLINICAL")
                print(f"Processing Time: {processing_time:.2f}s\n")

            return {
                "original_query": query,
                "expanded_query": expanded_query,
                "intents": [],
                "is_clinical": False,
                "rejected_reason": intent_result.get("rejected_reason", "Query is not clinical"),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_seconds": processing_time
            }

        intents = intent_result.get("intents", [])

        total_queries = sum(len(i.get("final_queries", [])) for i in intents)
        total_sub_natures = sum(len(i.get("sub_nature", [])) for i in intents)
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        if verbose:
            print("Status: CLINICAL")
            print(f"Intents: {len(intents)} | Sub-natures: {total_sub_natures} | Queries: {total_queries}")
            print(f"Processing Time: {processing_time:.2f}s\n")

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "abbreviations_expanded": expansion_result.get("abbreviations_expanded", []),
            "concepts_added": expansion_result.get("concepts_added", []),
            "intents": intents,
            "is_clinical": True
        }


if __name__ == "__main__":
    pipeline = ContextualIntentPipeline(
        project=PROJECT_ID,
        location="us-central1",
        model=MODEL_VERSION
    )

    test_queries = [
        "I am searching for diagnostic test reports",
        "I am searching for pulmonary function test reports"
    ]

    for idx, query in enumerate(test_queries, start=1):
        print("\n" + "="*80)
        print(f"Query {idx}/{len(test_queries)}")
        print("="*80)

        result = pipeline.run(query, verbose=True)
        output_filename = f"query_result_{idx}.json"

        with open(output_filename, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Saved to: {output_filename}")
