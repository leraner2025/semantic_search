import json
import re
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# ======================================================
# UPDATED — SHORTENED PROMPTS (Functionally Identical)
# ======================================================

QUERY_EXPANSION_PROMPT = """
You are an expert medical AI assistant.

TASK: Expand the user’s query into a complete, explicit clinical description.

RULES:
1. Expand all abbreviations (HTN → Hypertension, DM → Diabetes Mellitus, SOB → Shortness of Breath, etc.).
2. Clarify vague medical terms using standard clinical language.
3. Add only clinically reasonable implicit context.
4. No hallucination, no new conditions, no added actions.
5. Maintain original intent; do not broaden scope.
6. No action verbs unless present in original query.

Return ONLY JSON:
{
  "expanded_query": "...",
  "abbreviations_expanded": [...],
  "concepts_added": [...]
}

User Input: {query}
"""

INTENT_EXTRACTION_PROMPT = """
You are a clinical intent extraction engine.

STEP 1 — Determine if query is clinical.
If NOT clinical (weather, sports, entertainment, non-medical tech, travel, food), return:
{
  "is_clinical": false,
  "reason": "Query is not clinical",
  "intents": []
}

STEP 2 — If clinical, extract ALL distinct clinical intents.

Each intent must contain:
- intent_title
- description
- nature (primary characteristic)
- sub_nature: deep hierarchical taxonomy (≥3 levels)
- final_queries: ≥5 specific atomic CUI-targeting queries

SUB_NATURE FORMAT:
{
  "category": "Level2 >> Level3 >> Level4",
  "elements": [atomic concepts],
  "entities": [entity types]
}

RULES:
- Taxonomy must go: Broad Domain → Specialty → Category → Concept → Atomic elements.
- Final queries must come ONLY from atomic elements.
- Queries must be concise (2–5 words).
- Remove filler words ("history of", "associated with", articles, etc.).
- Queries must target 2–15 CUIs.
- MUST NOT return intents without final_queries.

RETURN JSON:
{
  "is_clinical": true,
  "original_query": "{original_query}",
  "expanded_query": "{expanded_query}",
  "total_intents_detected": number,
  "intents": [...]
}

User Input: {expanded_query}
Timestamp: {timestamp}
"""

QUERY_REFINEMENT_PROMPT = """
You are a medical query optimizer.

TASK: Refine each intent's final_queries for maximum conciseness and specificity.

RULES:
1. Concise: remove unnecessary words (“history of”, articles, filler).
2. Specific: maintain atomic, ontology-searchable medical terms.
3. Complete: do not delete clinically relevant dimensions.
4. No duplicates; merge queries only if identical in meaning.
5. Maintain medical accuracy.

Return ONLY JSON:
{
  "intents": [... refined intents ...],
  "refinements_made": [...],
  "total_queries": number,
  "optimization_summary": "..."
}

Input: {intents_json}
"""

# ======================================================
# PIPELINE (Unchanged)
# ======================================================

class ContextualIntentPipeline:
    """
    Production-ready clinical intent extraction pipeline using pure LLM reasoning.
    """
    
    def __init__(self, project: str, location: str = "us-central1", 
                 model: str = "gemini-2.0-flash-exp", 
                 enable_refinement: bool = False):

        aiplatform.init(project=project, location=location)
        self.model = GenerativeModel(model)
        self.enable_refinement = enable_refinement
        
    def _call_model(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        try:
            if hasattr(self.model, "session") and self.model.session is not None:
                self.model.session.reset()
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                    "top_k": 40
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
            
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            
            print(f"Failed to parse JSON from response: {text[:200]}")
            return {}
    
    def _validate_json_structure(self, data: Dict, required_keys: List[str]) -> bool:
        return all(key in data for key in required_keys)
    
    # ======================================================
    # STEP 1: Query Expansion
    # ======================================================
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        raw_response = self._call_model(prompt)
        data = self._safe_json(raw_response)
        
        if not self._validate_json_structure(data, ["expanded_query"]):
            return {
                "expanded_query": query,
                "abbreviations_expanded": [],
                "concepts_added": []
            }
        
        return {
            "expanded_query": data.get("expanded_query", query),
            "abbreviations_expanded": data.get("abbreviations_expanded", []),
            "concepts_added": data.get("concepts_added", [])
        }
    
    # ======================================================
    # STEP 2: Intent Extraction
    # ======================================================
    
    def extract_intents(self, original_query: str, expanded_query: str) -> Dict[str, Any]:
        prompt = INTENT_EXTRACTION_PROMPT.format(
            original_query=original_query,
            expanded_query=expanded_query,
            timestamp=datetime.utcnow().isoformat()
        )
        
        raw_response = self._call_model(prompt, max_tokens=2048)
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
        
        if not self._validate_json_structure(data, ["intents"]):
            return {
                "intents": [],
                "total_intents_detected": 0,
                "error": "Failed to extract intents"
            }
        
        intents = data.get("intents", [])
        validated = []
        
        for intent in intents:
            if "final_queries" not in intent or not intent["final_queries"]:
                continue
            validated.append(intent)
        
        return {
            "intents": validated,
            "total_intents_detected": data.get("total_intents_detected", len(validated)),
            "is_clinical": True,
            "original_query": data.get("original_query", original_query),
            "expanded_query": data.get("expanded_query", expanded_query)
        }
    
    # ======================================================
    # STEP 3: Refinement (unchanged)
    # ======================================================
    
    def refine_queries(self, intents: List[Dict]) -> Dict[str, Any]:
        if not self.enable_refinement or not intents:
            return {
                "intents": intents,
                "refinements_made": [],
                "total_queries": sum(len(i.get('final_queries', [])) for i in intents)
            }
        
        prompt = QUERY_REFINEMENT_PROMPT.format(intents_json=json.dumps(intents, indent=2))
        raw_response = self._call_model(prompt, max_tokens=8192)
        data = self._safe_json(raw_response)
        
        if not self._validate_json_structure(data, ["intents"]):
            return {
                "intents": intents,
                "refinements_made": ["Refinement failed - using original"],
                "total_queries": sum(len(i.get('final_queries', [])) for i in intents)
            }
        
        return {
            "intents": data.get("intents", intents),
            "refinements_made": data.get("refinements_made", []),
            "total_queries": data.get("total_queries", 0)
        }
    
    # ======================================================
    # FULL PIPELINE EXECUTION
    # ======================================================
    
    def run(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        start_time = datetime.utcnow()
        
        if verbose:
            print(f"Original Query: {query}")
        
        expansion = self.expand_query(query)
        expanded_query = expansion["expanded_query"]
        
        if verbose:
            print(f"Expanded Query: {expanded_query}")
        
        intent_result = self.extract_intents(query, expanded_query)
        
        if not intent_result.get("is_clinical", True):
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "original_query": query,
                "expanded_query": expanded_query,
                "intents": [],
                "is_clinical": False,
                "rejected_reason": intent_result.get("rejected_reason"),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_seconds": processing_time
            }
        
        intents = intent_result["intents"]
        
        if self.enable_refinement and intents:
            refinement = self.refine_queries(intents)
            intents = refinement["intents"]
        
        total_queries = sum(len(i.get('final_queries', [])) for i in intents)
        total_sub_natures = sum(len(i.get('sub_nature', [])) for i in intents)
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "abbreviations_expanded": expansion.get("abbreviations_expanded", []),
            "concepts_added": expansion.get("concepts_added", []),
            "intents": intents,
            "is_clinical": True,
            "statistics": {
                "total_intents": len(intents),
                "total_sub_natures": total_sub_natures,
                "total_queries": total_queries,
                "avg_queries_per_intent": round(total_queries / len(intents), 2) if intents else 0
            },
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    PROJECT_ID = PROJECT_ID  # Replace with your actual project ID
    
    pipeline = ContextualIntentPipeline(
        project=PROJECT_ID,
        location="us-central1",
        model="gemini-2.0-flash-exp",
        enable_refinement=False
    )
    
    test_queries = [
        "The patient has a family history of diabetes and heart disease. They want their labs from the past year reviewed. They are also concerned about fatigue and shortness of breath."
    ]
    
    for idx, query in enumerate(test_queries, 1):
        print("\n" + "="*80)
        print(f"Query {idx}")
        print("="*80)
        
        result = pipeline.run(query, verbose=True)
        
        output_filename = f"query_result_{idx}.json"
        with open(output_filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Saved to: {output_filename}")
