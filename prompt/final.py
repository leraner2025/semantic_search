import json
import re
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# ======================================================
# IMPORT MODULARIZED PROMPTS
# ======================================================
from prompt.query_expansion import QUERY_EXPANSION_PROMPT
from prompt.intent_extraction import INTENT_EXTRACTION_PROMPT
from prompt.query_refinement import QUERY_REFINEMENT_PROMPT


class ContextualIntentPipeline:
    """
    Production-ready clinical intent extraction pipeline using pure LLM reasoning.
    No hardcoded logic - fully dynamic and adaptable to any medical query.
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
        validated_intents = []

        for idx, intent in enumerate(intents):
            if "final_queries" not in intent or not intent["final_queries"]:
                print(f"⚠️  Warning: Intent {idx + 1} '{intent.get('intent_title', 'Unknown')}' has no final_queries. Skipping.")
                continue

            validated_intents.append(intent)

        return {
            "intents": validated_intents,
            "total_intents_detected": data.get("total_intents_detected", len(validated_intents)),
            "is_clinical": True,
            "original_query": data.get("original_query", original_query),
            "expanded_query": data.get("expanded_query", expanded_query)
        }

    # ======================================================
    # STEP 3: Query Refinement (Optional)
    # ======================================================

    def refine_queries(self, intents: List[Dict]) -> Dict[str, Any]:
        if not self.enable_refinement or not intents:
            return {
                "intents": intents,
                "refinements_made": [],
                "total_queries": sum(len(intent.get('final_queries', [])) for intent in intents)
            }

        prompt = QUERY_REFINEMENT_PROMPT.format(intents_json=json.dumps(intents, indent=2))
        raw_response = self._call_model(prompt, max_tokens=8192)
        data = self._safe_json(raw_response)

        if not self._validate_json_structure(data, ["intents"]):
            return {
                "intents": intents,
                "refinements_made": ["Refinement failed - using original"],
                "total_queries": sum(len(intent.get('final_queries', [])) for intent in intents)
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

        expansion_result = self.expand_query(query)
        expanded_query = expansion_result["expanded_query"]

        if verbose:
            print(f"Expanded Query: {expanded_query}")

        intent_result = self.extract_intents(query, expanded_query)

        if not intent_result.get("is_clinical", True):
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            if verbose:
                print(f"Status: NON-CLINICAL (rejected)")
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

        if self.enable_refinement and intents:
            refinement_result = self.refine_queries(intents)
            intents = refinement_result["intents"]

        total_queries = sum(len(intent.get('final_queries', [])) for intent in intents)
        total_sub_natures = sum(len(intent.get('sub_nature', [])) for intent in intents)
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        if verbose:
            print(f"Status: CLINICAL")
            print(f"Intents: {len(intents)} | Sub-natures: {total_sub_natures} | Queries: {total_queries}")
            print(f"Processing Time: {processing_time:.2f}s\n")

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "abbreviations_expanded": expansion_result.get("abbreviations_expanded", []),
            "concepts_added": expansion_result.get("concepts_added", []),
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
    PROJECT_ID = PROJECT_ID

    pipeline = ContextualIntentPipeline(
        project=PROJECT_ID,
        location="us-central1",
        model="gemini-2.0-flash-exp",
        enable_refinement=False
    )

    test_queries = [
        "The patient has a family history of diabetes and heart disease.They also want their lab results from the past year reviewed. Additionally, they are concerned about recent episodes of fatigue and shortness of breath."
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {idx}/{len(test_queries)}")
        print(f"{'='*80}")

        result = pipeline.run(query, verbose=True)

        output_filename = f"query_result_{idx}.json"
        with open(output_filename, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Saved to: {output_filename}")
