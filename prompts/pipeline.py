from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel
import json
from datetime import datetime
from query_expansion_prompt import query_expansion_prompt
from intent_entity_prompt import intent_entity_prompt
from summarization_prompt import summarization_prompt

# Initialize Vertex AI
vertexai_init(project="YOUR_PROJECT_ID", location="us-central1")
model = GenerativeModel("gcp-medical-llm-Pro")

# User query history
user_query_history = {}
MAX_HISTORY = 20

def medical_pipeline(user_id: str, query: str):
    """Full pipeline execution with JSON file output."""

    # -----------------------------
    # STEP 1 — Query Expansion
    # -----------------------------
    past_queries = user_query_history.get(user_id, [])[-MAX_HISTORY:]

    filled_expansion_prompt = query_expansion_prompt.format(
        user_id=user_id,
        previous_queries=past_queries,
        query=query
    )
    expansion_raw = model.generate_content(filled_expansion_prompt).text
    expansion_data = json.loads(expansion_raw)
    expanded_query = expansion_data["Expanded_Query"]

    user_query_history.setdefault(user_id, []).append(query)

    # -----------------------------
    # STEP 2 — Intent + Entity Extraction
    # -----------------------------
    filled_intent_entity_prompt = intent_entity_prompt.format(
        datetime=datetime.utcnow().isoformat(),
        expanded_query=expanded_query
    )
    ie_raw = model.generate_content(filled_intent_entity_prompt).text
    ie_data = json.loads(ie_raw)

    formatted_query = ie_data.get("formatted_query", expanded_query)

    # -----------------------------
    # STEP 3 — Summarization
    # -----------------------------
    filled_summary_prompt = summarization_prompt.format(
        formatted_query=formatted_query
    )
    summary_raw = model.generate_content(filled_summary_prompt).text
    summary_data = json.loads(summary_raw)

    # -----------------------------
    # FINAL JSON OUTPUT
    # -----------------------------
    output_json = {
        "input_query": query,
        "expanded_query": expanded_query,
        "intent": ie_data.get("intent"),
        "entities": ie_data.get("entities"),
        "formatted_query": formatted_query,
        "summary": summary_data.get("summary"),

        # 🔥 Code 2 will read this
        "cui_processing_input": {
            "text_for_ner": formatted_query
        }
    }

    # 🔥 Save JSON for code 2
    with open("pipeline_output.json", "w") as f:
        json.dump(output_json, f, indent=2)

    return output_json


# -----------------------------
# Example execution
# -----------------------------
if __name__ == "__main__":
    result = medical_pipeline("user42", "bp and labs?")
    print(json.dumps(result, indent=2))
    print("\nSaved as pipeline_output.json\n")
