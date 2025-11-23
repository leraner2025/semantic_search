
#pipeline.py (Unified Pipeline Call)
#✔ Imports all prompts  
#✔ Calls medical LLM once per step  
#✔ Produces **final unified output**

from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel
import json

from query_expansion_prompt import query_expansion_prompt
from intent_entity_prompt import intent_entity_prompt
from summarization_prompt import summarization_prompt

vertexai_init(project="YOUR_PROJECT_ID", location="us-central1")
model = GenerativeModel("gcp-medical-llm-Pro")  # or gemini-medical-1.0

user_query_history = {}
MAX_HISTORY = 20

def medical_pipeline(user_id: str, query: str):

    # -------------------------------------------
    # STEP 1 — Query Expansion
    # -------------------------------------------
    past = user_query_history.get(user_id, [])[-MAX_HISTORY:]
    filled_prompt_expansion = query_expansion_prompt.format(
        user_id=user_id,
        previous_queries=past,
        query=query
    )

    expansion_raw = model.generate_content(filled_prompt_expansion).text
    expansion = json.loads(expansion_raw)
    expanded_query = expansion["Expanded_Query"]

    # Update history
    user_query_history.setdefault(user_id, []).append(query)

    # -------------------------------------------
    # STEP 2 — Intent + Entity Extraction
    # -------------------------------------------
    from datetime import datetime
    filled_ie_prompt = intent_entity_prompt.format(
        datetime=datetime.utcnow().isoformat(),
        expanded_query=expanded_query
    )

    ie_raw = model.generate_content(filled_ie_prompt).text
    ie_data = json.loads(ie_raw)
    formatted_query = ie_data["formatted_query"]

    # -------------------------------------------
    # STEP 3 — Summarization
    # -------------------------------------------
    filled_summary_prompt = summarization_prompt.format(
        formatted_query=formatted_query
    )

    summary_raw = model.generate_content(filled_summary_prompt).text
    summary_data = json.loads(summary_raw)

    # -------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------
    return {
        "input_query": query,
        "expanded_query": expanded_query,
        "intent": ie_data["intent"],
        "entities": ie_data["entities"],
        "formatted_query": formatted_query,
        "summary": summary_data["summary"]
    }


# Example execution
if __name__ == "__main__":
    output = medical_pipeline("user42", "bp and labs?")
    print(json.dumps(output, indent=2))
