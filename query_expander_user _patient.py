#user level and broken at patient level

from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel
import json

# ------------------------------
# Vertex AI Initialization
# ------------------------------
vertexai_init(
    project="YOUR_PROJECT_ID",  
    location="us-central1"
)

model = GenerativeModel("gemini-1.5-pro-002")

# ------------------------------
# Query History Storage
# ------------------------------
# Structure: user_id -> patient_id -> list of past queries
query_history = {}

MAX_HISTORY = 15  # use last 15 queries per patient for context

# ------------------------------
# Query Expansion Function
# ------------------------------
def expand_query_user_patient(user_id: str, patient_id: str, new_query: str):
    """
    Expands a new query based on the user's past queries for a specific patient.
    
    Rules:
    - Only use past queries for this user & patient
    - Do NOT add new clinical facts or recommendations
    - Maintain user's style and reasoning progression
    """
    # Fetch past queries for this user + patient
    past_queries = query_history.get(user_id, {}).get(patient_id, [])[-MAX_HISTORY:]

    # ---- STEP 1: Summarize past queries ----
    summary_prompt = f"""
SYSTEM:
You are an AI summarizing a user's past queries for a SINGLE PATIENT
into structured insights.

PAST QUERIES:
{past_queries}

TASK:
Return JSON with:
1. "query_topics": main topics explored
2. "reasoning_flow": logical progression of questions
3. "writing_style": phrasing style (concise/verbose, shorthand/full terms)

RULES:
- Return ONLY JSON.
- Do NOT add clinical facts.
"""
    try:
        summary_raw = model.generate_content(summary_prompt).text
        summary = json.loads(summary_raw)
    except Exception:
        summary = {
            "query_topics": [],
            "reasoning_flow": [],
            "writing_style": "concise"
        }

    # ---- STEP 2: Expand the new query ----
    expand_prompt = f"""
SYSTEM:
You are a query expansion assistant.

TASK:
Expand the NEW query using ONLY:
- query_topics
- reasoning_flow
- writing_style

STRICT RULES:
- Do NOT add new clinical findings
- Do NOT add diagnoses, recommendations, or unrelated topics
- Only reorganize, clarify, or extend the query
- Preserve user's style and reasoning progression

--- INPUTS ---
NEW QUERY:
{new_query}

USER INSIGHTS:
{summary}

Return ONLY the expanded query.
"""
    expanded_query = model.generate_content(expand_prompt).text

    # ---- STEP 3: Update query history ----
    query_history.setdefault(user_id, {}).setdefault(patient_id, []).append(new_query)

    return expanded_query

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    user_id = "user_123"
    patient_id = "patient_456"

    # Seed history
    query_history[user_id] = {
        patient_id: [
            "Check last 3 hemoglobin values.",
            "Review iron studies from previous visit.",
            "Was ferritin low in past labs?",
            "Look at MCV trend over the past few months."
        ]
    }

    new_query = "What additional labs should I consider?"

    expanded = expand_query_user_patient(user_id, patient_id, new_query)
    print("\nExpanded Query:\n", expanded)
    print("\nUpdated Query History:\n", query_history[user_id][patient_id])
