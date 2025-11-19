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
# User Query History Store
# ------------------------------
user_query_history = {}  # user_id -> list of past queries
MAX_HISTORY = 15         # max number of past queries to consider


# ------------------------------
# Robust Expansion Function
# ------------------------------
def expand_query(user_id: str, new_query: str, max_history=MAX_HISTORY):
    """
    Expands a new query based on the user's past queries.
    Automatically updates the user's history.
    """

    # Fetch user history or initialize
    past_queries = user_query_history.get(user_id, [])

    # Limit history to last N queries
    past_queries_to_use = past_queries[-max_history:]

    # ---- STEP 1: ANALYZE PAST QUERIES ----
    analysis_prompt = f"""
SYSTEM:
Analyze a user's past queries to detect patterns, reasoning progression, and style.

PAST QUERIES:
{past_queries_to_use}

TASK:
Return JSON with:

1. "query_patterns": recurring topics/directions
2. "reasoning_progression": evolution of questioning
3. "user_style": phrasing style (concise/verbose, shorthand/full terms)

RULES:
- Return ONLY JSON
- Do NOT include patient-specific data or clinical facts
"""
    try:
        analysis_raw = model.generate_content(analysis_prompt).text
        analysis = json.loads(analysis_raw)
    except Exception:
        # Fallback in case of parsing issues
        analysis = {
            "query_patterns": [],
            "reasoning_progression": [],
            "user_style": "concise"
        }

    # ---- STEP 2: EXPAND NEW QUERY ----
    expand_prompt = f"""
SYSTEM:
You are a query expansion assistant.

Use ONLY:
- User's query patterns
- Reasoning progression
- Observed style

RULES:
- Do NOT add patient-specific data or clinical facts
- Expand ONLY via clarification, organization, continuity
- Match user's style

--- INPUTS ---
NEW QUERY:
{new_query}

EXTRACTED QUERY PATTERNS:
{analysis.get('query_patterns')}

USER REASONING PROGRESSION:
{analysis.get('reasoning_progression')}

USER STYLE:
{analysis.get('user_style')}

TASK:
Return ONLY the expanded query consistent with user's past queries.
"""

    expanded_query = model.generate_content(expand_prompt).text

    # ---- STEP 3: Update user history ----
    user_query_history.setdefault(user_id, []).append(new_query)

    return expanded_query


# ------------------------------
#  Example Usage
# ------------------------------
if __name__ == "__main__":
    user_id = "user_123"

    # Simulate past queries
    user_query_history[user_id] = [
        "Show hemoglobin trends over last 6 months.",
        "Review iron studies and ferritin.",
        "Highlight abnormal lab values from last visit."
    ]

    new_query = "What additional tests should I consider?"

    expanded = expand_query(user_id, new_query)
    print("\nExpanded Query:\n", expanded)

    # Check updated history
    print("\nUpdated User History:\n", user_query_history[user_id])
