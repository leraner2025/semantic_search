
from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel
import json


vertexai_init(
    project="YOUR_PROJECT_ID",  # <-- replace with your GCP project
    location="us-central1"      # <-- choose your region
)

# Load enterprise LLM
model = GenerativeModel("gemini-1.5-pro-002")


# ------------------------------
# LLM Pipeline
# ------------------------------
def expand_query_with_physician_history(new_query: str, past_queries: list):
    """
    Enterprise-ready LLM pipeline:
    STEP 1: Analyze physician's past queries for a patient.
    STEP 2: Expand the new query using:
            - Clinical patterns
            - Reasoning progression
            - Physician's style
    Rules:
    - No new clinical facts
    - No diagnoses or recommendations
    """

    # ---- STEP 1: ANALYZE PAST QUERIES ----
    analysis_prompt = f"""
SYSTEM:
You are an AI analyzing a physician’s previous queries about a SINGLE PATIENT
in a HIPAA-compliant environment.

PAST QUERIES:
{past_queries}

TASK:
Return structured JSON with:

1. "clinical_patterns": recurring medical topics or directions
2. "reasoning_progression": evolution of questioning
3. "physician_style": phrasing style (concise/verbose, shorthand/full terms)

RULES:
- Return ONLY proper JSON.
- Do NOT add clinical facts or interpret labs/symptoms.
"""

    analysis_raw = model.generate_content(analysis_prompt).text
    analysis = json.loads(analysis_raw)

    clinical_patterns = analysis["clinical_patterns"]
    reasoning_progression = analysis["reasoning_progression"]
    physician_style = analysis["physician_style"]

    # ---- STEP 2: EXPAND NEW QUERY ----
    expand_prompt = f"""
SYSTEM:
You are a clinical query expansion assistant.

Use ONLY:
- The physician’s clinical patterns
- Reasoning progression
- Observed style

RULES:
- Do NOT add new clinical findings.
- Do NOT add diagnoses, recommendations, or symptoms.
- Expand ONLY via clarification, organization, and continuity.
- Match the physician’s style.

--- INPUTS ---
NEW QUERY:
{new_query}

EXTRACTED CLINICAL PATTERNS:
{clinical_patterns}

PHYSICIAN REASONING PROGRESSION:
{reasoning_progression}

PHYSICIAN STYLE:
{physician_style}

TASK:
Return ONLY the expanded query, preserving clinical safety and style.
"""

    expanded_query = model.generate_content(expand_prompt).text
    return expanded_query


 Example Usage
# ------------------------------
if __name__ == "__main__":
    past_queries = [
        "Check last 3 hemoglobin values.",
        "Review iron studies from previous visit.",
        "Was ferritin low in past labs?",
        "Look at MCV trend over the past few months."
    ]

    new_query = "What additional tests should I check?"

    expanded = expand_query_with_physician_history(new_query, past_queries)
    print("\nExpanded Query:\n", expanded)
