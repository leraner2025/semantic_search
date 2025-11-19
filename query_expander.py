from vertexai.generative_models import GenerativeModel
import json

model = GenerativeModel("gemini-1.5-pro")


def expand_query_with_physician_history(new_query: str, past_queries: list):
    """
    Full LLM-only pipeline:
    STEP 1: Analyze physician's past queries for this patient
            → extract patterns, reasoning progression, style
    STEP 2: Expand the new query using ONLY:
            - new query
            - extracted insights
            - without adding ANY new clinical facts
    """
    
    # ---- STEP 1: ANALYZE PAST QUERIES ----
    analysis_prompt = f"""
SYSTEM:
You are an AI model analyzing a physician’s previous queries about a SINGLE PATIENT.

PAST QUERIES:
{past_queries}

TASK:
Analyze the past queries and return structured JSON with:

1. "clinical_patterns":
    The consistent medical topics or directions
    the physician has been exploring for THIS patient.

2. "reasoning_progression":
    How the physician’s line of questioning has evolved over time.

3. "physician_style":
    How the physician typically asks questions:
    - concise / verbose
    - direct / descriptive
    - shorthand vs full terminology
    - phrasing patterns

RULES:
- Return ONLY proper JSON.
- Do NOT add any clinical facts.
- Do NOT interpret lab values, symptoms, or conditions.
"""

    analysis_raw = model.generate_content(analysis_prompt).text
    analysis = json.loads(analysis_raw)

    clinical_patterns = analysis["clinical_patterns"]
    reasoning_progression = analysis["reasoning_progression"]
    physician_style = analysis["physician_style"]

    # ---- STEP 2: EXPAND THE NEW QUERY SAFELY ----
    expand_prompt = f"""
SYSTEM:
You are a clinical query expansion assistant.

You must expand the NEW query using ONLY:
- The physician’s clinical patterns
- Their reasoning progression
- Their observed style of writing

PAST QUERIES WERE ALREADY ANALYZED.
DO NOT use any clinical data not present in past queries or the new query.

STRICT RULES:
- NO new clinical findings.
- NO diagnoses or recommendations.
- NO invented values or symptoms.
- Only reorganize, clarify, and extend based on patterns.
- Follow the physician's style.

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
Expand the NEW query in a way consistent with the physician's style
and their ongoing line of investigation, but without adding new facts.

Return ONLY the expanded query.
"""

    expanded_query = model.generate_content(expand_prompt).text
    return expanded_query
