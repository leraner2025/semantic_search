# ----------------------------
# Install and Import Packages
# ----------------------------
!pip install --upgrade google-cloud-aiplatform

from vertexai import init
from vertexai.preview.generative_models import GenerativeModel
import pandas as pd
import time

# ----------------------------
# GCP Enterprise Configuration
# ----------------------------
PROJECT_ID = "your-gcp-project-id"       # Replace with your GCP project ID
LOCATION = "us-central1"                 # Replace with your region (e.g., us-central1, europe-west4)

init(project=PROJECT_ID, location=LOCATION)

# ----------------------------
# Query and CUIs
# ----------------------------
QUERY = "MRI of head"
CUI_LIST = [
    "C0407663", "C1540653", "C2456562", "C2122946", "C1551261", "C2456531", "C0373281",
    "C2945195", "C0203763", "C4029406", "C4489436", "C1525651", "C5411734", "C3836754",
    "C1524499", "C2456013", "C2551749", "C0993821", "C4723759", "C5696315", "C2043372",
    "C3515869", "C2321543", "C2456012", "C5577983", "C2456540", "C3866546", "C5411275",
    "C1629025", "C3860450", "C3694983", "C5224967", "C0411858", "C2125982", "C4067258",
    "C3866615", "C1864674", "C3826001", "C2319237"
]
BATCH_SIZE = 100
SLEEP_TIME = 2

# ----------------------------
# Gemini Pro Setup
# ----------------------------
model = GenerativeModel("gemini-pro")

def format_prompt(query, cuis):
    cui_lines = "\n".join([f"- {cui}" for cui in cuis])
    prompt = f"""
You are a biomedical expert evaluating semantic coverage.

User query: '{query}'

Your task:
- Interpret the query as a unified clinical intent, not just a set of keywords
- Evaluate whether the following CUIs collectively represent the full meaning and purpose of the query
- Consider the diagnostic goal, anatomical and procedural context, and any relevant variants

Return:
- Coverage score (0–1)
- A clear justification of how the CUIs preserve the full user intent

CUIs:
{cui_lines}
"""
    return prompt

# ----------------------------
# Batch Evaluation
# ----------------------------
responses = []

for i in range(0, len(CUI_LIST), BATCH_SIZE):
    batch = CUI_LIST[i:i + BATCH_SIZE]
    prompt = format_prompt(QUERY, batch)
    print(f"\n[Batch {i // BATCH_SIZE + 1}] Sending {len(batch)} CUIs to Gemini Pro...")

    try:
        response = model.generate_content(prompt)
        responses.append(response.text)
        print(response.text)
        time.sleep(SLEEP_TIME)
    except Exception as e:
        print(f"[Error] Gemini Pro failed: {e}")
        responses.append(f"[Error] Batch {i // BATCH_SIZE + 1}: {e}")

# ----------------------------
# Final Aggregation and Justification
# ----------------------------
final_prompt = f"""
You are a biomedical expert evaluating semantic coverage.

User query: '{QUERY}'

You are given multiple Gemini Pro evaluations of CUI batches. Each batch was scored and justified separately.

Your task:
- Interpret the query as a unified clinical intent
- Read all batch responses
- Return a final coverage score (0–1) for the full set of CUIs
- Provide a unified explanation of how the CUIs support the query
- Explain whether the CUIs fully justify the diagnostic and anatomical meaning of the query

Batch responses:
{chr(10).join(responses)}
"""

print("\n[Final Evaluation] Sending aggregated prompt to Gemini Pro...")
try:
    final_response = model.generate_content(final_prompt)
    print("\n Final Coverage Evaluation:\n")
    print(final_response.text)

    # Optional: Save to file
    with open("final_coverage_output.txt", "w") as f:
        f.write(final_response.text)

except Exception as e:
    print(f"[Error] Final evaluation failed: {e}")
