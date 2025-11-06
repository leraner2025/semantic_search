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
CUI_LIST = [...]  # Replace with your full list of 6000 CUIs
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

Given the user query: '{query}'

Evaluate whether the following CUIs collectively represent the full meaning of the query.

Instructions:
- Break down the query into key concepts (e.g., procedure, anatomy)
- Match each concept to relevant CUIs
- Return a coverage score (0–1)
- Provide a clear justification of how the CUIs support the query
- List any missing or redundant concepts

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

The user query is: '{QUERY}'

You are given multiple Gemini Pro evaluations of CUI batches. Each batch was scored and justified separately.

Your task:
- Read all batch responses
- Return a final coverage score (0–1) for the full set of CUIs
- Provide a unified explanation of how the CUIs support the query
- List any major missing or redundant concepts
- Explain whether the CUIs fully justify the user intent

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
