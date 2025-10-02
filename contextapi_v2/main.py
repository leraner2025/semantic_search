from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_docai_embedding_pipeline
import os
import subprocess
import requests
import json

app = FastAPI()

class QueryRequest(BaseModel):
    gcs_uri: str
    query_text: str

def fetch_docai_response(gcs_uri: str) -> str:
    # Get identity token
    tmp = subprocess.run(['gcloud', 'auth', 'print-identity-token'], stdout=subprocess.PIPE, universal_newlines=True)
    token = str(tmp.stdout).strip()

    # Set headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Set payload
    data = {
        "gcs_uri": gcs_uri
    }

    # Send POST request
    response = requests.post("https://api/", headers=headers, json=data)

    # Decode response
    try:
        return response.json()
    except json.JSONDecodeError:
        return response.content.decode("utf-8")

@app.post("/query")
def query_docai(request: QueryRequest):
    project_id = os.getenv("PROJECT_ID")
    cui_table = os.getenv("BQ_CUI_TABLE")

    # Fetch DocAI response content from GCS URI
    response_content = fetch_docai_response(request.gcs_uri)

    # Run embedding pipeline
    results = run_docai_embedding_pipeline(
        response_content=response_content,
        query_text=request.query_text,
        project_id=project_id,
        cui_table=cui_table
    )

    return {
        "results": [
            {
                "entity_id": e.entity_id,
                "text": e.text,
                "top_cuis": e.top_cuis or [],
                "similarity": e.sim_to_query
            }
            for e in results
        ]
    }






# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from pipeline import run_docai_embedding_pipeline
# import os

# app = FastAPI()

# class QueryRequest(BaseModel):
#     response_content: str
#     query_text: str

# @app.post("/query")
# def query_docai(request: QueryRequest):
#     project_id = os.getenv("PROJECT_ID")
#     cui_table = os.getenv("BQ_CUI_TABLE")

#     results = run_docai_embedding_pipeline(
#         response_content=request.response_content,
#         query_text=request.query_text,
#         project_id=project_id,
#         cui_table=cui_table
#     )

#     return {
#         "results": [
#             {
#                 "entity_id": e.entity_id,
#                 "text": e.text,
#                 "top_cuis": e.top_cuis or [],
#                 "similarity": e.sim_to_query
#             }
#             for e in results
#         ]
#     }
