from fastapi import FastAPI, HTTPException
from app.models import QueryRequest, QueryResponse
from app.pipeline import run_docai_embedding_pipeline
from config.settings import PROJECT_ID, BQ_CUI_TABLE
import subprocess
import requests
import json

app = FastAPI(
    title="Contextual API",
    description="Semantic search over DocAI-parsed PDFs using Vertex AI embeddings",
    version="1.0.0"
)

@app.post("/query-doc", response_model=QueryResponse)
def query_doc(request: QueryRequest):
    try:
        # Get identity token for DocAI authorization
        tmp = subprocess.run(['gcloud', 'auth', 'print-identity-token'], stdout=subprocess.PIPE, universal_newlines=True)
        token = str(tmp.stdout).strip()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        docai_payload = {
            "gcs_uri": request.gcs_uri,
            "individual_pages": request.individual_pages
        }

        # Call DocAI endpoint
        docai_response = requests.post("https://6789", headers=headers, json=docai_payload)
        if docai_response.status_code != 200:
            raise HTTPException(status_code=docai_response.status_code, detail="DocAI request failed")

        response_content = docai_response.content.decode("utf-8")

        # Run embedding pipeline
        entities = run_docai_embedding_pipeline(response_content, request.query, PROJECT_ID, BQ_CUI_TABLE)

        # Format response
        results = [
            {
                "entity_id": e.entity_id,
                "text": e.text,
                "top_cuis": [{"cui": cui, "score": score} for cui, score in (e.top_cuis or [])],
                "embedding": e.emb.tolist() if e.emb is not None else None,
                "similarity_to_query": e.sim_to_query
            }
            for e in entities
        ]

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
