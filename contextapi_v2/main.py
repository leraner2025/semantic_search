from fastapi import FastAPI, Request
from pydantic import BaseModel
from pipeline import run_docai_embedding_pipeline
import os

app = FastAPI()

class QueryRequest(BaseModel):
    response_content: str
    query_text: str

@app.post("/query")
def query_docai(request: QueryRequest):
    project_id = os.getenv("PROJECT_ID")
    cui_table = os.getenv("BQ_CUI_TABLE")

    results = run_docai_embedding_pipeline(
        response_content=request.response_content,
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
