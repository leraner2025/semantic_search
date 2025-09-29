from fastapi import FastAPI, Request
from pydantic import BaseModel
from pipeline import run_docai_embedding_pipeline
from config import PROJECT_ID, BQ_CUI_TABLE

app = FastAPI()

class DocAIRequest(BaseModel):
    response_content: str
    query_text: str

@app.post("/analyze")
def analyze_docai(req: DocAIRequest):
    entities = run_docai_embedding_pipeline(req.response_content, req.query_text, PROJECT_ID, BQ_CUI_TABLE)
    return [
        {
            "entity_id": e.entity_id,
            "text": e.text,
            "top_cuis": [{"cui": cui, "score": score} for cui, score in (e.top_cuis or [])],
            "similarity_to_query": e.sim_to_query
        }
        for e in entities
    ]
