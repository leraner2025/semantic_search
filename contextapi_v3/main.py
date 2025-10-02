# main.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from pipeline import run_docai_embedding_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    gcs_uri: str
    user_query: str

class EntityResponse(BaseModel):
    entity_id: str
    text: str
    cui: str
    sim_to_query: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the DocAI + Gemini API"}

@app.post("/query", response_model=List[EntityResponse])
def query_docai(request: QueryRequest):
    entities = run_docai_embedding_pipeline(request.gcs_uri, request.user_query)
    return [
        EntityResponse(
            entity_id=e.entity_id,
            text=e.text,
            cui=e.cui,
            sim_to_query=e.sim_to_query
        )
        for e in entities
    ]
