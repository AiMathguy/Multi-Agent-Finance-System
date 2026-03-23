from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.orchestrator import run_pipeline

router = APIRouter()

class RunRequest(BaseModel):
    prompt: str

@router.post("/run")
def run(req: RunRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt required")
    return run_pipeline(req.prompt)