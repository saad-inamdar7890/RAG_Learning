from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.rag_pipeline import RAGPipeline
from src.eval_engine import run_evaluation, EVAL_QUERIES

app = FastAPI(title="Ask My Docs API", description="Production RAG Backend")

pipeline = None
_executor = ThreadPoolExecutor(max_workers=1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("public/index.html")

# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    citation_number: int
    doc_id: str
    page_number: int
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

class EvalRequest(BaseModel):
    strategy: str   # "normal" | "semantic" | "parent_child"

class EvalQueryResult(BaseModel):
    query: str
    expected_topic: str
    answer: str
    passed: bool

class EvalResponse(BaseModel):
    strategy: str
    total: int
    passed: int
    score_pct: float
    results: list[EvalQueryResult]

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = RAGPipeline(ollama_model="llama3.1:8b")

# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result = pipeline.ask(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Evaluation endpoint ───────────────────────────────────────────────────────

@app.post("/api/evaluate", response_model=EvalResponse)
async def run_eval(request: EvalRequest):
    allowed = {"normal", "semantic", "parent_child"}
    if request.strategy not in allowed:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {allowed}")
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            _executor, run_evaluation, request.strategy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    passed = sum(1 for r in results if r["passed"])
    return {
        "strategy": request.strategy,
        "total": len(results),
        "passed": passed,
        "score_pct": round(passed / len(results) * 100, 1) if results else 0,
        "results": results,
    }

# ── Metadata ──────────────────────────────────────────────────────────────────

@app.get("/api/eval-queries")
def get_eval_queries():
    return {"queries": [q["query"] for q in EVAL_QUERIES]}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
