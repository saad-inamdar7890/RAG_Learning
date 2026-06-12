from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.rag_pipeline import RAGPipeline
from src.eval_engine import run_evaluation, EVAL_QUERIES
from src import metrics

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

class Trace(BaseModel):
    total_s: float
    steps: dict
    tokens: int

class Source(BaseModel):
    citation_number: int
    doc_id: str
    page_number: int
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    trace: Trace

class EvalRequest(BaseModel):
    strategy: str   # "normal" | "semantic" | "parent_child"

class CriterionResult(BaseModel):
    passed: bool
    detail: str

class CriteriaBreakdown(BaseModel):
    latency: CriterionResult
    citation: CriterionResult
    faithfulness: CriterionResult
    topical: CriterionResult

class EvalQueryResult(BaseModel):
    query: str
    expected_topic: str
    answer: str
    passed: bool
    latency_s: float
    criteria: CriteriaBreakdown

class EvalResponse(BaseModel):
    strategy: str
    total: int
    passed: int
    score_pct: float
    avg_latency_s: float
    criteria_pass_counts: dict
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
    total = len(results)
    avg_lat = round(sum(r.get("latency_s", 0) for r in results) / max(total, 1), 2)
    criteria_names = ["latency", "citation", "faithfulness", "topical"]
    criteria_counts = {
        k: sum(1 for r in results if r.get("criteria", {}).get(k, {}).get("passed", False))
        for k in criteria_names
    }
    return {
        "strategy": request.strategy,
        "total": total,
        "passed": passed,
        "score_pct": round(passed / total * 100, 1) if total else 0,
        "avg_latency_s": avg_lat,
        "criteria_pass_counts": criteria_counts,
        "results": results,
    }

# ── Metadata ──────────────────────────────────────────────────────────────────

@app.get("/api/eval-queries")
def get_eval_queries():
    return {"queries": [q["query"] for q in EVAL_QUERIES]}

@app.get("/api/metrics")
def get_metrics():
    """Returns p50/p95/p99 latencies, step averages, token counts."""
    return metrics.get_summary()

@app.get("/api/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
