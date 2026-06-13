from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.rag_pipeline import RAGPipeline
from src.eval_engine import generate_single_eval, judge_single_eval, EVAL_QUERIES
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

class SingleEvalGenerateRequest(BaseModel):
    strategy: str
    query: str

class SingleEvalGenerateResponse(BaseModel):
    answer: str
    latency_s: float
    retrieved_context: list[dict]
    expected_topic: str

class CriterionResult(BaseModel):
    passed: bool
    detail: str

class CriteriaBreakdown(BaseModel):
    latency: CriterionResult
    citation: CriterionResult
    faithfulness: CriterionResult
    topical: CriterionResult

class SingleEvalJudgeRequest(BaseModel):
    query: str
    answer: str
    expected_topic: str
    retrieved_context: list[dict]
    generate_latency_s: float

class SingleEvalJudgeResponse(BaseModel):
    passed: bool
    criteria: CriteriaBreakdown

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

@app.post("/api/eval-generate", response_model=SingleEvalGenerateResponse)
async def eval_generate_endpoint(req: SingleEvalGenerateRequest):
    allowed = {"normal", "semantic", "parent_child"}
    if req.strategy not in allowed:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {allowed}")
    
    # Find expected topic
    expected = next((item["expected_topic"] for item in EVAL_QUERIES if item["query"] == req.query), "Unknown")
    
    loop = asyncio.get_event_loop()
    try:
        res = await loop.run_in_executor(
            _executor, generate_single_eval, req.strategy, req.query
        )
        res["expected_topic"] = expected
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/eval-judge", response_model=SingleEvalJudgeResponse)
async def eval_judge_endpoint(req: SingleEvalJudgeRequest):
    loop = asyncio.get_event_loop()
    try:
        res = await loop.run_in_executor(
            _executor, judge_single_eval, req.query, req.answer, req.expected_topic, req.retrieved_context, req.generate_latency_s
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Metadata ──────────────────────────────────────────────────────────────────

@app.get("/api/eval-queries")
def get_eval_queries():
    return {"queries": EVAL_QUERIES}

@app.get("/api/metrics")
def get_metrics():
    """Returns p50/p95/p99 latencies, step averages, token counts."""
    return metrics.get_summary()

@app.get("/api/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
