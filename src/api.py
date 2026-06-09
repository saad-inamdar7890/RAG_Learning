from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.rag_pipeline import RAGPipeline

app = FastAPI(title="Ask My Docs API", description="Production RAG Backend")

# We will initialize the pipeline lazily or on startup
pipeline = None

# Allow CORS for the frontend
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

@app.on_event("startup")
def startup_event():
    global pipeline
    # Ensure Ollama is running and the model is downloaded
    pipeline = RAGPipeline(ollama_model="llama3.1:8b")

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.ask(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
