# app/main.py
from contextlib import asynccontextmanager
from fastapi    import FastAPI
from pydantic   import BaseModel
from app.model    import load_model
from app.analyzer import init_analyzer, analyze, analyze_with_explanation


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Connecting to HF API...")
    load_model()
    print("[Startup] Loading retriever and database...")
    init_analyzer()
    print("[Startup] ✅ Server is ready!")
    yield
    print("[Shutdown] Server shutting down.")


app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    """Judges ping this to check readiness."""
    return {"status": "ok"}


@app.post("/analyze")
def analyze_endpoint(request: PromptRequest):
    """
    HACKATHON GRADING ENDPOINT.
    Returns ONLY harmful boolean and articles list.
    Do NOT change this response format.
    """
    result = analyze(request.prompt)
    return {
        "harmful":  result["harmful"],
        "articles": result["articles"]
    }


@app.post("/explain")
def explain_endpoint(request: PromptRequest):
    """
    HUMAN-FRIENDLY ENDPOINT.
    Returns full analysis with reasoning,
    consequences, and improvement suggestions.
    """
    result = analyze_with_explanation(request.prompt)
    return result