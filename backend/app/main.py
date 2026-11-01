from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from .model_manager import add_model, check_ollama, provider_catalog, load_model, generate_text
from .watermark_engine import generate_watermarked, detect_watermark
from .experiment_engine import run_evaluation

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="LLM Watermark Lab API", version="0.1.0", description="Self-hosted research platform for LLM text watermarking.")

class LoadRequest(BaseModel):
    provider: str = "huggingface"
    model_id: str = Field(min_length=1, max_length=256)

class GenerateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=80, ge=1, le=256)

class ModelRequest(BaseModel):
    provider: str
    model_id: str = Field(min_length=1, max_length=256)
    base_url: str | None = None


class WatermarkGenerateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    gamma: float = Field(default=0.5, ge=0.05, le=0.95)
    delta: float = Field(default=2.0, ge=0.1, le=10)
    z_threshold: float = Field(default=4.0, ge=0.5, le=10)

class WatermarkDetectRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=20000)
    prompt_prefix: str | None = Field(default=None, max_length=8000)
    gamma: float = Field(default=0.5, ge=0.05, le=0.95)
    delta: float = Field(default=2.0, ge=0.1, le=10)
    z_threshold: float = Field(default=4.0, ge=0.5, le=10)

class EvaluationRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    prompt_count: int = Field(default=4, ge=1, le=10)
    seeds: list[int] = Field(default_factory=lambda: [11, 22], min_length=1, max_length=10)
    max_new_tokens: int = Field(default=96, ge=16, le=256)
    gamma: float = Field(default=0.5, ge=0.05, le=0.95)
    delta: float = Field(default=2.0, ge=0.1, le=10)
    z_threshold: float = Field(default=4.0, ge=0.5, le=10)
    attacks: list[str] = Field(default_factory=lambda: ["none", "word_deletion", "word_substitution", "lexical_paraphrase_proxy"], min_length=1, max_length=4)

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "llm-watermark-lab"}

@app.get("/api/providers")
def providers():
    return {"providers": provider_catalog()}

@app.get("/api/providers/ollama/status")
def ollama_status():
    return check_ollama()

@app.post("/api/models/validate")
def validate_model(body: ModelRequest):
    try:
        return add_model(body.provider, body.model_id, body.base_url)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/models/load")
def load_selected_model(body: LoadRequest):
    if body.provider != "huggingface": raise HTTPException(status_code=400, detail="Explicit model loading currently supports Hugging Face only.")
    try: return load_model(body.model_id)
    except Exception as exc: raise HTTPException(status_code=400, detail=f"Model load failed: {exc}") from exc

@app.post("/api/generate")
def generate(body: GenerateRequest):
    try: return generate_text(body.model_id, body.prompt, body.max_new_tokens)
    except (ValueError, RuntimeError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/watermark/generate")
def watermark_generate(body: WatermarkGenerateRequest):
    try:
        return generate_watermarked(body.model_id, body.prompt, body.max_new_tokens, body.gamma, body.delta, body.z_threshold)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/watermark/detect")
def watermark_detect(body: WatermarkDetectRequest):
    try:
        return detect_watermark(body.model_id, body.text, body.gamma, body.delta, body.z_threshold, body.prompt_prefix)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/experiments/evaluate")
def evaluate(body: EvaluationRequest):
    try:
        return run_evaluation(body.model_id, body.prompt_count, body.seeds,
                              body.max_new_tokens, body.gamma, body.delta,
                              body.z_threshold, body.attacks)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
