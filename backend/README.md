# Model Manager API

Install the backend dependencies from the repository root:

    python -m pip install -r backend/requirements.txt
    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000. API documentation is available at /docs.

Hugging Face model IDs are registered without downloading weights; actual loading is an explicit operation. Ollama is checked through its local tags endpoint. Hosted API providers are reported as configured only when their environment variable exists. This version intentionally does not expose API keys or claim that ordinary chat APIs support logits-level watermark embedding.
