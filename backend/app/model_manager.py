"""Provider discovery and capability checks for LLM Watermark Lab."""
from __future__ import annotations
import os, urllib.error, urllib.request
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str
    kind: str
    description: str
    watermark_embedding: str
    text_generation: bool = True

PROVIDERS = {
    "huggingface": ProviderInfo("huggingface", "Hugging Face (local)", "local", "Load a causal language model and tokenizer from the Hub or a local directory.", "supported_when_compatible"),
    "ollama": ProviderInfo("ollama", "Ollama", "local_service", "Connect to an Ollama server for ordinary text generation.", "not_supported_by_standard_generate_api"),
    "openai": ProviderInfo("openai", "OpenAI API", "api", "Hosted text generation; configure OPENAI_API_KEY.", "not_exposed_by_standard_chat_api"),
    "anthropic": ProviderInfo("anthropic", "Anthropic API", "api", "Hosted text generation; configure ANTHROPIC_API_KEY.", "not_exposed_by_standard_messages_api"),
    "deepseek": ProviderInfo("deepseek", "DeepSeek API", "api", "Hosted text generation; configure DEEPSEEK_API_KEY.", "not_exposed_by_standard_chat_api"),
}
API_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}

def provider_catalog() -> list[dict[str, Any]]:
    rows = []
    for key, item in PROVIDERS.items():
        row = asdict(item)
        env_name = API_ENV.get(key)
        row["configured"] = bool(os.getenv(env_name)) if env_name else True
        row["embedding_note"] = ("Requires a compatible causal LM, tokenizer and logits processor." if key == "huggingface" else "Can be used for unwatermarked comparison/paraphrasing; embedding is not claimed.")
        rows.append(row)
    return rows

def check_ollama(base_url: str | None = None) -> dict[str, Any]:
    url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as response:
            import json
            payload = json.loads(response.read().decode("utf-8"))
        models = [m.get("name") for m in payload.get("models", []) if m.get("name")]
        return {"connected": True, "url": url, "models": models}
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return {"connected": False, "url": url, "models": [], "error": str(exc)}

def inspect_huggingface_model(model_id: str) -> dict[str, Any]:
    if not model_id or len(model_id) > 256 or model_id.startswith("-"):
        raise ValueError("Provide a valid Hugging Face model ID or local model directory.")
    is_local = os.path.isdir(os.path.expanduser(model_id))
    return {"provider": "huggingface", "model_id": model_id, "source": "local_directory" if is_local else "huggingface_hub", "watermark_embedding": "candidate_requires_runtime_validation", "requirements": ["causal language model", "compatible tokenizer", "logits processing support"], "download_on_load": not is_local}

def load_huggingface_causal_model(model_id: str, **kwargs: Any):
    """Load only when explicitly requested; downloads Hub weights if not cached."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install optional model dependencies: pip install transformers torch") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not callable(getattr(model, "generate", None)):
        raise ValueError("Loaded model does not provide the Transformers generation interface.")
    model.eval()
    return tokenizer, model

def add_model(provider: str, model_id: str, base_url: str | None = None) -> dict[str, Any]:
    if provider not in PROVIDERS:
        raise ValueError("Unsupported provider: " + provider)
    if not model_id.strip():
        raise ValueError("Model ID/name cannot be empty.")
    if provider == "huggingface":
        result = inspect_huggingface_model(model_id.strip())
        result.update(status="registered_not_downloaded", message="Model registered. Weights download only when explicitly loaded.")
        return result
    if provider == "ollama":
        state = check_ollama(base_url)
        if not state["connected"]:
            raise RuntimeError("Cannot reach Ollama at " + state["url"] + ": " + state.get("error", "connection failed"))
        if model_id not in state["models"]:
            raise ValueError("Model is not installed in Ollama. Available: " + ", ".join(state["models"]))
        return {"provider": provider, "model_id": model_id, "status": "connected", "watermark_embedding": "not_supported_by_standard_generate_api"}
    env_name = API_ENV[provider]
    if not os.getenv(env_name):
        raise ValueError("Set " + env_name + " in your local .env file before using this provider.")
    return {"provider": provider, "model_id": model_id, "status": "credentials_configured", "watermark_embedding": "not_exposed_by_standard_api"}

_loaded = None
_load_lock = __import__("threading").Lock()

def load_model(model_id: str):
    global _loaded
    with _load_lock:
        if _loaded and _loaded["model_id"] == model_id:
            return {"status": "already_loaded", "provider": "huggingface", "model_id": model_id, "model_class": type(_loaded["model"]).__name__}
        tokenizer, model = load_huggingface_causal_model(model_id)
        _loaded = {"model_id": model_id, "tokenizer": tokenizer, "model": model}
        return {"status": "loaded", "provider": "huggingface", "model_id": model_id, "model_class": type(model).__name__, "device": str(next(model.parameters()).device)}

def generate_text(model_id: str, prompt: str, max_new_tokens: int = 80):
    if not prompt.strip(): raise ValueError("Prompt cannot be empty.")
    if not 1 <= max_new_tokens <= 256: raise ValueError("max_new_tokens must be between 1 and 256.")
    state = _loaded
    if not state or state["model_id"] != model_id: raise RuntimeError("Load this model before generating text.")
    import torch
    tokenizer, model = state["tokenizer"], state["model"]
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return {"model_id": model_id, "text": tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)}
