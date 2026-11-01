"""Integration with the upstream Kirchenbauer et al. watermark implementation."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any

UPSTREAM = Path(__file__).resolve().parents[2] / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from . import model_manager


def _loaded_state() -> dict[str, Any]:
    state = model_manager._loaded
    if not state:
        raise RuntimeError("Load a Hugging Face model before using watermark features.")
    return state


def _config(gamma: float, delta: float, z_threshold: float) -> tuple[dict[str, Any], int]:
    if not 0.05 <= gamma <= 0.95:
        raise ValueError("gamma must be between 0.05 and 0.95")
    if not 0.1 <= delta <= 10:
        raise ValueError("delta must be between 0.1 and 10")
    if not 0.5 <= z_threshold <= 10:
        raise ValueError("z_threshold must be between 0.5 and 10")
    raw_key = os.getenv("WATERMARK_RESEARCH_KEY")
    if not raw_key:
        raise RuntimeError("WATERMARK_RESEARCH_KEY is not configured in the server .env file.")
    try:
        secret_key = int(raw_key)
    except ValueError as exc:
        raise RuntimeError("WATERMARK_RESEARCH_KEY must be an integer.") from exc
    return {"gamma": gamma, "delta": delta, "z_threshold": z_threshold}, secret_key


def generate_watermarked(model_id: str, prompt: str, max_new_tokens: int = 100,
                         gamma: float = 0.5, delta: float = 2.0,
                         z_threshold: float = 4.0) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256.")
    state = _loaded_state()
    if state["model_id"] != model_id:
        raise RuntimeError("The requested model is not currently loaded.")
    params, secret_key = _config(gamma, delta, z_threshold)
    import torch
    tokenizer, model = state["tokenizer"], state["model"]
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    vocab = list(range(int(model.config.vocab_size)))
    processor = WatermarkLogitsProcessor(vocab=vocab, gamma=params["gamma"],
                                         delta=params["delta"], hash_key=secret_key)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=0.8, top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                                logits_processor=[processor])
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    detection = detect_watermark(model_id, text, gamma, delta, z_threshold, prompt_prefix=prompt)
    return {"model_id": model_id, "prompt": prompt, "generated_text": text,
            "watermark": {"gamma": gamma, "delta": delta,
            "z_threshold": z_threshold}, "detection": detection}


def detect_watermark(model_id: str, text: str, gamma: float = 0.5,
                     delta: float = 2.0, z_threshold: float = 4.0,
                     prompt_prefix: str | None = None) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Text to detect cannot be empty.")
    state = _loaded_state()
    if state["model_id"] != model_id:
        raise RuntimeError("Load the same model/tokenizer used to generate this text before detection.")
    params, secret_key = _config(gamma, delta, z_threshold)
    tokenizer = state["tokenizer"]
    vocab = list(range(int(state["model"].config.vocab_size)))
    detector = WatermarkDetector(vocab=vocab, gamma=params["gamma"],
                                 delta=params["delta"], hash_key=secret_key,
                                 device=next(state["model"].parameters()).device,
                                 tokenizer=tokenizer, z_threshold=params["z_threshold"],
                                 normalizers=[], ignore_repeated_bigrams=True)
    try:
        if prompt_prefix:
            prefix_ids = tokenizer(prompt_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            text_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if prefix_ids.numel():
                token_ids = __import__("torch").cat((prefix_ids[-1:].to(detector.device), text_ids.to(detector.device)))
            else:
                token_ids = text_ids.to(detector.device)
            scores = detector.detect(tokenized_text=token_ids)
        else:
            scores = detector.detect(text=text)
    except (ZeroDivisionError, AssertionError, IndexError) as exc:
        raise ValueError("Text is too short to score. Provide a longer passage.") from exc
    result = {"prediction": bool(scores.get("prediction", False)),
              "num_tokens_scored": int(scores["num_tokens_scored"]),
              "num_green_tokens": int(scores["num_green_tokens"]),
              "green_fraction": float(scores["green_fraction"]),
              "z_score": float(scores["z_score"]),
              "p_value": float(scores["p_value"]),
              "threshold": params["z_threshold"], "gamma": params["gamma"]}
    if result["prediction"] and "confidence" in scores:
        result["confidence"] = float(scores["confidence"])
    return result
