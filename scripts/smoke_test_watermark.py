#!/usr/bin/env python3
"""CPU smoke test for the authors' extended watermark implementation.

Downloads distilgpt2 on first run. This is an integration check, not a
statistically meaningful experiment or a reproduction of the paper's results.
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))
from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--prompt",
        default="Artificial intelligence is changing education because",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading {args.model} on {device}; first run downloads model weights.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    vocab = list(tokenizer.get_vocab().values())
    gamma, delta = 0.25, 2.0
    watermark_key = int(os.environ.get("WATERMARK_RESEARCH_KEY", "15485863"))
    scheme = f"ff-anchored_minhash_prf-4-True-{watermark_key}"

    watermark = WatermarkLogitsProcessor(
        vocab=vocab,
        gamma=gamma,
        delta=delta,
        seeding_scheme=scheme,
    )
    with torch.inference_mode():
        watermarked_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor=LogitsProcessorList([watermark]),
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        plain_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = input_ids.shape[-1]
    watermarked_text = tokenizer.decode(
        watermarked_ids[0, prompt_length:], skip_special_tokens=True
    )
    plain_text = tokenizer.decode(plain_ids[0, prompt_length:], skip_special_tokens=True)

    detector = WatermarkDetector(
        vocab=vocab,
        gamma=gamma,
        seeding_scheme=scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )
    watermarked_result = detector.detect(watermarked_text)
    plain_result = detector.detect(plain_text)

    print("\n=== Watermarked sample ===\n", watermarked_text)
    print("\n=== Unwatermarked control ===\n", plain_text)
    print("\n=== Detector comparison ===")
    for label, result in (("watermarked", watermarked_result), ("control", plain_result)):
        print(
            f"{label}: prediction={result.get('prediction')}, "
            f"z={result.get('z_score'):.3f}, "
            f"green_fraction={result.get('green_fraction'):.3f}, "
            f"tokens_scored={result.get('num_tokens_scored')}"
        )
    print("\nSmoke test completed. Repeat with many prompts/seeds before drawing conclusions.")


if __name__ == "__main__":
    main()
