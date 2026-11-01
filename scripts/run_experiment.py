#!/usr/bin/env python3
"""Run a reproducible watermark generation, detection and robustness study."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))

from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor
from src.attacks import apply_attack
from src.metrics import bootstrap_rate_ci, classification_metrics

DEFAULT_PROMPTS = [
    "Artificial intelligence is changing education because",
    "Renewable energy technologies can help cities by",
    "A reliable software development process should",
    "Climate change affects agriculture through",
    "The benefits of scientific research include",
    "Modern communication networks are important because",
    "Students can improve their learning outcomes when",
    "A secure computer system should always",
    "Machine learning models are useful for",
    "The future of transportation may depend on",
    "Universities can support innovation by",
    "Data privacy is important in digital services because",
    "Cloud computing allows organizations to",
    "Robotics can improve manufacturing by",
    "A healthy environment requires communities to",
    "Open source software helps developers",
    "The purpose of statistical testing is",
    "Good engineering design begins with",
    "Digital learning platforms can support students by",
    "Watermarking generated text may be useful when",
]

ATTACKS = ["none", "word_deletion", "word_substitution", "lexical_paraphrase_proxy"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate(model, tokenizer, prompt: str, seed: int, max_new_tokens: int,
             watermark_processor=None) -> str:
    set_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded["attention_mask"].to("cpu")
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if watermark_processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([watermark_processor])
    with torch.inference_mode():
        output = model.generate(**kwargs)
    return tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()


def safe_detect(detector, text: str) -> dict:
    """Return a negative/insufficient-evidence result for texts too short to score."""
    try:
        return detector.detect(text)
    except ValueError as exc:
        if "Must have at least" not in str(exc):
            raise
        return {
            "num_tokens_scored": 0, "num_green_tokens": 0,
            "green_fraction": float("nan"), "z_score": -1_000_000.0,
            "p_value": 1.0, "prediction": False,
            "insufficient_text": True,
        }


def perplexity(model, tokenizer, text: str) -> float:
    """Reference-model perplexity; a proxy, not a complete text-quality measure."""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to("cpu")
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=input_ids)
    return float(torch.exp(out.loss).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--seeds", default="11,22")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--lengths", default=None, help="Comma-separated generation lengths, e.g. 48,96,160 (defaults to --max-new-tokens)")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--attacks", default=",".join(ATTACKS))
    args = parser.parse_args()

    if not 1 <= args.prompts <= len(DEFAULT_PROMPTS):
        parser.error(f"--prompts must be between 1 and {len(DEFAULT_PROMPTS)}")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    lengths = ([int(x.strip()) for x in args.lengths.split(",") if x.strip()]
               if args.lengths else [args.max_new_tokens])
    if not lengths or any(x < 8 or x > 512 for x in lengths):
        parser.error("--lengths values must be between 8 and 512")
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    unknown = set(attacks) - set(ATTACKS)
    if unknown:
        parser.error(f"Unknown attacks: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"Loading {args.model} on CPU; CUDA available={torch.cuda.is_available()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to("cpu")
    model.eval()
    vocab = list(tokenizer.get_vocab().values())

    gamma, delta, threshold = 0.25, 2.0, 4.0
    watermark_key = int(os.environ.get("WATERMARK_RESEARCH_KEY", "15485863"))
    scheme = f"ff-anchored_minhash_prf-4-True-{watermark_key}"
    key_id = hashlib.sha256(str(watermark_key).encode()).hexdigest()[:12]
    if "WATERMARK_RESEARCH_KEY" not in os.environ:
        print("WARNING: using the upstream default watermark key; set WATERMARK_RESEARCH_KEY for your own research key.", flush=True)
    detector = WatermarkDetector(
        vocab=vocab, gamma=gamma, seeding_scheme=scheme,
        device=torch.device("cpu"), tokenizer=tokenizer,
        z_threshold=threshold, normalizers=[], ignore_repeated_ngrams=True,
    )

    rows = []
    attack_rows = []
    base_prompts = DEFAULT_PROMPTS[:args.prompts]
    total = len(lengths) * len(base_prompts) * len(seeds) * 2
    completed = 0

    for generation_length in lengths:
      for prompt_idx, prompt in enumerate(base_prompts):
        for seed in seeds:
            for label, is_watermarked in (("control", False), ("watermarked", True)):
                processor = None
                if is_watermarked:
                    processor = WatermarkLogitsProcessor(
                        vocab=vocab, gamma=gamma, delta=delta,
                        seeding_scheme=scheme,
                    )
                text = generate(model, tokenizer, prompt, seed, generation_length, processor)
                detected = safe_detect(detector, text)
                ppl = float("nan") if args.skip_perplexity else perplexity(model, tokenizer, text)
                row = {
                    "run_id": run_id,
                    "prompt_id": prompt_idx,
                    "seed": seed,
                    "max_new_tokens": generation_length,
                    "label": label,
                    "is_watermarked": int(is_watermarked),
                    "text": text,
                    "word_count": len(text.split()),
                    "num_tokens_scored": detected.get("num_tokens_scored"),
                    "num_green_tokens": detected.get("num_green_tokens"),
                    "green_fraction": detected.get("green_fraction"),
                    "z_score": detected.get("z_score"),
                    "p_value": detected.get("p_value"),
                    "prediction": int(bool(detected.get("prediction"))),
                    "perplexity": ppl,
                }
                rows.append(row)
                if is_watermarked:
                    for attack_idx, attack in enumerate(attacks):
                        attacked = apply_attack(text, attack, seed=seed + attack_idx + prompt_idx)
                        result = safe_detect(detector, attacked)
                        attack_rows.append({
                            "run_id": run_id, "prompt_id": prompt_idx, "seed": seed,
                            "max_new_tokens": generation_length,
                            "attack": attack, "text": attacked,
                            "word_count": len(attacked.split()),
                            "num_tokens_scored": result.get("num_tokens_scored"),
                            "green_fraction": result.get("green_fraction"),
                            "z_score": result.get("z_score"),
                            "p_value": result.get("p_value"),
                            "prediction": int(bool(result.get("prediction"))),
                        })
                completed += 1
                print(f"[{completed}/{total}] {label} prompt={prompt_idx} seed={seed} "
                      f"z={row['z_score']:.3f} detected={row['prediction']} "
                      f"words={row['word_count']}", flush=True)

    def write_csv(path: Path, records: list[dict]) -> None:
        if not records:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    samples_path = output_dir / f"samples_{run_id}.csv"
    attacks_path = output_dir / f"attacks_{run_id}.csv"
    write_csv(samples_path, rows)
    write_csv(attacks_path, attack_rows)

    labels = [r["is_watermarked"] for r in rows]
    predictions = [r["prediction"] for r in rows]
    scores = [r["z_score"] for r in rows]
    overall = classification_metrics(labels, predictions, scores)
    positive = [r for r in rows if r["is_watermarked"] == 1]
    negative = [r for r in rows if r["is_watermarked"] == 0]
    tpr_ci = bootstrap_rate_ci([r["prediction"] for r in positive])
    fpr_ci = bootstrap_rate_ci([r["prediction"] for r in negative])

    length_summary = {}
    for generation_length in lengths:
        group = [r for r in rows if r["max_new_tokens"] == generation_length]
        length_summary[str(generation_length)] = classification_metrics(
            [r["is_watermarked"] for r in group],
            [r["prediction"] for r in group],
            [r["z_score"] for r in group],
        )

    attack_summary = {}
    for attack in attacks:
        group = [r for r in attack_rows if r["attack"] == attack]
        if group:
            lo, hi = bootstrap_rate_ci([r["prediction"] for r in group])
            scored = [r for r in group if r["num_tokens_scored"] > 0 and np.isfinite(r["z_score"])]
            green = [r["green_fraction"] for r in scored if np.isfinite(r["green_fraction"])]
            attack_summary[attack] = {
                "n": len(group),
                "scored_n": len(scored),
                "insufficient_text_n": len(group) - len(scored),
                "detection_rate": float(np.mean([r["prediction"] for r in group])),
                "detection_rate_95ci": [lo, hi],
                "mean_z_score": float(np.mean([r["z_score"] for r in scored])) if scored else None,
                "mean_green_fraction": float(np.mean(green)) if green else None,
            }

    def finite_mean(records):
        values = [r["perplexity"] for r in records if np.isfinite(r["perplexity"])]
        return float(np.mean(values)) if values else None

    quality = {
        "control_mean_perplexity": finite_mean(negative),
        "watermarked_mean_perplexity": finite_mean(positive),
        "note": "Reference-model perplexity is only a proxy; compare matched prompts and lengths.",
    }
    report = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": "cpu",
        "configuration": {
            "gamma": gamma, "delta": delta, "seeding_scheme": scheme,
            "key_id": key_id,
            "z_threshold": threshold, "ignore_repeated_ngrams": True,
            "generation_lengths": lengths, "seeds": seeds,
            "prompt_count": len(base_prompts), "attacks": attacks,
        },
        "classification": overall,
        "length_sensitivity": length_summary,
        "true_positive_rate_95ci": list(tpr_ci),
        "false_positive_rate_95ci": list(fpr_ci),
        "attack_robustness": attack_summary,
        "quality_proxy": quality,
        "artifacts": {"samples_csv": str(samples_path), "attacks_csv": str(attacks_path)},
        "caveats": [
            "Small smoke-scale runs are not sufficient for strong scientific claims.",
            "Lexical paraphrase proxy is not a semantic paraphrasing attack.",
            "Detection is evidence for a matching watermark configuration, not proof of authorship.",
        ],
    }
    report_path = output_dir / f"report_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    print("\n=== CLASSIFICATION ===")
    print(json.dumps(overall, indent=2))
    print(f"TPR 95% bootstrap CI: {tpr_ci}")
    print(f"FPR 95% bootstrap CI: {fpr_ci}")
    print("\n=== ATTACK ROBUSTNESS ===")
    print(json.dumps(attack_summary, indent=2))
    print("\n=== QUALITY PROXY ===")
    print(json.dumps(quality, indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
