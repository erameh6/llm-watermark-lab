"""Deterministic text perturbations for watermark robustness experiments.

The lexical paraphrase proxy is deliberately not presented as a true semantic
paraphrase. Use a separately documented paraphrase model for that experiment.
"""
from __future__ import annotations

import random
import re

SYNONYMS = {
    "important": "significant", "help": "assist", "use": "utilize",
    "show": "demonstrate", "make": "create", "change": "alter",
    "large": "substantial", "small": "minor", "good": "effective",
    "bad": "poor", "learn": "study", "students": "learners",
    "student": "learner", "system": "framework", "method": "approach",
    "problem": "challenge", "fast": "rapid", "slow": "gradual",
    "increase": "raise", "decrease": "reduce", "result": "outcome",
    "different": "distinct", "need": "require", "many": "numerous",
}


def delete_words(text: str, rate: float = 0.10, seed: int = 0) -> str:
    words = text.split()
    if len(words) < 3:
        return text
    rng = random.Random(seed)
    kept = [w for w in words if rng.random() >= rate]
    return " ".join(kept) if len(kept) >= 2 else text


def substitute_words(text: str, rate: float = 0.10, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = text.split()
    for i, word in enumerate(words):
        match = re.match(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$", word)
        if not match or rng.random() >= rate:
            continue
        prefix, core, suffix = match.groups()
        replacement = SYNONYMS.get(core.lower())
        if replacement:
            if core[:1].isupper():
                replacement = replacement.capitalize()
            words[i] = prefix + replacement + suffix
    return " ".join(words)


def lexical_paraphrase_proxy(text: str, rate: float = 0.20, seed: int = 0) -> str:
    """A limited synonym-replacement proxy, not a semantic paraphraser."""
    return substitute_words(text, rate=rate, seed=seed)


def apply_attack(text: str, attack: str, seed: int = 0) -> str:
    if attack == "none":
        return text
    if attack == "word_deletion":
        return delete_words(text, seed=seed)
    if attack == "word_substitution":
        return substitute_words(text, seed=seed)
    if attack == "lexical_paraphrase_proxy":
        return lexical_paraphrase_proxy(text, seed=seed)
    raise ValueError(f"Unknown attack: {attack}")
