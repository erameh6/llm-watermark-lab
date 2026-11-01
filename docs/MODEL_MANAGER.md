# Model Manager design

## Provider categories

- Hugging Face local/Hub: primary route for watermark embedding, conditional on a causal LM, compatible tokenizer and generation-time logits processing.
- Ollama: model discovery and ordinary text generation. The standard generate endpoint is not considered logits-compatible.
- Hosted chat APIs: comparison generation and paraphrasing only unless a provider exposes a documented low-level token-generation interface.

## Security

- API credentials are read only from server-side environment variables.
- Never return credentials from an endpoint or put them in frontend code.
- Keep .env out of version control.
- Use a separate watermark secret per installation/research run. Do not publish it with experiment outputs.
- Model weights are cached outside the source repository.

## Current API

- GET /api/health
- GET /api/providers
- GET /api/providers/ollama/status
- POST /api/models/validate

The POST endpoint validates provider configuration. It does not imply that a model has been downloaded or loaded. Hugging Face loading is available as a lazy Python function in backend/app/model_manager.py; a future UI action will invoke it with explicit resource limits.


## Kirchenbauer green-list watermark integration

The application exposes two research endpoints for a loaded Hugging Face causal language model:

- `POST /api/watermark/generate` applies the upstream `WatermarkLogitsProcessor` during `model.generate`.
- `POST /api/watermark/detect` scores text with the matching tokenizer, vocabulary, watermark parameters and server-side secret key.

The browser Model Manager includes controls for gamma (green-list fraction), delta (logit bias), and the detector z-score threshold. Watermarked generation uses sampling (`temperature=0.8`, `top_p=0.9`) so the output is not artificially deterministic. The generated continuation is also sent to the detector for an immediate diagnostic score.

**Important research constraints:** use the same model/tokenizer, `WATERMARK_RESEARCH_KEY`, gamma and seeding scheme for generation and detection. The key is read only on the server and is never returned by the API. Changing the key invalidates detection of text generated with the previous key. Detection is statistical evidence for this scheme/key, not proof of authorship. Very short texts and repetitive generations have low statistical power; a negative result does not establish that text is unwatermarked. Hosted APIs and standard Ollama generation are not routed through this embedding implementation.

The implementation imports the Apache-2.0-licensed upstream code in `upstream/watermark_processor.py` and its associated normalization modules; see `upstream/LICENSE.md` and the upstream project attribution in the root README.

## Reproducible robustness evaluation

The **Robustness evaluation** panel runs a paired CPU experiment using the currently loaded Hugging Face model and the same `simple_1` green-list watermark implementation as the interactive watermark feature. It generates matched unwatermarked controls and watermarked samples for each prompt/seed pair, detects both, then attacks each watermarked sample using:

- `none` — unchanged watermarked text (baseline)
- `word_deletion` — deterministic deletion perturbation
- `word_substitution` — deterministic synonym replacement
- `lexical_paraphrase_proxy` — limited synonym replacement proxy; it is not a neural or human semantic paraphrase

The evaluator reports accuracy, precision, recall/detection rate, specificity, F1, false-positive rate, Wilson score 95% intervals, attack-specific detection rates, and scorable sample counts. Generation uses matched sampling settings, a fixed requested token count, EOS suppression, and a mild repetition penalty to reduce early-stop and repetitive-output artefacts. For watermark detection of the unmodified generated passage, it scores the original generated token IDs and includes the final prompt token as the seed anchor; this avoids decode/re-tokenize mismatches and correctly aligns the first generated token with the watermark's seeding scheme.

Each run writes three persistent files under `results/`: `evaluation_samples_<run>.csv`, `evaluation_attacks_<run>.csv`, and `evaluation_report_<run>.json`. The browser displays the summary and artifact paths. For statistically meaningful conclusions, increase prompt and seed counts and evaluate multiple models. Wilson intervals are computed for binary rates; prompt/seed observations may still be correlated. The lexical proxy is not a substitute for semantic paraphrase evaluation. Use a separately documented paraphrase model and inspect semantic preservation before drawing robustness conclusions.
