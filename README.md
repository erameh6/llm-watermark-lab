# LLM Text Watermarking Research

Research workspace for reproducing and evaluating the statistical watermark from:

- Kirchenbauer et al., *A Watermark for Large Language Models* (2023): https://arxiv.org/abs/2301.10226
- Official author implementation: [`upstream/`](upstream/) — https://github.com/jwkirchenbauer/lm-watermarking

## Project goals

1. Reproduce a baseline soft watermark with a fixed, documented configuration.
2. Generate watermarked and unwatermarked text using the same model/tokenizer.
3. Detect watermarks with the matching secret key and report statistical scores.
4. Evaluate false positives, detection power, text quality, length sensitivity and robustness to edits.
5. Compare one carefully scoped improvement against the baseline.

## Server notes

This workspace is on an Ubuntu VPS with Python 3.12, approximately 11 GiB RAM and no NVIDIA GPU detected. Start with a small causal language model and CPU inference. Do not attempt to load multi-billion-parameter models on this machine without checking memory requirements first. For larger models, use a remote GPU runtime and keep experiment outputs here.

## Layout

- `upstream/` — unmodified clone of the authors' reference implementation.
- `src/` — project-specific experiment and analysis code.
- `configs/` — versioned experiment configurations.
- `data/` — datasets; keep private or licensed source data out of Git.
- `results/`, `figures/` — metrics and plots.
- `notebooks/` — exploratory analysis.
- `docs/` — research notes and methodology.
- `tests/` — reproducibility and unit tests.

## Environment setup

```bash
cd ~/llm-text-watermarking
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CPU-only PyTorch on this VPS, install from the official CPU wheel index if the default PyPI wheel attempts to pull CUDA packages:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## First baseline configuration

See [`configs/baseline.yaml`](configs/baseline.yaml). The initial reference settings are gamma=0.25, delta=2.0, `selfhash` seeding and repeated n-gram filtering at detection, following the authors' updated implementation guidance. Record the exact model, tokenizer revision, library versions, random seeds and generation settings for every run.

## Important experimental controls

- Keep the model and tokenizer identical between watermarked and control generation.
- Use separate prompts or a pre-registered split for tuning and final evaluation.
- Evaluate human-written and unwatermarked model text as negative controls.
- Report confidence intervals and false-positive rates, not accuracy alone.
- Test multiple text lengths; very short samples have limited statistical power.
- Treat detection as statistical evidence of a particular watermark/key, not proof of a person's authorship.
- Never publish the secret key used for any deployment-style demonstration. Use a research-only key in experiments and document key handling.

## Status

- Workspace created at `/home/ubuntu/llm-text-watermarking`.
- Official author repository cloned into `upstream/`.
- Python 3.12 virtual environment created at `.venv/`.
- CPU-only PyTorch and core research dependencies installed.
- `scripts/smoke_test_watermark.py` added for a quick generation/detection integration check.
- `scripts/run_experiment.py` added for paired watermarked/control generation, detection metrics, confidence intervals, length sensitivity, perturbation attacks and reference-model perplexity.
- `scripts/plot_results.py` added for detector, length and attack plots.
- Unit tests cover metrics and deterministic perturbations.
- A small end-to-end run completed successfully (2 prompts, 1 seed, 32/64 generated-token settings; 8 samples total). The report and CSV files are saved under `results/`.

## Run experiments

```bash
cd ~/llm-text-watermarking
source .venv/bin/activate
source .env  # private key is stored locally and excluded from Git

# Quick generation/detection check
python scripts/smoke_test_watermark.py --max-new-tokens 96

# Small end-to-end experiment
python scripts/run_experiment.py --prompts 4 --seeds 11,22 --lengths 48,96

# Generate plots from a report
python scripts/plot_results.py results/report_<RUN_ID>.json

# Run tests
python -m pytest -q
```

Useful options: `--prompts 1..20`, `--seeds 11,22,33`, `--lengths 48,96,160`, `--skip-perplexity`, and `--attacks none,word_deletion,word_substitution,lexical_paraphrase_proxy`.

The lexical paraphrase option is a synonym-substitution proxy, not a true semantic paraphrasing attack. Perplexity is a limited reference-model quality proxy. Use larger held-out datasets and a real paraphrasing system before making research claims.

The VPS has no NVIDIA GPU detected, so inference currently runs on CPU. Start with small batches; use a GPU host for larger models and full-scale evaluations.

## Semantic paraphrasing robustness experiment

A separate, model-based attack is implemented in `scripts/run_semantic_paraphrase.py`. It uses `Vamsi/T5_Paraphrase_Paws` to generate paraphrases and `sentence-transformers/all-MiniLM-L6-v2` embeddings to estimate semantic similarity. The MiniLM score is an automatic proxy and does not replace human review.

```bash
cd ~/llm-text-watermarking
source .venv/bin/activate
source .env
python scripts/run_semantic_paraphrase.py \
  --input results/samples_<BASELINE_RUN_ID>.csv \
  --batch-size 4

python scripts/plot_semantic_paraphrase.py \
  results/semantic_paraphrase_<RUN_ID>.csv
```

The semantic attack script reads already-generated watermarked samples, paraphrases eligible texts (at least 12 whitespace-delimited words), re-runs the watermark detector, and reports semantic cosine similarity and lexical Jaccard overlap. It does not regenerate unwatermarked controls. Empty model outputs are counted as failures in the all-output detection rate and excluded from valid-output semantic similarity and paired comparisons.

Completed CPU run (1 Nov 2026): 114 eligible source samples; 113 non-empty paraphrase outputs. On valid pairs, watermark detection decreased from 82.30% before paraphrasing to 59.29% after paraphrasing. Mean MiniLM cosine similarity was 0.875 and mean lexical Jaccard overlap was 0.720. These are preliminary results from DistilGPT-2 outputs and one paraphrasing model, not publication-grade evidence; inspect generated examples and conduct human semantic-preservation review.

Artifacts: `results/semantic_paraphrase_20261101T150421Z.csv`, `results/semantic_paraphrase_report_20261101T150421Z.json`, and `results/figures/semantic_paraphrase/`.

## Model Manager (initial implementation)

The self-hosted Model Manager is available in the browser when the backend is running.

Install the lightweight web dependencies and start the service:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r backend/requirements.txt
    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000. API documentation is available at http://127.0.0.1:8000/docs.

Supported provider registrations:
- Hugging Face: register a Hub model ID or local model directory. Registration does not download weights; the model loader is an explicit Python operation.
- Ollama: check the local Ollama service and verify that a named model is installed.
- OpenAI, Anthropic and DeepSeek: check whether the relevant API key is configured in the server environment. Keys are never returned by the API.

Watermark capability is reported separately from text-generation capability. The Kirchenbauer method requires access to token-generation logits and a compatible tokenizer. Standard hosted chat APIs and Ollama's ordinary generate endpoint are not advertised as watermark-embedding providers.

### Docker quick start

    cp .env.example .env
    # Set a private WATERMARK_RESEARCH_KEY in .env
    docker compose up --build

Then open http://localhost:8000. Model weights are stored in a persistent Docker volume, not in the image. The first image build is large because it installs the CPU research stack.

The Model Manager is an initial integration slice. It currently provides provider discovery, registration/validation and a browser UI. The next implementation step is connecting the Hugging Face loader to the existing Kirchenbauer generation pipeline and adding working generation adapters for comparison-only API providers.


## Interactive watermark workflow

After loading a compatible Hugging Face causal LM in Model Manager, use **Kirchenbauer watermarking** to generate watermarked text or score a passage. The implementation uses the upstream Apache-2.0 `WatermarkLogitsProcessor` and `WatermarkDetector` from `upstream/`. Keep the same model/tokenizer, gamma, delta and private `WATERMARK_RESEARCH_KEY` for both operations. Detection is statistical evidence, not proof of authorship; short or repetitive passages may be inconclusive. See [Model Manager documentation](docs/MODEL_MANAGER.md).

## Running robustness studies

Use the **Robustness evaluation** panel after loading a Hugging Face model. It creates paired watermarked/control generations and applies deterministic word deletion, substitution, and a clearly labelled lexical paraphrase proxy. Results are saved persistently in `results/` as CSV and JSON.

For a neural semantic-paraphrase follow-up on the generated sample CSV, run (inside the Docker container):

```bash
docker compose exec web python scripts/run_platform_semantic_paraphrase.py \\
  --input results/evaluation_samples_RUN_ID.csv \\
  --model distilgpt2 --gamma 0.5 --delta 2.0 --z-threshold 4.0
```

This downloads/uses `Vamsi/T5_Paraphrase_Paws` and `sentence-transformers/all-MiniLM-L6-v2`; first execution may take several minutes on CPU. It reports detection before/after paraphrasing plus automatic semantic-cosine and lexical-overlap proxies. Inspect the CSV examples and do not treat embedding cosine as a human semantic judgement.

## GitHub Codespaces

See [README_CODESPACES.md](README_CODESPACES.md) for one-click Codespaces setup. The Codespaces-specific Compose override exposes port 8000 to the Codespaces port forwarder; the normal VPS Compose file remains bound to 127.0.0.1.
