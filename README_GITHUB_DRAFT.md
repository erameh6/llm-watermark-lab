# LLM Watermark Lab

An open-source, self-hosted research platform for studying statistical watermarks in LLM-generated text.

## Quick start (Docker)

    git clone https://github.com/USERNAME/llm-watermark-lab.git
    cd llm-watermark-lab
    cp .env.example .env
    # Edit .env and set a private WATERMARK_RESEARCH_KEY.
    docker compose up --build

Open http://localhost:8000. Interactive API documentation: http://localhost:8000/docs.

The first Docker build installs the CPU research stack and may take several minutes. Model weights are not baked into the image; Hugging Face weights are cached in a persistent Docker volume and downloaded only when a model is explicitly loaded.

## Model providers

- Hugging Face: register a model ID or local directory. Compatible causal LMs can be loaded with Transformers. Actual watermark embedding requires a compatible tokenizer and generation-time logits processing.
- Ollama: discover installed models through the local Ollama service. The standard Ollama generate API is treated as generation-only; no logits-level watermark capability is claimed.
- OpenAI, Anthropic and DeepSeek: configure credentials in the local .env file. Standard chat APIs are treated as comparison/paraphrasing providers, not watermark-embedding providers.

Never commit .env, API keys, watermark secrets, model weights, or private datasets.

## Current status

The Model Manager is an initial working slice: provider catalogue, Hugging Face model registration without automatic download, Ollama connectivity/model discovery, API-key presence checks, and a browser interface. The next integration step is wiring compatible Hugging Face models into the existing Kirchenbauer generation pipeline and adding actual API generation adapters.

## Research attribution

The watermark implementation in upstream/ is based on the authors' official repository for Kirchenbauer et al., *A Watermark for Large Language Models* (2023). See upstream/LICENSE.md and upstream/README.md. Cite the original paper in research outputs.
