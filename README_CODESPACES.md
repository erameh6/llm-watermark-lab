# Run LLM Watermark Lab in GitHub Codespaces

1. Open the repository on GitHub.
2. Select **Code → Codespaces → Create codespace on main**.
3. Wait for the development container to build. The first build installs CPU-only PyTorch and research dependencies and can take several minutes.
4. Codespaces forwards port **8000** and opens the application in a browser. If it does not open automatically, open the **Ports** tab and visit port 8000.
5. API documentation is available at `/docs`.

## Configuration

- FastAPI runs in the Docker Compose `web` service.
- Port 8000 is forwarded privately by default.
- A private `WATERMARK_RESEARCH_KEY` is generated locally in the Codespace before Compose starts.
- `.env` is excluded from Git and has owner-only permissions.
- Hugging Face model downloads use a named Docker volume.
- Experiment CSV/JSON output is written to `results/` in the workspace.
- Unit tests run after the Codespace is created.

## Resource guidance

Use a Codespace with at least **4 cores and 16 GB RAM** for practical CPU model loading and small experiments. Smaller configurations may run the API and detector but can run out of memory when loading models. Larger models and neural paraphrasing are substantially faster with a GPU; standard Codespaces CPU machines do not provide a CUDA GPU.

The first image build is large because the project installs CPU PyTorch and research dependencies. The first model run also downloads model weights. Both can take time and consume Codespaces storage.

## Secrets and model providers

Never commit `.env`. The Codespace creates a random research key automatically. For hosted provider integrations or gated Hugging Face models, use Codespaces secrets and expose them only to the Codespace when needed. Never put API keys in source files, notebooks, screenshots, or experiment reports.

The current provider catalogue includes OpenAI, Anthropic and DeepSeek, but hosted-provider generation adapters are not fully implemented. Hugging Face causal language models are the supported watermark-generation route because the algorithm needs generation-time logits. Standard Ollama and hosted chat endpoints are not assumed to expose those logits.

## Persistence and lifecycle

A Codespace is a development environment, not a permanent public server. Stop unused Codespaces and push important source changes to GitHub. Back up experiment outputs you want to keep; do not rely on a stopped or deleted Codespace as the only copy.

## Local Docker alternative

```bash
cp .env.example .env
# Set WATERMARK_RESEARCH_KEY to a private integer if running manually.
docker compose up --build
```

Then open http://localhost:8000.
