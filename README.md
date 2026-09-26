# LLM Agentic Post-Training Toolkit

A Python toolkit for building, training, and evaluating tool-using language-model workflows.

## Structure

- `src/llm_posttrain/` — runtime, tools, reward calculation, training, evaluation, and serving code
- `scripts/` — dataset preparation, training, evaluation, and local serving entry points
- `configs/` — model and runtime configuration
- `tests/` — unit and integration tests
- `web/` — minimal browser client for the local serving API

Model weights, datasets, generated outputs, and experiment artifacts are intentionally kept outside
the repository. Configure their paths before running training or evaluation commands.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/phase01.txt
python -m pip install -e .
```

Install the serving dependencies only when the local API is needed:

```bash
python -m pip install -r requirements-serving.txt
```

## Tests

```bash
pytest -q
```

## Local serving

```bash
PYTHONPATH=src python scripts/serve_api.py \
  --backend adapter \
  --adapter-path /path/to/adapter \
  --host 127.0.0.1 \
  --port 8000
```

The service exposes a health endpoint at `/health`. Keep credentials, model weights, and private
datasets outside the repository.
