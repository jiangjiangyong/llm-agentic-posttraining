# LLM Agentic Post-Training Toolkit

This repository is a reproducible toolkit for adapting language models to tool-use workflows and building reliable multi-step agents.

The project covers:

- model loading, PEFT/QLoRA, SFT, DPO, and a minimal agentic RL path;
- tool schemas, agent runtime, code environments, rewards, traces, and benchmarks;
- typed state, checkpoints, failure classification, recovery, and long-horizon evaluation;
- Transformers/PEFT serving, FastAPI endpoints, health checks, and a small web client.

~~~text
Data / Post-training → Model Adapter → Agent Runtime → State → Tools
    → Checkpoint → Failure Detection → Recovery → Evaluation → Serving
~~~

## Repository layout

- src/llm_posttrain/ — V1 model, runtime, tool, training, evaluation, and serving code
- scripts/ — dataset, training, evaluation, audit, and serving entry points
- configs/ — model and runtime configuration
- tests/ — unit and regression tests
- v2_development/ — the isolated V2 development layer for model and agent engineering
- web/ — a minimal local client for the serving API
- docs/PROJECT_INTRODUCTION.md — detailed project overview and handoff guide

Model weights, datasets, generated outputs, experiment artifacts, credentials, and virtual environments are intentionally kept outside the repository.

## Setup

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/phase01.txt
python -m pip install -e .
~~~

Serving dependencies are optional:

~~~bash
python -m pip install -r requirements-serving.txt
~~~

## Tests

~~~bash
PYTHONPATH=src pytest -q
cd v2_development
python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src python scripts/smoke_cpu.py
~~~

Training and GPU evaluation should only be started after checking the relevant configuration, external model/data paths, CUDA, PyTorch, Transformers, and PEFT versions.

## Local serving

~~~bash
PYTHONPATH=src python scripts/serve_api.py \
  --backend adapter \
  --adapter-path /path/to/adapter \
  --host 127.0.0.1 \
  --port 8000
~~~

The service exposes a health endpoint at /health. Keep model weights, private datasets, credentials, and runtime outputs outside the repository.
