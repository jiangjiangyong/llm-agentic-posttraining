from __future__ import annotations

import argparse

from llm_posttrain.config import load_yaml
from llm_posttrain.models.adapter_runner import build_adapter_runner
from llm_posttrain.models.loader import build_runner
from llm_posttrain.serving.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LLM Agentic Runtime API.")
    parser.add_argument("--model-config", default="configs/base_model.yaml")
    parser.add_argument("--backend", choices=("base", "adapter"), default="adapter")
    parser.add_argument("--adapter-path", default="models/adapters/sft_qwen3_1.7b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--web-dir", default="web")
    args = parser.parse_args()

    config = load_yaml(args.model_config)
    if args.backend == "base":
        runner = build_runner(config)
        runner.max_new_tokens = args.max_new_tokens
        adapter_path = None
    else:
        runner = build_adapter_runner(
            config,
            args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        adapter_path = args.adapter_path

    app = create_app(
        runner,
        backend_name=args.backend,
        model_path=str(config["model"]["local_path"]),
        adapter_path=adapter_path,
        max_steps=args.max_steps,
        web_dir=args.web_dir,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
