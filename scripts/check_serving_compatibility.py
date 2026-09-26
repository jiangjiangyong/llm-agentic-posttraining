from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    vllm_installed = importlib.util.find_spec("vllm") is not None
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "vllm_installed": vllm_installed,
        "recommended_backend": "transformers_fallback",
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        report.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "cuda_version": torch.version.cuda,
            }
        )
        if vllm_installed and capability >= (8, 0):
            report["recommended_backend"] = "vllm_candidate_after_smoke_test"
            report["reason"] = "vLLM is installed and the GPU is at least compute capability 8.0; run an isolated smoke test before switching."
        elif vllm_installed:
            report["reason"] = "vLLM is installed, but this GPU has an older compute capability; keep Transformers fallback until compatibility is proven."
        else:
            report["reason"] = "vLLM is not installed in the project environment; use the tested Transformers fallback."
    else:
        report["reason"] = "CUDA is unavailable; use CPU or remote GPU before serving the model."

    output_path = Path("artifacts/serving_compatibility.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
