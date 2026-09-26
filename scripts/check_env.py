from __future__ import annotations

import importlib
import platform
import sys


def main() -> int:
    print("=== Runtime ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")

    required = [
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "safetensors",
        "yaml",
        "tqdm",
        "pytest",
    ]
    failed = []

    print("=== Libraries ===")
    for module_name in required:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"{module_name}: {version}")
        except Exception as exc:
            failed.append(module_name)
            print(f"{module_name}: ERROR {type(exc).__name__}: {exc}")

    import torch

    print("=== PyTorch ===")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(
                f"GPU {index}: {props.name}; "
                f"VRAM={props.total_memory / 1024**3:.2f} GiB; "
                f"compute_capability={props.major}.{props.minor}"
            )
        torch.zeros(1, device="cuda")
        print("cuda smoke: PASS")

    if failed:
        print(f"Required imports failed: {', '.join(failed)}")
        return 1
    print("ENV_CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
