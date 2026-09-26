from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


def has_model_files(path: Path) -> bool:
    has_config = (path / "config.json").exists()
    has_tokenizer = (path / "tokenizer_config.json").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and has_tokenizer and has_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--local-dir", default="models/base/Qwen3-1.7B-Base")
    parser.add_argument("--cache-dir", default="models/cache/modelscope")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    cache_dir = Path(args.cache_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if has_model_files(local_dir) and not args.force:
        print(f"Model already exists: {local_dir}")
        return

    print(f"Downloading {args.repo_id} via ModelScope")
    resolved = snapshot_download(
        model_id=args.repo_id,
        cache_dir=str(cache_dir),
        local_dir=str(local_dir),
        max_workers=args.max_workers,
    )
    print(f"Model path: {resolved}")
    print(f"Model files complete: {has_model_files(local_dir)}")


if __name__ == "__main__":
    main()
