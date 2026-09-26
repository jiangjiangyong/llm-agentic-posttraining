from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/evaluation/base_smoke.jsonl")
    parser.add_argument("--output", default="artifacts/eval_manifest.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    content = dataset_path.read_bytes()
    records = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = {
        "dataset": dataset_path.as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "num_samples": len(records),
        "purpose": "baseline evaluation only; never use for training",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
