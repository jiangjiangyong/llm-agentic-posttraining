from pathlib import Path

from llm_posttrain.config import get_required, load_yaml


def test_load_base_config() -> None:
    config = load_yaml(Path("configs/base_model.yaml"))
    assert config["model"]["repo_id"] == "Qwen/Qwen3-1.7B-Base"
    assert get_required(config, "evaluation", "dataset_path") == "data/evaluation/base_smoke.jsonl"


def test_missing_config_key() -> None:
    try:
        get_required({}, "model", "local_path")
    except KeyError as exc:
        assert "model.local_path" in str(exc)
    else:
        raise AssertionError("Expected missing config key to raise")
