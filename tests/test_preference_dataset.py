import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from build_preference_dataset import make_rule_negative, score_candidate, split_stratified
from create_preference_source import build_records


def test_source_gold_and_rule_negative_quality() -> None:
    records = build_records(16, seed=7)
    for index, record in enumerate(records):
        chosen = record["gold"]["response"]
        rejected = make_rule_negative(record, index)
        assert score_candidate(record, chosen)["passed"] is True
        assert score_candidate(record, rejected)["passed"] is False


def test_preference_split_is_stratified() -> None:
    records = build_records(32, seed=11)
    pairs = [
        {
            "id": record["id"],
            "category": record["category"],
            "prompt": record["messages"],
            "chosen": record["gold"]["response"],
            "rejected": make_rule_negative(record, index),
        }
        for index, record in enumerate(records)
    ]
    train, valid = split_stratified(pairs, ratio=0.2, seed=11)
    assert len(train) + len(valid) == len(pairs)
