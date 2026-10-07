from __future__ import annotations

from pathlib import Path


FROZEN_FINAL_TEST_POLICY = "FROZEN_FINAL_TEST_DO_NOT_MINE_FAILURES"


def is_frozen_final_test(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.name == "final_test.jsonl" or "final_test" in candidate.parts


def assert_no_final_test_input(path: str | Path, purpose: str) -> None:
    if is_frozen_final_test(path):
        raise PermissionError(
            f"{purpose} refuses frozen final test input {path}; "
            "final_test is evaluation-only and cannot enter training, preference, or failure mining."
        )


def assert_final_test_eval_allowed(path: str | Path, acknowledged: bool) -> None:
    if is_frozen_final_test(path) and not acknowledged:
        raise PermissionError(
            f"Formal evaluation of frozen final test {path} requires --allow-final-test-eval."
        )
