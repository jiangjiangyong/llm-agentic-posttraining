from __future__ import annotations

from scripts.run_code_agent_showcase import run_showcase


def test_showcase_runs_real_environment_and_resolver() -> None:
    result = run_showcase()

    assert result.semantic_success is True
    assert result.strict_success is True
    assert result.workspace_files == ["research/showcase.md"]
    resolution = result.steps[1]["content_ref_resolutions"][0]
    assert resolution["applied"] is True
